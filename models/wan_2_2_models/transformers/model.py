# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from models.wan_2_2_models.transformers.attention import flash_attention, attention


__all__ = ['WanModel']


def _device_type(tensor_or_device):
    """Return the device type string ('xpu', 'cuda', 'cpu') for autocast."""
    if isinstance(tensor_or_device, torch.device):
        return tensor_or_device.type
    if isinstance(tensor_or_device, torch.Tensor):
        return tensor_or_device.device.type
    return str(tensor_or_device)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    if grid_sizes.shape[-1] == 3:
        # split freqs
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, shape in enumerate(grid_sizes.tolist()):
        if len(shape)==3:
            f,h,w = shape
            seq_len = f * h * w

            # precompute multipliers
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
                seq_len, n, -1, 2))
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                                dim=-1).reshape(seq_len, 1, -1)
        else:
            seq_len = int(shape[0])
            # precompute multipliers
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
                seq_len, n, -1, 2))
            freqs_i = freqs[:seq_len].unsqueeze(1)  # shape l,1,d
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_precomputed_1d(x, freqs):
    seq_len = freqs.size(0)
    x_main = torch.view_as_complex(
        x[:, :seq_len].to(torch.float64).reshape(x.size(0), seq_len, x.size(2), -1, 2)
    )
    x_main = torch.view_as_real(x_main * freqs.view(1, seq_len, 1, -1)).flatten(3)
    x_main = x_main.float()

    if seq_len == x.size(1):
        return x_main
    return torch.cat([x_main, x[:, seq_len:]], dim=1)

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 cross_attn_dim=None,
                 fused_qkv=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        if cross_attn_dim is None:
            cross_attn_dim = dim

        # layers
        # self.q = nn.Linear(dim, dim)
        # self.k = nn.Linear(cross_attn_dim, dim)
        # self.v = nn.Linear(cross_attn_dim, dim)

        self.cross_attn_dim = cross_attn_dim
        self.fused_qkv = fused_qkv and cross_attn_dim == dim
        # layers
        if self.fused_qkv:
            self.qkv = nn.Linear(dim, dim * 3)
            self.q = None
            self.k = None
            self.v = None
        else:
            self.qkv = None
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(cross_attn_dim, dim)
            self.v = nn.Linear(cross_attn_dim, dim)

        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.fused_qkv:
            q_weight_key = prefix + "q.weight"
            k_weight_key = prefix + "k.weight"
            v_weight_key = prefix + "v.weight"
            q_bias_key = prefix + "q.bias"
            k_bias_key = prefix + "k.bias"
            v_bias_key = prefix + "v.bias"
            qkv_weight_key = prefix + "qkv.weight"
            qkv_bias_key = prefix + "qkv.bias"

            if (
                qkv_weight_key not in state_dict
                and q_weight_key in state_dict
                and k_weight_key in state_dict
                and v_weight_key in state_dict
            ):
                state_dict[qkv_weight_key] = torch.cat(
                    [
                        state_dict.pop(q_weight_key),
                        state_dict.pop(k_weight_key),
                        state_dict.pop(v_weight_key),
                    ],
                    dim=0,
                )

            if (
                qkv_bias_key not in state_dict
                and q_bias_key in state_dict
                and k_bias_key in state_dict
                and v_bias_key in state_dict
            ):
                state_dict[qkv_bias_key] = torch.cat(
                    [
                        state_dict.pop(q_bias_key),
                        state_dict.pop(k_bias_key),
                        state_dict.pop(v_bias_key),
                    ],
                    dim=0,
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, seq_lens, grid_sizes, freqs, precomputed_freqs=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            if self.fused_qkv:
                q, k, v = self.qkv(x).chunk(3, dim=-1)
                q = self.norm_q(q).view(b, s, n, d)
                k = self.norm_k(k).view(b, s, n, d)
                v = v.view(b, s, n, d)
            else:
                q = self.norm_q(self.q(x)).view(b, s, n, d)
                k = self.norm_k(self.k(x)).view(b, s, n, d)
                v = self.v(x).view(b, s, n, d)

            return q, k, v

        q, k, v = qkv_fn(x)

        if precomputed_freqs is None:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
        else:
            q = rope_apply_precomputed_1d(q, precomputed_freqs)
            k = rope_apply_precomputed_1d(k, precomputed_freqs)
        x = attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def compute_kv(self, context):
        b, n, d = context.size(0), self.num_heads, self.head_dim
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        return k, v

    def forward(self, x, context, context_lens, kv_cache=None, return_kv=False):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if kv_cache is None:
            k, v = self.compute_kv(context)
        else:
            k, v = kv_cache

        # compute attention
        x = attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)

        if return_kv:
            return x, (k, v)

        return x


class WanAttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 cross_attn_dim=None,
                 fused_self_attn_qkv=True,
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            fused_qkv=fused_self_attn_qkv,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps, cross_attn_dim=cross_attn_dim)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        cross_attn_kv_cache=None,
        return_cross_attn_kv=False,
        self_attn_precomputed_freqs=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast(_device_type(e), dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs, precomputed_freqs=self_attn_precomputed_freqs)
        with torch.amp.autocast(_device_type(x), dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        new_cross_attn_kv = None
        if return_cross_attn_kv:
            cross_attn_out, new_cross_attn_kv = self.cross_attn(
                self.norm3(x),
                context,
                context_lens,
                kv_cache=cross_attn_kv_cache,
                return_kv=True,
            )
        else:
            cross_attn_out = self.cross_attn(
                self.norm3(x),
                context,
                context_lens,
                kv_cache=cross_attn_kv_cache,
                return_kv=False,
            )
        x = x + cross_attn_out

        y = self.ffn(
            self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast(_device_type(x), dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
        if return_cross_attn_kv:
            return x, new_cross_attn_kv
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast(_device_type(x), dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 max_seq_len=1024,
                 use_ae=False,
                 action_in_dim=16,
                 action_dim=1536,
                 action_num_heads=24,
                 action_ffn_dim=4096,
                 action_max_seq_len=60,
                 fused_self_attn_qkv=True,
                 enable_action_cross_attn_kv_cache=False,
                 enable_action_rope_cache=True,):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.fused_self_attn_qkv = fused_self_attn_qkv
        self.enable_action_cross_attn_kv_cache = enable_action_cross_attn_kv_cache
        self.enable_action_rope_cache = enable_action_rope_cache

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps, fused_self_attn_qkv=fused_self_attn_qkv) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(max_seq_len, d - 4 * (d // 6)),
            rope_params(max_seq_len, 2 * (d // 6)),
            rope_params(max_seq_len, 2 * (d // 6))
        ],
                               dim=1)
        
        self.use_ae = use_ae
        if self.use_ae:
            self.action_dim = action_dim
            self.action_proj_in = nn.Linear(action_in_dim, action_dim)
            self.action_blocks = nn.ModuleList([
                WanAttentionBlock(action_dim, action_ffn_dim, action_num_heads, window_size, qk_norm,
                                cross_attn_norm, eps, cross_attn_dim=dim, fused_self_attn_qkv=fused_self_attn_qkv) for _ in range(num_layers)
            ])
            self.action_time_embedding = nn.Sequential(
                nn.Linear(freq_dim, action_dim), nn.SiLU(), nn.Linear(action_dim, action_dim))
            self.action_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(action_dim, action_dim * 6))
            
            assert (action_dim % action_num_heads) == 0 and (action_dim // action_num_heads) % 2 == 0
            action_d = action_dim // action_num_heads
            self.action_freqs = torch.cat([
                rope_params(action_max_seq_len, action_d),
            ],
                                dim=1)
            self.action_head = Head(action_dim, action_in_dim, [1], eps)
        self.gradient_checkpointing = False

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        action_states: torch.Tensor = None,
        action_timestep: torch.LongTensor = None,
        return_video: bool = True,
        return_action: bool = False,
        store_buffer=False,
        video_states_buffer=None,
        action_context_kv_cache=None,
        history_action_state: torch.Tensor = None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        device = self.patch_embedding.weight.device
        if return_video or store_buffer:
            if store_buffer:
                video_states_buffer = []
            # params
            if self.freqs.device != device:
                self.freqs = self.freqs.to(device)

            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            # embeddings
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            assert seq_lens.max() <= seq_len
            x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                        dim=1) for u in x
            ])

            # time embeddings
            if t.dim() == 1:
                t = t.expand(t.size(0), seq_len)
            with torch.amp.autocast(_device_type(device), dtype=torch.float32):
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # breakpoint()
            # context
            context_lens = None
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
            
            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens)
            
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, **kwargs)

                return custom_forward

        if return_action:
            if self.action_freqs.device != device:
                self.action_freqs = self.action_freqs.to(device)
            assert self.use_ae
            if not video_states_buffer:
                assert store_buffer or return_video

            if self.enable_action_cross_attn_kv_cache and action_context_kv_cache is None and store_buffer:
                action_context_kv_cache = [None for _ in range(len(self.action_blocks))]

            if history_action_state is not None:
                action_states = torch.cat((history_action_state, action_states), dim=1)
                action_timestep = torch.cat((torch.zeros_like(action_timestep[:,0:1]), action_timestep), dim=1)
            action_states = self.action_proj_in(action_states)
            action_seq_len = action_states.shape[1]
            with torch.amp.autocast(_device_type(device), dtype=torch.float32):
                action_bt = action_timestep.size(0)
                action_timestep = action_timestep.flatten()
                action_e = self.action_time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            action_timestep).unflatten(0, (action_bt, action_seq_len)).float())
                action_e0 = self.action_time_projection(action_e).unflatten(2, (6, self.action_dim))
                assert action_e.dtype == torch.float32 and action_e0.dtype == torch.float32
            action_seq_lens = torch.tensor([action_seq_len for _ in range(action_bt)])
            action_grid_sizes = torch.stack(
                [torch.tensor([action_states.shape[1]], dtype=torch.long) for _ in action_states])
            action_precomputed_rope_freqs = None
            if self.enable_action_rope_cache:
                action_precomputed_rope_freqs = self.action_freqs[:action_seq_len]

            def create_action_custom_forward(module):
                def action_custom_forward(*inputs):
                    return module(*inputs)

                return action_custom_forward


        for block_idx, block in enumerate(self.blocks):
            if self.training and self.gradient_checkpointing:
                if return_video or store_buffer:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        use_reentrant=False,
                    )
                    if store_buffer:
                        video_states_buffer.append(x.clone())
                else:
                    x = video_states_buffer[block_idx]
                if return_action:
                    action_states = torch.utils.checkpoint.checkpoint(
                        create_action_custom_forward(self.action_blocks[block_idx]),
                        action_states,
                        action_e0,
                        action_seq_lens,
                        action_grid_sizes,
                        self.action_freqs,
                        x,
                        None
                    )
            else:
                if return_video or store_buffer:
                    x = block(x, **kwargs)
                    if store_buffer:
                        video_states_buffer.append(x.clone())
                else:
                    x = video_states_buffer[block_idx]

                if return_action:
                    cached_kv = None if action_context_kv_cache is None else action_context_kv_cache[block_idx]
                    should_build_kv_cache = (
                        self.enable_action_cross_attn_kv_cache and store_buffer and cached_kv is None
                    )

                    if should_build_kv_cache:
                        action_states, cached_kv = self.action_blocks[block_idx](
                            x=action_states,
                            e=action_e0,
                            seq_lens=action_seq_lens,
                            grid_sizes=action_grid_sizes,
                            freqs=self.action_freqs,
                            context=x,
                            context_lens=None,
                            cross_attn_kv_cache=None,
                            return_cross_attn_kv=True,
                            self_attn_precomputed_freqs=action_precomputed_rope_freqs,
                        )
                        action_context_kv_cache[block_idx] = cached_kv
                    else:
                        action_states = self.action_blocks[block_idx](
                            x=action_states,
                            e=action_e0,
                            seq_lens=action_seq_lens,
                            grid_sizes=action_grid_sizes,
                            freqs=self.action_freqs,
                            context=x,
                            context_lens=None,
                            cross_attn_kv_cache=cached_kv,
                            return_cross_attn_kv=False,
                            self_attn_precomputed_freqs=action_precomputed_rope_freqs,
                        )
        # for block in self.blocks:
        #     x = block(x, **kwargs)
        final_output = {}
        if store_buffer:
            final_output['video_states_buffer'] = video_states_buffer
            if action_context_kv_cache is not None:
                final_output['action_context_kv_cache'] = action_context_kv_cache
        
        if return_video:
            # head
            x = self.head(x, e)

            # unpatchify
            x = self.unpatchify(x, grid_sizes)
            final_output['video'] = [u.float() for u in x]
        
        if return_action:
            action_states = self.action_head(action_states, action_e)
            if history_action_state is not None:
                action_states = action_states[:, 1:]
            
            final_output['action'] = action_states
        
        return final_output

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
