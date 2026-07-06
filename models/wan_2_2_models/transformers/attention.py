# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

from contextlib import nullcontext
from typing import Optional
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
    print("Flash attention 3 is available")
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False
    print("Flash attention 3 is not available")

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
    print("Flash attention 2 is available")
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
    print("Flash attention 2 is not available")

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'mot_attention',
]

_ATTENTION_IMPL = "auto"
_SDPA_BACKEND = "auto"
_FLASH_ATTN_VERSION = "auto"


def set_attention_backend(attention_impl="auto", sdpa_backend="auto", flash_attn_version="auto"):
    global _ATTENTION_IMPL, _SDPA_BACKEND, _FLASH_ATTN_VERSION
    _ATTENTION_IMPL = attention_impl
    _SDPA_BACKEND = sdpa_backend
    _FLASH_ATTN_VERSION = str(flash_attn_version)


def _resolve_sdpa_backend(name):
    if name in (None, "auto"):
        return None

    attention_module = getattr(torch.nn, "attention", None)
    if attention_module is None or not hasattr(attention_module, "SDPBackend"):
        return None

    backend_enum = attention_module.SDPBackend
    name_map = {
        "flash": "FLASH_ATTENTION",
        "efficient": "EFFICIENT_ATTENTION",
        "math": "MATH",
        "cudnn": "CUDNN_ATTENTION",
    }
    attr = name_map.get(name)
    if attr is None or not hasattr(backend_enum, attr):
        return None
    return getattr(backend_enum, attr)


def _sdpa_context():
    backend = _resolve_sdpa_backend(_SDPA_BACKEND)
    if backend is None:
        return nullcontext()

    attention_module = getattr(torch.nn, "attention", None)
    if attention_module is None or not hasattr(attention_module, "sdpa_kernel"):
        warnings.warn(
            "Requested SDPA backend is unavailable in this PyTorch build; falling back to auto."
        )
        return nullcontext()
    return attention_module.sdpa_kernel(backend)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type in ('cuda', 'xpu') and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            #deterministic=deterministic)[0].unflatten(0, (b, lq))
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=None,
    fa_version=None,
):
    use_sdpa = _ATTENTION_IMPL == "sdpa"
    use_flash_attn = _ATTENTION_IMPL == "flash_attn"

    flash_attn_available = FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE
    requested_fa_version = fa_version if fa_version is not None else _FLASH_ATTN_VERSION
    if requested_fa_version == "auto":
        requested_fa_version = None
    elif requested_fa_version is not None:
        requested_fa_version = int(requested_fa_version)

    requested_fa_available = flash_attn_available
    if requested_fa_version == 2:
        requested_fa_available = FLASH_ATTN_2_AVAILABLE
    elif requested_fa_version == 3:
        requested_fa_available = FLASH_ATTN_3_AVAILABLE

    if not use_sdpa and (
        (use_flash_attn and requested_fa_available)
        or (_ATTENTION_IMPL == "auto" and requested_fa_available)
    ):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=requested_fa_version,
        )
    else:
        if use_flash_attn and not requested_fa_available:
            warnings.warn(
                "Requested flash_attn version is unavailable. Falling back to SDPA."
            )
        elif use_flash_attn and not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
            warnings.warn(
                "Requested flash_attn implementation, but flash_attn is unavailable. Falling back to SDPA."
            )
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        # Use input tensor dtype if no explicit dtype given (preserves float32 on non-CUDA devices)
        effective_dtype = dtype if dtype is not None else q.dtype
        q = q.transpose(1, 2).to(effective_dtype)
        k = k.transpose(1, 2).to(effective_dtype)
        v = v.transpose(1, 2).to(effective_dtype)

        if q_scale is not None:
            q = q * q_scale

        with _sdpa_context():
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out

