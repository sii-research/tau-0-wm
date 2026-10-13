# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Regression tests for the device-agnostic (XPU/CUDA/CPU) support review.

Covers the four compatibility issues reported by the maintainers on the
"Add Intel XPU support with device-agnostic backend selection" PR:

1. flash_attention(): ``dtype=None`` must resolve to a half dtype instead of
   tripping ``assert dtype in half_dtypes`` (breaks the auto/flash_attn paths).
2. attention(): the SDPA path must stay dtype-compatible with the output
   projection when autocast is disabled. RoPE upcasts q/k to float32 while the
   model weights (and the value projection) are bfloat16, so the compute dtype
   must anchor on ``v`` and not on the float32 query.
3. Wan2_2_VAE / Wan2_2_VAE_Batch: with ``device=None`` the resolved device must
   also be used for the normalization buffers and model placement (the original
   ``None`` left everything on the CPU).
4. WanTI2V: an explicitly passed device (the ``--device`` flag of the inference
   servers) must take precedence over independent auto-detection, so the
   pipeline and the models always land on the same device.

All tests are device-agnostic: they run on CPU and, when available, on the
first XPU or CUDA device.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.wan_2_2_models.transformers import attention as attention_module
from models.wan_2_2_models.transformers.attention import attention, flash_attention
from models.wan_2_2_models.transformers.model import WanSelfAttention

_ATTENTION_IMPL_ATTR = "_ATTENTION_IMPL"


def best_device():
    """First XPU/CUDA device if available, else CPU (mirrors auto-detection)."""
    if torch.xpu.is_available():
        return torch.device("xpu", 0)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


class FlashAttentionDtypeResolutionTests(unittest.TestCase):
    """Issue 1: dtype=None must not trip the half-dtype assertion."""

    def setUp(self):
        self.device = best_device()
        self._patchers = [
            mock.patch.object(attention_module, "FLASH_ATTN_2_AVAILABLE", True),
            mock.patch.object(attention_module, "FLASH_ATTN_3_AVAILABLE", False),
            mock.patch.object(attention_module, _ATTENTION_IMPL_ATTR, "auto"),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patchers])

    def _install_fake_flash_attn(self, recorder):
        class _Module:
            flash_attn_varlen_func = staticmethod(
                lambda q, k, v, **kw: recorder["fn"](q, k, v))
        patcher = mock.patch.object(attention_module, "flash_attn",
                                    _Module, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    @unittest.skipIf(best_device().type == "cpu",
                     "flash_attention asserts a cuda/xpu device")
    def test_flash_attention_resolves_none_dtype(self):
        recorded = {}

        def fake_varlen(q, k, v, **kwargs):
            recorded["dtypes"] = (q.dtype, k.dtype, v.dtype)
            return q  # [B*Lq, N, C]

        self._install_fake_flash_attn({"fn": fake_varlen})

        b, l, n, c = 1, 8, 2, 32
        q = torch.randn(b, l, n, c, device=self.device, dtype=torch.float32)
        k = torch.randn(b, l, n, c, device=self.device, dtype=torch.float32)
        v = torch.randn(b, l, n, c, device=self.device, dtype=torch.float32)

        # Regression: this used to raise AssertionError (None not in half dtypes).
        out = flash_attention(q, k, v, dtype=None)

        self.assertEqual(out.shape, (b, l, n, c))
        # Output comes back in the caller's dtype (original q dtype), matching
        # the pre-XPU CUDA behaviour.
        self.assertEqual(out.dtype, torch.float32)
        # The kernels ran in half precision.
        for d in recorded["dtypes"]:
            self.assertIn(d, (torch.float16, torch.bfloat16))

    @unittest.skipIf(best_device().type == "cpu",
                     "flash_attention asserts a cuda/xpu device")
    def test_attention_auto_impl_with_dtype_none_routes_to_flash(self):
        def fake_varlen(q, k, v, **kwargs):
            return q

        self._install_fake_flash_attn({"fn": fake_varlen})

        b, l, n, c = 1, 8, 2, 32
        q = torch.randn(b, l, n, c, device=self.device, dtype=torch.float32)
        k = torch.randn(b, l, n, c, device=self.device, dtype=torch.float32)
        v = torch.randn(b, l, n, c, device=self.device, dtype=torch.float32)

        # Default _ATTENTION_IMPL is "auto": with flash available this used to
        # crash on the assert before ever reaching the kernel.
        out = attention(q, k, v, dtype=None)
        self.assertEqual(out.shape, (b, l, n, c))
        self.assertEqual(out.dtype, torch.float32)


class SdpaDtypeCompatibilityTests(unittest.TestCase):
    """Issue 2: SDPA output must stay compatible with the output projection."""

    def setUp(self):
        self.device = best_device()
        # Force the SDPA implementation for determinism.
        patcher = mock.patch.object(attention_module, "_ATTENTION_IMPL", "sdpa")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_sdpa_dtype_anchors_on_value_dtype(self):
        # Simulates the real stack with bfloat16 weights and autocast disabled:
        # q/k are float32 because rope_apply always returns float, while v keeps
        # the bfloat16 of the value projection.
        q = torch.randn(1, 8, 2, 32, device=self.device, dtype=torch.float32)
        k = torch.randn(1, 8, 2, 32, device=self.device, dtype=torch.float32)
        v = torch.randn(1, 8, 2, 32, device=self.device, dtype=torch.bfloat16)

        out = attention(q, k, v)  # dtype=None

        self.assertEqual(out.dtype, torch.bfloat16)
        # The output projection must accept the attention output.
        proj = nn.Linear(64, 64, device=self.device,
                         dtype=torch.bfloat16)
        y = proj(out.flatten(2))
        self.assertEqual(y.dtype, torch.bfloat16)

    def test_sdpa_preserves_float32_models(self):
        # Float32 models (e.g. on XPU/CPU) must keep computing in float32.
        q = torch.randn(1, 8, 2, 32, device=self.device, dtype=torch.float32)
        k = torch.randn(1, 8, 2, 32, device=self.device, dtype=torch.float32)
        v = torch.randn(1, 8, 2, 32, device=self.device, dtype=torch.float32)

        out = attention(q, k, v)

        self.assertEqual(out.dtype, torch.float32)
        proj = nn.Linear(64, 64, device=self.device, dtype=torch.float32)
        proj(out.flatten(2))  # must not raise

    def test_self_attention_bf16_weights_without_autocast(self):
        # End-to-end reviewer reproduction: a real WanSelfAttention with
        # bfloat16 weights, run without autocast. The RoPE path upcasts q/k to
        # float32; the block must still return bfloat16 so self.o() works.
        torch.manual_seed(0)
        attn = (
            WanSelfAttention(dim=64, num_heads=2, qk_norm=True, fused_qkv=False)
            .to(device=self.device, dtype=torch.bfloat16)
            .eval()
        )
        x = torch.randn(1, 16, 64, device=self.device, dtype=torch.bfloat16)
        grid_sizes = torch.tensor([[4, 2, 2]])
        freqs = torch.randn(1024, 16, device=self.device)

        with torch.no_grad():
            out = attn(x, seq_lens=None, grid_sizes=grid_sizes, freqs=freqs)

        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, (1, 16, 64))


class VaeDevicePlacementTests(unittest.TestCase):
    """Issue 3: VAE constructors must use the *resolved* device."""

    def setUp(self):
        from models.wan_2_2_models.vae import vae2_2

        self.vae2_2 = vae2_2
        self.device = best_device()

        def fake_video_vae(**kwargs):
            # Tiny stand-in with a parameter so placement is observable.
            return nn.Linear(4, 4)

        patcher = mock.patch.object(vae2_2, "_video_vae", fake_video_vae)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _check_placement(self, vae, expected_type):
        self.assertEqual(vae.device, expected_type)
        self.assertEqual(vae.scale[0].device.type, expected_type)
        self.assertEqual(vae.scale[1].device.type, expected_type)
        self.assertEqual(next(vae.model.parameters()).device.type, expected_type)

    def test_wan22_vae_resolves_none_device(self):
        vae = self.vae2_2.Wan2_2_VAE(vae_pth=None, device=None)
        self._check_placement(vae, self.device.type)

    def test_wan22_vae_honors_explicit_device(self):
        vae = self.vae2_2.Wan2_2_VAE(vae_pth=None, device="cpu")
        self._check_placement(vae, "cpu")

    def test_wan22_vae_batch_resolves_none_device(self):
        vae = self.vae2_2.Wan2_2_VAE_Batch(vae_pth=None, device=None)
        self._check_placement(vae, self.device.type)

    def test_wan22_vae_batch_honors_explicit_device(self):
        vae = self.vae2_2.Wan2_2_VAE_Batch(vae_pth=None, device="cpu")
        self._check_placement(vae, "cpu")


class WanTi2vDeviceSelectionTests(unittest.TestCase):
    """Issue 4: WanTI2V must honor the caller's device."""

    def setUp(self):
        try:
            from models.wan_2_2_models.pipeline.textimage2video import WanTI2V
        except Exception as exc:  # torchvision/PIL/einops missing, etc.
            raise unittest.SkipTest(f"WanTI2V import failed: {exc}")
        self.WanTI2V = WanTI2V

    def _make(self, **kwargs):
        # Components are only stored, never touched, in __init__.
        return self.WanTI2V(None, None, None, **kwargs)

    def test_explicit_device_wins_over_auto_detection(self):
        # On a host with an accelerator, the explicit device must still win;
        # otherwise the pipeline lands on a different device than the models.
        pipe = self._make(device="cpu")
        self.assertEqual(pipe.device.type, "cpu")
        self.assertIsNone(pipe.device.index)

    def test_explicit_torch_device_object(self):
        pipe = self._make(device=torch.device("cpu"))
        self.assertEqual(pipe.device, torch.device("cpu"))

    def test_auto_detection_without_explicit_device(self):
        pipe = self._make()
        expected = best_device().type
        self.assertEqual(pipe.device.type, expected)


class ServerDeviceSelectionTests(unittest.TestCase):
    """The --device flag must reach init_distributed_and_get_device."""

    def test_cpu_flag_honored_even_with_accelerator(self):
        try:
            from web_infer_utils.server import init_distributed_and_get_device
        except Exception as exc:
            raise unittest.SkipTest(f"server import failed: {exc}")

        # No RANK/WORLD_SIZE in the environment -> non-distributed branch.
        device, *_ = init_distributed_and_get_device(device_type="cpu")
        self.assertEqual(device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
