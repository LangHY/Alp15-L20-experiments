"""Run on the inference server: python -m unittest compression.test_w4a8 -v.

CPU tests verify format/storage/reference mathematics. CUDA tests exercise
the actual native Triton integer kernel and skip when CUDA/Triton is absent.
Dense-model error is reported, not used as a fabricated quality guarantee.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from compression.w4a8 import (
    GROUP_SIZE, PackedW4A8Linear, pack_int4, quantize_activation_reference,
    triton, unpack_int4,
)


def explicit_integer_reference(module, x):
    """Independent groupwise specification, CPU INT32 dot / FP32 scales."""
    k, n = module.in_features, module.out_features
    xf = x.detach().cpu().float().reshape(-1, k)
    packed = module.qweight.cpu().to(torch.int16)
    weight = torch.empty(n, k, dtype=torch.int32)
    for parity in (0, 1):
        nibble = (packed >> (4 * parity)) & 15
        weight[:, parity::2] = torch.where(nibble >= 8, nibble - 16, nibble)
    result = torch.zeros(xf.shape[0], n, dtype=torch.float32)
    clean = torch.where(torch.isfinite(xf), xf, 0.0)
    for group, start in enumerate(range(0, k, GROUP_SIZE)):
        tile = clean[:, start:start + GROUP_SIZE]
        amax = tile.abs().max(dim=1).values
        scale = torch.where(amax == 0, 1.0,
                            (amax / 127.0).clamp_min(torch.finfo(torch.float32).tiny))
        activation = (tile / scale[:, None]).round().clamp(-127, 127).to(torch.int32)
        dot = activation @ weight[:, start:start + GROUP_SIZE].T
        result += (dot.float() * scale[:, None]) * module.scales.cpu().float()[None, :, group]
    if module.bias is not None:
        result += module.bias.cpu().float()
    result[~torch.isfinite(xf).all(-1)] = float("nan")
    return result.to(device=x.device, dtype=module.original_dtype).reshape(*x.shape[:-1], n)


class PackedFormatTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(234)

    def test_pack_unpack_exact_bytes_and_signed_values(self):
        values = torch.tensor([[-7, -1, 0, 1, 6, 7, -2, -3]], dtype=torch.int8)
        packed = pack_int4(values)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.tolist(), [[0xF9, 0x10, 0x76, 0xDE]])
        torch.testing.assert_close(unpack_int4(packed), values, rtol=0, atol=0)
        every = torch.arange(-7, 8, dtype=torch.int8).repeat(2).reshape(3, 10)
        torch.testing.assert_close(unpack_int4(pack_int4(every)), every, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            pack_int4(torch.tensor([-8, 0], dtype=torch.int8))
        with self.assertRaises(ValueError):
            pack_int4(torch.tensor([0], dtype=torch.int8))
        with self.assertRaises(TypeError):
            pack_int4(torch.zeros(2))

    def test_storage_reduction_and_no_resident_float_weight(self):
        linear = nn.Linear(256, 64, bias=False, dtype=torch.bfloat16)
        packed = PackedW4A8Linear.from_linear(linear)
        self.assertEqual(packed.qweight.shape, (64, 128))
        self.assertEqual(packed.scales.shape, (64, 2))
        self.assertEqual(packed.scales.dtype, torch.bfloat16)
        original_bytes = linear.weight.numel() * linear.weight.element_size()
        self.assertEqual(packed.storage_bytes(), 64 * 256 // 2 + 64 * 2 * 2)
        self.assertGreater(original_bytes / packed.storage_bytes(), 3.8)
        self.assertLess(original_bytes / packed.storage_bytes(), 4.0)
        self.assertEqual(list(packed.parameters()), [])
        self.assertFalse(hasattr(packed, "weight"))
        self.assertEqual(set(packed.state_dict()), {"qweight", "scales"})
        before = packed.storage_bytes()
        debug_weight = packed.dequantize()
        self.assertEqual(debug_weight.shape, linear.weight.shape)
        self.assertEqual(packed.storage_bytes(), before)
        self.assertFalse(any(t.shape == linear.weight.shape for t in packed.buffers()))

    def test_rtn_quantizes_using_actual_stored_bf16_scale(self):
        linear = nn.Linear(128, 64, bias=False, dtype=torch.float32)
        packed = PackedW4A8Linear.from_linear(linear)
        expected = (linear.weight.detach().reshape(64, 1, 128) /
                    packed.scales.float().unsqueeze(-1)).round().clamp(-7, 7).flatten(1).to(torch.int8)
        torch.testing.assert_close(unpack_int4(packed.qweight), expected, rtol=0, atol=0)

    def test_activation_rounding_ties_zero_and_finite_scales(self):
        x = torch.zeros(3, 128)
        x[0, :7] = torch.tensor([127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
        x[2] = torch.finfo(torch.float32).tiny / 128
        quantized, scales = quantize_activation_reference(x)
        self.assertEqual(quantized[0, 0, :7].tolist(), [127, 0, 2, 2, 0, -2, -2])
        self.assertEqual(scales[0, 0].item(), 1.0)
        self.assertEqual(scales[1, 0].item(), 1.0)
        self.assertTrue(bool(torch.isfinite(scales).all()))
        self.assertTrue(bool((scales > 0).all()))
        self.assertEqual(quantized[1].abs().sum().item(), 0)

    def test_reference_bias_prefix_shapes_and_integer_math(self):
        for dtype in (torch.float32, torch.bfloat16):
            linear = nn.Linear(256, 64, bias=True, dtype=dtype)
            packed = PackedW4A8Linear.from_linear(linear)
            x = torch.randn(2, 3, 256, dtype=dtype)
            expected = explicit_integer_reference(packed, x)
            torch.testing.assert_close(packed.reference(x), expected, rtol=0, atol=0)
            with self.assertRaisesRegex(RuntimeError, "reference"):
                packed(x)
            with self.assertRaises(ValueError):
                packed.reference(torch.ones(2, 255, dtype=dtype))
            self.assertEqual(packed.reference(torch.empty(0, 256, dtype=dtype)).shape, (0, 64))
            self.assertEqual(packed.reference(x[0, 0]).shape, (64,))

    def test_zero_weight_and_invalid_values(self):
        linear = nn.Linear(128, 64, bias=True, dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.fill_(0.25)
        packed = PackedW4A8Linear.from_linear(linear)
        self.assertTrue(bool((packed.scales == 1).all()))
        x = torch.zeros(3, 128, dtype=torch.bfloat16)
        torch.testing.assert_close(packed.reference(x), torch.full((3, 64), 0.25, dtype=torch.bfloat16))
        x[1, 0] = float("nan")
        x[2, 0] = float("inf")
        result = packed.reference(x)
        self.assertTrue(bool(torch.isnan(result[1:]).all()))
        with torch.no_grad():
            linear.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            PackedW4A8Linear.from_linear(linear)

    def test_dimension_validation_and_explicit_metadata(self):
        for k, n in ((127, 64), (128, 63), (0, 64)):
            with self.assertRaises(ValueError):
                PackedW4A8Linear(k, n)
        module = PackedW4A8Linear.from_linear(nn.Linear(128, 64, bias=True, dtype=torch.bfloat16))
        metadata = json.loads(json.dumps(module.metadata()))
        restored = PackedW4A8Linear.from_metadata(metadata)
        restored.load_state_dict(module.state_dict(), strict=True)
        x = torch.randn(2, 128, dtype=torch.bfloat16)
        torch.testing.assert_close(restored.reference(x), module.reference(x), rtol=0, atol=0)
        metadata["group_size"] = 64
        with self.assertRaises(ValueError):
            PackedW4A8Linear.from_metadata(metadata)

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "safetensors unavailable")
    def test_safetensors_save_reload(self):
        for bias in (False, True):
            module = PackedW4A8Linear.from_linear(nn.Linear(256, 64, bias=bias, dtype=torch.bfloat16))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "packed.safetensors"
                module.save(path)
                restored = PackedW4A8Linear.load(path)
            self.assertEqual(restored.metadata(), module.metadata())
            for name, value in module.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            x = torch.randn(3, 256, dtype=torch.bfloat16)
            torch.testing.assert_close(restored.reference(x), module.reference(x), rtol=0, atol=0)

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "safetensors unavailable")
    def test_safetensors_rejects_invalid_packed_format(self):
        from safetensors.torch import save_file
        module = PackedW4A8Linear(128, 64)
        for corrupt in ("scale", "reserved_nibble", "dtype"):
            tensors = {name: value.clone() for name, value in module.state_dict().items()}
            if corrupt == "scale":
                tensors["scales"][0, 0] = 0
            elif corrupt == "reserved_nibble":
                tensors["qweight"][0, 0] = 8
            else:
                tensors["scales"] = tensors["scales"].float()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "invalid.safetensors"
                save_file(tensors, str(path), metadata={"w4a8": json.dumps(module.metadata())})
                with self.assertRaises(ValueError):
                    PackedW4A8Linear.load(path)


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip is None and triton is not None,
                     "NVIDIA CUDA and Triton required")
class NativeKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(987)

    def test_native_kernel_matches_explicit_integer_reference(self):
        # FP32 tests tighten arithmetic tolerance; BF16 allows one output ULP.
        for dtype, rtol, atol in ((torch.float32, 2e-6, 2e-6),
                                  (torch.bfloat16, 0.008, 0.004)):
            for bias in (False, True):
                linear = nn.Linear(256, 64, bias=bias, device="cuda", dtype=dtype)
                packed = PackedW4A8Linear.from_linear(linear)
                for m in (1, 15, 16, 17, 33):
                    with self.subTest(dtype=dtype, bias=bias, m=m):
                        x = torch.randn(m, 256, device="cuda", dtype=dtype)
                        expected = explicit_integer_reference(packed, x)
                        with mock.patch.object(packed, "dequantize", side_effect=AssertionError("dense fallback")):
                            actual = packed(x)
                        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                        torch.testing.assert_close(packed.reference(x), expected, rtol=0, atol=0)
                        self.assertEqual(actual.dtype, dtype)
                        self.assertFalse(hasattr(packed, "weight"))

    def test_native_kernel_prefix_noncontiguous_zero_invalid_and_empty(self):
        linear = nn.Linear(256, 128, bias=True, device="cuda", dtype=torch.bfloat16)
        packed = PackedW4A8Linear.from_linear(linear)
        x = torch.randn(2, 3, 512, device="cuda", dtype=torch.bfloat16)[..., ::2]
        self.assertFalse(x.is_contiguous())
        x[0, 0].zero_()
        actual = packed(x)
        torch.testing.assert_close(actual, explicit_integer_reference(packed, x), rtol=0.008, atol=0.004)
        torch.testing.assert_close(actual[0, 0], packed.bias, rtol=0, atol=0)
        self.assertEqual(packed(x[0, 0]).shape, (128,))
        self.assertEqual(packed(torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)).shape, (0, 128))
        x[0, 1, 0] = float("nan")
        x[1, 0, 0] = float("inf")
        invalid_output = packed(x)
        self.assertTrue(bool(torch.isnan(invalid_output[0, 1]).all()))
        self.assertTrue(bool(torch.isnan(invalid_output[1, 0]).all()))
        self.assertTrue(bool(torch.isfinite(invalid_output[0, 0]).all()))

    def test_native_activation_ties_match_integer_reference(self):
        linear = nn.Linear(128, 64, bias=False, device="cuda", dtype=torch.float32)
        packed = PackedW4A8Linear.from_linear(linear)
        x = torch.zeros(17, 128, device="cuda")
        x[:, :7] = torch.tensor([127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], device="cuda")
        torch.testing.assert_close(packed(x), explicit_integer_reference(packed, x), rtol=2e-6, atol=2e-6)

    def test_dense_bf16_quantization_error_is_reported(self):
        linear = nn.Linear(256, 64, bias=False, device="cuda", dtype=torch.bfloat16)
        packed = PackedW4A8Linear.from_linear(linear)
        x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            dense = linear(x).float()
            actual = packed(x).float()
        mse = (actual - dense).square().mean().item()
        dense_energy = dense.square().mean().item()
        self.assertTrue(bool(torch.isfinite(actual).all()))
        print(f"\nRTN W4A8 tiny dense comparison: MSE={mse:.8g}, "
              f"relative_MSE={mse / max(dense_energy, 1e-30):.8g}; "
              "this is numerical smoke data, not model quality or speed evidence")


if __name__ == "__main__":
    unittest.main(verbosity=2)


