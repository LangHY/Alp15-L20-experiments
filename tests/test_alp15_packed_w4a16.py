"""CPU packed INT4 数学、存储与安全重载测试；不读取真实 checkpoint。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import safetensors
except ImportError:
    torch = None

pilot = None
if torch is not None:
    path = Path(__file__).resolve().parents[1] / "scripts/alp15_packed_w4a16.py"
    spec = importlib.util.spec_from_file_location("alp15_packed_w4a16", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)


@unittest.skipIf(torch is None, "须用已有 torch+safetensors 环境执行")
class PackedW4A16Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261008)

    def linear(self, *, bias=True, out_features=19):
        return torch.nn.Linear(256, out_features, bias=bias)

    def test_known_twos_complement_packing(self):
        values = torch.tensor([[-7, -1, 0, 1, 6, 7, -2, 2]], dtype=torch.int8)
        packed = pilot.pack_signed_int4(values)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.tolist(), [[0xF9, 0x10, 0x76, 0x2E]])
        torch.testing.assert_close(pilot.unpack_signed_int4(packed), values)
        for bad in (torch.tensor([-8, 0]), torch.tensor([0, 8]), torch.tensor([1]), torch.tensor([0., 1.])):
            with self.assertRaises(ValueError):
                pilot.pack_signed_int4(bad)
        with self.assertRaises(ValueError):
            pilot.unpack_signed_int4(torch.tensor([8], dtype=torch.uint8))

    def test_zero_groups_and_bias(self):
        linear = self.linear(out_features=2)
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.copy_(torch.tensor([1.5, -2.5]))
        model = pilot.PackedW4A16Linear.from_linear(linear)
        self.assertTrue(torch.equal(model.qweight, torch.zeros_like(model.qweight)))
        self.assertTrue(torch.equal(model.scales, torch.ones_like(model.scales)))
        torch.testing.assert_close(model(torch.randn(3, 256)), torch.tensor([[1.5, -2.5]]).expand(3, 2))

    def test_exact_rtn_groups_uses_stored_scale(self):
        linear = self.linear(bias=False, out_features=1)
        with torch.no_grad():
            linear.weight[:, :128].copy_(torch.tensor([-7., -1., 0., 1., 7., 2., 3., -3.]).repeat(16))
            linear.weight[:, 128:].copy_(2 * linear.weight[:, :128])
        model = pilot.PackedW4A16Linear.from_linear(linear)
        torch.testing.assert_close(model.scales, torch.tensor([[1., 2.]], dtype=torch.float16))
        torch.testing.assert_close(model.dequantize(torch.float32), linear.weight, rtol=0, atol=0)
        random = self.linear(bias=False)
        model = pilot.PackedW4A16Linear.from_linear(random, storage_dtype=torch.bfloat16)
        groups = random.weight.detach().reshape(19, 2, 128)
        expected_scale = (groups.abs().amax(-1) / 7).to(torch.bfloat16)
        expected_q = torch.round(groups / expected_scale.float().unsqueeze(-1)).clamp(-7, 7).to(torch.int8)
        torch.testing.assert_close(pilot.unpack_signed_int4(model.qweight).reshape_as(expected_q), expected_q)

    def test_forward_reference_tiled_and_explicit_dequant_equal(self):
        for dtype in (torch.float16, torch.bfloat16):
            for bias in (False, True):
                model = pilot.PackedW4A16Linear.from_linear(self.linear(bias=bias), storage_dtype=dtype,
                                                           tile_out_features=4)
                for shape in ((256,), (3, 256), (2, 3, 256), (0, 256)):
                    x = torch.randn(shape)
                    expected = torch.nn.functional.linear(x, model.dequantize(torch.float32),
                                                          None if model.bias is None else model.bias.float())
                    torch.testing.assert_close(model.reference(x), expected, rtol=0, atol=0)
                    torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-6)

    def test_persistent_storage_savings_no_dense_copy(self):
        linear = self.linear(out_features=512)
        model = pilot.PackedW4A16Linear.from_linear(linear)
        expected = 512 * 256 // 2 + 512 * (256 // 128) * 2 + 512 * 2
        self.assertEqual(model.storage_bytes(), expected)
        self.assertEqual(set(model.state_dict()), {"qweight", "scales", "bias"})
        self.assertEqual(list(model.parameters()), [])
        self.assertLess(model.storage_bytes(), (linear.weight.numel() + linear.bias.numel()) * 2 * .27)
        before = model.dequantize(torch.float32)
        with torch.no_grad():
            linear.weight.fill_(123)
            linear.bias.fill_(456)
        torch.testing.assert_close(model.dequantize(torch.float32), before, rtol=0, atol=0)
        self.assertEqual(model.storage_report()["persistent_tensor_bytes"], expected)

    def test_dtype_transfer_preserves_bytes_rejects_scale_damage(self):
        model = pilot.PackedW4A16Linear.from_linear(self.linear())
        original = model.qweight.clone()
        model.to(dtype=torch.bfloat16).half().to(device="cpu")
        self.assertEqual(model.qweight.dtype, torch.uint8)
        torch.testing.assert_close(model.qweight, original)
        with self.assertRaises(ValueError):
            model.float()
        self.assertEqual(model.scales.dtype, torch.float16)
        with self.assertRaises(ValueError):
            model._apply(lambda t: t.float())
        self.assertEqual(model.qweight.dtype, torch.uint8)

    def test_invalid_weight_bias_shapes_storage_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for field in ("weight", "bias"):
                linear = self.linear()
                with torch.no_grad():
                    getattr(linear, field).flatten()[0] = value
                with self.assertRaises(ValueError):
                    pilot.PackedW4A16Linear.from_linear(linear)
        with self.assertRaises(ValueError):
            pilot.PackedW4A16Linear.from_linear(torch.nn.Linear(129, 2))
        for dtype in (torch.float32, torch.int8):
            with self.assertRaises(ValueError):
                pilot.PackedW4A16Linear.from_linear(self.linear(), storage_dtype=dtype)
        with self.assertRaises(ValueError):
            pilot.PackedW4A16Linear.from_linear(self.linear(), group_size=64)
        model = pilot.PackedW4A16Linear.from_linear(self.linear())
        for scales in (torch.zeros_like(model.scales), torch.full_like(model.scales, float("nan")),
                       model.scales[:, :1], model.scales.float()):
            with self.assertRaises(ValueError):
                pilot.PackedW4A16Linear(model.qweight, scales)
        corrupt = model.qweight.clone()
        corrupt[0, 0] = 8
        with self.assertRaises(ValueError):
            pilot.PackedW4A16Linear(corrupt, model.scales)
        for tile in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                pilot.PackedW4A16Linear.from_linear(self.linear(), tile_out_features=tile)
        with self.assertRaises(ValueError):
            model(torch.randn(3, 255))

    def test_tiny_nonzero_and_overflow_weights(self):
        linear = self.linear(bias=False)
        with torch.no_grad():
            linear.weight.fill_(1e-20)
        model = pilot.PackedW4A16Linear.from_linear(linear)
        self.assertTrue((model.scales > 0).all().item())
        self.assertTrue(torch.isfinite(model.dequantize()).all().item())
        with torch.no_grad():
            linear.weight.fill_(1e10)
        with self.assertRaises(ValueError):
            pilot.PackedW4A16Linear.from_linear(linear, storage_dtype=torch.float16)

    def test_safe_serialization_roundtrip_rejects_corrupt_format(self):
        from safetensors.torch import save_file
        for dtype in (torch.float16, torch.bfloat16):
            for bias in (False, True):
                model = pilot.PackedW4A16Linear.from_linear(self.linear(bias=bias), storage_dtype=dtype,
                                                           tile_out_features=3)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "packed.safetensors"
                    model.save(path)
                    restored = pilot.PackedW4A16Linear.load(path)
                    self.assertEqual(restored.tile_out_features, 3)
                    self.assertEqual(restored.storage_report(), model.storage_report())
                    for name, tensor in model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
                    x = torch.randn(3, 256)
                    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
                    save_file(model.state_dict(), str(path), metadata={"format": "not-this-format"})
                    with self.assertRaises(ValueError):
                        pilot.PackedW4A16Linear.load(path)
                    metadata = {"format": pilot.FORMAT, "group_size": "128", "in_features": "256",
                                "out_features": "19", "tile_out_features": "3"}
                    save_file({**model.state_dict(), "dense_weight": torch.zeros(19, 256)},
                              str(path), metadata=metadata)
                    with self.assertRaises(ValueError):
                        pilot.PackedW4A16Linear.load(path)
                    save_file(model.state_dict(), str(path), metadata={**metadata, "in_features": "128"})
                    with self.assertRaises(ValueError):
                        pilot.PackedW4A16Linear.load(path)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "无 CUDA，跳过 GPU 数学复核")
    def test_cuda_bf16_reference(self):
        linear = self.linear().to(device="cuda", dtype=torch.bfloat16)
        model = pilot.PackedW4A16Linear.from_linear(linear, storage_dtype=torch.bfloat16).to("cuda")
        x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(x, model.dequantize(), model.bias)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
