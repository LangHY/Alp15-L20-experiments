"""隔离 AWQ 算子诊断边界；CPU 格式检查不代表 CUDA 内核验收。"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from alp15_awq_full_packed import PackedAWQLinear


class AWQKernelProbeTests(unittest.TestCase):
    def test_fused_candidate_refuses_cpu_without_fallback(self):
        kernel = importlib.import_module("alp15_awq_kernel")
        matrix = PackedAWQLinear(torch.zeros((1, 64), dtype=torch.uint8),
                                 torch.ones(1, 1, dtype=torch.bfloat16),
                                 torch.ones(128, dtype=torch.bfloat16))
        candidate = kernel.FusedAWQLinear.from_reference(matrix)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            candidate(torch.ones(1, 128, dtype=torch.bfloat16))

    def test_cpu_import_does_not_import_triton(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import sys; import benchmark_alp15_awq_kernel; import alp15_awq_kernel; "
                                 "assert 'triton' not in sys.modules"],
                                cwd=SCRIPTS, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_matrix_hash_mismatch_is_rejected(self):
        probe = importlib.import_module("benchmark_alp15_awq_kernel")
        matrix = PackedAWQLinear(torch.zeros((1, 64), dtype=torch.uint8),
                                 torch.ones(1, 1), torch.ones(128, dtype=torch.bfloat16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.safetensors"
            matrix.save(path)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                probe.load_reference_matrix(path, expected_sha256="0" * 64)

    def test_cuda_benchmark_refuses_cpu_execution(self):
        probe = importlib.import_module("benchmark_alp15_awq_kernel")
        matrix = PackedAWQLinear(torch.zeros((1, 64), dtype=torch.uint8),
                                 torch.ones(1, 1), torch.ones(128, dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            probe.benchmark_matrix(matrix, device="cpu", m_values=[1],
                                   dtype=torch.bfloat16, warmup=1, repeats=2, seed=1)

    def test_existing_packed_file_is_loaded_without_requantization(self):
        probe = importlib.import_module("benchmark_alp15_awq_kernel")
        # 已知 nibble 顺序：低位 -7，高位 +7；amax=7 -> 直接保留整数。
        qweight = torch.full((3, 64), 0x79, dtype=torch.uint8)
        original = PackedAWQLinear(qweight, torch.tensor([[7.], [0.], [7.]]),
                                   torch.ones(128, dtype=torch.bfloat16),
                                   torch.tensor([1., -2., 3.], dtype=torch.bfloat16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.safetensors"
            original.save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            matrix, receipt = probe.load_reference_matrix(path, expected_sha256=digest)
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(matrix.state_dict()[name], value))
            inputs = torch.ones(5, 128, dtype=torch.bfloat16)
            inputs[:, 0::2] = 2
            expected = torch.tensor([-447., -2., -445.], dtype=torch.bfloat16).expand(5, -1)
            torch.testing.assert_close(matrix(inputs), expected, rtol=0, atol=0)
            self.assertEqual(receipt["sha256"], digest)
            self.assertEqual(receipt["shape"], [3, 128])


if os.environ.get("ALP15_KERNEL_REQUIRE_CUDA") == "1":
    class AWQKernelCUDATests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            if not torch.cuda.is_available():
                raise RuntimeError("ALP15_KERNEL_REQUIRE_CUDA=1 需要实际 CUDA；不得 skip 后算通过")
            cls.kernel = importlib.import_module("alp15_awq_kernel")

        def known_matrix(self):
            return PackedAWQLinear(torch.full((3, 64), 0x1F, dtype=torch.uint8),
                                   torch.tensor([[7.], [0.], [7.]], dtype=torch.bfloat16),
                                   torch.tensor([2., 1.] * 64, dtype=torch.bfloat16),
                                   torch.tensor([1., -2., 3.], dtype=torch.bfloat16)).cuda()

        def test_signed_nibbles_zero_group_nontrivial_prescale_bias_odd_tails(self):
            matrix = self.known_matrix()
            fused = self.kernel.FusedAWQLinear.from_reference(matrix)
            torch.testing.assert_close(fused.dequantize_for_validation(), matrix.dequantize(torch.bfloat16),
                                       rtol=0, atol=0)
            for shape in ((1, 128), (5, 128), (17, 128), (33, 128), (2, 3, 128)):
                with self.subTest(shape=shape):
                    inputs = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
                    expected = torch.tensor([-63., -2., -61.], device="cuda", dtype=torch.bfloat16)
                    expected = expected.expand(*shape[:-1], 3)
                    torch.testing.assert_close(fused(inputs), expected, rtol=0, atol=0)

        def test_random_math_and_exact_dequant_bridge(self):
            from alp15_packed_w4a16 import pack_signed_int4
            generator = torch.Generator().manual_seed(20261008)
            integers = torch.randint(-7, 8, (19, 256), generator=generator, dtype=torch.int8)
            maxima = (0.01 + torch.rand(19, 2, generator=generator)).to(torch.bfloat16)
            maxima[0, 1] = 0
            scale = (0.3 + torch.rand(256, generator=generator)).to(torch.bfloat16)
            matrix = PackedAWQLinear(pack_signed_int4(integers), maxima, scale).cuda()
            fused = self.kernel.FusedAWQLinear.from_reference(matrix)
            self.assertTrue(torch.equal(fused.dequantize_for_validation(), matrix.dequantize(torch.bfloat16)))
            for m in (1, 7, 17, 33):
                inputs = torch.randn(m, 256, generator=generator).to(device="cuda", dtype=torch.bfloat16)
                torch.testing.assert_close(fused(inputs), matrix(inputs), rtol=0.01, atol=0.02)

        def test_uses_callers_cuda_stream(self):
            matrix = self.known_matrix()
            fused = self.kernel.FusedAWQLinear.from_reference(matrix)
            # 先编译，排除首次JIT的CPU时间意外掩盖跨流竞态。
            fused(torch.ones(1, 128, device="cuda", dtype=torch.bfloat16))
            torch.cuda.synchronize()
            inputs = torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16)
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(1000000)
                inputs.fill_(1)
                output = fused(inputs)
            stream.synchronize()
            expected = torch.tensor([[-63., -2., -61.]], device="cuda", dtype=torch.bfloat16)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

        def test_distinct_weight_buffers_do_not_share_cached_dense_state(self):
            original = self.known_matrix()
            other = PackedAWQLinear(torch.zeros_like(original.qweight).cpu(), original.amax.cpu(),
                                    original.pre_scale.cpu(), original.bias.cpu()).cuda()
            first = self.kernel.FusedAWQLinear.from_reference(original)
            second = self.kernel.FusedAWQLinear.from_reference(other)
            inputs = torch.ones(1, 128, device="cuda", dtype=torch.bfloat16)
            torch.testing.assert_close(first(inputs), original(inputs), rtol=0, atol=0)
            torch.testing.assert_close(second(inputs), other.bias.view(1, -1), rtol=0, atol=0)
            torch.testing.assert_close(first(inputs), original(inputs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
