"""纯数学/边界测试；不读取真实模型、不访问网络、不运行 GPU。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/alp15_lowrank_pilot.py"
SPEC = importlib.util.spec_from_file_location("alp15_lowrank_pilot", SCRIPT)
assert SPEC and SPEC.loader
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)

try:
    import torch
except ImportError:
    torch = None


class BoundaryAndBudgetTests(unittest.TestCase):
    def test_actual_budget_and_linked_width(self):
        for rank in (320, 896, 1024, 1216):
            params = pilot.lowrank_parameter_count(4096, 5376, rank)
            width = pilot.matched_channel_width(4096, 5376, rank)
            self.assertEqual(params, 3 * rank * 9472)
            self.assertLessEqual(width * 3 * 4096, params)
            self.assertGreater((width + 1) * 3 * 4096, params)

    def test_invalid_ranks_rejected(self):
        for rank in (-1, 0, 4097, True, 1.5):
            with self.assertRaises(ValueError):
                pilot.lowrank_parameter_count(4096, 5376, rank)

    def test_unequal_clip_lengths_restored(self):
        records = [{"clip_id": "a", "rows": 3}, {"clip_id": "b", "rows": 7}]
        bounds = pilot.clip_boundaries(records, 10)
        self.assertEqual([(r["start"], r["stop"]) for r in bounds], [(0, 3), (3, 10)])
        with self.assertRaises(ValueError):
            pilot.clip_boundaries(records, 12)
        with self.assertRaises(ValueError):
            pilot.clip_boundaries([records[0], records[0]], 6)
        with self.assertRaises(ValueError):
            pilot.clip_boundaries([{"clip_id": "a", "rows": 0}], 1)

    def test_deterministic_clip_split_mutual_exclusion(self):
        clips = [f"clip_{i:02d}" for i in range(80)]
        fit, evaluation = pilot.split_clips(clips, clips, ["monitor"])
        self.assertEqual((len(fit), len(evaluation)), (64, 16))
        self.assertFalse(set(fit) & set(evaluation))
        self.assertEqual(set(fit + evaluation), set(clips))
        self.assertEqual((fit, evaluation), pilot.split_clips(list(reversed(clips)), clips, []))
        with self.assertRaises(ValueError):
            pilot.split_clips(clips, clips, [clips[0]])
        with self.assertRaises(ValueError):
            pilot.split_clips(clips[:-1], clips, [])
        with self.assertRaises(ValueError):
            pilot.split_clips(clips, clips[:-1], [])
        with self.assertRaises(ValueError):
            pilot.split_clips(clips, clips, [], fit_count=63)

    def test_parameter_count_rejects_dense_base_and_fake_rank(self):
        def tensor(shape):
            return SimpleNamespace(shape=shape, numel=lambda: shape[0] * shape[1])
        factors = {"gate_proj": (tensor((7, 2)), tensor((2, 5))),
                   "up_proj": (tensor((7, 2)), tensor((2, 5))),
                   "down_proj": (tensor((5, 2)), tensor((2, 7)))}
        self.assertEqual(pilot.verify_factor_parameter_count(factors, 5, 7, 2), 72)
        with self.assertRaises(ValueError):
            pilot.verify_factor_parameter_count({**factors, "dense_base": tensor((7, 5))}, 5, 7, 2)
        with self.assertRaises(ValueError):
            pilot.verify_factor_parameter_count(factors, 5, 7, 1)
        with self.assertRaises(ValueError):
            pilot.verify_factor_parameter_count({**factors, "gate_proj": (*factors["gate_proj"], tensor((7, 5)))}, 5, 7, 2)


@unittest.skipIf(torch is None, "当前 CPU 环境没有 torch；须在既有服务器环境完成")
class FactorMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261007)

    def test_fullrank_plain_and_scaled_reconstruction(self):
        weight = torch.randn(7, 5)
        inputs = torch.randn(23, 5)
        for scale in (None, pilot.activation_scale(inputs)):
            b, a = pilot.factors_from_svd(pilot.compute_svd(weight, scale), 5)
            torch.testing.assert_close(b @ a, weight, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(pilot.factor_linear(inputs, (b, a)), inputs @ weight.T, rtol=3e-5, atol=1e-5)

    def test_complete_gated_ffn_fullrank(self):
        weights = {"gate_proj": torch.randn(7, 5), "up_proj": torch.randn(7, 5), "down_proj": torch.randn(5, 7)}
        factors = {name: pilot.factors_from_svd(pilot.compute_svd(weight), 5) for name, weight in weights.items()}
        x = torch.randn(11, 5)
        torch.testing.assert_close(pilot.factor_ffn(x, factors), pilot.dense_ffn(x, weights), rtol=1e-4, atol=5e-5)

    def test_nonfinite_weights_inputs_scales_rejected(self):
        weight = torch.randn(7, 5)
        for bad in (float("nan"), float("inf")):
            altered = weight.clone()
            altered[0, 0] = bad
            with self.assertRaises(ValueError):
                pilot.compute_svd(altered)
            with self.assertRaises(ValueError):
                pilot.activation_scale(altered)
        for scale in (torch.zeros(5), -torch.ones(5), torch.ones(6), torch.full((5,), float("nan"))):
            with self.assertRaises(ValueError):
                pilot.compute_svd(weight, scale)

    def test_bad_rank_and_bf16_real_factor_shapes(self):
        decomposition = pilot.compute_svd(torch.randn(7, 5))
        for rank in (0, -1, 6, True):
            with self.assertRaises(ValueError):
                pilot.factors_from_svd(decomposition, rank)
        b, a = pilot.factors_from_svd(decomposition, 2, torch.bfloat16)
        self.assertEqual((b.shape, a.shape), (torch.Size([7, 2]), torch.Size([2, 5])))
        self.assertEqual((b.dtype, a.dtype), (torch.bfloat16, torch.bfloat16))

    def test_scale_formula_and_clamp(self):
        x = torch.tensor([[0., 1., -9.], [0., -3., 7.]])
        torch.testing.assert_close(pilot.activation_scale(x), torch.tensor([1e-6, 2. ** .5, 8. ** .5]))

    def test_historical_channel_score_literal_expected(self):
        features = torch.tensor([[1., -2., 3.], [-3., 4., 1.]])
        down = torch.tensor([[1., 2., 0.], [2., 0., 3.]])
        # 二阶矩为 [5,10,5]，列平方范数为 [5,4,9]，独立手算结果为 [25,40,45]。
        scores = pilot.channel_scores(features.square().mean(0), down)
        torch.testing.assert_close(scores, torch.tensor([25., 40., 45.], dtype=torch.float64))
        self.assertEqual(torch.argsort(scores, descending=True, stable=True).tolist(), [2, 1, 0])
        with self.assertRaises(ValueError):
            pilot.channel_scores(torch.tensor([-1., 2., 3.]), down)


if __name__ == "__main__":
    unittest.main()
