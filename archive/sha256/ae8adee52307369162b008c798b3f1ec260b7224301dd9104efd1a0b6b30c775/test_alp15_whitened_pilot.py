"""阶段2纯 CPU 数学测试；不加载真实权重或访问 GPU/网络。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/alp15_whitened_pilot.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("alp15_whitened_pilot", SCRIPT)
assert SPEC and SPEC.loader
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "本地无 torch；须在既有服务器环境运行全部 CPU 数学测试")
class WhitenedMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261007)
        torch.set_num_threads(8)

    def test_cholesky_and_fullrank_reconstruction(self):
        x = torch.randn(29, 5, dtype=torch.float64)
        gram = x.T @ x / len(x)
        lower, lam = pilot.regularized_cholesky(gram)
        self.assertAlmostEqual(lam, 1e-4 * gram.trace().item() / 5)
        torch.testing.assert_close(lower @ lower.T, gram + lam * torch.eye(5, dtype=torch.float64))
        weight = torch.randn(7, 5)
        factors = pilot.balanced_factors(pilot.whitened_svd(weight, lower), 5)
        torch.testing.assert_close(factors[0] @ factors[1], weight, rtol=3e-5, atol=5e-6)

    def test_ridge_known_literal(self):
        # Z=[[1,0],[0,2]]，Y=[[3],[8]]，Gram=diag(.5,2)，mu=.000125。
        ztz = torch.tensor([[1., 0.], [0., 4.]], dtype=torch.float64)
        zty = torch.tensor([[3.], [16.]], dtype=torch.float64)
        solution, mu = pilot.ridge_closed_form(ztz, zty, 2)
        self.assertAlmostEqual(mu, .000125)
        expected = torch.tensor([[1.5 / .500125], [8. / 2.000125]], dtype=torch.float64)
        torch.testing.assert_close(solution, expected, rtol=1e-12, atol=1e-12)

    def test_complete_gated_fullrank_output_and_parameter_count(self):
        base = pilot.base
        x = torch.randn(31, 5)
        weights = {"gate_proj": torch.randn(7, 5), "up_proj": torch.randn(7, 5), "down_proj": torch.randn(5, 7)}
        features = base.ffn_features(x, weights)
        x_lower, _ = pilot.regularized_cholesky(x.double().T @ x.double() / len(x))
        feature_lower, _ = pilot.regularized_cholesky(features.double().T @ features.double() / len(x))
        factors = {name: pilot.balanced_factors(pilot.whitened_svd(weight, feature_lower if name == "down_proj" else x_lower), 5) for name, weight in weights.items()}
        self.assertEqual(base.verify_factor_parameter_count(factors, 5, 7, 5), 180)
        torch.testing.assert_close(base.factor_ffn(x, factors), base.dense_ffn(x, weights), rtol=2e-4, atol=2e-4)

    def test_nonfinite_invalid_rank_and_zero_gram_rejected(self):
        for bad in (float("nan"), float("inf")):
            gram = torch.eye(3, dtype=torch.float64)
            gram[0, 0] = bad
            with self.assertRaises(ValueError):
                pilot.regularized_cholesky(gram)
            with self.assertRaises(ValueError):
                pilot.ridge_closed_form(gram, torch.ones(3, 2), 7)
        with self.assertRaises(ValueError):
            pilot.regularized_cholesky(torch.zeros(3, 3))
        with self.assertRaises(ValueError):
            pilot.ridge_closed_form(torch.zeros(3, 3), torch.ones(3, 2), 7)
        lower, _ = pilot.regularized_cholesky(torch.eye(3))
        decomposition = pilot.whitened_svd(torch.randn(4, 3), lower)
        for rank in (0, -1, 4, True):
            with self.assertRaises(ValueError):
                pilot.balanced_factors(decomposition, rank)
        with self.assertRaises(ValueError):
            pilot.whitened_svd(torch.full((4, 3), float("nan")), lower)

    def test_cholesky_failure_has_no_regularization_retry(self):
        # 正 trace 但负特征值；固定微小 lambda 无法修复，必须抛异常。
        with self.assertRaises(RuntimeError):
            pilot.regularized_cholesky(torch.diag(torch.tensor([-1., 2.], dtype=torch.float64)))


class FrozenSpecificationTests(unittest.TestCase):
    def test_layers_ranks_and_fixed_regularization(self):
        self.assertEqual(pilot.LAYERS, (6, 35))
        self.assertEqual(pilot.RANKS, (896, 1024))
        self.assertEqual(pilot.REGULARIZATION_RATIO, 1e-4)
        self.assertEqual(pilot.base.SEED, 20261007)
        self.assertEqual(pilot.base.sha256_file(Path(pilot.base.__file__)), pilot.BASE_CODE_SHA256)


if __name__ == "__main__":
    unittest.main()
