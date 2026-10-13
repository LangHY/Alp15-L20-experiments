"""Run on the inference server: python -m unittest compression.test_mlp_pruning."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from compression.mlp_pruning import (
    apply_overlay, cache_teacher_io, compute_activation_scores,
    load_pruned_mlp_state, prune_mlp, save_overlay, select_channels,
    train_down_projection,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden=4, intermediate=8, dtype=torch.float32):
        super().__init__()
        self.hidden_size = hidden
        self.intermediate_size = intermediate
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=dtype)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MLPPruningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_structural_pruning_matches_dense_channel_mask(self):
        mlp = TinyMLP()
        x = torch.randn(3, 5, 4)
        ids = torch.tensor([0, 2, 5, 7])
        activ = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
        mask = torch.zeros(8)
        mask[ids] = 1
        expected = mlp.down_proj(activ * mask).detach()
        act_fn = mlp.act_fn
        report = prune_mlp(mlp, ids)
        torch.testing.assert_close(mlp(x), expected)
        self.assertIs(mlp.act_fn, act_fn)
        self.assertEqual(mlp.hidden_size, 4)
        self.assertEqual(mlp.intermediate_size, 4)
        self.assertEqual(tuple(mlp.gate_proj.weight.shape), (4, 4))
        self.assertEqual(tuple(mlp.down_proj.weight.shape), (4, 4))
        self.assertEqual(report["kept_channels"], ids.tolist())
        self.assertEqual(report["original_intermediate_size"], 8)

    def test_score_formula_and_stable_ties(self):
        mlp = TinyMLP()
        x = torch.randn(15, 4)
        activ = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
        expected = activ.square().mean(0) * mlp.down_proj.weight.square().sum(0)
        scores = compute_activation_scores(mlp, x, batch_size=4)
        torch.testing.assert_close(scores, expected.detach().double(), rtol=1e-6, atol=1e-9)
        self.assertFalse(scores.requires_grad)
        self.assertEqual(select_channels(torch.tensor([2., 1., 2., 2.]), 2).tolist(), [0, 2])

    def test_invalid_channels_and_bias_rejected_before_mutation(self):
        for ids in ([1, 1], [2, 0], [-1, 2], [8], [0.5, 1.5], []):
            mlp = TinyMLP()
            gate = mlp.gate_proj
            with self.assertRaises(ValueError):
                prune_mlp(mlp, ids)
            self.assertIs(mlp.gate_proj, gate)
        mlp = TinyMLP()
        mlp.down_proj = nn.Linear(8, 4, bias=True)
        with self.assertRaises(ValueError):
            compute_activation_scores(mlp, torch.randn(3, 4))

    def test_teacher_cache_and_state_loading(self):
        dense = TinyMLP()
        x = torch.randn(2, 7, 4)
        cached_x, cached_y = cache_teacher_io(dense, x, batch_size=3)
        torch.testing.assert_close(cached_y, dense(x).detach().reshape(-1, 4))
        self.assertFalse(cached_y.requires_grad)
        torch.testing.assert_close(cached_x, x.reshape(-1, 4))
        pruned = copy.deepcopy(dense)
        ids = [0, 2, 6]
        prune_mlp(pruned, ids)
        state = {name: value.clone() for name, value in pruned.state_dict().items()}
        reloaded = copy.deepcopy(dense)
        load_pruned_mlp_state(reloaded, state, ids)
        torch.testing.assert_close(reloaded(x), pruned(x))

    def test_distillation_improves_and_freezes_parameters(self):
        mlp = TinyMLP(intermediate=6)
        prune_mlp(mlp, [0, 1, 3, 5])
        x = torch.randn(96, 4)
        vx = torch.randn(32, 4)
        planted = torch.randn(4, 4) * 0.2
        with torch.no_grad():
            y = F.linear(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x), planted)
            vy = F.linear(mlp.act_fn(mlp.gate_proj(vx)) * mlp.up_proj(vx), planted)
            mlp.down_proj.weight.zero_()
        gate, up = mlp.gate_proj.weight.detach().clone(), mlp.up_proj.weight.detach().clone()
        result = train_down_projection(mlp, x, y, vx, vy, steps=100,
                                       batch_size=24, lr=0.03, eval_every=5)
        self.assertLess(result["final_val_nmse"], result["baseline_val_nmse"] * 0.8)
        self.assertGreater(result["best_step"], 0)
        torch.testing.assert_close(mlp.gate_proj.weight, gate, rtol=0, atol=0)
        torch.testing.assert_close(mlp.up_proj.weight, up, rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in mlp.parameters()))
        self.assertTrue(mlp.training)

    def test_baseline_wins_against_overtraining_in_export_dtype(self):
        mlp = TinyMLP(dtype=torch.bfloat16)
        prune_mlp(mlp, [0, 2, 4, 6])
        x = torch.randn(24, 4).bfloat16()
        vx = torch.randn(12, 4).bfloat16()
        with torch.no_grad():
            y, vy = -mlp(x), mlp(vx)
        original = mlp.down_proj.weight.detach().clone()
        result = train_down_projection(mlp, x, y, vx, vy, steps=12,
                                       batch_size=12, lr=0.1, eval_every=1)
        self.assertEqual(result["baseline_val_nmse"], 0.0)
        self.assertEqual(result["final_val_nmse"], 0.0)
        self.assertEqual(result["best_step"], 0)
        torch.testing.assert_close(mlp.down_proj.weight, original, rtol=0, atol=0)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in mlp.parameters()))

    def test_zero_steps_preserves_weights(self):
        mlp = TinyMLP()
        x = torch.randn(7, 4)
        original = mlp.down_proj.weight.detach().clone()
        y = mlp(x).detach()
        result = train_down_projection(mlp, x, y, x.clone(), y.clone(), steps=0)
        self.assertEqual(result["completed_steps"], 0)
        torch.testing.assert_close(mlp.down_proj.weight, original, rtol=0, atol=0)

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "optional safetensors unavailable")
    def test_overlay_roundtrip_and_independent_base_hash(self):
        dense = TinyMLP()
        pruned = copy.deepcopy(dense)
        report = prune_mlp(pruned, [0, 2, 6])
        with torch.no_grad():
            pruned.down_proj.weight.add_(0.03)
        x = torch.randn(13, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = save_overlay({0: pruned}, {0: report}, Path(directory), base_sha256="a" * 64)
            fresh = copy.deepcopy(dense)
            with self.assertRaises(ValueError):
                apply_overlay({0: fresh}, path, expected_base_sha256="b" * 64)
            self.assertEqual(fresh.intermediate_size, 8)
            apply_overlay({0: fresh}, path, expected_base_sha256="a" * 64)
            torch.testing.assert_close(fresh(x), pruned(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()


