"""CPU checks: python -m unittest compression.test_joint_mlp_distill."""
import copy
import json
import math
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch

from compression.joint_mlp_distill import train_joint_mlp
from compression.mlp_pruning import load_pruned_mlp_state, prune_mlp
from compression.test_mlp_pruning import TinyMLP


class JointMLPDistillationTests(unittest.TestCase):
    def test_raw_step_log_matches_updates_and_marks_cpu_memory_unavailable(self):
        mlp = TinyMLP(dtype=torch.bfloat16)
        x, vx = self._inputs()
        with torch.no_grad():
            y, vy = mlp(x) * 1.1, mlp(vx) * 1.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'steps.jsonl'
            result = train_joint_mlp(mlp, x, y, vx, vy, steps=3, lr=0.001,
                                     eval_every=2, metrics_path=path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            steps = [row for row in rows if row['event'] == 'step']
            self.assertEqual([r['step'] for r in steps], [1, 2, 3])
            self.assertEqual([r['step'] for r in steps if 'val_nmse' in r], [2, 3])
            self.assertEqual(rows[0]['event'], 'baseline')
            self.assertEqual(rows[-1]['event'], 'completed')
            self.assertEqual(result['step_history'], [{k: v for k, v in row.items() if k != 'event'} for row in steps])
            for row in steps:
                self.assertEqual(row['lr'], 0.001)
                self.assertTrue(math.isfinite(row['gradient_norm']))
                self.assertIsNone(row['cuda_allocated_bytes'])
            with self.assertRaises(FileExistsError):
                train_joint_mlp(mlp, x, y, vx, vy, steps=1, metrics_path=path)

    def test_nonfinite_failure_is_recorded_without_claiming_completion(self):
        mlp = TinyMLP(dtype=torch.bfloat16)
        x, vx = self._inputs()
        with torch.no_grad():
            y, vy = mlp(x), mlp(vx)
        vy[0, 0] = float('nan')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failed.jsonl'
            with self.assertRaises(FloatingPointError):
                train_joint_mlp(mlp, x, y, vx, vy, metrics_path=path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[-1]['event'], 'failed')
            self.assertTrue(rows[-1]['restores_baseline'])
            self.assertFalse(any(row['event'] == 'completed' for row in rows))

    def setUp(self):
        torch.manual_seed(321)

    def _inputs(self, dtype=torch.bfloat16):
        return torch.randn(48, 4).to(dtype), torch.randn(24, 4).to(dtype)

    def _assert_frozen_bf16(self, mlp):
        self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in mlp.parameters()))
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in mlp.parameters()))

    def test_joint_training_updates_all_three_and_improves_exported_validation(self):
        mlp = TinyMLP(dtype=torch.bfloat16)
        prune_mlp(mlp, [0, 2, 4, 6])
        teacher = copy.deepcopy(mlp)
        with torch.no_grad():
            teacher.gate_proj.weight.mul_(1.3)
            teacher.up_proj.weight.mul_(0.8)
            teacher.down_proj.weight.mul_(1.5)
        x, vx = self._inputs()
        with torch.no_grad():
            y, vy = teacher(x), teacher(vx)
        original = {name: tensor.detach().clone() for name, tensor in mlp.state_dict().items()}
        result = train_joint_mlp(mlp, x, y, vx, vy, steps=100, batch_size=24,
                                 lr=0.003, eval_every=5, seed=42)
        self.assertLess(result['final_val_nmse'], result['baseline_val_nmse'])
        self.assertGreater(result['best_step'], 0)
        self.assertEqual(result['export_dtype'], 'torch.bfloat16')
        for name, value in mlp.state_dict().items():
            self.assertFalse(torch.equal(value, original[name]), f'{name} did not update')
        with torch.no_grad():
            prediction = mlp(vx).float()
            target = vy.float()
            actual = ((prediction - target).square().sum() / target.square().sum()).item()
        self.assertAlmostEqual(result['final_val_nmse'], actual, places=6)
        self.assertTrue(mlp.training)
        self._assert_frozen_bf16(mlp)

    def test_untrained_bf16_baseline_wins_when_training_worsens_validation(self):
        mlp = TinyMLP(dtype=torch.bfloat16)
        x, vx = self._inputs()
        with torch.no_grad():
            y, vy = -mlp(x), mlp(vx)
        original = {name: tensor.detach().clone() for name, tensor in mlp.state_dict().items()}
        result = train_joint_mlp(mlp, x, y, vx, vy, steps=8, batch_size=24,
                                 lr=0.1, eval_every=1)
        self.assertEqual(result['baseline_val_nmse'], 0.0)
        self.assertEqual(result['final_val_nmse'], 0.0)
        self.assertEqual(result['best_step'], 0)
        for name, value in mlp.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self._assert_frozen_bf16(mlp)

    def test_nonfinite_update_restores_original_baseline_not_earlier_candidate(self):
        mlp = TinyMLP(dtype=torch.bfloat16)
        teacher = copy.deepcopy(mlp)
        with torch.no_grad():
            teacher.down_proj.weight.mul_(1.5)
        x, vx = self._inputs()
        with torch.no_grad():
            y, vy = teacher(x), teacher(vx)
        original = {name: tensor.detach().clone() for name, tensor in mlp.state_dict().items()}
        adam_step = torch.optim.Adam.step
        calls = 0

        def corrupt_second_update(optimizer, *args, **kwargs):
            nonlocal calls
            result = adam_step(optimizer, *args, **kwargs)
            calls += 1
            if calls == 2:
                with torch.no_grad():
                    optimizer.param_groups[0]['params'][0].view(-1)[0] = float('nan')
            return result

        with mock.patch.object(torch.optim.Adam, 'step', corrupt_second_update):
            with self.assertRaises(FloatingPointError):
                train_joint_mlp(mlp, x, y, vx, vy, steps=4, batch_size=24,
                                lr=0.003, eval_every=1)
        self.assertEqual(calls, 2)
        for name, value in mlp.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self._assert_frozen_bf16(mlp)

    def test_nonfinite_teacher_target_restores_baseline_and_freezes(self):
        mlp = TinyMLP(dtype=torch.bfloat16)
        x, vx = self._inputs()
        with torch.no_grad():
            y, vy = mlp(x), mlp(vx)
        vy[0, 0] = float('nan')
        original = {name: tensor.detach().clone() for name, tensor in mlp.state_dict().items()}
        with self.assertRaises(FloatingPointError):
            train_joint_mlp(mlp, x, y, vx, vy, steps=2)
        for name, value in mlp.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self._assert_frozen_bf16(mlp)

    def test_trained_structural_state_roundtrip(self):
        dense = TinyMLP(dtype=torch.bfloat16)
        mlp = copy.deepcopy(dense)
        ids = [0, 2, 4, 6]
        prune_mlp(mlp, ids)
        teacher = copy.deepcopy(mlp)
        with torch.no_grad():
            teacher.down_proj.weight.mul_(1.2)
        x, vx = self._inputs()
        with torch.no_grad():
            y, vy = teacher(x), teacher(vx)
        train_joint_mlp(mlp, x, y, vx, vy, steps=12, lr=0.003, eval_every=2)
        state = {name: tensor.detach().clone() for name, tensor in mlp.state_dict().items()}
        load_pruned_mlp_state(dense, state, ids)
        with torch.no_grad():
            torch.testing.assert_close(dense(vx), mlp(vx), rtol=0, atol=0)
        self._assert_frozen_bf16(mlp)

    def test_zero_steps_restores_original_mode_and_weights(self):
        mlp = TinyMLP(dtype=torch.bfloat16).eval()
        x, vx = self._inputs()
        with torch.no_grad():
            y, vy = mlp(x), mlp(vx)
        original = {name: tensor.detach().clone() for name, tensor in mlp.state_dict().items()}
        result = train_joint_mlp(mlp, x, y, vx, vy, steps=0)
        self.assertEqual(result['completed_steps'], 0)
        self.assertFalse(mlp.training)
        for name, value in mlp.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self._assert_frozen_bf16(mlp)


if __name__ == '__main__':
    unittest.main()
