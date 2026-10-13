"""KV/少步诊断公开门禁；不会加载模型或调用 GPU。"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from alp15_kv_steps_probe import validate_reference_pairs, numerical_gate, summarize_complete, storage_inventory_by_component, cache_candidate_allowed, CANDIDATE_DECISION


class FrozenPairsTests(unittest.TestCase):
    def fixture(self):
        return [{'clip_id': f'clip-{clip:02}', 'seed': seed, 'ADE_m': 1.0,
                 'initial_noise_sha256': f'{clip * 3 + seed:064x}'}
                for clip in range(20) for seed in (42, 43, 44)]

    def test_accepts_fixed_twenty_clip_blocked_pairs(self):
        teacher, awq = self.fixture(), self.fixture()
        clips, refs = validate_reference_pairs(teacher, awq)
        self.assertEqual(len(clips), 20)
        self.assertEqual(len(refs['teacher']), 60)

    def test_refuses_missing_duplicate_unpaired_and_nonfinite_runs(self):
        teacher, awq = self.fixture(), self.fixture()
        for kind in ('missing', 'duplicate', 'noise', 'nonfinite'):
            changed = copy.deepcopy(awq)
            if kind == 'missing': changed.pop()
            elif kind == 'duplicate': changed[-1] = changed[0]
            elif kind == 'noise': changed[0]['initial_noise_sha256'] = 'f' * 64
            else: changed[0]['ADE_m'] = float('nan')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_reference_pairs(teacher, changed)


class NumericalGateTests(unittest.TestCase):
    def test_candidate_policy_preserves_strict_and_refuses_pairing_changes(self):
        gate = {'passed': False, 'noise_paired': True, 'prefix_paired': True}
        self.assertFalse(cache_candidate_allowed(gate))
        with self.assertRaises(ValueError):
            cache_candidate_allowed(gate, 'new_candidate')
        self.assertTrue(cache_candidate_allowed(gate, 'new_candidate', CANDIDATE_DECISION))
        for field in ('noise_paired', 'prefix_paired'):
            changed = dict(gate, **{field: False})
            self.assertFalse(cache_candidate_allowed(changed, 'new_candidate', CANDIDATE_DECISION))

    def test_gate_requires_trajectory_and_ade_and_noise_agreement(self):
        reference = {'pred_xyz': [[1., 0., 0.], [2., 0., 0.]],
                     'ADE_m': 1., 'initial_noise_sha256': 'a' * 64}
        self.assertTrue(numerical_gate(reference, copy.deepcopy(reference))['passed'])
        for field, value in [('ADE_m', 1.002), ('initial_noise_sha256', 'b' * 64),
                             ('pred_xyz', [[1.2, 0., 0.], [2., 0., 0.]])]:
            changed = copy.deepcopy(reference)
            changed[field] = value
            self.assertFalse(numerical_gate(reference, changed)['passed'])

    def test_summary_cannot_remove_failed_or_missing_trajectories(self):
        rows = [{'clip_id': clip, 'seed': seed, 'ADE_m': ade, 'FDE_m': ade * 2,
                 'e2e_ms': 100., 'status': 'completed', 'teacher_ADE_m': 4.}
                for clip, ade in [('a', 1.), ('b', 3.)] for seed in (42, 43, 44)]
        report = summarize_complete(rows, ['a', 'b'])
        self.assertEqual(report['independent_clips'], 2)
        self.assertEqual(report['mean_ADE_m'], 2.)
        self.assertEqual(report['clip_ADE_p95_m'], 2.9)
        self.assertEqual(report['paired_teacher_mean_ADE_m'], 4.)
        self.assertEqual(report['relative_teacher_ADE_loss'], -.5)
        self.assertEqual(report['e2e_max_ms'], 100.)
        self.assertEqual(report['requests_over_300ms'], 0)
        with self.assertRaises(ValueError): summarize_complete(rows[:-1], ['a', 'b'])
        rows[0]['status'] = 'failed'
        with self.assertRaises(ValueError): summarize_complete(rows, ['a', 'b'])

    def test_summary_requires_paired_teacher_and_counts_tail_timeouts(self):
        rows = [{'clip_id': 'a', 'seed': seed, 'ADE_m': 1.16, 'FDE_m': 2.,
                 'e2e_ms': latency, 'status': 'completed', 'teacher_ADE_m': 1.}
                for seed, latency in zip((42, 43, 44), (100., 300., 500.))]
        report = summarize_complete(rows, ['a'])
        self.assertFalse(report['development_ADE_within_15pct'])
        self.assertEqual(report['requests_over_300ms'], 1)
        self.assertEqual(report['e2e_max_ms'], 500.)
        for invalid in (float('nan'), 0.):
            changed = copy.deepcopy(rows)
            for row in changed: row['teacher_ADE_m'] = invalid
            with self.assertRaises(ValueError): summarize_complete(changed, ['a'])


class StorageBoundaryTests(unittest.TestCase):
    def test_inventory_counts_shared_storage_once_and_reports_component_aliases(self):
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            self.skipTest('需服务器 torch CPU；本地跳过不能作数值通过')
        import torch
        from torch import nn
        model = nn.Module()
        model.vlm = nn.Module()
        model.vlm.lm_head = nn.Linear(2, 2, bias=False)
        model.vlm.model = nn.Module()
        model.vlm.model.language_model = nn.Module()
        model.vlm.model.language_model.embed_tokens = nn.Embedding(2, 2)
        model.vlm.model.language_model.embed_tokens.weight = model.vlm.lm_head.weight
        model.expert = nn.Linear(2, 1, bias=False)
        result = storage_inventory_by_component(model)
        self.assertEqual(result['persistent_storage_bytes'], 24)
        self.assertEqual(result['groups']['expert']['storage_bytes'], 8)
        self.assertEqual(len(result['shared_storages']), 1)


if __name__ == '__main__':
    unittest.main()
