"""纯CPU审计公开接口测试；真实冻结结果与合成反例明确分开。"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_alp15_precision as audit

INPUTS = {
    'teacher': ('precision_20261009/teacher_development_result.json', '67b328a1ecac9de414b47680ef6476c96f6cb30dab0c736a500a902970cebaa0'),
    'kv': ('jetson_300ms_20261008/kv_candidate_full20_v1/result.json', 'db04c2ed3226e2c3c988c3f6ebd9218bd6150759d3334d25d7e193c7693ef84e'),
    'reference': ('jetson_300ms_20261009/hybrid_reference20_v3/result.json', 'e83ef209381c6e166e862795e65fc21681528fe77d90c3bf48bd4d9e048e3b59'),
    'hybrid': ('jetson_300ms_20261009/hybrid_full20_v3/result.json', 'a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6'),
}


class PrecisionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = {k: ROOT / 'docs/baselines/evidence' / value[0] for k, value in INPUTS.items()}
        cls.hashes = {k: value[1] for k, value in INPUTS.items()}
        cls.frozen = {k: audit.parse_json(path.read_bytes()) for k, path in cls.paths.items()}

    def altered(self, change):
        data = copy.deepcopy(self.frozen)
        change(data)
        with self.assertRaises(audit.AuditError):
            audit.audit(data, self.hashes)

    def arguments(self, output):
        args = []
        for key in INPUTS:
            args.extend(['--' + key, str(self.paths[key]), '--' + key + '-sha256', self.hashes[key]])
        return args + ['--output', str(output)]

    def test_real_frozen_results_reproduce_literal_means_and_60_exact_pairs(self):
        before = {k: audit.sha(p.read_bytes()) for k, p in self.paths.items()}
        result = audit.audit(self.frozen, self.hashes)
        self.assertTrue(result['integrity_passed'])
        self.assertEqual(result['pair_exact_count'], 60)
        self.assertEqual((len(result['per_request']), len(result['per_clip'])), (480, 160))
        self.assertAlmostEqual(result['summaries']['hybrid']['ADE_clip']['mean'], 1.282724219933152, places=14)
        self.assertAlmostEqual(result['summaries']['hybrid']['teacher_mean_ADE_m'], 1.2323994789272548, places=14)
        self.assertAlmostEqual(result['summaries']['hybrid']['relative_teacher_ADE_loss'], .0408347632942061, places=14)
        self.assertIsNone(result['summaries']['hybrid']['teacher_FDE'])
        self.assertIsNone(result['independent_precision_pass'])
        self.assertEqual(before, {k: audit.sha(p.read_bytes()) for k, p in self.paths.items()})

    def test_real_KV_gate_is_stable_when_runtime_sum_uses_legacy_left_fold(self):
        # 在任意Python版本强制重放3.9的逐项float求和；修复前会命中真实审计错误。
        def legacy_sum(values, start=0):
            for value in values:
                start += value
            return start
        rows = {(r['variant'], r['clip_id'], r['seed']): r for r in self.frozen['kv']['records']}
        key = ('06e20fbf-c4c7-4f47-8374-9a54f61f310b', 42)
        a = [x for point in rows[('fullprefill_10steps',) + key]['pred_xyz'] for x in point]
        b = [x for point in rows[('actualKV_10steps',) + key]['pred_xyz'] for x in point]
        old = legacy_sum((x-y)**2 for x, y in zip(a, b)) / legacy_sum(x*x for x in a)
        stable = math.fsum((x-y)**2 for x, y in zip(a, b)) / math.fsum(x*x for x in a)
        self.assertEqual(old, 1.1041936546822382e-08)
        self.assertEqual(stable, 1.1041936546822378e-08)
        with patch.object(audit, 'sum', legacy_sum, create=True):
            result = audit.audit(self.frozen, self.hashes)
        self.assertTrue(result['integrity_passed'])
        self.assertEqual(result['pair_exact_count'], 60)

    def test_real_cli_writes_all_artifacts_and_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'audit'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(self.arguments(output)), 0)
            self.assertEqual({p.name for p in output.iterdir()}, {'audit.json', 'input_manifest.json', 'per_request.csv', 'per_clip.csv', 'report.md', 'run.log'})
            manifest = json.loads((output / 'input_manifest.json').read_text())
            self.assertEqual(manifest['decision_sha256'], audit.DECISION_SHA)
            for key, info in manifest['inputs'].items():
                self.assertEqual(info['sha256_before'], info['sha256_after_audit'])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                audit.main(self.arguments(output))
            self.assertEqual(exc.exception.code, 2)

    def test_cli_wrong_sha_saves_failed_audit_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failed'
            args = self.arguments(output)
            args[args.index('--teacher-sha256') + 1] = '0' * 64
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(args), 1)
            result = json.loads((output / 'audit.json').read_text())
            self.assertFalse(result['integrity_passed'])
            self.assertEqual(result['summaries'], {})
            self.assertIn('SHA mismatch', result['diagnostics'][0])

    def test_cli_malformed_records_save_first_failure_for_all_four_inputs(self):
        cases = [('null', None), ('missing', 'MISSING'), ('string', 'bad'),
                 ('dict', {}), ('bool', True), ('non_dict_element', [42])]
        for label in INPUTS:
            for kind, value in cases:
                with self.subTest(input=label, structure=kind), tempfile.TemporaryDirectory() as directory:
                    changed = copy.deepcopy(self.frozen[label])
                    if kind == 'missing':
                        changed.pop('records')
                    else:
                        changed['records'] = value
                    source = Path(directory) / 'malformed.json'
                    source.write_text(json.dumps(changed), encoding='utf-8')
                    output = Path(directory) / 'failed'
                    args = self.arguments(output)
                    args[args.index('--' + label) + 1] = str(source)
                    args[args.index('--' + label + '-sha256') + 1] = audit.sha(source.read_bytes())
                    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/audit_alp15_precision.py')] + args,
                                               capture_output=True, text=True)
                    self.assertEqual(completed.returncode, 1)
                    self.assertNotIn('Traceback', completed.stderr)
                    result = json.loads((output / 'audit.json').read_text())
                    manifest = json.loads((output / 'input_manifest.json').read_text())
                    self.assertFalse(result['integrity_passed'])
                    self.assertEqual(result['summaries'], {})
                    self.assertTrue(result['diagnostics'])
                    self.assertNotIn('SHA mismatch', result['diagnostics'][0])
                    self.assertEqual(manifest['inputs'][label]['sha256_before'], audit.sha(source.read_bytes()))
                    self.assertTrue({'audit.json', 'report.md', 'input_manifest.json', 'run.log'}.issubset({p.name for p in output.iterdir()}))

    def test_cli_oversized_integer_ADE_and_coordinates_save_numeric_failure(self):
        for label in INPUTS:
            for field in ('ADE_m', 'pred_xyz'):
                with self.subTest(input=label, field=field), tempfile.TemporaryDirectory() as directory:
                    changed = copy.deepcopy(self.frozen[label])
                    if field == 'ADE_m':
                        changed['records'][0]['ADE_m'] = 10 ** 400
                    else:
                        changed['records'][0]['pred_xyz'][0][0] = 10 ** 400
                    source = Path(directory) / 'oversized.json'
                    source.write_text(json.dumps(changed), encoding='utf-8')
                    output = Path(directory) / 'failed'
                    args = self.arguments(output)
                    args[args.index('--' + label) + 1] = str(source)
                    args[args.index('--' + label + '-sha256') + 1] = audit.sha(source.read_bytes())
                    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/audit_alp15_precision.py')] + args,
                                               capture_output=True, text=True)
                    self.assertEqual(completed.returncode, 1)
                    self.assertNotIn('Traceback', completed.stderr)
                    result = json.loads((output / 'audit.json').read_text())
                    manifest = json.loads((output / 'input_manifest.json').read_text())
                    self.assertFalse(result['integrity_passed'])
                    self.assertEqual(result['summaries'], {})
                    self.assertIn(field, result['diagnostics'][0])
                    self.assertIn('finite float', result['diagnostics'][0])
                    self.assertEqual(manifest['inputs'][label]['sha256_after_audit'], audit.sha(source.read_bytes()))
                    self.assertTrue({'audit.json', 'report.md', 'input_manifest.json', 'run.log'}.issubset({p.name for p in output.iterdir()}))

    def test_parser_rejects_duplicate_keys_invalid_json_nonfinite_constants_and_overflow(self):
        for raw in ('{"a":1,"a":2}', '{', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(audit.AuditError):
                audit.parse_json(raw)

    def test_missing_duplicate_extra_seeds_and_wrong_teacher_group(self):
        changes = [lambda d: d['kv']['records'].pop(),
                   lambda d: d['kv']['records'].append(copy.deepcopy(d['kv']['records'][0])),
                   lambda d: d['hybrid']['records'][0].update(seed=45),
                   lambda d: d['teacher']['records'][0].update(prefix='student8'),
                   lambda d: d['teacher']['records'].pop()]
        for change in changes:
            with self.subTest(change=change):
                self.altered(change)

    def test_teacher_ADE_swapped_bindings_fail_even_when_mean_is_unchanged(self):
        def swap(data):
            rows = data['kv']['records']
            other = next(row for row in rows if row['teacher_ADE_m'] != rows[0]['teacher_ADE_m'])
            rows[0]['teacher_ADE_m'], other['teacher_ADE_m'] = other['teacher_ADE_m'], rows[0]['teacher_ADE_m']
        self.altered(swap)

    def test_short_trajectory_bad_coordinate_nonfinite_bool_and_negative_error(self):
        changes = [lambda d: d['hybrid']['records'][0]['pred_xyz'].pop(),
                   lambda d: d['hybrid']['records'][0]['pred_xyz'][0].pop(),
                   lambda d: d['hybrid']['records'][0]['pred_xyz'][0].__setitem__(0, float('nan')),
                   lambda d: d['kv']['records'][0].update(ADE_m=True),
                   lambda d: d['kv']['records'][0].update(FDE_m=-1),
                   lambda d: d['kv']['records'][0].update(e2e_ms=float('inf'))]
        for change in changes:
            with self.subTest(change=change):
                self.altered(change)

    def test_coordinate_prefix_noise_tokens_and_boundary_mismatch(self):
        changes = [lambda d: d['hybrid']['records'][0]['pred_xyz'][0].__setitem__(0, 99.),
                   lambda d: d['hybrid']['records'][0].update(prefix_sha256='0' * 64),
                   lambda d: d['hybrid']['records'][0].update(initial_noise_sha256='0' * 64),
                   lambda d: d['hybrid']['records'][0]['generated_token_ids'].__setitem__(0, 0),
                   lambda d: d['hybrid']['records'][0].update(generated_token_ids=[]),
                   lambda d: d['hybrid']['records'][0].update(future_start_token_id=True),
                   lambda d: d['hybrid']['records'][0].pop('prefix_length')]
        for change in changes:
            with self.subTest(change=change):
                self.altered(change)

    def test_changed_protection_code_input_dynamic_source_and_reference_binding(self):
        changes = [lambda d: d['hybrid'].update(protected_sha256_after='0' * 64),
                   lambda d: d['hybrid']['code_sha256'].__setitem__(next(iter(d['hybrid']['code_sha256'])), '0' * 64),
                   lambda d: d['hybrid']['input_sha256'].__setitem__(next(iter(d['hybrid']['input_sha256'])), '0' * 64),
                   lambda d: d['hybrid']['dynamic_import_sources']['_remote_module_non_scriptable'].update(source_sha256='0' * 64),
                   lambda d: d['hybrid']['paired_reference'].update(sha256='0' * 64)]
        for change in changes:
            with self.subTest(change=change):
                self.altered(change)

    def test_missing_dispatch_module_fake_totals_zero_fused_and_duplicate_targets(self):
        changes = [lambda d: d['hybrid']['records'][0]['dispatch']['modules'].pop(next(iter(d['hybrid']['records'][0]['dispatch']['modules']))),
                   lambda d: d['hybrid']['records'][0]['dispatch']['totals'].update(m1_fused=1),
                   lambda d: next(iter(d['hybrid']['records'][0]['dispatch']['modules'].values())).update(m1_fused=0),
                   lambda d: d['hybrid']['hybrid_install']['target_names'].__setitem__(0, d['hybrid']['hybrid_install']['target_names'][1])]
        for change in changes:
            with self.subTest(change=change):
                self.altered(change)

    def test_synchronously_falsified_252_target_names_are_rejected(self):
        def replace(data):
            for label in ('hybrid', 'reference'):
                names = data[label]['hybrid_install']['target_names']
                old, fake = names[0], 'vlm.model.language_model.layers.99.mlp.down_proj'
                names[0] = fake
                for row in data[label]['records']:
                    modules = row['dispatch']['modules']
                    modules[fake] = modules.pop(old)
        self.altered(replace)

    def test_saved_pair_gate_and_KV_numerical_gate_are_crosschecked(self):
        self.altered(lambda d: d['hybrid']['strict_pair_gates'][0].update(passed=False))
        self.altered(lambda d: d['kv']['numerical_gates'][0].update(absolute_ADE_delta_m=0))
        self.altered(lambda d: d['reference']['original_helper_gates'].pop())

    def test_synthetic_ratio_uses_ratio_of_clip_means_not_mean_of_ratios(self):
        data = copy.deepcopy(self.frozen)
        clips = sorted({r['clip_id'] for r in data['teacher']['records']})
        first = set(clips[:10])
        for row in data['teacher']['records']:
            row['ADE_m'] = 1. if row['clip_id'] in first else 9.
        for label in ('kv', 'reference', 'hybrid'):
            for row in data[label]['records']:
                row['teacher_ADE_m'] = 1. if row['clip_id'] in first else 9.
                row['ADE_m'] = 2. if row['clip_id'] in first else 9.
        # 合成的KV10数组完全等于fullprefill10，冻结门禁值均为零。
        full = {(r['clip_id'], r['seed']): r for r in data['kv']['records'] if r['variant'] == 'fullprefill_10steps'}
        for row in data['kv']['records']:
            if row['variant'] == 'actualKV_10steps':
                ref = full[row['clip_id'], row['seed']]
                row['pred_xyz'], row['prefix_sha256'] = ref['pred_xyz'], ref['prefix_sha256']
        for gate in data['kv']['numerical_gates']:
            gate.update(trajectory_nmse=0., absolute_ADE_delta_m=0., max_abs_xyz_delta_m=0., noise_paired=True,
                        prefix_paired=True, passed=True, numerical_gate_passed=True)
        result = audit.audit(data, self.hashes)
        self.assertAlmostEqual(result['summaries']['hybrid']['relative_teacher_ADE_loss'], .1)
        self.assertEqual(result['summaries']['hybrid']['clips_over_15pct'], 10)
        self.assertTrue(result['summaries']['hybrid']['development_point_estimate_within_15pct'])

    def test_quantile_linear_interpolation_has_independent_literal_expected_value(self):
        self.assertAlmostEqual(audit.quantile([0., 10., 20., 30.], .95), 28.5, places=13)

    def test_zero_teacher_clip_has_null_ratio_but_zero_overall_fails(self):
        data = copy.deepcopy(self.frozen)
        clip = data['teacher']['records'][0]['clip_id']
        for row in data['teacher']['records']:
            if row['clip_id'] == clip:
                row['ADE_m'] = 0.
        for label in ('kv', 'reference', 'hybrid'):
            for row in data[label]['records']:
                if row['clip_id'] == clip:
                    row['teacher_ADE_m'] = 0.
        result = audit.audit(data, self.hashes)
        selected = next(row for row in result['per_clip'] if row['candidate'] == 'hybrid' and row['clip_id'] == clip)
        self.assertIsNone(selected['ADE_ratio'])
        for row in data['teacher']['records']:
            row['ADE_m'] = 0.
        for label in ('kv', 'reference', 'hybrid'):
            for row in data[label]['records']:
                row['teacher_ADE_m'] = 0.
        with self.assertRaisesRegex(audit.AuditError, 'overall teacher ADE is zero'):
            audit.audit(data, self.hashes)


if __name__ == '__main__':
    unittest.main()
