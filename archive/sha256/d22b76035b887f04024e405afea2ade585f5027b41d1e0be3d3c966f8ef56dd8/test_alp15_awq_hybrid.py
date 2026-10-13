"""CPU 合成回归：不能作为真实252矩阵CUDA、轨迹或性能验收。"""
from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import alp15_awq_hybrid as hybrid
from alp15_awq_full_packed import PackedAWQLinear


def reference(bias=True):
    return PackedAWQLinear(torch.full((3, 64), 0x79, dtype=torch.uint8),
        torch.tensor([[7.], [0.], [7.]], dtype=torch.bfloat16),
        torch.tensor([2., 1.] * 64, dtype=torch.bfloat16),
        torch.tensor([1., -2., 3.], dtype=torch.bfloat16) if bias else None)


def trajectory(clip='synthetic', seed=42):
    return {'clip_id': clip, 'seed': seed, 'decision_id': hybrid.DECISION_ID, 'steps': 10,
        'status': 'completed', 'pred_xyz': [[1., 2., 0.]] * 64, 'ADE_m': 1., 'FDE_m': 2.,
        'initial_noise_sha256': 'a' * 64, 'prefix_sha256': 'b' * 64,
        'generated_token_ids': [10, 20], 'future_start_token_id': 20, 'prefix_length': 5}


def provenance():
    result = {key: {} for key in ('code_sha256', 'runtime_source_sha256', 'input_sha256')}
    result.update(decision_id=hybrid.DECISION_ID, steps=10, request_route='fullprefill',
                  text_seed=42, trajectory_seeds=[42, 43, 44])
    return result


class MockFused(nn.Module):
    """只模拟分派及storage共享；不模拟CUDA数值kernel。"""
    def __init__(self, original):
        super().__init__()
        for name in ('qweight', 'amax', 'pre_scale', 'bias'):
            self.register_buffer(name, getattr(original, name))
        self.out_features = original.out_features

    def forward(self, inputs):
        return torch.full((*inputs.shape[:-1], self.out_features), 123., dtype=inputs.dtype)


class HybridTests(unittest.TestCase):
    def test_only_actual_m1_uses_lazy_factory(self):
        original = reference()
        candidate = hybrid.HybridAWQLinear.from_reference(original)
        with mock.patch.object(hybrid, '_make_fused', side_effect=MockFused) as factory:
            with mock.patch.object(original, 'forward', wraps=original.forward) as forward:
                for m in (2, 17, 256, 3086):
                    value = torch.ones((1, m, 128), dtype=torch.bfloat16)
                    torch.testing.assert_close(candidate(value), original(value), rtol=0, atol=0)
                    self.assertIsNone(candidate.fused)
                self.assertEqual(factory.call_count, 0)
                self.assertEqual(forward.call_count, 8)
                output = candidate(torch.ones(1, 1, 128, dtype=torch.bfloat16))
                self.assertTrue(torch.equal(output, torch.full((1, 1, 3), 123., dtype=torch.bfloat16)))
                candidate(torch.ones(128, dtype=torch.bfloat16))
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(forward.call_count, 8)
        self.assertEqual(candidate.dispatch, {'m1_fused': 2, 'm_gt1_reference': 4, 'disabled_reference': 0})

    def test_disabled_is_original_bitwise_for_all_m_and_bias(self):
        for has_bias in (False, True):
            original = reference(has_bias)
            candidate = hybrid.HybridAWQLinear.from_reference(original, enabled=False)
            with mock.patch.object(hybrid, '_make_fused', side_effect=AssertionError('must remain lazy')):
                for m in (1, 2, 17, 256, 3086):
                    value = torch.randn(m, 128).bfloat16()
                    self.assertTrue(torch.equal(candidate(value), original(value)))
            self.assertIsNone(candidate.fused)
            self.assertEqual(candidate.dispatch['disabled_reference'], 5)

    def test_buffer_objects_and_unique_storage_are_shared_after_factory(self):
        original = reference()
        candidate = hybrid.HybridAWQLinear.from_reference(original)
        before = candidate.storage_bytes()
        with mock.patch.object(hybrid, '_make_fused', side_effect=MockFused):
            candidate(torch.ones(1, 128, dtype=torch.bfloat16))
        for name in ('qweight', 'amax', 'pre_scale', 'bias'):
            self.assertIs(getattr(original, name), getattr(candidate, name))
            self.assertIs(getattr(original, name), getattr(candidate.fused, name))
        self.assertEqual(before, candidate.storage_bytes())
        self.assertEqual(before, original.storage_bytes())
        self.assertEqual(len(list(candidate.parameters())), 0)
        self.assertFalse(any('dense' in key for key in candidate.state_dict()))

    def test_enabled_m1_cpu_refuses_without_fallback(self):
        original = reference()
        candidate = hybrid.HybridAWQLinear.from_reference(original)
        with mock.patch.object(original, 'forward', side_effect=AssertionError('no fallback')):
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                candidate(torch.ones(1, 128, dtype=torch.bfloat16))
        self.assertEqual(candidate.dispatch['m1_fused'], 0)

    def test_invalid_input_and_reference_are_explicit(self):
        for value in (torch.ones(1, 127), torch.ones(0, 128), torch.tensor(1.),
                      torch.ones(1, 128, dtype=torch.int64), torch.ones(1, 128, device='meta')):
            with self.subTest(shape=value.shape):
                with self.assertRaises(ValueError):
                    hybrid.HybridAWQLinear.from_reference(reference(), False)(value)
        with self.assertRaises(TypeError): hybrid.HybridAWQLinear.from_reference(nn.Linear(128, 3))
        with self.assertRaises(TypeError): hybrid.HybridAWQLinear.from_reference(reference(), 'yes')

    def test_m1_validation_does_not_cast_or_make_contiguous(self):
        import alp15_awq_kernel
        original = reference()
        candidate = hybrid.HybridAWQLinear.from_reference(original)
        # 原kernel先拒绝CPU；mock保留输入对象，以证明wrapper未cast/contiguous。
        observed = []
        class Observe(nn.Module):
            def forward(self, value):
                observed.append(value)
                raise ValueError('unsupported original input')
        with mock.patch.object(hybrid, '_make_fused', return_value=Observe()):
            values = [torch.ones(1, 256, dtype=torch.bfloat16)[:, ::2],
                      torch.ones(1, 128, dtype=torch.float32),
                      torch.ones(1, 128, dtype=torch.bfloat16, requires_grad=True)]
            for value in values:
                with self.assertRaisesRegex(ValueError, 'unsupported'):
                    candidate(value)
                self.assertIs(observed[-1], value)

    def test_disabled_and_large_m_never_import_candidate_or_triton(self):
        program = '''
import sys, importlib.abc, torch
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('triton') or fullname == 'alp15_awq_kernel':
            raise AssertionError('unexpected candidate import: ' + fullname)
sys.meta_path.insert(0, Block())
from alp15_awq_hybrid import HybridAWQLinear
from alp15_awq_full_packed import PackedAWQLinear
original = PackedAWQLinear(torch.zeros((1,64),dtype=torch.uint8),torch.ones(1,1,dtype=torch.bfloat16),torch.ones(128,dtype=torch.bfloat16))
HybridAWQLinear.from_reference(original,False)(torch.ones(1,128,dtype=torch.bfloat16))
HybridAWQLinear.from_reference(original)(torch.ones(2,128,dtype=torch.bfloat16))
assert 'triton' not in sys.modules and 'alp15_awq_kernel' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', program], cwd=SCRIPTS, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.packed = self.root / 'packed'
        self.packed.mkdir()
        self.model = nn.Module()
        self.model.vlm = nn.Module()
        self.model.vlm.model = nn.Module()
        self.model.vlm.model.language_model = nn.Module()
        self.model.vlm.model.language_model.layers = nn.ModuleList([nn.Module() for _ in range(36)])
        self.model.expert = nn.Linear(2, 2)
        self.model.vlm.model.visual = nn.Linear(2, 2)
        for layer in self.model.vlm.model.language_model.layers:
            layer.self_attn, layer.mlp = nn.Module(), nn.Module()
        self.rows, self.asset_fixtures = [], {}
        for name in sorted(hybrid.TARGET_NAMES):
            matrix = reference()
            owner, attribute = name.rsplit('.', 1)
            setattr(self.model.get_submodule(owner), attribute, matrix)
            path = self.packed / (name.replace('.', '_') + '.safetensors')
            # 本机无safetensors时仍验证真实SHA和installer边界；读取明确mock，
            # 不把合成fixture称为真实safetensors格式/CUDA验收。
            path.write_bytes(b'CPU mocked packed fixture: ' + name.encode())
            self.asset_fixtures[str(path)] = copy.deepcopy(matrix.state_dict())
            self.rows.append({'name': name[len('vlm.model.'):], 'shape': [3, 128], 'file': str(path),
                'sha256': hybrid.sha256_file(path), 'amax_dtype': 'torch.bfloat16', 'pre_scale_dtype': 'torch.bfloat16'})
        self.write_source()
        def load_fixture(path, device='cpu'):
            self.assertEqual(str(device), 'cpu')
            state = self.asset_fixtures[str(path)]
            return PackedAWQLinear(**{key: value.clone() for key, value in state.items()})
        self.load_mock = mock.patch.object(PackedAWQLinear, 'load', side_effect=load_fixture)
        self.load_mock.start()

    def tearDown(self):
        self.load_mock.stop()
        self.temp.cleanup()

    def write_source(self):
        path = self.root / 'result.json'
        path.write_text(json.dumps({'status': 'completed_awq_development_pilot', 'packed_layers': self.rows}))
        self.sha = hybrid.sha256_file(path)

    def manifest(self):
        with mock.patch.object(hybrid, 'AWQ_RESULT_SHA256', self.sha):
            return hybrid.build_packed_manifest(self.root)

    def test_installs_exact252_preserving_protected_modules_and_buffers(self):
        expert = copy.deepcopy(self.model.expert.state_dict())
        visual = copy.deepcopy(self.model.vlm.model.visual.state_dict())
        original = {name: self.model.get_submodule(name) for name in hybrid.TARGET_NAMES}
        with mock.patch.object(hybrid, 'AWQ_RESULT_SHA256', self.sha):
            receipt = hybrid.install_hybrid_awq(self.model, self.manifest(), enabled=False)
        self.assertEqual(receipt['coverage'], '252/252')
        self.assertEqual(receipt['packed_bytes_before'], receipt['packed_bytes_after'])
        for name, reference_module in original.items():
            self.assertIs(self.model.get_submodule(name).reference, reference_module)
            self.assertIsNone(self.model.get_submodule(name).fused)
        for key in expert: self.assertTrue(torch.equal(expert[key], self.model.expert.state_dict()[key]))
        for key in visual: self.assertTrue(torch.equal(visual[key], self.model.vlm.model.visual.state_dict()[key]))
        self.assertEqual(hybrid.dispatch_inventory(self.model)['module_count'], 252)

    def test_duplicate_missing_extra_shape_and_dtype_are_rejected(self):
        base = copy.deepcopy(self.rows)
        bad_sets = [base[:-1], base + [copy.deepcopy(base[0])],
                    [dict(base[0], name='visual.proj')] + base[1:],
                    [dict(base[0], shape=[3, 127])] + base[1:],
                    [dict(base[0], amax_dtype='torch.float32')] + base[1:]]
        for rows in bad_sets:
            self.rows = rows
            self.write_source()
            with self.subTest(rows=len(rows)), mock.patch.object(hybrid, 'AWQ_RESULT_SHA256', self.sha):
                with self.assertRaises(ValueError): hybrid.build_packed_manifest(self.root)

    def test_manifest_must_match_original_artifact(self):
        manifest = self.manifest()
        manifest['layers'][0]['shape'] = [4, 128]
        with mock.patch.object(hybrid, 'AWQ_RESULT_SHA256', self.sha):
            with self.assertRaisesRegex(ValueError, '逐条'): hybrid.validate_packed_manifest(manifest)
        with self.assertRaisesRegex(ValueError, '冻结'): hybrid.validate_packed_manifest(self.manifest())

    def test_wrong_asset_hash_and_format_rejected(self):
        row = self.rows[0]
        row['sha256'] = '0' * 64
        self.write_source()
        with mock.patch.object(hybrid, 'AWQ_RESULT_SHA256', self.sha):
            with self.assertRaisesRegex(ValueError, 'SHA256'): hybrid.build_packed_manifest(self.root)

    def test_mismatched_last_buffer_leaves_entire_model_unmodified(self):
        name = sorted(hybrid.TARGET_NAMES)[-1]
        self.model.get_submodule(name).pre_scale[0] = 3
        with mock.patch.object(hybrid, 'AWQ_RESULT_SHA256', self.sha):
            with self.assertRaisesRegex(ValueError, 'buffer'):
                hybrid.install_hybrid_awq(self.model, self.manifest())
        self.assertTrue(all(type(self.model.get_submodule(name)) is PackedAWQLinear for name in hybrid.TARGET_NAMES))


class RequestGateTests(unittest.TestCase):
    def test_import_receipt_survives_generated_module_owner_process_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            captured = dynamic_subprocess_receipt(Path(folder).resolve())
            original = next(iter(captured['dynamic_import_sources'].values()))['original_path']
            self.assertFalse(Path(original).exists())  # 真实Torch atexit清理已发生。
            hybrid.verify_import_provenance(captured)

    def test_unknown_tmp_source_is_not_ignored_or_snapshotted_as_torch(self):
        runner = importlib.import_module('run_alp15_awq_hybrid_eval')
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder).resolve() / 'unknown_module.py'
            source.write_text('value=1\n')
            module = types.ModuleType('_unknown_fixture_module')
            module.__file__ = str(source)
            with mock.patch.dict(sys.modules, {'_unknown_fixture_module': module}):
                inventory = runner.imported_source_receipt(Path(folder) / 'snapshots', {}, {})
            self.assertIn(str(source), inventory)
            source.unlink()
            with self.assertRaises(FileNotFoundError):
                hybrid.verify_hash_inventory({str(source): inventory[str(source)]}, 'actual_import_sha256')

    def test_missing_dynamic_original_is_not_reconstructed_from_generator(self):
        with tempfile.TemporaryDirectory() as folder:
            module = types.ModuleType(hybrid.TORCH_REMOTE_NAME)
            module.__file__ = str(Path(folder) / (hybrid.TORCH_REMOTE_NAME + '.py'))
            with self.assertRaises(FileNotFoundError):
                hybrid.snapshot_torch_remote_module(module, Path(folder) / 'snapshots')
            self.assertFalse((Path(folder) / 'snapshots').exists())

    def test_strict_gate_cannot_use_actual_kv_exemption(self):
        runner = importlib.import_module('run_alp15_awq_hybrid_eval')
        base = trajectory()
        self.assertTrue(runner.strict_pair_gate(base, copy.deepcopy(base))['passed'])
        for key, value in [('ADE_m', 1.1), ('generated_token_ids', [11, 20]),
                           ('prefix_sha256', 'c' * 64),
                           ('initial_noise_sha256', 'c' * 64), ('prefix_length', 6),
                           ('pred_xyz', [[9., 2., 0.]] * 64)]:
            changed = dict(base, **{key: value})
            self.assertFalse(runner.strict_pair_gate(base, changed)['passed'])
        with self.assertRaisesRegex(ValueError, '边界'):
            runner.strict_pair_gate(base, dict(base, future_start_token_id=21))

    def test_same_provenance_rejects_each_changed_source_and_step(self):
        runner = importlib.import_module('run_alp15_awq_hybrid_eval')
        base = provenance()
        runner.assert_same_provenance(base, base)
        for key in base:
            changed = dict(base, **{key: 'changed'})
            with self.assertRaises(RuntimeError): runner.assert_same_provenance(base, changed)

    def test_expansion_requires_new_complete_strict_smoke(self):
        runner = importlib.import_module('run_alp15_awq_hybrid_eval')
        base = provenance()
        rows = [dict(trajectory(clip, seed), passed=True)
                for clip in ('a', 'b') for seed in (42, 43, 44)]
        smoke = dict(base, status='completed_awq_hybrid_development_eval', awq_forward='hybrid_m1',
            clips=['a', 'b'], strict_pair_gates=copy.deepcopy(rows),
            original_helper_gates=copy.deepcopy(rows), records=copy.deepcopy(rows))
        # 此测试只测缺失/重复/旧状态的early gate；完整正例在BoundCaptureTests。
        for key in ('strict_pair_gates', 'original_helper_gates', 'records'):
            broken = copy.deepcopy(smoke)
            broken[key][-1] = broken[key][0]
            with self.assertRaises(RuntimeError): runner.validate_smoke(broken, base, ['a', 'b', 'c'])
        broken = dict(smoke, status='completed_kv_steps_development_probe')
        with self.assertRaises(RuntimeError): runner.validate_smoke(broken, base, ['a', 'b', 'c'])

    def test_one_point_and_nan_in_either_trajectory_are_rejected(self):
        runner = importlib.import_module('run_alp15_awq_hybrid_eval')
        for invalid in (dict(trajectory(), pred_xyz=[[1., 2., 0.]]),
                        dict(trajectory(), pred_xyz=[[float('nan'), 2., 0.]] * 64),
                        dict(trajectory(), ADE_m=float('nan')),
                        dict(trajectory(), decision_id='old_actualKV_exemption')):
            with self.assertRaises(ValueError): runner.strict_pair_gate(trajectory(), invalid)
            with self.assertRaises(ValueError): runner.strict_pair_gate(invalid, trajectory())

    def test_source_receipt_rejects_changed_old_kernel(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError): hybrid.frozen_source_receipt(folder)
            for name in hybrid.FROZEN_SOURCE:
                (Path(folder) / name).write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'SHA256'): hybrid.frozen_source_receipt(folder)


class BoundCaptureTests(unittest.TestCase):
    """独立JSON/tensor失败fixture，验证来源门禁；不生成GPU通过证据。"""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.folder = self.root / 'reference2'
        self.folder.mkdir()
        self.capture = self.folder / 'decode_activations'
        self.capture.mkdir()
        self.runner = importlib.import_module('run_alp15_awq_hybrid_eval')
        self.base = provenance()
        for key in ('code_sha256', 'runtime_source_sha256', 'input_sha256', 'actual_import_sha256'):
            path = self.root / (key + '.fixture')
            path.write_bytes(key.encode())
            self.base[key] = {str(path): hybrid.sha256_file(path)}
        self.base.update(status='completed_awq_hybrid_development_eval', exit_code=0,
            source_snapshot_directory=str(self.folder / 'runtime_source_snapshots'), dynamic_import_sources={}, virtual_import_sources={},
            awq_forward='reference', clips=['a', 'b'], protected_sha256_before='c' * 64,
            protected_sha256_after='c' * 64,
            records=[trajectory(clip, seed) for clip in ('a', 'b') for seed in (42, 43, 44)])
        rows = []
        for name in sorted(hybrid.TARGET_NAMES):
            path = self.capture / (name.replace('.', '_') + '.pt')
            torch.save(torch.ones(1, 128, dtype=torch.bfloat16), path)
            rows.append({'name': name, 'file': path.name, 'sha256': hybrid.sha256_file(path),
                         'shape': [1, 128], 'dtype': 'torch.bfloat16'})
        self.base['decode_activation_capture'] = self.runner.write_final_capture_manifest(
            {'folder': str(self.capture), 'activations': rows, 'clip_id': 'b'}, self.base, self.folder)
        self.result_path = self.folder / 'result.json'
        self.result_path.write_text(json.dumps(self.base))
        self.runner.write_final_reference_receipt(self.base, self.folder)
        self.manifest_path = self.capture / 'manifest.json'
        self.receipt_path = self.folder / 'reference_final_receipt.json'

    def tearDown(self): self.temp.cleanup()

    def refresh_hash_binding(self, manifest=None, reference_result=None):
        # 故意让错误语义具有一致SHA，确保门禁不只是发现文件被编辑。
        if manifest is not None: self.manifest_path.write_text(json.dumps(manifest))
        reference_result = reference_result or json.loads(self.result_path.read_text())
        reference_result['decode_activation_capture']['sha256'] = hybrid.sha256_file(self.manifest_path)
        self.result_path.write_text(json.dumps(reference_result))
        receipt = json.loads(self.receipt_path.read_text())
        receipt['activation_manifest_sha256'] = hybrid.sha256_file(self.manifest_path)
        receipt['reference_result_sha256'] = hybrid.sha256_file(self.result_path)
        self.receipt_path.write_text(json.dumps(receipt))

    def probe_fixture(self):
        actual, hashes, manifest, baseline = hybrid.load_bound_capture(self.manifest_path, self.folder)
        probe = {'status': 'passed_all252_cuda_m1_probe_with_actual', 'coverage': '252/252',
            'actual_activation_coverage': '252/252', 'decision_id': hybrid.DECISION_ID,
            'matrices': [{'name': name, 'status': 'passed', 'activations': [{'kind': 'actual_decode_bf16'}]}
                         for name in sorted(hybrid.TARGET_NAMES)],
            'source_sha256': dict(baseline['code_sha256']), 'input_sha256': hashes,
            'activation_provenance': {key: baseline[key] for key in ('code_sha256', 'runtime_source_sha256',
                'input_sha256', 'actual_import_sha256', 'dynamic_import_sources', 'source_snapshot_directory', 'virtual_import_sources')},
            'activation_binding': {'manifest': str(self.manifest_path),
                'manifest_sha256': hybrid.sha256_file(self.manifest_path),
                'reference_result': str(self.result_path), 'reference_result_sha256': hybrid.sha256_file(self.result_path),
                'final_receipt_sha256': hybrid.sha256_file(self.receipt_path)}}
        return probe

    def test_non_circular_success_binding_and_probe_positive_fixture(self):
        actual, hashes, manifest, result = hybrid.load_bound_capture(self.manifest_path, self.folder)
        self.assertEqual(len(actual), 252)
        self.assertNotIn('reference_result_sha256', manifest)
        self.assertNotIn('final_receipt_sha256', result['decode_activation_capture'])
        self.runner.validate_probe_binding(self.probe_fixture(), self.base, self.base, self.result_path)

    def test_full252_binding_survives_actual_torch_generator_process_exit(self):
        captured = dynamic_subprocess_receipt(self.folder / 'runtime_source_snapshots')
        manifest = json.loads(self.manifest_path.read_text())
        for key in ('actual_import_sha256', 'source_snapshot_directory', 'dynamic_import_sources', 'virtual_import_sources'):
            self.base[key] = captured[key]
            manifest[key] = captured[key]
        self.refresh_hash_binding(manifest, self.base)
        self.base = json.loads(self.result_path.read_text())
        actual, _, _, _ = hybrid.load_bound_capture(self.manifest_path, self.folder)
        self.assertEqual(len(actual), 252)
        self.runner.validate_probe_binding(self.probe_fixture(), self.base, self.base, self.result_path)

    def test_legacy_v2_capture_cannot_be_relabelled_or_backfilled(self):
        manifest = json.loads(self.manifest_path.read_text())
        manifest['schema'] = 'alp15-m1-activations-v2'
        self.refresh_hash_binding(manifest)
        with self.assertRaisesRegex(ValueError, '旧结果不可补造'):
            hybrid.load_bound_capture(self.manifest_path, self.folder)

    def test_orphan_capture_without_receipt_is_rejected(self):
        self.receipt_path.unlink()
        with self.assertRaises(FileNotFoundError): hybrid.load_bound_capture(self.manifest_path)

    def test_failed_reference_even_consistent_receipt_is_rejected(self):
        failed = dict(self.base, status='failed', exit_code=1)
        self.refresh_hash_binding(reference_result=failed)
        with self.assertRaisesRegex(ValueError, '成功reference'):
            hybrid.load_bound_capture(self.manifest_path)

    def test_runtime_file_drift_is_rejected(self):
        path = next(iter(self.base['runtime_source_sha256']))
        Path(path).write_bytes(b'changed runtime')
        with self.assertRaisesRegex(ValueError, '漂移'):
            hybrid.load_bound_capture(self.manifest_path)

    def test_actual_import_drift_is_rejected(self):
        path = next(iter(self.base['actual_import_sha256']))
        Path(path).write_bytes(b'changed import')
        with self.assertRaisesRegex(ValueError, '漂移'):
            hybrid.load_bound_capture(self.manifest_path)

    def test_partial_coverage_even_consistent_sha_is_rejected(self):
        manifest = json.loads(self.manifest_path.read_text())
        manifest['activations'].pop()
        self.refresh_hash_binding(manifest)
        with self.assertRaisesRegex(ValueError, '完整252'):
            hybrid.load_bound_capture(self.manifest_path)

    def test_duplicate_activation_file_cannot_stand_in_for_another_layer(self):
        manifest = json.loads(self.manifest_path.read_text())
        manifest['activations'][1]['file'] = manifest['activations'][0]['file']
        manifest['activations'][1]['sha256'] = manifest['activations'][0]['sha256']
        self.refresh_hash_binding(manifest)
        with self.assertRaisesRegex(ValueError, 'activation名字/文件'):
            hybrid.load_bound_capture(self.manifest_path)

    def test_finalization_source_failure_revokes_completed_status_and_receipt(self):
        self.receipt_path.unlink()
        Path(next(iter(self.base['runtime_source_sha256']))).write_bytes(b'changed before final bind')
        with self.assertRaisesRegex(ValueError, '漂移'):
            self.runner.write_final_reference_receipt(self.base, self.folder)
        terminal = json.loads(self.result_path.read_text())
        self.assertEqual(terminal['status'], 'failed')
        self.assertEqual(terminal['exit_code'], 1)
        with self.assertRaises(ValueError): hybrid.load_bound_capture(self.manifest_path)

    def test_capture_runtime_must_equal_completed_reference_runtime(self):
        manifest = json.loads(self.manifest_path.read_text())
        manifest['runtime_source_sha256'] = dict(self.base['code_sha256'])
        self.refresh_hash_binding(manifest)
        with self.assertRaisesRegex(ValueError, '来源不配对'):
            hybrid.load_bound_capture(self.manifest_path)

    def test_wrong_paired_reference_is_rejected(self):
        other = self.root / 'wrong_reference.json'
        other.write_text('{}')
        with self.assertRaisesRegex(ValueError, '错误paired'):
            hybrid.load_bound_capture(self.manifest_path, other)
        with self.assertRaisesRegex(ValueError, '错误paired'):
            self.runner.validate_probe_binding(self.probe_fixture(), self.base, self.base, other)

    def test_random_only_and_false_actual_coverage_cannot_admit_candidate(self):
        probe = self.probe_fixture()
        for changes in ({'status': 'passed_all252_random_only_cuda_diagnostic'},
                        {'actual_activation_coverage': '251/252'}):
            with self.assertRaisesRegex(RuntimeError, 'random-only'):
                self.runner.validate_probe_binding(dict(probe, **changes), self.base, self.base, self.result_path)

    def test_probe_runtime_or_reference_binding_drift_rejected(self):
        probe = self.probe_fixture()
        changed = copy.deepcopy(probe)
        changed['activation_provenance']['runtime_source_sha256'] = self.base['code_sha256']
        with self.assertRaisesRegex(RuntimeError, '漂移'):
            self.runner.validate_probe_binding(changed, self.base, self.base, self.result_path)
        changed = copy.deepcopy(probe)
        changed['activation_binding']['reference_result_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'paired reference'):
            self.runner.validate_probe_binding(changed, self.base, self.base, self.result_path)

    def test_complete_smoke_recomputes_pair_gate_and_rejects_saved_pass_for_one_point(self):
        probe = self.probe_fixture()
        probe_path = self.root / 'probe_result.json'
        probe_path.write_text(json.dumps(probe))
        gates = [self.runner.strict_pair_gate(row, row) for row in self.base['records']]
        smoke = dict(self.base, awq_forward='hybrid_m1', strict_pair_gates=gates,
            original_helper_gates=[dict(gate) for gate in gates],
            paired_reference={'result': str(self.result_path), 'sha256': hybrid.sha256_file(self.result_path)},
            matrix_probe_evidence={'result': str(probe_path), 'sha256': hybrid.sha256_file(probe_path)})
        self.runner.validate_smoke(smoke, self.base, ['a', 'b', 'c'])
        damaged = copy.deepcopy(smoke)
        damaged['records'][0]['pred_xyz'] = [[1., 2., 0.]]
        with self.assertRaisesRegex(ValueError, '64'):
            self.runner.validate_smoke(damaged, self.base, ['a', 'b', 'c'])


def dynamic_subprocess_receipt(directory):
    """CPU生成小源码后真正退出进程；没有RPC初始化、模型或GPU。"""
    program = '''
import json,sys
from pathlib import Path
from torch.distributed.nn.jit import instantiator
instantiator.instantiate_non_scriptable_remote_module_template()
from run_alp15_awq_hybrid_eval import imported_source_receipt
metadata={}
virtual={}
inventory=imported_source_receipt(sys.argv[1],metadata,virtual)
paths={row['snapshot_path'] for row in metadata.values()}
paths.update(entry['path'] for row in metadata.values() for entry in row['generators'].values())
paths.update(row['implementation_path'] for row in virtual.values())
print(json.dumps({'actual_import_sha256':{p:inventory[p] for p in paths},
    'source_snapshot_directory':str(Path(sys.argv[1]).resolve()),'dynamic_import_sources':metadata,'virtual_import_sources':virtual}))
'''
    result = subprocess.run([sys.executable, '-c', program, str(directory)], cwd=SCRIPTS,
                            capture_output=True, text=True)
    if result.returncode != 0: raise AssertionError(result.stderr)
    return json.loads(result.stdout)


class DynamicSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name).resolve()
        self.receipt = dynamic_subprocess_receipt(self.folder / 'snapshot_a')

    def tearDown(self): self.temp.cleanup()

    def test_two_processes_keep_same_stable_identity_without_same_temp_path(self):
        other = dynamic_subprocess_receipt(self.folder / 'snapshot_b')
        hybrid.assert_dynamic_sources_equal(self.receipt, other)
        a = self.receipt['dynamic_import_sources'][hybrid.TORCH_REMOTE_NAME]
        b = other['dynamic_import_sources'][hybrid.TORCH_REMOTE_NAME]
        self.assertNotEqual(a['original_path'], b['original_path'])
        self.assertNotEqual(a['snapshot_path'], b['snapshot_path'])

    def test_snapshot_tamper_and_missing_mapping_are_rejected(self):
        metadata = self.receipt['dynamic_import_sources'][hybrid.TORCH_REMOTE_NAME]
        broken = copy.deepcopy(self.receipt)
        broken['dynamic_import_sources'] = {}
        with self.assertRaisesRegex(ValueError, '完整映射'):
            hybrid.verify_import_provenance(broken)
        Path(metadata['snapshot_path']).write_bytes(b'altered snapshot')
        with self.assertRaisesRegex(ValueError, '漂移'):
            hybrid.verify_import_provenance(self.receipt)

    def test_generator_tamper_and_forged_module_identity_are_rejected(self):
        changed = copy.deepcopy(self.receipt)
        changed['dynamic_import_sources'][hybrid.TORCH_REMOTE_NAME]['module_name'] = 'forged_module'
        with self.assertRaisesRegex(ValueError, '身份'):
            hybrid.verify_import_provenance(changed)
        changed = copy.deepcopy(self.receipt)
        row = changed['dynamic_import_sources'][hybrid.TORCH_REMOTE_NAME]
        row['generators']['torch.distributed.nn.jit.instantiator']['sha256'] = '0' * 64
        row['stable_identity_sha256'] = hybrid._dynamic_identity(row)
        with self.assertRaisesRegex(ValueError, '生成器'):
            hybrid.verify_import_provenance(changed)

    def test_live_producer_requires_original_even_with_valid_snapshot(self):
        with self.assertRaises(FileNotFoundError):
            hybrid.verify_import_provenance(self.receipt, require_live=True)


if __name__ == '__main__':
    unittest.main()
