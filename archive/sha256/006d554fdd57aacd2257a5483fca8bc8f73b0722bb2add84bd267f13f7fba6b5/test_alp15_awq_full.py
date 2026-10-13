"""AWQ 公开边界：数据隔离与真实 packed 持久化/数值一致性。"""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from alp15_awq_full import validate_calibration_manifest

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, '需服务器推理环境的 torch，不能把本机跳过当通过')
class AWQPackedBoundaryTests(unittest.TestCase):
    def test_calibration_callback_runs_under_outer_memory_phase_without_reset(self):
        from alp15_awq_full import run_calibration_operation
        # 与服务器同一实际MemoryRecorder，不mock内部active/phase逻辑。
        directory = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(directory/'compression'))
        sys.path.insert(0, str(directory/'docs/baselines/evidence/quant_route_20261008/full20/code'))
        from runtime_memory_record import MemoryRecorder
        import json
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'memory.jsonl'
            recorder = MemoryRecorder(path, device=None)
            with recorder.phase('modelopt_awq'):
                value, timing = run_calibration_operation(recorder, 'calibration/pass1/train-a', lambda: 'sampled')
                self.assertEqual(value, 'sampled')
                self.assertGreaterEqual(timing['elapsed_s'], 0)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row['phase'] for row in rows], ['modelopt_awq'])
            self.assertEqual(rows[0]['status'], 'completed')

    def test_actual_saved_modelopt_weight_is_reconstructed_exactly(self):
        import torch
        from safetensors.torch import load_file
        from alp15_awq_full_packed import PackedAWQLinear
        directory = Path(os.environ.get('ALP15_AWQ_DIAGNOSTICS_DIR',
            str(Path(__file__).resolve().parents[1] / 'docs/baselines/evidence/quant_route_20261008/awq_api_smoke')))
        if not (directory / 'diagnostics.safetensors').exists():
            self.skipTest('此前真实ModelOpt诊断证据未上传；完整runner仍逐矩阵执行此门禁')
        evidence = load_file(str(directory / 'diagnostics.safetensors'))
        source = load_file(str(directory / 'source.safetensors'))
        module = PackedAWQLinear.from_quantized(evidence['quantized_weight'],
            evidence['weight_quantizer_amax'], evidence['pre_quant_scale'], source['bias'])
        self.assertTrue(torch.equal(module.dequantize(), evidence['quantized_weight']))
        self.assertEqual(module.storage_bytes(), 9088)

    def test_prescale_amax_and_zero_groups_survive_save_load(self):
        import torch
        from alp15_awq_full_packed import PackedAWQLinear
        # 手算样例：weight 第一行=1，第二行全零，输入全1，prescale全2。
        fake_weight = torch.cat((torch.ones(1, 128), torch.zeros(1, 128))).to(torch.bfloat16)
        amax = torch.tensor([[7.], [0.]], dtype=torch.bfloat16)
        prescale = torch.full((128,), 2., dtype=torch.bfloat16)
        bias = torch.tensor([3., 4.], dtype=torch.bfloat16)
        module = PackedAWQLinear.from_quantized(fake_weight, amax, prescale, bias)
        x = torch.ones(1, 128)
        torch.testing.assert_close(module(x), torch.tensor([[259., 4.]]), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.safetensors'
            module.save(path)
            loaded = PackedAWQLinear.load(path)
            torch.testing.assert_close(loaded(x), torch.tensor([[259., 4.]]), rtol=0, atol=0)
            self.assertTrue(torch.equal(loaded.amax, amax))
            self.assertTrue(torch.equal(loaded.pre_scale, prescale))
            self.assertEqual(loaded.storage_bytes(), 392)

    def test_packed_rejects_invalid_scale_and_weight_not_on_calibrated_grid(self):
        import torch
        from alp15_awq_full_packed import PackedAWQLinear
        weight = torch.ones(2, 128, dtype=torch.bfloat16)
        amax = torch.full((2, 1), 7., dtype=torch.bfloat16)
        prescale = torch.ones(128, dtype=torch.bfloat16)
        for kind in ('negative_amax', 'zero_prescale', 'off_grid', 'nonfinite', 'illegal_nibble'):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                if kind == 'negative_amax':
                    PackedAWQLinear.from_quantized(weight, -amax, prescale)
                elif kind == 'zero_prescale':
                    PackedAWQLinear.from_quantized(weight, amax, prescale * 0)
                elif kind == 'off_grid':
                    PackedAWQLinear.from_quantized(weight * .25, amax, prescale)
                elif kind == 'nonfinite':
                    PackedAWQLinear.from_quantized(weight * float('nan'), amax, prescale)
                else:
                    PackedAWQLinear(torch.full((2, 64), 8, dtype=torch.uint8), amax, prescale)


class CalibrationBoundaryTests(unittest.TestCase):
    def test_manifest_keeps_training_clips_out_of_development_evaluation(self):
        manifest = {'schema': 'alp15-awq-calibration-v1', 'role': 'training',
                    'examples': [{'clip_id': 'train-a', 'path': '/tmp/train-a.pt', 'sha256': 'a'*64}],
                    'evaluation_clips': ['eval-a']}
        self.assertEqual(validate_calibration_manifest(manifest)['calibration_clips'], ['train-a'])

    def test_manifest_rejects_overlap_duplicate_and_unverifiable_inputs(self):
        base = {'schema': 'alp15-awq-calibration-v1', 'role': 'training',
                'examples': [{'clip_id': 'train-a', 'path': '/tmp/train-a.pt', 'sha256': 'a'*64}],
                'evaluation_clips': ['eval-a']}
        import copy
        for kind in ('overlap', 'duplicate', 'no_hash', 'empty_eval', 'eval_duplicate', 'no_provenance'):
            manifest = copy.deepcopy(base)
            if kind == 'overlap':
                manifest['evaluation_clips'] = ['train-a']
            elif kind == 'duplicate':
                manifest['examples'] *= 2
            elif kind == 'no_hash':
                manifest['examples'][0]['sha256'] = 'not-a-hash'
            elif kind == 'empty_eval':
                manifest['evaluation_clips'] = []
            elif kind == 'eval_duplicate':
                manifest['evaluation_clips'] *= 2
            else:
                manifest['examples'][0]['clip_id'] = ''
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_calibration_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
