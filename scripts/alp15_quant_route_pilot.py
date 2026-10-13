"""离线配对路线筛查；真实packed存储，参考反量化forward，不是部署速度测试。"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def summary(rows):
    by_clip = {}
    for row in rows:
        by_clip.setdefault(row['clip_id'], []).append(row)
    means = [statistics.mean(r['ADE_m'] for r in v) for v in by_clip.values()]
    return {'clips': len(means), 'trajectories': len(rows),
            'mean_ADE_m': statistics.mean(means),
            'mean_FDE_m': statistics.mean(statistics.mean(r['FDE_m'] for r in v) for v in by_clip.values()),
            'clip_mean_ADE_median_m': statistics.median(means),
            'clip_mean_ADE_p95_m': percentile(means, .95), 'clip_mean_ADE_max_m': max(means),
            'mean_paired_ADE_difference_m': statistics.mean(r['paired_ADE_difference_m'] for r in rows),
            'percentile_method': 'linear interpolation at (n-1)*q; clip means, not seeds as independent samples'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clips', type=int, choices=(2, 20), default=2)
    parser.add_argument('--variants', nargs='+', choices=('best8', 'teacher'), default=['best8', 'teacher'])
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / 'compression'))
    sys.path.insert(0, str(root / 'scripts'))
    import torch
    from alp15_packed_w4a16 import PackedW4A16Linear
    from controlled_8b_diagnostic import generate_prefix, sampled, digest, TrajectoryBoundaryError
    from run_mlp_experiment import (file_digest, load_overlay, _enable_offline_mode, _block_network,
        _verify_model, _verify_dataset, _verify_config_dir, _verify_config_source,
        _load_local_model_and_processor, _load_offline_dataset_interface,
        MODEL_VLM_NAME, COSMOS_REVISION, QWEN_REVISION, DATASET_REVISION)
    from runtime_memory_record import MemoryRecorder, storage_inventory

    _enable_offline_mode()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = root / 'experiments/controlled8_expanded20_20261002_125000'
    parent = root / 'experiments/mlp_newdata8_recovery_20261002_165238'
    dense_ref = json.loads((reference / 'result.json').read_text())
    best_ref = json.loads((parent / 'selected_task_validation.json').read_text())
    # 旧版step400结果没有status字段，完成态在controlled_recovery_result.json。
    recovery = json.loads((parent / 'controlled_recovery_result.json').read_text())
    if dense_ref['status'] != 'completed' or recovery['status'] != 'completed':
        raise RuntimeError('Frozen task reference is not complete')
    if best_ref['step'] != 400 or best_ref['missing_markers'] != 0 or best_ref['trajectory_seeds'] != [42, 43, 44]:
        raise RuntimeError('Selected best8 task schema/completion mismatch')
    identities = {'source_commit': '36aeb4c5938cbc2eb2aed33b22434773da4ab639',
                  'checkpoint_revision': '7aba8293c09993f2e125c6819df05d7fa3e873ea',
                  'dataset_revision': DATASET_REVISION}
    for key, expected in identities.items():
        if dense_ref[key] != expected:
            raise RuntimeError('Frozen identity mismatch: ' + key)
    refs = {'teacher': {(r['clip_id'], r['seed']): r for r in dense_ref['records']
                        if r['model'] == 'dense' and r['prefix'] == 'teacher'},
            'best8': {(r['clip_id'], r['seed']): r for r in best_ref['records']}}
    if set(refs['teacher']) != set(refs['best8']) or len(refs['teacher']) != 60:
        raise RuntimeError('Expected exactly paired 20 clips times 3 seeds')
    for key in refs['teacher']:
        if refs['teacher'][key]['initial_noise_sha256'] != refs['best8'][key]['initial_noise_sha256']:
            raise RuntimeError('Frozen references are not noise paired')
    clips = sorted({k[0] for k in refs['teacher']})[:args.clips]
    examples = {}
    cache_hashes = {}
    for path in sorted((reference / 'teacher_prefix_cache').glob('*.pt')):
        item = torch.load(path, weights_only=True, map_location='cpu')
        if item['clip_id'] in clips:
            examples[item['clip_id']] = item
            cache_hashes[str(path)] = file_digest(path)
    if set(examples) != set(clips):
        raise RuntimeError('Missing offline prompt example')
    if recovery['selected_step'] != 400 or recovery['params'] != 8020878578:
        raise RuntimeError('Incorrect best8 parent')
    overlay = Path(recovery['overlay_directory'])
    result = {'status': 'running', 'argv': sys.argv, **identities,
              'clips': clips, 'trajectory_seeds': [42, 43, 44], 'text_seed': 42,
              'scope': 'Repeated-use development set, not independent test; no new official baseline',
              'method': 'RTN symmetric group128 packed INT4 / FP16 scales / BF16 activations; not AWQ or ModelOpt',
              'runtime': 'Reference dequantization+F.linear; not optimized W4A16 kernel',
              'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name(),
                              'source_HEAD': subprocess.check_output(['git', '-C', str(root / 'alpamayo1.5'), 'rev-parse', 'HEAD'], text=True).strip()},
              'inputs_sha256': {str(reference/'result.json'): file_digest(reference/'result.json'),
                               str(parent/'selected_task_validation.json'): file_digest(parent/'selected_task_validation.json'),
                               str(parent/'controlled_recovery_result.json'): file_digest(parent/'controlled_recovery_result.json'),
                               str(overlay/'manifest.json'): file_digest(overlay/'manifest.json'), **cache_hashes},
              'code_sha256': {}, 'records': [], 'failures': [], 'variants': {},
              'missing_metrics': ['Orin execution, unified whole-system peak, energy, collisions, optimized latency',
                                  'Original packed-only low-peak loader; this pilot loads original BF16 first',
                                  'Independent held-out generalization', 'Training metrics: no optimizer'],
              'timing_limitations': 'Concurrent unrelated GPU training; diagnostic timestamps only, not candidate speed ranking'}
    (out / 'code').mkdir()
    import shutil
    for path in [Path(__file__), Path(__file__).with_name('alp15_packed_w4a16.py'),
                 root/'compression/controlled_8b_diagnostic.py', root/'compression/expert_interface_distill.py',
                 root/'compression/run_mlp_experiment.py', root/'compression/runtime_memory_record.py',
                 root/'scripts/test_alpamayo15_offline_compat.py', root/'compression/mlp_pruning.py']:
        shutil.copy2(path, out/'code'/path.name)
        result['code_sha256'][str(path)] = file_digest(path)
    if result['environment']['source_HEAD'] != identities['source_commit']:
        raise RuntimeError('Official source HEAD changed')
    recorder = MemoryRecorder(out / 'memory_phases.jsonl', device=torch.device('cuda'))
    started = time.perf_counter()

    def save():
        result['elapsed_seconds'] = time.perf_counter() - started
        write_json(out / 'result.json', result)

    def guarded_digest(model):
        h = hashlib.sha256()
        quantized_prefixes = tuple(f'vlm.model.language_model.layers.{i}.{section}.{projection}.'
            for i in range(36) for section, projections in (
                ('mlp', ('gate_proj', 'up_proj', 'down_proj')),
                ('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')))
            for projection in projections)
        for name, value in model.state_dict().items():
            if name.startswith(quantized_prefixes):
                continue
            h.update(name.encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    @torch.no_grad()
    def evaluate(model, interface, variant, label, selected_clips):
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        for clip in selected_clips:
            times = {}
            def stage(name, fn):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                value = fn()
                torch.cuda.synchronize()
                times[name] = (time.perf_counter() - begin) * 1000
                return value
            with recorder.phase(f'{label}/{clip}', model):
                data = stage('data_decode_ms', lambda: load_physical_aiavdataset(clip, t0_us=5_100_000, avdi=interface, maybe_stream=False))
                try:
                    generated, actual = stage('generation_ms', lambda: generate_prefix(model, examples[clip]))
                except TrajectoryBoundaryError as exc:
                    result['failures'].append({'variant': label, **exc.details})
                    save()
                    raise
                del actual
                inputs = {k: v.cuda() for k, v in generated['inputs'].items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = stage('full_prefill_ms', lambda: model.vlm.model(**inputs, use_cache=True, return_dict=True))
                base_ms = times['generation_ms'] + times['full_prefill_ms']
                for seed in (42, 43, 44):
                    measured = stage('trajectory_sample_ms', lambda: sampled(model, output, inputs, data, seed))
                    frozen = refs[variant][clip, seed]
                    if measured['initial_noise_sha256'] != frozen['initial_noise_sha256']:
                        raise RuntimeError('Trajectory RNG no longer paired')
                    pred = torch.tensor(measured['pred_xyz'])[:, :2]
                    gt = data['ego_future_xyz'][0, 0, :, :2].float().cpu()
                    distances = torch.linalg.vector_norm(pred - gt, dim=-1)
                    fde = distances[-1].item()
                    if not torch.isfinite(distances).all() or not all(math.isfinite(v) for v in (fde, measured['ADE_m'], frozen['ADE_m'])):
                        raise RuntimeError('Nonfinite trajectory/reference metric')
                    row = {'variant': label, 'clip_id': clip, 'seed': seed, **measured,
                           'FDE_m': fde, 'max_displacement_m': distances.max().item(),
                           'reference_ADE_m': frozen['ADE_m'],
                           'paired_ADE_difference_m': measured['ADE_m'] - frozen['ADE_m'],
                           'generated_tokens': generated['generated_tokens'], 'prefix_length': inputs['input_ids'].shape[1],
                           'prefix_sha256': digest(inputs['input_ids']),
                           'diagnostic_stage_ms': dict(times),
                           'diagnostic_compute_ms': base_ms + times['trajectory_sample_ms']}
                    if label == 'best8_bf16_reload' and abs(row['paired_ADE_difference_m']) > 1e-4:
                        raise RuntimeError('Best8 independent reload differs from frozen task reference')
                    result['records'].append(row)
                    save()
                    print('QUANT_ROUTE', label, clip, seed, row['ADE_m'], row['FDE_m'], flush=True)
                del output, inputs, data
                gc.collect()
                torch.cuda.empty_cache()

    save()
    try:
        with _block_network(), torch.no_grad():
            with recorder.phase('input_verification'):
                config = _verify_model(root/'models')
                _verify_dataset(root/'alp15-test-clip-delivery-20260917')
                configs = root/'alp15-configs-delivery-20260919'
                cosmos = configs/'dependencies/configs/Cosmos-Reason2-8B'
                qwen = configs/'dependencies/configs/Qwen3-VL-2B-Instruct'
                for directory, label, repo, revision, source_name in [
                    (cosmos, 'Cosmos', MODEL_VLM_NAME, COSMOS_REVISION, 'cosmos-reason2-source.json'),
                    (qwen, 'Qwen', 'Qwen/Qwen3-VL-2B-Instruct', QWEN_REVISION, 'qwen3-vl-source.json')]:
                    _verify_config_dir(directory, label)
                    _verify_config_source(directory, label, repo, revision, source_name, configs/'manifests')
                interface = _load_offline_dataset_interface(root/'alp15-test-clip-delivery-20260917')
            for variant in args.variants:
                gc.collect()
                torch.cuda.empty_cache()
                with recorder.phase(variant+'/load_bf16'):
                    model, processor = _load_local_model_and_processor(root/'models', cosmos, qwen, config)
                    model.eval().requires_grad_(False)
                    if variant == 'best8':
                        load_overlay(model, overlay)
                if variant == 'best8':
                    evaluate(model, interface, variant, 'best8_bf16_reload', clips[:2])
                original_storage = storage_inventory(model)
                guard = guarded_digest(model)
                manifest = []
                packed_dir = out/'packed'/variant
                packed_dir.mkdir(parents=True)
                with recorder.phase(variant+'/pack_and_reload', model):
                    for index, layer in enumerate(model.vlm.model.language_model.layers):
                        targets = [('mlp', n) for n in ('gate_proj', 'up_proj', 'down_proj')]
                        targets += [('self_attn', n) for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
                        for section, name in targets:
                            container = getattr(layer, section)
                            linear = getattr(container, name)
                            if not isinstance(linear, torch.nn.Linear) or linear.in_features % 128:
                                raise RuntimeError(f'Unsupported target {index}/{section}/{name}')
                            packed = PackedW4A16Linear.from_linear(linear, storage_dtype=torch.float16)
                            path = packed_dir/f'layer{index:02d}_{section}_{name}.safetensors'
                            packed.save(path)
                            reloaded = PackedW4A16Linear.load(path, device=linear.weight.device)
                            for key, value in packed.state_dict().items():
                                if not torch.equal(value, reloaded.state_dict()[key]):
                                    raise RuntimeError('Packed reload is not bitwise identical')
                            manifest.append({'layer': index, 'section': section, 'projection': name,
                                             'shape': list(linear.weight.shape), 'original_bytes': linear.weight.numel()*linear.weight.element_size(),
                                             'packed_buffer_bytes': reloaded.storage_bytes(), 'file': str(path),
                                             'file_bytes': path.stat().st_size, 'sha256': file_digest(path)})
                            setattr(container, name, reloaded)
                            del linear, packed, reloaded
                        torch.cuda.empty_cache()
                if len(manifest) != 36 * 7:
                    raise RuntimeError('Unexpected quantization target count')
                if guarded_digest(model) != guard:
                    raise RuntimeError('Protected model modules changed')
                write_json(packed_dir/'manifest.json', {'method': result['method'], 'matrices': manifest,
                           'guarded_sha256': guard, 'base_checkpoint_revision': identities['checkpoint_revision']})
                result['variants'][variant] = {'original_storage': original_storage,
                    'packed_storage': storage_inventory(model), 'matrix_count': len(manifest),
                    'packed_manifest_sha256': file_digest(packed_dir/'manifest.json'), 'guarded_sha256': guard}
                save()
                evaluate(model, interface, variant, variant+'_w4a16_rtn', clips)
                if guarded_digest(model) != guard:
                    raise RuntimeError('Protected weights changed during inference')
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
            labels = sorted({r['variant'] for r in result['records']})
            result['summaries'] = {label: summary([r for r in result['records'] if r['variant']==label]) for label in labels}
            result['status'] = 'completed'
            save()
    except BaseException as exc:
        result['status'] = 'failed'
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        save()
        raise
    finally:
        try:
            rows = result['records']
            if rows:
                with (out/'metrics.csv').open('x', newline='') as stream:
                    columns = ['variant', 'clip_id', 'seed', 'ADE_m', 'FDE_m', 'paired_ADE_difference_m', 'generated_tokens', 'diagnostic_compute_ms']
                    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(rows)
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                (out/'curves').mkdir(exist_ok=False)
                for label in sorted({r['variant'] for r in rows}):
                    values = [r for r in rows if r['variant']==label]
                    plt.plot(range(len(values)), [r['ADE_m'] for r in values], marker='.', label=label)
                plt.xlabel('Recorded paired clip/seed index (not independent scenes)')
                plt.ylabel('Trajectory ADE (m)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(out/'curves/trajectory_ADE.png')
                plt.savefig(out/'curves/trajectory_ADE.svg')
                plt.close()
                shutil.copy2(out/'metrics.csv', out/'curves/metrics.csv')
            result['pilot_curve_export_status'] = 'completed' if rows else 'no_recorded_metrics'
        except Exception as exc:
            result['pilot_curve_export_status'] = 'failed'
            result['pilot_curve_export_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        # 既有项目记录导出器也执行；不能伪造缺失训练曲线。
        try:
            with (out/'curve_export.log').open('w') as log:
                exported = subprocess.run([sys.executable, str(root/'compression/export_curves.py'),
                    '--experiment', str(out), '--output', str(out/'recorded_export_curves')],
                    stdout=log, stderr=subprocess.STDOUT)
            result['existing_curve_export_exit_code'] = exported.returncode
        except Exception as exc:
            result['existing_curve_export_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        save()
    print('QUANT_ROUTE_COMPLETED', json.dumps(result['summaries']), flush=True)


if __name__ == '__main__':
    requested_output = Path(sys.argv[sys.argv.index('--output') + 1]) if '--output' in sys.argv else None
    output_existed = requested_output.exists() if requested_output is not None else True
    try:
        main()
    except BaseException as exc:
        # Preflight也必须留痕；不得覆盖已有目录中的历史结果。
        if requested_output is not None and not output_existed:
            output = requested_output
            if output.is_dir() and not (output/'result.json').exists():
                write_json(output/'result.json', {'status': 'failed_preflight', 'argv': sys.argv,
                    'error': {'type': type(exc).__name__, 'message': str(exc)}})
        raise
