"""Validation-only MLP width search using the first experiment's frozen teacher cache.

This script never collects a new dense baseline and never evaluates holdout clips.
Run the selected overlay's holdout and fresh-process reload checks separately.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from mlp_pruning import prune_mlp, select_channels, train_down_projection
from layerwise_allocation import validate_layer_widths
from run_mlp_experiment import (
    COSMOS_REVISION, DATASET_REVISION, MODEL_VLM_NAME, QWEN_REVISION,
    _block_network, _enable_offline_mode, _load_local_model_and_processor,
    _load_offline_dataset_interface, _verify_config_dir, _verify_config_source,
    _verify_dataset, _verify_model, evaluate, file_digest, protected_digest,
    write_json,
)

SOURCE_COMMIT = '36aeb4c5938cbc2eb2aed33b22434773da4ab639'
CHECKPOINT_REVISION = '7aba8293c09993f2e125c6819df05d7fa3e873ea'
ORIGINAL_WIDTH = 12288
NUM_LAYERS = 36


def _widths(value):
    try:
        widths = [int(part) for part in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('widths must be comma-separated integers') from exc
    if not widths or len(set(widths)) != len(widths):
        raise argparse.ArgumentTypeError('widths must be nonempty and unique')
    if any(not 128 <= width < ORIGINAL_WIDTH or width % 128 for width in widths):
        raise argparse.ArgumentTypeError('each width must be a multiple of 128 in [128, 12160]')
    return sorted(widths, reverse=True)


def _read_teacher(experiment):
    split_path = experiment / 'split.json'
    teacher_path = experiment / 'teacher_calibration.json'
    provenance_path = experiment / 'provenance.json'
    split = json.loads(split_path.read_text())
    teacher = json.loads(teacher_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    first_stage_path = experiment / 'result.json'
    first_stage = json.loads(first_stage_path.read_text())
    if provenance.get('source_commit') != SOURCE_COMMIT:
        raise RuntimeError('First experiment source commit differs from the frozen source')
    if provenance.get('checkpoint_revision') != CHECKPOINT_REVISION:
        raise RuntimeError('First experiment checkpoint revision differs')
    if provenance.get('dataset_revision') != DATASET_REVISION:
        raise RuntimeError('First experiment dataset revision differs')
    if first_stage.get('status') != 'experiment_completed':
        raise RuntimeError('The established first compression stage must be complete')
    if first_stage.get('original_params') != provenance['original_params']:
        raise RuntimeError('First-stage original parameter counts are inconsistent')
    # Only stage parameter counts are carried forward; its holdout measurements
    # never participate in width selection.
    provenance['first_stage_params'] = first_stage['pruned_params']
    train_ids, val_ids, holdout_ids = (
        split['calibration'], split['validation'], split['holdout'])
    if len(train_ids) != 4 or len(val_ids) != 2 or len(holdout_ids) != 4:
        raise RuntimeError('Expected the established 4 calibration / 2 validation / 4 holdout split')
    all_ids = train_ids + val_ids + holdout_ids
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError('Teacher experiment splits overlap or contain duplicate clips')
    records = teacher['validation']
    if len(records) != 2 or {item['clip_id'] for item in records} != set(val_ids):
        raise RuntimeError('Dense validation records do not match the established validation split')
    dense_ade = [item['minADE_m'] for item in records]
    if any(not math.isfinite(value) or value < 0 for value in dense_ade):
        raise RuntimeError('Dense validation minADE is invalid')
    score_tensors, artifact_hashes = [], []
    for path in (split_path, teacher_path, provenance_path, first_stage_path):
        artifact_hashes.append({'file': str(path.relative_to(experiment)), 'sha256': file_digest(path)})
    for index in range(NUM_LAYERS):
        score_path = experiment / f'scores_{index:02d}.pt'
        score = torch.load(score_path, map_location='cpu', weights_only=True)
        if not isinstance(score, torch.Tensor) or tuple(score.shape) != (ORIGINAL_WIDTH,):
            raise RuntimeError(f'Wrong original score shape for layer {index}')
        if not torch.isfinite(score).all() or (score < 0).any():
            raise RuntimeError(f'Nonfinite or negative score for layer {index}')
        score_tensors.append(score)
        for path in (score_path,
                     experiment / 'teacher_cache' / f'train_{index:02d}.pt',
                     experiment / 'teacher_cache' / f'val_{index:02d}.pt'):
            if not path.is_file():
                raise RuntimeError(f'Missing established teacher artifact: {path}')
            artifact_hashes.append({'file': str(path.relative_to(experiment)),
                                    'sha256': file_digest(path)})
    return split, provenance, records, sum(dense_ade) / len(dense_ade), score_tensors, artifact_hashes


def _cache_inputs(experiment, index):
    train = torch.load(experiment / 'teacher_cache' / f'train_{index:02d}.pt',
                       map_location='cpu', weights_only=True)
    val = torch.load(experiment / 'teacher_cache' / f'val_{index:02d}.pt',
                     map_location='cpu', weights_only=True)
    for label, cache in (('train', train), ('val', val)):
        if set(cache) != {'x', 'y'}:
            raise RuntimeError(f'Invalid {label} teacher cache keys for layer {index}')
        x, y = cache['x'], cache['y']
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise RuntimeError(f'Invalid {label} teacher tensors for layer {index}')
        if x.ndim != 2 or x.shape != y.shape or x.shape[-1] != 4096 or x.shape[0] == 0:
            raise RuntimeError(f'Invalid {label} teacher cache dimensions for layer {index}')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise RuntimeError(f'Nonfinite {label} teacher cache for layer {index}')
    return train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--teacher-experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--widths', type=_widths, default=_widths('5376,1408'))
    parser.add_argument('--allocation-json', type=Path,
                        help='Calibration-only per-layer allocation; overrides uniform widths')
    parser.add_argument('--stop-after-first-accepted', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Stop at the first validation-accepted stage (default); '
                             'disable explicitly for a complete multi-width sweep')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--joint-mlp', action='store_true',
                        help='Distill all three MLP projections instead of down_proj only')
    args = parser.parse_args()
    trainer = train_down_projection
    if args.joint_mlp:
        from joint_mlp_distill import train_joint_mlp
        trainer = train_joint_mlp
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.eval_every < 1:
        parser.error('steps, batch-size, lr and eval-every must be positive')
    root = args.root.resolve()
    teacher_experiment = args.teacher_experiment.resolve()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError('The sweep output directory must be new; refusing to overwrite artifacts')
    _enable_offline_mode()
    torch.set_num_threads(8)
    split, teacher_provenance, dense_records, dense_mean, scores, teacher_hashes = _read_teacher(teacher_experiment)
    allocation = None
    if args.allocation_json:
        allocation = json.loads(args.allocation_json.read_text(encoding='utf-8'))
        validate_layer_widths(allocation.get('layer_widths'))
        if allocation.get('format_version') != 1 or allocation.get('ranking_uses_holdout') is not False or allocation.get('ranking_uses_validation') is not False:
            raise RuntimeError('Allocation must document calibration-only importance measurements')
        if Path(allocation['teacher_experiment']).resolve() != teacher_experiment:
            raise RuntimeError('Allocation was measured against a different teacher experiment')
        if sum(allocation['layer_widths']) != allocation['sum_widths'] or allocation['sum_widths'] > allocation['channel_budget']:
            raise RuntimeError('Allocation violates the recorded channel budget')
        profile_dir = args.allocation_json.resolve().parent
        if file_digest(profile_dir / 'provenance.json') != allocation['profile_provenance_sha256'] or file_digest(profile_dir / 'measurement_progress.json') != allocation['measurement_sha256']:
            raise RuntimeError('Importance measurement provenance SHA mismatch')
        args.widths = [allocation['name']]
    out.mkdir(parents=True)
    if allocation:
        write_json(out / 'allocation.json', allocation)
    code_dir = out / 'code'
    code_dir.mkdir()
    import shutil
    for name in ('run_width_sweep.py', 'joint_mlp_distill.py', 'mlp_pruning.py', 'layerwise_allocation.py', 'run_mlp_experiment.py'):
        shutil.copy2(Path(__file__).parent / name, code_dir / name)
    write_json(out / 'split.json', split)
    write_json(out / 'teacher_artifacts.json', {
        'experiment': str(teacher_experiment), 'artifacts': teacher_hashes,
        'scores_source': 'First experiment calibration clips only; no validation/holdout score collection'})
    validation_limit = dense_mean * 1.15
    configs = root / 'alp15-configs-delivery-20260919'
    cosmos = configs / 'dependencies/configs/Cosmos-Reason2-8B'
    qwen = configs / 'dependencies/configs/Qwen3-VL-2B-Instruct'
    dataset = root / 'alp15-test-clip-delivery-20260917'
    results = []
    with _block_network():
        config = _verify_model(root / 'models')
        _verify_dataset(dataset)
        for directory, label, repo, revision, manifest in (
            (cosmos, 'Cosmos', MODEL_VLM_NAME, COSMOS_REVISION, 'cosmos-reason2-source.json'),
            (qwen, 'Qwen', 'Qwen/Qwen3-VL-2B-Instruct', QWEN_REVISION, 'qwen3-vl-source.json')):
            _verify_config_dir(directory, label)
            _verify_config_source(directory, label, repo, revision, manifest, configs / 'manifests')
        interface = _load_offline_dataset_interface(dataset)
        if any(clip not in interface.clip_index.index for clip in split['validation']):
            raise RuntimeError('An established validation clip is missing from the frozen dataset')
        model, processor = _load_local_model_and_processor(root / 'models', cosmos, qwen, config)
        model.eval().requires_grad_(False)
        layers = model.vlm.model.language_model.layers
        if len(layers) != NUM_LAYERS or len(model.expert.layers) != NUM_LAYERS:
            raise RuntimeError('Unexpected layer count in frozen model')
        if any(layer.mlp.intermediate_size != ORIGINAL_WIDTH for layer in layers):
            raise RuntimeError('The loaded checkpoint must have dense original-width MLPs')
        if any(parameter.dtype != torch.bfloat16 for layer in layers for parameter in layer.mlp.parameters()):
            raise RuntimeError('Expected BF16 dense MLP templates')
        original_params = sum(parameter.numel() for parameter in model.parameters())
        original_vlm_params = sum(parameter.numel() for parameter in model.vlm.parameters())
        if original_params != teacher_provenance['original_params']:
            raise RuntimeError('Dense model parameter count differs from the established teacher')
        protected_before = protected_digest(model)
        if protected_before != teacher_provenance['protected_sha256_before']:
            raise RuntimeError('Dense model protected weights differ from the established teacher')
        provenance = {
            'source_commit': SOURCE_COMMIT, 'checkpoint_revision': CHECKPOINT_REVISION,
            'dataset_revision': DATASET_REVISION, 'protected_sha256_before': protected_before,
            'original_params': original_params, 'original_vlm_params': original_vlm_params,
            'first_stage_params': teacher_provenance['first_stage_params'],
            'initialization': 'Every candidate starts from the immutable original dense 11B teacher; '
                              'no chained pruning of the previously accepted 10B student',
            'teacher_experiment': str(teacher_experiment), 'widths': args.widths,
            'allocation': allocation,
            'allocation_json_sha256': file_digest(args.allocation_json) if allocation else None,
            'code_sha256': {p.name: file_digest(p) for p in code_dir.iterdir()},
            'stop_after_first_accepted': args.stop_after_first_accepted,
            'steps_per_layer': args.steps, 'batch_size': args.batch_size, 'lr': args.lr,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'selection': 'Smallest model with validation mean minADE <= 1.15 * cached dense mean',
            'validation_dense_mean_minADE_m': dense_mean,
            'validation_limit_minADE_m': validation_limit,
            'distillation': ('Dense cached teacher MLP outputs; all three MLP projections FP32 Adam; '
                             'actual BF16 validation best' if args.joint_mlp else
                             'Dense cached teacher MLP outputs; only down_proj FP32 Adam; actual BF16 validation best'),
            'holdout_policy': 'No holdout evaluations or holdout-based selection in this script'}
        write_json(out / 'provenance.json', provenance)
        # Keep one immutable original-width CPU template per layer. The rest of the
        # model stays on GPU; each candidate is rebuilt from these dense templates.
        templates = [layer.mlp.to('cpu') for layer in layers]
        torch.cuda.empty_cache()
        for width in args.widths:
            layer_widths = allocation['layer_widths'] if allocation else [width] * NUM_LAYERS
            candidate_dir = out / f'width_{width}'
            overlay_dir = candidate_dir / 'distilled'
            overlay_dir.mkdir(parents=True)
            candidate = {'keep_width': width, 'directory': str(candidate_dir),
                         'layer_widths': layer_widths,
                         'overlay_directory': str(overlay_dir), 'status': 'running'}
            start = time.perf_counter()
            training = []
            manifest = {
                'method': 'activation-down-column-L2 structured MLP channel pruning',
                'base_checkpoint_revision': CHECKPOINT_REVISION,
                'protected_sha256': protected_before, 'keep_width': width,
                'layer_widths': layer_widths,
                'distillation_steps_per_layer': args.steps, 'layers': []}
            try:
                for index, layer in enumerate(layers):
                    layer_width = layer_widths[index]
                    keep = select_channels(scores[index], layer_width)
                    restored = copy.deepcopy(templates[index])
                    if restored.intermediate_size != ORIGINAL_WIDTH:
                        raise RuntimeError('Dense CPU template was mutated')
                    prune_mlp(restored, keep)
                    layer.mlp = restored.to(device='cuda', dtype=torch.bfloat16)
                    layer.mlp.eval().requires_grad_(False)
                    train, val = _cache_inputs(teacher_experiment, index)
                    logging_args = {'metrics_path': candidate_dir / 'training_steps' / f'layer_{index:02d}.jsonl'} if args.joint_mlp else {}
                    report = trainer(
                        layer.mlp, train['x'], train['y'], val['x'], val['y'],
                        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                        seed=42 + index, eval_every=args.eval_every, **logging_args)
                    report['layer'] = index
                    if report['final_val_nmse'] > report['baseline_val_nmse']:
                        raise RuntimeError(f'Export-dtype local validation worsened for layer {index}')
                    if any(parameter.dtype != torch.bfloat16 or parameter.requires_grad
                           for parameter in layer.mlp.parameters()):
                        raise RuntimeError('Distilled MLP must be BF16 and frozen')
                    training.append(report)
                    tensor_path = overlay_dir / f'mlp_{index:02d}.safetensors'
                    save_file({key: value.detach().cpu().contiguous()
                               for key, value in layer.mlp.state_dict().items()}, str(tensor_path))
                    manifest['layers'].append({
                        'layer': index, 'original_width': ORIGINAL_WIDTH, 'keep_width': layer_width,
                        'keep_indices': keep.tolist(),
                        'calibration_token_count': train['x'].shape[0],
                        'sha256': file_digest(tensor_path)})
                    write_json(candidate_dir / 'distillation_progress.json', training)
                    print(f'SWEEP width={width} layer={index} '
                          f'local_val={report["baseline_val_nmse"]:.8g}->{report["final_val_nmse"]:.8g} '
                          f'best_step={report["best_step"]}', flush=True)
                    del train, val, restored
                    gc.collect()
                    torch.cuda.empty_cache()
                write_json(overlay_dir / 'manifest.json', manifest)
                model.eval().requires_grad_(False)
                protected_after = protected_digest(model)
                if protected_after != protected_before:
                    raise RuntimeError('Protected attention/vision/expert weights changed')
                validation = []
                for clip in split['validation']:
                    record = evaluate(model, processor, interface, clip)
                    validation.append(record)
                    write_json(candidate_dir / 'validation.json', validation)
                    print(f'SWEEP VALIDATION width={width} clip={clip} '
                          f'ADE={record["minADE_m"]:.6f}', flush=True)
                mean_ade = sum(record['minADE_m'] for record in validation) / len(validation)
                candidate_params = sum(parameter.numel() for parameter in model.parameters())
                candidate_vlm_params = sum(parameter.numel() for parameter in model.vlm.parameters())
                if allocation and candidate_params != allocation['actual_total_params']:
                    raise RuntimeError('Actual layerwise parameter count differs from the planned budget')
                candidate.update({
                    'status': 'completed', 'protected_sha256_after': protected_after,
                    'params': candidate_params, 'vlm_params': candidate_vlm_params,
                    'original_dense_params': original_params,
                    'previous_10B_stage_params': teacher_provenance['first_stage_params'],
                    'removed_fraction_vs_previous_10B_stage':
                        1 - candidate_params / teacher_provenance['first_stage_params'],
                    'removed_fraction': 1 - candidate_params / original_params,
                    'mlp_removed_fraction': 1 - sum(layer_widths) / (NUM_LAYERS * ORIGINAL_WIDTH),
                    'bf16_weight_gib': candidate_params * 2 / 2**30,
                    'validation': validation, 'validation_mean_minADE_m': mean_ade,
                    'validation_dense_mean_minADE_m': dense_mean,
                    'validation_limit_minADE_m': validation_limit,
                    'validation_pass': mean_ade <= validation_limit,
                    'local_distillation_layer_reports': training,
                    'elapsed_seconds': time.perf_counter() - start})
            except Exception as exc:
                candidate.update({'status': 'failed', 'error_type': type(exc).__name__,
                                  'error': str(exc), 'completed_layers': len(training),
                                  'validation_pass': False,
                                  'elapsed_seconds': time.perf_counter() - start})
                print(f'SWEEP FAILED width={width}: {type(exc).__name__}: {exc}', flush=True)
                # A protected-weight change invalidates the shared model for later
                # candidates; ordinary per-candidate failures can be rebuilt safely.
                if protected_digest(model) != protected_before:
                    write_json(candidate_dir / 'result.json', candidate)
                    raise RuntimeError('Protected weights changed; cannot continue the sweep') from exc
            results.append(candidate)
            write_json(candidate_dir / 'result.json', candidate)
            write_json(out / 'progress.json', {'candidates': results, 'selection_uses_holdout': False})
            gc.collect()
            torch.cuda.empty_cache()
            if args.stop_after_first_accepted and candidate.get('validation_pass'):
                print(f'SWEEP STAGE ACCEPTED width={width}; later widths require the next stage', flush=True)
                break
        passing = [candidate for candidate in results
                   if candidate['status'] == 'completed' and candidate['validation_pass']]
        selected = min(passing, key=lambda candidate: (candidate['params'], candidate['keep_width'])) if passing else None
        selection = {
            'status': 'selected_pending_holdout_and_reload' if selected else 'no_validation_passing_candidate',
            'criterion': 'Minimum parameters subject to validation mean minADE <= 1.15 * dense validation mean',
            'selection_uses_holdout': False, 'dense_validation': dense_records,
            'requested_widths': args.widths,
            'evaluated_widths': [candidate['keep_width'] for candidate in results],
            'deferred_widths': [width for width in args.widths
                                if width not in {candidate['keep_width'] for candidate in results}],
            'original_dense_params': original_params,
            'previous_10B_stage_params': teacher_provenance['first_stage_params'],
            'dense_validation_mean_minADE_m': dense_mean,
            'validation_limit_minADE_m': validation_limit,
            'selected': None if selected is None else {
                key: selected[key] for key in ('keep_width', 'params', 'vlm_params',
                    'layer_widths',
                    'overlay_directory', 'validation_mean_minADE_m', 'validation_pass')},
            'required_next_checks': ['Fresh-process selected overlay load',
                                     'Evaluate the four established untouched holdout clips',
                                     'Compare holdout predictions against stored dense holdout outputs'],
            'limitations': ['Only two internal validation clips from one official train chunk drive selection',
                            'Local teacher-state distillation does not eliminate student distribution shift',
                            'Validation selection is not independent holdout accuracy evidence',
                            'No Jetson deployment or deployment latency claim']}
        write_json(out / 'selection.json', selection)
        write_json(out / 'result.json', {**selection, 'candidates': results})
        print(json.dumps(selection, indent=2), flush=True)
    return 0 if selected else 2


if __name__ == '__main__':
    sys.exit(main())
