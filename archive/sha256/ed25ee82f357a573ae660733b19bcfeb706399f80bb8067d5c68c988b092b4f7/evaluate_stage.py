"""Evaluate an already-selected compression stage on untouched internal holdout clips.

Uses stored dense and prior-10B predictions; it never reruns those baselines or
changes selection/hyperparameters. --repeat-reload launches a second fresh process.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from layerwise_allocation import overlay_widths

from run_mlp_experiment import (
    COSMOS_REVISION, DATASET_REVISION, MODEL_VLM_NAME, QWEN_REVISION,
    _block_network, _enable_offline_mode, _load_local_model_and_processor,
    _load_offline_dataset_interface, _verify_config_dir, _verify_config_source,
    _verify_dataset, _verify_model, evaluate, file_digest, load_overlay,
    protected_digest, write_json,
)

SOURCE_COMMIT = '36aeb4c5938cbc2eb2aed33b22434773da4ab639'
CHECKPOINT_REVISION = '7aba8293c09993f2e125c6819df05d7fa3e873ea'


def _records(records, clips, label):
    if not isinstance(records, list) or len(records) != len(clips):
        raise RuntimeError(f'{label} must contain exactly the established four holdout clips')
    by_clip = {record['clip_id']: record for record in records}
    if len(by_clip) != len(records) or set(by_clip) != set(clips):
        raise RuntimeError(f'{label} holdout IDs are duplicated or differ from the frozen split')
    ordered = [by_clip[clip] for clip in clips]
    for record in ordered:
        ade = record['minADE_m']
        if not math.isfinite(ade) or ade < 0 or record.get('t0_us') != 5_100_000:
            raise RuntimeError(f'{label} holdout metric or timestamp is invalid')
        if not isinstance(record.get('pred_xyz_sha256'), str) or len(record['pred_xyz_sha256']) != 64:
            raise RuntimeError(f'{label} prediction SHA256 is missing or malformed')
        if 'cot' not in record:
            raise RuntimeError(f'{label} CoC record is missing')
    return ordered


def _mean(records):
    return sum(record['minADE_m'] for record in records) / len(records)


def _reload_match(reference, current, clips):
    reference_records = _records(reference['current_holdout'], clips, 'Reference reload')
    checks = []
    for old, new in zip(reference_records, current, strict=True):
        if old['clip_id'] != new['clip_id']:
            raise RuntimeError('Fresh-process reload holdout order differs')
        if old['pred_xyz_sha256'] != new['pred_xyz_sha256']:
            raise RuntimeError(f'Fresh-process XYZ SHA256 differs for {new["clip_id"]}')
        if old['cot'] != new['cot']:
            raise RuntimeError(f'Fresh-process CoC differs for {new["clip_id"]}')
        if old['minADE_m'] != new['minADE_m']:
            raise RuntimeError(f'Fresh-process minADE differs for {new["clip_id"]}')
        if old.get('pred_xyz_shape') != new.get('pred_xyz_shape'):
            raise RuntimeError(f'Fresh-process trajectory shape differs for {new["clip_id"]}')
        checks.append({'clip_id': new['clip_id'], 'xyz_sha256_identical': True,
                       'cot_identical': True, 'minADE_identical': True})
    return checks


def _report(payload, path, extra=''):
    mean = payload['mean_minADE_m']
    lines = [
        '# Independent stage holdout evaluation', '',
        f'Selected MLP width: {payload["keep_width"]}; total parameters: {payload["params"]:,}.', '',
        'Selection was fixed using the two validation clips before these four holdout evaluations.', '',
        '| Model | Total parameters | Holdout mean minADE (m) |',
        '| --- | ---: | ---: |',
        f'| Original dense teacher | {payload["original_dense_params"]:,} | {mean["dense"]:.6f} |',
        f'| Prior 10B student | {payload["previous_10B_stage_params"]:,} | {mean["prior_10B"]:.6f} |',
        f'| Selected stage | {payload["params"]:,} | {mean["current"]:.6f} |', '',
        '| Holdout clip | Dense minADE (m) | Prior 10B minADE (m) | Current minADE (m) |',
        '| --- | ---: | ---: | ---: |',
    ]
    for item in payload['per_clip_comparison']:
        lines.append(f'| {item["clip_id"]} | {item["dense_minADE_m"]:.6f} | '
                     f'{item["prior_10B_minADE_m"]:.6f} | {item["current_minADE_m"]:.6f} |')
    lines += ['', f'Current holdout mean is within 15% of dense: {payload["holdout_mean_within_15pct_dense"]}.',
              'This is a post-selection observation; it does not change the selected model.']
    if payload['reload_verification'] is not None:
        lines += ['', 'Fresh-process reload passed exact trajectory SHA256, CoC, and minADE checks on all four clips.']
    elif extra:
        lines += ['', extra]
    else:
        lines += ['', 'Fresh-process reload verification has not yet been run.']
    lines += ['', 'Limits: four internal holdout clips from one official training chunk; '
              'no official validation/generalization or hardware deployment claim.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--teacher-experiment', type=Path, required=True)
    parser.add_argument('--stage-experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat-reload', action='store_true')
    parser.add_argument('--reference-evaluation', type=Path)
    args = parser.parse_args()
    if args.repeat_reload and args.reference_evaluation:
        parser.error('--repeat-reload and --reference-evaluation cannot be combined')
    root, teacher_dir, stage_dir, out = (
        args.root.resolve(), args.teacher_experiment.resolve(),
        args.stage_experiment.resolve(), args.output.resolve())
    if out.exists():
        raise RuntimeError('Evaluation output directory must be new; refusing to overwrite artifacts')
    split_path = teacher_dir / 'split.json'
    teacher_provenance_path = teacher_dir / 'provenance.json'
    teacher_result_path = teacher_dir / 'result.json'
    selection_path = stage_dir / 'selection.json'
    split = json.loads(split_path.read_text())
    teacher_provenance = json.loads(teacher_provenance_path.read_text())
    teacher_result = json.loads(teacher_result_path.read_text())
    selection = json.loads(selection_path.read_text())
    if teacher_provenance.get('source_commit') != SOURCE_COMMIT:
        raise RuntimeError('Teacher source commit differs from frozen source')
    if teacher_provenance.get('checkpoint_revision') != CHECKPOINT_REVISION:
        raise RuntimeError('Teacher checkpoint revision differs')
    if teacher_provenance.get('dataset_revision') != DATASET_REVISION:
        raise RuntimeError('Teacher dataset revision differs')
    if teacher_result.get('status') != 'experiment_completed':
        raise RuntimeError('Prior compression experiment is incomplete')
    if teacher_result.get('original_params') != teacher_provenance['original_params']:
        raise RuntimeError('Teacher and prior-stage parameter counts are inconsistent')
    stage_split = json.loads((stage_dir / 'split.json').read_text())
    if stage_split != split:
        raise RuntimeError('Selected stage uses a different frozen split')
    all_clips = split['calibration'] + split['validation'] + split['holdout']
    if len(split['calibration']) != 4 or len(split['validation']) != 2 or len(split['holdout']) != 4:
        raise RuntimeError('Expected the established 4/2/4 clip split')
    if len(set(all_clips)) != len(all_clips):
        raise RuntimeError('Established train/validation/holdout splits overlap')
    if selection.get('selection_uses_holdout') is not False:
        raise RuntimeError('The selected stage must not use holdout for model selection')
    selected = selection.get('selected')
    if selection.get('status') != 'selected_pending_holdout_and_reload' or not selected:
        raise RuntimeError('No validation-accepted stage has been selected')
    if selected.get('validation_pass') is not True:
        raise RuntimeError('Selected stage did not pass its validation gate')
    cached_teacher = json.loads((teacher_dir / 'teacher_calibration.json').read_text())
    cached_validation = cached_teacher['validation']
    if len(cached_validation) != 2 or {r['clip_id'] for r in cached_validation} != set(split['validation']):
        raise RuntimeError('Stored dense validation clips differ from the frozen validation split')
    dense_val = _mean(cached_validation)
    selected_val = selected['validation_mean_minADE_m']
    if not math.isfinite(selected_val) or selected_val < 0 or not math.isfinite(dense_val):
        raise RuntimeError('Selected/dense validation metric is invalid')
    if selection['dense_validation_mean_minADE_m'] != dense_val:
        raise RuntimeError('Selection dense validation baseline differs from the stored teacher')
    if selection['validation_limit_minADE_m'] != dense_val * 1.15 or selected_val > dense_val * 1.15:
        raise RuntimeError('Selected stage fails the documented 15% validation criterion')
    overlay_dir = Path(selected['overlay_directory']).resolve()
    try:
        overlay_dir.relative_to(stage_dir)
    except ValueError as exc:
        raise RuntimeError('Selected overlay must belong to the selected stage experiment') from exc
    manifest_path = overlay_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['base_checkpoint_revision'] != CHECKPOINT_REVISION:
        raise RuntimeError('Selected overlay checkpoint revision differs')
    if manifest['protected_sha256'] != teacher_provenance['protected_sha256_before']:
        raise RuntimeError('Selected overlay protected weights differ from the established teacher')
    if manifest.get('keep_width') != selected['keep_width']:
        raise RuntimeError('Selected width differs from the overlay manifest')
    if [record['layer'] for record in manifest['layers']] != list(range(36)):
        raise RuntimeError('Selected overlay must contain all 36 layers exactly once')
    overlay_widths(manifest, selected)
    dense_path, prior_path = teacher_dir / 'dense_holdout.json', teacher_dir / 'distilled_holdout.json'
    dense = _records(json.loads(dense_path.read_text()), split['holdout'], 'Stored dense')
    prior = _records(json.loads(prior_path.read_text()), split['holdout'], 'Stored prior 10B')
    reference_path = args.reference_evaluation.resolve() if args.reference_evaluation else None
    reference = json.loads(reference_path.read_text()) if reference_path else None
    overlay_sha = file_digest(manifest_path)
    selection_sha = file_digest(selection_path)
    if reference is not None:
        if reference.get('status') not in ('holdout_evaluation_completed', 'fresh_process_reload_passed'):
            raise RuntimeError('Reference evaluation did not complete successfully')
        if reference.get('overlay_manifest_sha256') != overlay_sha or reference.get('selection_sha256') != selection_sha:
            raise RuntimeError('Reference evaluation uses a different overlay or selection')
        if reference.get('params') != selected['params'] or reference.get('keep_width') != selected['keep_width']:
            raise RuntimeError('Reference evaluation parameter count or width differs')
    out.mkdir(parents=True)
    _enable_offline_mode()
    torch.set_num_threads(8)
    configs = root / 'alp15-configs-delivery-20260919'
    cosmos = configs / 'dependencies/configs/Cosmos-Reason2-8B'
    qwen = configs / 'dependencies/configs/Qwen3-VL-2B-Instruct'
    dataset = root / 'alp15-test-clip-delivery-20260917'
    with _block_network():
        config = _verify_model(root / 'models')
        _verify_dataset(dataset)
        for directory, label, repo, revision, config_manifest in (
            (cosmos, 'Cosmos', MODEL_VLM_NAME, COSMOS_REVISION, 'cosmos-reason2-source.json'),
            (qwen, 'Qwen', 'Qwen/Qwen3-VL-2B-Instruct', QWEN_REVISION, 'qwen3-vl-source.json')):
            _verify_config_dir(directory, label)
            _verify_config_source(directory, label, repo, revision, config_manifest, configs / 'manifests')
        interface = _load_offline_dataset_interface(dataset)
        model, processor = _load_local_model_and_processor(root / 'models', cosmos, qwen, config)
        model.eval().requires_grad_(False)
        original_params = sum(parameter.numel() for parameter in model.parameters())
        if original_params != teacher_provenance['original_params']:
            raise RuntimeError('Fresh dense model parameter count differs from the established teacher')
        loaded_manifest = load_overlay(model, overlay_dir)
        if loaded_manifest != manifest:
            raise RuntimeError('Overlay manifest changed during loading')
        params = sum(parameter.numel() for parameter in model.parameters())
        vlm_params = sum(parameter.numel() for parameter in model.vlm.parameters())
        if params != selected['params'] or vlm_params != selected['vlm_params']:
            raise RuntimeError('Fresh-process model parameter counts differ from the selected stage')
        if protected_digest(model) != teacher_provenance['protected_sha256_before']:
            raise RuntimeError('Fresh-process protected attention/vision/expert weights differ')
        inputs = [split_path, teacher_provenance_path, teacher_result_path, selection_path,
                  manifest_path, dense_path, prior_path, teacher_dir / 'teacher_calibration.json']
        if reference_path:
            inputs.append(reference_path)
        provenance = {
            'source_commit': SOURCE_COMMIT, 'checkpoint_revision': CHECKPOINT_REVISION,
            'dataset_revision': DATASET_REVISION, 'teacher_experiment': str(teacher_dir),
            'stage_experiment': str(stage_dir), 'overlay_directory': str(overlay_dir),
            'protected_sha256': manifest['protected_sha256'], 'params': params, 'vlm_params': vlm_params,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'input_artifacts': [{'file': str(path), 'sha256': file_digest(path)} for path in inputs],
            'holdout_policy': 'Selection fixed before evaluation; no hyperparameter changes or dense baseline reruns'}
        write_json(out / 'provenance.json', provenance)
        current = []
        for clip in split['holdout']:
            record = evaluate(model, processor, interface, clip)
            current.append(record)
            write_json(out / 'current_holdout.json', current)
            print(f'STAGE HOLDOUT width={selected["keep_width"]} clip={clip} '
                  f'ADE={record["minADE_m"]:.6f}', flush=True)
        current = _records(current, split['holdout'], 'Current stage')
        protected_after = protected_digest(model)
        if protected_after != manifest['protected_sha256']:
            raise RuntimeError('Holdout evaluation changed protected weights')
        if file_digest(selection_path) != selection_sha or file_digest(manifest_path) != overlay_sha:
            raise RuntimeError('Stage selection or overlay manifest changed during evaluation')
        checks = _reload_match(reference, current, split['holdout']) if reference is not None else None
        comparison = []
        for dense_record, prior_record, record in zip(dense, prior, current, strict=True):
            comparison.append({
                'clip_id': record['clip_id'],
                'dense_minADE_m': dense_record['minADE_m'],
                'prior_10B_minADE_m': prior_record['minADE_m'], 'current_minADE_m': record['minADE_m'],
                'dense_pred_xyz_sha256': dense_record['pred_xyz_sha256'],
                'prior_10B_pred_xyz_sha256': prior_record['pred_xyz_sha256'],
                'current_pred_xyz_sha256': record['pred_xyz_sha256'],
                'dense_cot': dense_record['cot'], 'prior_10B_cot': prior_record['cot'], 'current_cot': record['cot']})
        payload = {
            'status': 'fresh_process_reload_passed' if checks is not None else 'holdout_evaluation_completed',
            'keep_width': selected['keep_width'], 'params': params, 'vlm_params': vlm_params,
            'original_dense_params': original_params,
            'previous_10B_stage_params': teacher_result['pruned_params'],
            'selection_sha256': selection_sha, 'overlay_manifest_sha256': overlay_sha,
            'protected_sha256_after': protected_after, 'selection_uses_holdout': False,
            'mean_minADE_m': {'dense': _mean(dense), 'prior_10B': _mean(prior), 'current': _mean(current)},
            'holdout_mean_within_15pct_dense': _mean(current) <= _mean(dense) * 1.15,
            'per_clip_comparison': comparison, 'dense_holdout': dense, 'prior_10B_holdout': prior,
            'current_holdout': current, 'reload_verification': checks,
            'limitations': ['Four internal holdout clips from a single official training chunk',
                            'No official validation/generalization or hardware deployment claim']}
        write_json(out / 'evaluation.json', payload)
        _report(payload, out / 'report.md')
        print(json.dumps({key: payload[key] for key in ('status', 'keep_width', 'params',
              'mean_minADE_m', 'holdout_mean_within_15pct_dense')}, indent=2), flush=True)
        # Release every model reference before spawning a separate GPU process.
        del model, processor, interface
        gc.collect()
        torch.cuda.empty_cache()
    if args.repeat_reload:
        reload_out = out / 'reload'
        command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                   '--teacher-experiment', str(teacher_dir), '--stage-experiment', str(stage_dir),
                   '--output', str(reload_out), '--reference-evaluation', str(out / 'evaluation.json')]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            write_json(out / 'reload_verification.json', {'status': 'fresh_process_reload_failed',
                       'returncode': completed.returncode, 'output': str(reload_out)})
            return completed.returncode
        reloaded = json.loads((reload_out / 'evaluation.json').read_text())
        if reloaded.get('status') != 'fresh_process_reload_passed':
            raise RuntimeError('Child process completed without passing reload verification')
        write_json(out / 'reload_verification.json', {'status': 'fresh_process_reload_passed',
                   'evaluation': str(reload_out / 'evaluation.json'),
                   'evaluation_sha256': file_digest(reload_out / 'evaluation.json'),
                   'checks': reloaded['reload_verification']})
        _report(payload, out / 'report.md',
                'Fresh-process reload passed exact trajectory SHA256, CoC, and minADE checks on all four clips.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
