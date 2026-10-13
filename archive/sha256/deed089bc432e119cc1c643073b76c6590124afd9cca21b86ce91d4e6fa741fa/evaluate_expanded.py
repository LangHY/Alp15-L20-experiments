"""Paired dense/student evaluation on all remaining locally delivered valid clips.

The 20 new validation clips gate the next compression stage. The remaining new
holdout clips are reporting-only. All clips are from one official training chunk;
this is neither official validation nor evidence of broad generalization.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import math
import traceback
import zipfile
from pathlib import Path

import torch
from layerwise_allocation import overlay_widths

from run_mlp_experiment import (
    CLIP_ID, COSMOS_REVISION, DATASET_REVISION, FEATURES, MODEL_VLM_NAME, QWEN_REVISION,
    _block_network, _enable_offline_mode, _load_local_model_and_processor,
    _load_offline_dataset_interface, _verify_config_dir, _verify_config_source,
    _verify_dataset, _verify_model, evaluate, file_digest, load_overlay, protected_digest,
)

SOURCE_COMMIT = '36aeb4c5938cbc2eb2aed33b22434773da4ab639'
CHECKPOINT_REVISION = '7aba8293c09993f2e125c6819df05d7fa3e873ea'
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_RESAMPLES = 5000


def _write(path, value):
    """Atomic progress saves allow resuming after an interrupted write."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _artifact(path):
    return {'file': str(path.resolve()), 'sha256': file_digest(path)}


def _eligible_clips(interface, dataset, excluded):
    frame = interface.clip_index
    candidates = set(frame[(frame['chunk'] == 3119) & frame['clip_is_valid']].index.tolist())
    snapshot = dataset / 'datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots' / DATASET_REVISION
    for relative, _, _ in FEATURES.values():
        with zipfile.ZipFile(snapshot / relative) as packed:
            present = {Path(name).name[:36] for name in packed.namelist()}
        candidates.intersection_update(present)
    ordered = sorted(candidates - set(excluded),
                     key=lambda clip: hashlib.sha256(('alp15-expanded-v1:' + clip).encode()).hexdigest())
    if len(ordered) < 20:
        raise RuntimeError(f'Only {len(ordered)} new complete valid clips are available; need 20 validation clips')
    return {
        'expanded_validation': ordered[:20], 'expanded_holdout': ordered[20:],
        'excluded_clips': sorted(set(excluded)), 'eligible_clip_count_before_exclusion': len(candidates),
        'expanded_clip_count': len(ordered), 'chunk': 3119,
        'ordering': "sha256('alp15-expanded-v1:' + clip_id), ascending",
        'note': 'Clip-disjoint internal expansion within official train chunk3119; not official validation.',
    }


def _validate_progress(progress, clips, identity):
    if progress.get('identity') != identity:
        raise RuntimeError('Resume progress uses different input artifacts or model identity')
    for phase in ('dense', 'student'):
        records = progress[phase]
        if not set(records).issubset(set(clips)):
            raise RuntimeError(f'{phase} resume progress contains clips outside the saved expanded split')
        for clip, record in records.items():
            if record.get('status') not in ('completed', 'failed'):
                raise RuntimeError(f'Invalid {phase} resume record status for {clip}')
            if record['status'] == 'completed':
                result = record['result']
                if result['clip_id'] != clip or result.get('t0_us') != 5_100_000:
                    raise RuntimeError(f'{phase} cached record clip/timestamp mismatch')
                if not math.isfinite(result['minADE_m']) or result['minADE_m'] < 0:
                    raise RuntimeError(f'{phase} cached ADE is invalid')
                if len(result.get('pred_xyz_sha256', '')) != 64 or 'cot' not in result:
                    raise RuntimeError(f'{phase} cached prediction hash or CoC is missing')
                if type(result.get('trajectory_start_missing')) is not bool:
                    raise RuntimeError(f'{phase} cached generation-marker status is unknown')


def _distribution(values):
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    return {'mean': tensor.mean().item(), 'median': torch.quantile(tensor, 0.5).item(),
            'p90': torch.quantile(tensor, 0.9).item()}


def _paired_summary(clips, progress):
    paired = []
    for clip in clips:
        dense, student = progress['dense'].get(clip), progress['student'].get(clip)
        if not dense or not student or dense['status'] != 'completed' or student['status'] != 'completed':
            continue
        d, s = dense['result'], student['result']
        paired.append({'clip_id': clip, 'dense_minADE_m': d['minADE_m'],
                       'student_minADE_m': s['minADE_m'],
                       'delta_minADE_m': s['minADE_m'] - d['minADE_m'],
                       'dense_pred_xyz_sha256': d['pred_xyz_sha256'],
                       'student_pred_xyz_sha256': s['pred_xyz_sha256'],
                       'dense_cot': d['cot'], 'student_cot': s['cot']})
    deltas = [record['delta_minADE_m'] for record in paired]
    interval = None
    if deltas:
        tensor = torch.tensor(deltas, dtype=torch.float64)
        generator = torch.Generator(device='cpu').manual_seed(BOOTSTRAP_SEED)
        means = []
        for first in range(0, BOOTSTRAP_RESAMPLES, 100):
            count = min(100, BOOTSTRAP_RESAMPLES - first)
            indices = torch.randint(len(deltas), (count, len(deltas)), generator=generator)
            means.append(tensor[indices].mean(dim=1))
        quantiles = torch.quantile(torch.cat(means), torch.tensor([0.025, 0.975], dtype=torch.float64))
        interval = {'lower_m': quantiles[0].item(), 'upper_m': quantiles[1].item(),
                    'resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED,
                    'method': 'Paired clip bootstrap percentile CI for mean(student ADE - dense ADE)'}
    return {
        'requested_clips': len(clips), 'completed_pairs': len(paired),
        'dense_minADE_m': _distribution([record['dense_minADE_m'] for record in paired]),
        'student_minADE_m': _distribution([record['student_minADE_m'] for record in paired]),
        'delta_minADE_m': _distribution(deltas), 'bootstrap_mean_delta_ci95': interval,
        'student_wins': sum(value < -1e-12 for value in deltas),
        'student_losses': sum(value > 1e-12 for value in deltas),
        'ties': sum(abs(value) <= 1e-12 for value in deltas), 'paired_records': paired,
    }


def _run_phase(model, processor, interface, clips, phase, progress, output):
    role_by_clip = {clip: role for role in ('expanded_validation', 'expanded_holdout')
                    for clip in progress['split'][role]}
    for clip in clips:
        previous = progress[phase].get(clip)
        if previous and previous['status'] == 'completed':
            print(f'EXPANDED SKIP phase={phase} clip={clip}', flush=True)
            continue
        attempt = (previous or {}).get('attempts', 0) + 1
        warnings = []
        class MarkerWarnings(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                if 'No <traj_future_start> token found' in message:
                    warnings.append(message)
        logger = logging.getLogger('alpamayo1_5.models.alpamayo1_5')
        handler = MarkerWarnings()
        logger.addHandler(handler)
        try:
            result = evaluate(model, processor, interface, clip)
            result['trajectory_start_missing'] = bool(warnings)
            result['trajectory_generation_warnings'] = warnings
            progress[phase][clip] = {'status': 'completed', 'attempts': attempt,
                                     'role': role_by_clip[clip], 'result': result}
            print(f'EXPANDED phase={phase} role={role_by_clip[clip]} clip={clip} '
                  f'ADE={result["minADE_m"]:.6f}', flush=True)
        except Exception as exc:
            progress[phase][clip] = {'status': 'failed', 'attempts': attempt,
                                     'role': role_by_clip[clip], 'error_type': type(exc).__name__,
                                     'error': str(exc), 'traceback': traceback.format_exc()}
            print(f'EXPANDED FAILED phase={phase} clip={clip}: {type(exc).__name__}: {exc}', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            logger.removeHandler(handler)
        _write(output / 'progress.json', progress)
        _write(output / f'{phase}_progress.json', progress[phase])


def _report(result, path):
    lines = ['# Expanded paired evaluation', '',
             f'Status: {result["status"]}. Selected MLP width: {result["keep_width"]}.', '',
             'All previous baseline, calibration, validation, and holdout clips were excluded.', '',
             '| Split | Completed pairs / requested | Dense mean ADE (m) | Student mean ADE (m) | '
             'Mean paired difference (m) | Student wins / losses / ties |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    interval_lines = []
    for role in ('expanded_validation', 'expanded_holdout'):
        item = result[role]
        dense = item['dense_minADE_m']
        student = item['student_minADE_m']
        delta = item['delta_minADE_m']
        means = ('—', '—', '—') if dense is None else (
            f'{dense["mean"]:.6f}', f'{student["mean"]:.6f}', f'{delta["mean"]:.6f}')
        lines.append(f'| {role} | {item["completed_pairs"]} / {item["requested_clips"]} | '
                     f'{means[0]} | {means[1]} | {means[2]} | '
                     f'{item["student_wins"]} / {item["student_losses"]} / {item["ties"]} |')
        interval = item['bootstrap_mean_delta_ci95']
        if interval:
            interval_lines += ['', f'{role} mean paired difference 95% bootstrap interval: '
                               f'[{interval["lower_m"]:.6f}, {interval["upper_m"]:.6f}] m.']
    lines += interval_lines
    gate = result['next_6B_stage_gate']
    lines += ['', f'Expanded validation gate for proceeding to 6B: {gate["pass"]}.',
              'The gate requires a complete run and validation mean student ADE ≤ 1.15 × dense ADE. '
              'Expanded holdout measurements do not change this gate or any hyperparameters.', '',
              'The 15% tolerance is a preset threshold for this experiment, not a deployment safety standard.', '',
              'Mean, median, p90, individual predictions/CoC/hashes, paired deltas, and the fixed-seed '
              '5000-resample bootstrap results are saved in JSON.', '',
              'These are internal clip-disjoint samples from one official training chunk, not official '
              'validation. Bootstrap intervals describe this clip set and do not guarantee generalization.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--teacher-experiment', type=Path, required=True)
    parser.add_argument('--stage-experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true',
                        help='Reuse successful records and retry failed/missing clips in an existing output')
    parser.add_argument('--dense-reference', type=Path,
                        help='Reuse a completed expansion teacher phase on the identical frozen split')
    args = parser.parse_args()
    root, teacher_dir, stage_dir, out = (args.root.resolve(), args.teacher_experiment.resolve(),
                                        args.stage_experiment.resolve(), args.output.resolve())
    if args.resume and not out.is_dir():
        raise RuntimeError('--resume requires the existing expanded-evaluation output directory')
    if not args.resume and out.exists():
        raise RuntimeError('Output directory must be new unless --resume is supplied')
    old_split_path = teacher_dir / 'split.json'
    teacher_provenance_path = teacher_dir / 'provenance.json'
    selection_path = stage_dir / 'selection.json'
    stage_split_path = stage_dir / 'split.json'
    old_split = json.loads(old_split_path.read_text())
    teacher_provenance = json.loads(teacher_provenance_path.read_text())
    selection = json.loads(selection_path.read_text())
    if teacher_provenance.get('source_commit') != SOURCE_COMMIT:
        raise RuntimeError('Teacher source commit differs from the frozen source')
    if teacher_provenance.get('checkpoint_revision') != CHECKPOINT_REVISION:
        raise RuntimeError('Teacher checkpoint revision differs')
    if teacher_provenance.get('dataset_revision') != DATASET_REVISION:
        raise RuntimeError('Teacher dataset revision differs')
    if json.loads(stage_split_path.read_text()) != old_split:
        raise RuntimeError('Selected stage and teacher use different established splits')
    if selection.get('status') != 'selected_pending_holdout_and_reload':
        raise RuntimeError('No selected validation-accepted student is available')
    selected = selection.get('selected')
    if not selected or selected.get('validation_pass') is not True:
        raise RuntimeError('Expanded evaluation requires a selected validation-accepted student')
    if selection.get('selection_uses_holdout') is not False:
        raise RuntimeError('The prior selection must not use holdout clips')
    saved_val = selected['validation_mean_minADE_m']
    saved_dense_val = selection['dense_validation_mean_minADE_m']
    saved_limit = selection['validation_limit_minADE_m']
    if any(not math.isfinite(value) or value < 0 for value in (saved_val, saved_dense_val, saved_limit)):
        raise RuntimeError('Saved selected validation metrics are invalid')
    if saved_limit != 1.15 * saved_dense_val or saved_val > saved_limit:
        raise RuntimeError('Selected student fails its saved validation criterion')
    overlay_dir = Path(selected['overlay_directory']).resolve()
    try:
        overlay_dir.relative_to(stage_dir)
    except ValueError as exc:
        raise RuntimeError('Overlay must belong to the selected stage experiment') from exc
    overlay_manifest_path = overlay_dir / 'manifest.json'
    manifest = json.loads(overlay_manifest_path.read_text())
    if manifest['base_checkpoint_revision'] != CHECKPOINT_REVISION:
        raise RuntimeError('Overlay checkpoint revision differs')
    if manifest['protected_sha256'] != teacher_provenance['protected_sha256_before']:
        raise RuntimeError('Overlay protected weights differ from the established teacher')
    if manifest['keep_width'] != selected['keep_width']:
        raise RuntimeError('Selected width differs from the overlay manifest')
    old_clips = old_split['calibration'] + old_split['validation'] + old_split['holdout']
    if tuple(map(len, (old_split['calibration'], old_split['validation'], old_split['holdout']))) != (4, 2, 4):
        raise RuntimeError('Expected the established 4/2/4 split')
    if len(set(old_clips)) != 10 or CLIP_ID in old_clips:
        raise RuntimeError('Established splits overlap or include the original baseline clip')
    excluded = [CLIP_ID] + old_clips
    inputs = [old_split_path, teacher_provenance_path, stage_split_path, selection_path, overlay_manifest_path]
    overlay_files = []
    expected_layer_widths = overlay_widths(manifest, selected)
    if [record['layer'] for record in manifest['layers']] != list(range(36)):
        raise RuntimeError('Overlay must include the complete 36-layer ordered mapping')
    for record in manifest['layers']:
        if record['original_width'] != 12288 or record['keep_width'] != expected_layer_widths[record['layer']]:
            raise RuntimeError('Overlay layer dimensions do not match the original/selected widths')
        path = overlay_dir / f'mlp_{record["layer"]:02d}.safetensors'
        digest = file_digest(path)
        if digest != record['sha256']:
            raise RuntimeError(f'Overlay tensor SHA256 differs: {path}')
        overlay_files.append({'file': str(path), 'sha256': digest})
    identity = {'source_commit': SOURCE_COMMIT, 'checkpoint_revision': CHECKPOINT_REVISION,
                'evaluation_schema': 'generation_markers_v2',
                'dataset_revision': DATASET_REVISION, 'keep_width': selected['keep_width'],
                'params': selected['params'], 'vlm_params': selected['vlm_params'],
                'input_artifacts': [_artifact(path) for path in inputs], 'overlay_files': overlay_files}
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
        split = _eligible_clips(interface, dataset, excluded)
        clips = split['expanded_validation'] + split['expanded_holdout']
        if args.resume:
            if json.loads((out / 'split.json').read_text()) != split:
                raise RuntimeError('Existing expanded split differs from the current frozen eligible clip set')
            if json.loads((out / 'provenance.json').read_text())['identity'] != identity:
                raise RuntimeError('Resume provenance uses different frozen inputs or selected student')
            progress = json.loads((out / 'progress.json').read_text())
            if progress.get('split') != split:
                raise RuntimeError('Resume progress split differs')
            _validate_progress(progress, clips, identity)
        else:
            out.mkdir(parents=True)
            _write(out / 'split.json', split)
            progress = {'identity': identity, 'split': split, 'dense': {}, 'student': {}}
            if args.dense_reference:
                reference_dir = args.dense_reference.resolve()
                reference_path = reference_dir / 'progress.json'
                reference = json.loads(reference_path.read_text())
                if reference['split'] != split:
                    raise RuntimeError('Teacher reference split differs from this expansion')
                for key in ('source_commit', 'checkpoint_revision', 'dataset_revision'):
                    if reference['identity'][key] != identity[key]:
                        raise RuntimeError('Teacher reference frozen identity differs')
                reference_result = json.loads((reference_dir / 'result.json').read_text())
                if reference_result['protected_sha256_after'] != manifest['protected_sha256']:
                    raise RuntimeError('Teacher reference protected weights differ')
                dense_records = copy.deepcopy(reference['dense'])
                if set(dense_records) != set(clips) or any(
                        r['status'] != 'completed' for r in dense_records.values()):
                    raise RuntimeError('Teacher reference must cover every new clip successfully')
                audit = reference_dir / 'generation_audit.json'
                audit_rows = json.loads(audit.read_text()) if audit.exists() else []
                if audit.exists():
                    audit_provenance = json.loads((reference_dir / 'generation_audit_provenance.json').read_text())
                    if audit_provenance['observed_clips_per_phase'] != {'dense': len(clips), 'student': len(clips)}:
                        raise RuntimeError('Legacy generation audit does not cover the complete expansion')
                    if file_digest(reference_dir / 'generation_audit_source.log') != audit_provenance['source_log_sha256']:
                        raise RuntimeError('Legacy generation audit source hash differs')
                warning_by_clip = {r['clip_id']: r['warnings'] for r in audit_rows if r['phase'] == 'dense'}
                for clip, record in dense_records.items():
                    if 'trajectory_start_missing' not in record['result']:
                        if not audit.exists():
                            raise RuntimeError('Legacy teacher references require a saved generation audit')
                        record['result']['trajectory_generation_warnings'] = warning_by_clip.get(clip, [])
                        record['result']['trajectory_start_missing'] = clip in warning_by_clip
                progress['dense'] = dense_records
                _validate_progress(progress, clips, identity)
                _write(out / 'dense_reference.json', {
                    'progress': _artifact(reference_path),
                    'result': _artifact(reference_dir / 'result.json'),
                    'generation_audit': _artifact(audit) if audit.exists() else None})
            _write(out / 'progress.json', progress)
            _write(out / 'provenance.json', {
                'identity': identity, 'root': str(root), 'teacher_experiment': str(teacher_dir),
                'stage_experiment': str(stage_dir), 'overlay_directory': str(overlay_dir),
                'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
                'next_stage_gate': 'Complete expanded run and expanded validation mean student ADE <= 1.15*dense',
                'gate_interpretation': '15% is this experiment preset tolerance, not a deployment safety standard',
                'holdout_policy': 'Expanded holdout is reporting-only; no parameter/selection changes',
                'note': 'Original baseline and prior 10 clips are never reevaluated'})
        _write(out / 'status.json', {'status': 'running', 'phase': 'dense', 'requested_new_clips': len(clips)})
        model, processor = _load_local_model_and_processor(root / 'models', cosmos, qwen, config)
        model.eval().requires_grad_(False)
        original_params = sum(parameter.numel() for parameter in model.parameters())
        if original_params != teacher_provenance['original_params']:
            raise RuntimeError('Dense model parameter count differs from the established teacher')
        if protected_digest(model) != manifest['protected_sha256']:
            raise RuntimeError('Dense protected attention/vision/expert weights differ')
        _run_phase(model, processor, interface, clips, 'dense', progress, out)
        if protected_digest(model) != manifest['protected_sha256']:
            raise RuntimeError('Expanded dense evaluation changed protected weights')
        _write(out / 'status.json', {'status': 'running', 'phase': 'student', 'requested_new_clips': len(clips)})
        loaded = load_overlay(model, overlay_dir)
        if loaded != manifest:
            raise RuntimeError('Overlay manifest changed during loading')
        params = sum(parameter.numel() for parameter in model.parameters())
        vlm_params = sum(parameter.numel() for parameter in model.vlm.parameters())
        if params != selected['params'] or vlm_params != selected['vlm_params']:
            raise RuntimeError('Loaded student parameter counts differ from selection')
        _run_phase(model, processor, interface, clips, 'student', progress, out)
        protected_after = protected_digest(model)
        if protected_after != manifest['protected_sha256']:
            raise RuntimeError('Expanded student evaluation changed protected weights')
        if identity['input_artifacts'] != [_artifact(path) for path in inputs]:
            raise RuntimeError('Frozen experiment metadata changed during expanded evaluation')
        validation = _paired_summary(split['expanded_validation'], progress)
        holdout = _paired_summary(split['expanded_holdout'], progress)
        failures = [{'phase': phase, 'clip_id': clip, **record}
                    for phase in ('dense', 'student') for clip, record in progress[phase].items()
                    if record['status'] == 'failed']
        complete = not failures and validation['completed_pairs'] == 20 and (
            holdout['completed_pairs'] == len(split['expanded_holdout']))
        dense_mean = validation['dense_minADE_m']['mean'] if validation['dense_minADE_m'] else None
        student_mean = validation['student_minADE_m']['mean'] if validation['student_minADE_m'] else None
        threshold = None if dense_mean is None else dense_mean * 1.15
        gate_pass = complete and student_mean is not None and student_mean <= threshold
        marker_counts = {
            role: {phase: sum(
                progress[phase].get(clip, {}).get('result', {}).get('trajectory_start_missing', False)
                for clip in split[role]) for phase in ('dense', 'student')}
            for role in ('expanded_validation', 'expanded_holdout')}
        gate_pass = gate_pass and marker_counts['expanded_validation']['student'] == 0
        result = {
            'status': 'completed' if complete else 'incomplete', 'keep_width': selected['keep_width'],
            'params': params, 'vlm_params': vlm_params, 'original_dense_params': original_params,
            'protected_sha256_after': protected_after, 'expanded_validation': validation,
            'expanded_holdout': holdout, 'failed_evaluations': failures,
            'missing_trajectory_start_counts': marker_counts,
            'next_6B_stage_gate': {'pass': gate_pass, 'requires_complete_run': True,
                'requires_no_student_validation_missing_trajectory_start': True,
                'uses_holdout_metrics': False, 'validation_dense_mean_minADE_m': dense_mean,
                'validation_student_mean_minADE_m': student_mean, 'limit_minADE_m': threshold,
                'interpretation': 'Preset tolerance for this experiment; not a deployment safety standard',
                'criterion': 'Expanded validation mean student minADE <= 1.15 * paired dense mean; '
                             'no student validation trajectory-start failures'},
            'limitations': ['Same official training chunk; this is not the official validation split',
                            'Clip-level bootstrap does not model correlations between clips',
                            'No generalization or hardware deployment guarantee']}
        _write(out / 'result.json', result)
        _write(out / 'status.json', {'status': result['status'], 'requested_new_clips': len(clips),
                                   'completed_pairs': validation['completed_pairs'] + holdout['completed_pairs'],
                                   'next_6B_stage_gate_pass': gate_pass})
        _report(result, out / 'report.md')
        print(json.dumps({'status': result['status'], 'new_clips': len(clips),
                          'next_6B_stage_gate': result['next_6B_stage_gate']}, indent=2), flush=True)
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
