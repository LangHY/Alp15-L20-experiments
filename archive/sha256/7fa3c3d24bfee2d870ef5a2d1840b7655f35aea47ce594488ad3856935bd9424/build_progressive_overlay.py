"""Prepare a conservative calibration-only allocation from accepted 8B weights."""
import argparse
import copy
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from layerwise_allocation import allocate_widths
from mlp_pruning import select_channels
from run_mlp_experiment import file_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--teacher-experiment', type=Path, required=True)
    parser.add_argument('--parent-experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError('Refusing to overwrite a preparation')
    profile = json.loads((args.profile / 'importance.json').read_text())
    source_plan = json.loads((args.profile / 'allocation_5376.json').read_text())
    if profile['status'] != 'completed' or source_plan['ranking_uses_holdout'] or source_plan['ranking_uses_validation']:
        raise RuntimeError('Completed calibration-only importance measurements required')
    if file_digest(args.profile / 'provenance.json') != source_plan['profile_provenance_sha256'] or file_digest(args.profile / 'measurement_progress.json') != source_plan['measurement_sha256']:
        raise RuntimeError('Importance measurement identity changed')
    parent = json.loads((args.parent_experiment / 'result.json').read_text())
    selected = parent['selected']
    expanded = json.loads((args.parent_experiment / 'expanded_checks/result.json').read_text())
    if not expanded['next_6B_stage_gate']['pass'] or selected['params'] != 8_020_878_578:
        raise RuntimeError('This progressive path must start from the accepted uniform 8B joint student')
    source = Path(selected['overlay_directory'])
    old_manifest = json.loads((source / 'manifest.json').read_text())
    protected_layers = [r['layer'] for r in profile['layers'] if r['protected']]
    curves = [[r for r in rows if r['width'] <= 5376] for rows in profile['curves']]
    plan = allocate_widths(curves, channel_budget=36 * 4480, minimum_width=3584,
        protected_layers=protected_layers, protected_minimum=5376)
    total = source_plan['fixed_non_mlp_params'] + 3 * 4096 * plan['sum_widths']
    label = 'progressive_4480'
    weights = out / f'width_{label}' / 'distilled'
    weights.mkdir(parents=True)
    manifest = copy.deepcopy(old_manifest)
    manifest.update({'method': 'Importance-guided conservative progressive pruning of accepted 8B joint weights',
        'keep_width': label, 'layer_widths': plan['layer_widths'],
        'parent_overlay_manifest_sha256': file_digest(source / 'manifest.json')})
    for index, width in enumerate(plan['layer_widths']):
        old = old_manifest['layers'][index]
        path = source / f'mlp_{index:02d}.safetensors'
        if file_digest(path) != old['sha256']:
            raise RuntimeError('Accepted parent weight SHA mismatch')
        score_path = args.teacher_experiment / f'scores_{index:02d}.pt'
        keep = select_channels(torch.load(score_path, weights_only=True), width)
        lookup = {channel: position for position, channel in enumerate(old['keep_indices'])}
        if any(i not in lookup for i in keep.tolist()):
            raise RuntimeError('New channels are not a subset of the accepted parent; refusing to reconstruct missing weights')
        indices = torch.tensor([lookup[i] for i in keep.tolist()])
        state = load_file(str(path))
        new = {'gate_proj.weight': state['gate_proj.weight'][indices].contiguous(),
            'up_proj.weight': state['up_proj.weight'][indices].contiguous(),
            'down_proj.weight': state['down_proj.weight'][:, indices].contiguous()}
        target = weights / path.name
        save_file(new, str(target))
        manifest['layers'][index].update({'keep_width': width, 'keep_indices': keep.tolist(),
            'sha256': file_digest(target), 'parent_weight_sha256': old['sha256'],
            'channel_score_sha256': file_digest(score_path)})
    write_json(weights / 'manifest.json', manifest)
    plan.update({'name': label, 'format_version': 1, 'actual_total_params': total,
        'ranking_uses_holdout': False, 'ranking_uses_validation': False,
        'profile_provenance_sha256': source_plan['profile_provenance_sha256'],
        'measurement_sha256': source_plan['measurement_sha256'], 'parent_experiment': str(args.parent_experiment),
        'maximum_width': 5376, 'status': 'prepared_not_evaluated'})
    write_json(out / 'allocation.json', plan)
    split = json.loads((args.teacher_experiment / 'split.json').read_text())
    write_json(out / 'split.json', split)
    candidate = {'keep_width': label, 'layer_widths': plan['layer_widths'], 'params': total,
        'vlm_params': total - 2_279_238_656 - 1_353_730, 'overlay_directory': str(weights),
        'directory': str(weights.parent), 'status': 'prepared_not_evaluated',
        'validation_mean_minADE_m': None, 'validation_pass': False}
    write_json(out / 'result.json', {'status': 'prepared_not_evaluated', 'candidates': [candidate],
        'original_dense_params': parent['original_dense_params'],
        'previous_10B_stage_params': parent['previous_10B_stage_params'], 'selected': None,
        'selection_uses_holdout': False})
    print('PROGRESSIVE_PREPARED', json.dumps({'params': total, 'widths': plan['layer_widths'],
        'note': 'Preparation only; no trajectory accuracy or acceptance claim'}), flush=True)


if __name__ == '__main__':
    main()
