"""Measure all 36 dense MLPs before allocating any student pruning budget.

Uses only the frozen calibration cache. Validation and holdout features are never
used for ranking, allocation, or this profiling run. No rollout baseline is run.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from layerwise_allocation import allocate_widths
from run_width_sweep import _read_teacher, ORIGINAL_WIDTH, CHECKPOINT_REVISION
from run_mlp_experiment import (
    _enable_offline_mode, _block_network, _verify_model, _verify_config_dir,
    _verify_config_source, _load_local_model_and_processor, file_digest,
    protected_digest, write_json, MODEL_VLM_NAME, COSMOS_REVISION, QWEN_REVISION,
)

GRID = [1408, 2560, 3072, 3584, 4096, 5376, 7168, 9216, 12288]


@torch.no_grad()
def measure_curve(mlp, cache, scores, widths, batch_size):
    x, y = cache['x'], cache['y']
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] != 4096 or not x.shape[0]:
        raise ValueError('Invalid calibration cache shape')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite calibration cache')
    ranked = torch.argsort(scores, descending=True, stable=True)
    indices = {w: ranked[:w].sort().values.to('cuda') for w in widths}
    down = {w: mlp.down_proj.weight.index_select(1, indices[w]) for w in widths if w != ORIGINAL_WIDTH}
    error = dict.fromkeys(widths, 0.0)
    target_energy, input_energy = 0.0, 0.0
    started = time.perf_counter()
    with torch.autocast('cuda', enabled=False):
        for first in range(0, x.shape[0], batch_size):
            inputs = x[first:first + batch_size].to('cuda', torch.bfloat16)
            target = y[first:first + batch_size].to('cuda', torch.float32)
            features = mlp.act_fn(mlp.gate_proj(inputs)) * mlp.up_proj(inputs)
            target_energy += target.square().sum().item()
            input_energy += inputs.float().square().sum().item()
            for width in widths:
                prediction = (mlp.down_proj(features) if width == ORIGINAL_WIDTH else
                              F.linear(features.index_select(1, indices[width]), down[width])).float()
                error[width] += (prediction - target).square().sum().item()
    if not target_energy > 0 or not input_energy > 0:
        raise ValueError('Calibration cache has zero input or output energy')
    rows = [{'width': w, 'calibration_nmse': error[w] / target_energy,
             'input_relative_error': error[w] / input_energy,
             'squared_error_sum': error[w], 'target_squared_energy': target_energy,
             'input_squared_energy': input_energy, 'calibration_tokens': x.shape[0],
             'removed_fraction': 1 - w / ORIGINAL_WIDTH} for w in widths]
    if any(not math.isfinite(r['calibration_nmse']) for r in rows):
        raise ValueError('Nonfinite sensitivity measurement')
    return rows, time.perf_counter() - started


def make_plots(out, curves, importance, plans):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
    plt.rcParams['svg.fonttype'] = 'path'
    figs = []
    fig, axes = plt.subplots(6, 6, figsize=(18, 16), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot([r['width'] for r in curves[i]], [r['calibration_nmse'] for r in curves[i]], marker='.')
        ax.set_title(f'第 {i} 层', fontproperties=font)
        ax.grid(alpha=.25)
    fig.supxlabel('保留的 MLP 通道数', fontproperties=font)
    fig.supylabel('剪枝前敏感性评估：校准集局部 NMSE（未蒸馏）', fontproperties=font)
    figs.append(('layer_sensitivity_curves', fig))
    fig, ax = plt.subplots(figsize=(15, 5))
    rows = sorted(importance, key=lambda r: r['layer'])
    colors = ['#d95f02' if r['protected'] else '#3182bd' for r in rows]
    ax.bar([r['layer'] for r in rows], [r['sensitivity_nmse_at_5376'] for r in rows], color=colors)
    ax.set_xlabel('MLP 层编号', fontproperties=font)
    ax.set_ylabel('统一保留 5376 通道时的校准 NMSE', fontproperties=font)
    ax.set_title('逐层剪枝敏感性；橙色为本轮保护的前四分之一层（局部指标）', fontproperties=font)
    figs.append(('layer_importance', fig))
    fig, ax = plt.subplots(figsize=(15, 5))
    for plan in plans:
        ax.plot(range(36), plan['layer_widths'], marker='.', label=f"{plan['actual_total_params']/1e9:.3f}B 分配")
    ax.axhline(5376, color='grey', ls='--', label='旧 8B 统一宽度')
    ax.axhline(1408, color='red', ls=':', label='旧 6B 统一宽度')
    ax.set_xlabel('MLP 层编号', fontproperties=font)
    ax.set_ylabel('保留通道数', fontproperties=font)
    ax.set_title('基于测量结果的非均匀宽度分配', fontproperties=font)
    ax.legend(prop=font)
    figs.append(('layer_width_allocation', fig))
    artifacts = []
    for name, fig in figs:
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            path = out / f'{name}.{suffix}'
            fig.savefig(path, dpi=150)
            artifacts.append({'file': path.name, 'sha256': file_digest(path)})
        plt.close(fig)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--teacher-experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    root, teacher, out = args.root.resolve(), args.teacher_experiment.resolve(), args.output.resolve()
    if out.exists():
        raise RuntimeError('Refusing to overwrite an existing profile')
    _enable_offline_mode()
    torch.set_num_threads(8)
    split, provenance, _, _, scores, hashes = _read_teacher(teacher)
    out.mkdir(parents=True)
    code = out / 'code'
    code.mkdir()
    for name in ('profile_layer_importance.py', 'layerwise_allocation.py', 'mlp_pruning.py', 'run_width_sweep.py'):
        shutil.copy2(Path(__file__).parent / name, code / name)
    write_json(out / 'provenance.json', {
        'checkpoint_revision': CHECKPOINT_REVISION, 'teacher_provenance': provenance,
        'teacher_artifacts': hashes, 'calibration_clips': split['calibration'],
        'ranking_uses_validation': False, 'ranking_uses_holdout': False,
        'grid': GRID, 'seed': 42, 'batch_size': args.batch_size,
        'metric': 'Local untrained MLP output squared error / teacher MLP output squared energy',
        'dense_reproduction_nmse_tolerance': 1e-3,
        'limitations': ['Four calibration clips from one chunk', 'Cached teacher inputs, not student inputs',
                        'Local sensitivity proxy, not causal trajectory importance',
                        'Dense candidate checks BF16 cache reproduction with NMSE <= 1e-3; different token batch shapes can change BF16 reductions',
                        'Raw dense reproduction error is retained; no subtraction or invented zero floor'],
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'argv': __import__('sys').argv,
        'code': [{'file': p.name, 'sha256': file_digest(p)} for p in code.iterdir()]})
    curves = []
    with _block_network():
        config = _verify_model(root / 'models')
        configs = root / 'alp15-configs-delivery-20260919'
        cosmos = configs / 'dependencies/configs/Cosmos-Reason2-8B'
        qwen = configs / 'dependencies/configs/Qwen3-VL-2B-Instruct'
        for directory, label, repo, rev, manifest in (
            (cosmos, 'Cosmos', MODEL_VLM_NAME, COSMOS_REVISION, 'cosmos-reason2-source.json'),
            (qwen, 'Qwen', 'Qwen/Qwen3-VL-2B-Instruct', QWEN_REVISION, 'qwen3-vl-source.json')):
            _verify_config_dir(directory, label)
            _verify_config_source(directory, label, repo, rev, manifest, configs / 'manifests')
        model, processor = _load_local_model_and_processor(root / 'models', cosmos, qwen, config)
        model.eval().requires_grad_(False)
        if protected_digest(model) != provenance['protected_sha256_before']:
            raise RuntimeError('Dense protected weights mismatch')
        layers = model.vlm.model.language_model.layers
        if len(layers) != 36 or any(l.mlp.intermediate_size != ORIGINAL_WIDTH for l in layers):
            raise RuntimeError('Expected original dense 36-layer model')
        for i, layer in enumerate(layers):
            cache = torch.load(teacher / 'teacher_cache' / f'train_{i:02d}.pt', map_location='cpu', weights_only=True)
            rows, elapsed = measure_curve(layer.mlp, cache, scores[i], GRID, args.batch_size)
            # A failed dense-cache reproduction invalidates all sensitivity scores.
            if rows[-1]['calibration_nmse'] > 1e-3:
                raise RuntimeError(f'Dense calibration cache mismatch at layer {i}: {rows[-1]}')
            curves.append(rows)
            write_json(out / 'measurement_progress.json', {
                'status': 'profiling', 'completed_layers': len(curves), 'curves': curves})
            print(f'IMPORTANCE layer={i} nmse5376={next(r["calibration_nmse"] for r in rows if r["width"]==5376):.6g} elapsed={elapsed:.2f}', flush=True)
            del cache
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    rank = sorted(range(36), key=lambda i: (-next(r['calibration_nmse'] for r in curves[i] if r['width']==5376), i))
    protected = rank[:9]
    importance = []
    for i, curve in enumerate(curves):
        reference = next(r for r in curve if r['width'] == 5376)
        importance.append({'layer': i, 'rank': rank.index(i)+1, 'protected': i in protected,
                           'sensitivity_nmse_at_5376': reference['calibration_nmse'],
                           'input_relative_error_at_5376': reference['input_relative_error'],
                           'calibration_tokens': reference['calibration_tokens']})
    fixed_params = provenance['original_params'] - 36 * ORIGINAL_WIDTH * 3 * 4096
    plans = []
    for mean_width in (5376, 3584):
        plan = allocate_widths(curves, channel_budget=36 * mean_width, minimum_width=2560,
                               protected_layers=protected, protected_minimum=5376)
        plan.update({'name': f'layerwise_{mean_width}', 'format_version': 1,
                     'actual_total_params': fixed_params + 3 * 4096 * plan['sum_widths'],
                     'target_total_params': fixed_params + 3 * 4096 * 36 * mean_width,
                     'fixed_non_mlp_params': fixed_params, 'reference_uniform_width': mean_width,
                     'teacher_experiment': str(teacher), 'profile_provenance_sha256': file_digest(out / 'provenance.json'),
                     'measurement_sha256': file_digest(out / 'measurement_progress.json'),
                     'protected_policy': 'Top 9 measured local NMSE layers at width 5376 have a 5376 floor; all other layers have a 2560 floor',
                     'ranking_uses_holdout': False, 'ranking_uses_validation': False})
        write_json(out / f"allocation_{mean_width}.json", plan)
        plans.append(plan)
    for name, records in [('layer_importance.csv', importance),
                           ('sensitivity_measurements.csv', [dict(layer=i, **r) for i, c in enumerate(curves) for r in c])]:
        with (out / name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_json(out / 'importance.json', {'status': 'completed', 'layers': importance, 'curves': curves,
                                        'plans': plans, 'charts': make_plots(out, curves, importance, plans)})
    print('IMPORTANCE_COMPLETE', json.dumps({'protected_layers': protected,
          'plans': [{'name': p['name'], 'params': p['actual_total_params'], 'widths': p['layer_widths']} for p in plans]}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
