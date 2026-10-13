"""First offline structured MLP pruning and block distillation experiment.

Uses an immutable Alpamayo source/checkpoint and an independent MLP overlay.
Internal clip holdouts are not the official dataset validation split.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import time
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from mlp_pruning import prune_mlp, train_down_projection, load_pruned_mlp_state
from test_alpamayo15_offline_compat import (
    CLIP_ID, DATASET_REVISION, FEATURES, _enable_offline_mode, _block_network,
    _verify_model, _verify_dataset, _verify_config_dir, _verify_config_source,
    _load_offline_dataset_interface, _load_local_model_and_processor,
    MODEL_VLM_NAME, COSMOS_REVISION, QWEN_REVISION,
)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def protected_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith('vlm.model.language_model.layers.') and '.mlp.' in name:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def evaluate(model, processor, interface, clip):
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    data = load_physical_aiavdataset(clip, t0_us=5_100_000, avdi=interface, maybe_stream=False)
    messages = helper.create_message(data['image_frames'].flatten(0, 1), camera_indices=data['camera_indices'])
    encoded = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors='pt')
    inputs = helper.to_device({'tokenized_data': encoded, 'ego_history_xyz': data['ego_history_xyz'],
        'ego_history_rot': data['ego_history_rot']}, 'cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        xyz, rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            inputs, top_p=0.98, temperature=0.6, num_traj_samples=1,
            max_generation_length=256, return_extra=True)
    torch.cuda.synchronize()
    latency = (time.perf_counter() - start) * 1000
    pred = xyz[0, 0, 0, :, :2].float().cpu()
    gt = data['ego_future_xyz'][0, 0, :, :2].float().cpu()
    ade = torch.linalg.vector_norm(pred - gt, dim=-1).mean().item()
    if not math.isfinite(ade) or not torch.isfinite(xyz).all() or not torch.isfinite(rot).all():
        raise RuntimeError(f'Nonfinite trajectory for {clip}')
    if tuple(xyz.shape) != (1, 1, 1, 64, 3) or tuple(rot.shape) != (1, 1, 1, 64, 3, 3):
        raise RuntimeError('Unexpected trajectory shapes')
    return {'clip_id': clip, 't0_us': 5_100_000, 'minADE_m': ade,
        'inference_ms': latency, 'cot': extra['cot'].tolist(),
        'pred_xyz': xyz[0, 0, 0].float().cpu().tolist(),
        'pred_xyz_sha256': hashlib.sha256(xyz.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        'pred_xyz_shape': list(xyz.shape), 'pred_rot_shape': list(rot.shape)}


def collect(model, processor, interface, clips, cache_dir, role, token_limit, score):
    layers = model.vlm.model.language_model.layers
    xs, ys = [[] for _ in layers], [[] for _ in layers]
    sums = [torch.zeros(layer.mlp.intermediate_size, dtype=torch.float64) for layer in layers]
    counts = [0 for _ in layers]
    phase_counts = {'prefill': 0, 'decode': 0}
    handles = []
    def hook_for(index):
        def hook(mlp, args, output):
            x, y = args[0].detach().reshape(-1, 4096), output.detach().reshape(-1, 4096)
            # Uniform prefill subset plus every autoregressive decode state.
            if x.shape[0] > token_limit:
                selected = torch.linspace(0, x.shape[0] - 1, token_limit, device=x.device).long()
                x, y = x[selected], y[selected]
            if index == 0:
                phase_counts['prefill' if args[0].shape[-2] > 1 else 'decode'] += x.shape[0]
            xs[index].append(x.cpu())
            ys[index].append(y.cpu())
            if score:
                with torch.no_grad():
                    z = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
                    sums[index] += z.float().square().sum(0).double().cpu()
                    counts[index] += z.shape[0]
        return hook
    for index, layer in enumerate(layers):
        handles.append(layer.mlp.register_forward_hook(hook_for(index)))
    records = []
    try:
        for clip in clips:
            result = evaluate(model, processor, interface, clip)
            records.append(result)
            print(f'CALIBRATION {role} {clip} ADE={result["minADE_m"]:.6f}', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    for index in range(len(layers)):
        torch.save({'x': torch.cat(xs[index]), 'y': torch.cat(ys[index])}, cache_dir / f'{role}_{index:02d}.pt')
        xs[index].clear()
        ys[index].clear()
    return records, sums, counts, phase_counts


def load_overlay(model, directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    layers = model.vlm.model.language_model.layers
    if manifest['base_checkpoint_revision'] != '7aba8293c09993f2e125c6819df05d7fa3e873ea':
        raise RuntimeError('Overlay checkpoint revision mismatch')
    if [record['layer'] for record in manifest['layers']] != list(range(len(layers))):
        raise RuntimeError('Overlay must contain every layer exactly once in order')
    if protected_digest(model) != manifest['protected_sha256']:
        raise RuntimeError('Overlay base protected weights do not match')
    for record in manifest['layers']:
        index = record['layer']
        path = directory / f'mlp_{index:02d}.safetensors'
        if file_digest(path) != record['sha256']:
            raise RuntimeError(f'Overlay hash mismatch: {path}')
        state = load_file(str(path), device='cpu')
        width = state['gate_proj.weight'].shape[0]
        keep = record['keep_indices']
        if len(keep) != width or len(set(keep)) != width or sorted(keep) != keep or min(keep) < 0 or max(keep) >= record['original_width']:
            raise RuntimeError('Invalid overlay channel mapping')
        if width != record['keep_width'] or set(state) != {'gate_proj.weight', 'up_proj.weight', 'down_proj.weight'}:
            raise RuntimeError('Invalid overlay dimensions or keys')
        load_pruned_mlp_state(layers[index].mlp, state, keep)
    model.eval().requires_grad_(False)
    if protected_digest(model) != manifest['protected_sha256']:
        raise RuntimeError('Overlay loading changed protected weights')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keep-width', type=int, default=9856)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--reload-only', choices=['pruned', 'distilled'])
    args = parser.parse_args()
    if not 1 <= args.keep_width < 12288 or args.keep_width % 128 != 0:
        parser.error('keep-width must be a multiple of 128 between 128 and 12160')
    if args.steps < 1:
        parser.error('steps must be positive')
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.reload_only and any((out / name).exists() for name in ('split.json', 'result.json', 'pruned', 'distilled')):
        raise RuntimeError('Refusing to mix results with an existing experiment')
    cache = out / 'teacher_cache'
    cache.mkdir(exist_ok=True)
    configs = root / 'alp15-configs-delivery-20260919'
    cosmos = configs / 'dependencies/configs/Cosmos-Reason2-8B'
    qwen = configs / 'dependencies/configs/Qwen3-VL-2B-Instruct'
    dataset = root / 'alp15-test-clip-delivery-20260917'
    _enable_offline_mode()
    torch.set_num_threads(8)
    with _block_network():
        config = _verify_model(root / 'models')
        _verify_dataset(dataset)
        for directory, label, repo, revision, manifest in (
            (cosmos, 'Cosmos', MODEL_VLM_NAME, COSMOS_REVISION, 'cosmos-reason2-source.json'),
            (qwen, 'Qwen', 'Qwen/Qwen3-VL-2B-Instruct', QWEN_REVISION, 'qwen3-vl-source.json')):
            _verify_config_dir(directory, label)
            _verify_config_source(directory, label, repo, revision, manifest, configs / 'manifests')
        interface = _load_offline_dataset_interface(dataset)
        frame = interface.clip_index
        candidates = sorted(frame[(frame['chunk'] == 3119) & frame['clip_is_valid']].index.tolist())
        snapshot = dataset / 'datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots' / DATASET_REVISION
        for relative, _, _ in FEATURES.values():
            with zipfile.ZipFile(snapshot / relative) as packed:
                present = {Path(name).name[:36] for name in packed.namelist()}
            candidates = [clip for clip in candidates if clip in present]
        candidates.remove(CLIP_ID)
        if len(candidates) < 10:
            raise RuntimeError('Not enough valid clips with all required features')
        # Fixed hashes avoid depending on original dataframe row order.
        candidates.sort(key=lambda clip: hashlib.sha256(('alp15-prune-v1:' + clip).encode()).hexdigest())
        split = {'calibration': candidates[:4], 'validation': candidates[4:6], 'holdout': candidates[6:10],
            'note': 'Internal clip-disjoint split within official train chunk3119, not official validation.',
            'excluded_baseline_clip': CLIP_ID}
        if args.reload_only:
            split = json.loads((out / 'split.json').read_text())
        else:
            write_json(out / 'split.json', split)
        model, processor = _load_local_model_and_processor(root / 'models', cosmos, qwen, config)
        model.eval().requires_grad_(False)
        layers = model.vlm.model.language_model.layers
        assert len(layers) == len(model.expert.layers) == 36
        assert all(layer.mlp.intermediate_size == 12288 for layer in layers)
        if args.reload_only:
            load_overlay(model, out / args.reload_only)
            results = [evaluate(model, processor, interface, clip) for clip in split['holdout']]
            write_json(out / f'{args.reload_only}_reload_eval.json', results)
            previous = json.loads((out / f'{args.reload_only}_holdout.json').read_text())
            for old, new in zip(previous, results, strict=True):
                if old['clip_id'] != new['clip_id'] or old['cot'] != new['cot']:
                    raise RuntimeError('Reload output ordering or CoC differs')
                torch.testing.assert_close(torch.tensor(old['pred_xyz']), torch.tensor(new['pred_xyz']), rtol=1e-3, atol=1e-4)
                if abs(old['minADE_m'] - new['minADE_m']) > 1e-4:
                    raise RuntimeError('Reload ADE differs beyond 1e-4 meters')
            print('FRESH PROCESS RELOAD PASS', flush=True)
            return
        original_params = sum(p.numel() for p in model.parameters())
        protected_before = protected_digest(model)
        write_json(out / 'provenance.json', {'source_commit': '36aeb4c5938cbc2eb2aed33b22434773da4ab639',
            'checkpoint_revision': '7aba8293c09993f2e125c6819df05d7fa3e873ea', 'dataset_revision': DATASET_REVISION,
            'protected_sha256_before': protected_before, 'original_params': original_params,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'calibration_tokens': 'Uniform 256 prefill positions plus every CoC decode position, equal token weight',
            'distillation': 'Local MLP feature MSE on dense-teacher inputs; down projection only; no whole-model logits KD'})
        calibration, sums, counts, phases = collect(model, processor, interface, split['calibration'], cache, 'train', 256, True)
        validation, _, _, val_phases = collect(model, processor, interface, split['validation'], cache, 'val', 256, False)
        write_json(out / 'teacher_calibration.json', {'calibration': calibration, 'validation': validation,
            'calibration_token_counts': phases, 'validation_token_counts': val_phases})
        dense_results = [evaluate(model, processor, interface, clip) for clip in split['holdout']]
        write_json(out / 'dense_holdout.json', dense_results)
        manifest = {'method': 'activation-down-column-L2 structured MLP channel pruning',
            'base_checkpoint_revision': '7aba8293c09993f2e125c6819df05d7fa3e873ea',
            'protected_sha256': protected_before, 'keep_width': args.keep_width, 'layers': []}
        pruned_dir, distilled_dir = out / 'pruned', out / 'distilled'
        pruned_dir.mkdir(exist_ok=True)
        distilled_dir.mkdir(exist_ok=True)
        for index, layer in enumerate(layers):
            mlp = layer.mlp
            score = (sums[index] / counts[index]) * mlp.down_proj.weight.float().square().sum(0).double().cpu()
            keep = torch.argsort(score, descending=True, stable=True)[:args.keep_width].sort().values
            prune_mlp(mlp, keep)
            record = {'layer': index, 'original_width': 12288, 'keep_width': args.keep_width,
                'keep_indices': keep.tolist(), 'calibration_token_count': counts[index]}
            manifest['layers'].append(record)
            torch.save(score, out / f'scores_{index:02d}.pt')
            save_file({k: v.detach().cpu().contiguous() for k, v in mlp.state_dict().items()}, str(pruned_dir / f'mlp_{index:02d}.safetensors'))
            record['sha256'] = file_digest(pruned_dir / f'mlp_{index:02d}.safetensors')
        write_json(pruned_dir / 'manifest.json', manifest)
        pruned_params = sum(p.numel() for p in model.parameters())
        pruned_results = [evaluate(model, processor, interface, clip) for clip in split['holdout']]
        write_json(out / 'pruned_holdout.json', pruned_results)
        training = []
        distilled_manifest = copy.deepcopy(manifest)
        for index, layer in enumerate(layers):
            train_data = torch.load(cache / f'train_{index:02d}.pt', weights_only=True)
            val_data = torch.load(cache / f'val_{index:02d}.pt', weights_only=True)
            report = train_down_projection(layer.mlp, train_data['x'], train_data['y'], val_data['x'], val_data['y'],
                steps=args.steps, batch_size=128, lr=3e-4, seed=42 + index)
            report['layer'] = index
            training.append(report)
            save_file({k: v.detach().cpu().contiguous() for k, v in layer.mlp.state_dict().items()}, str(distilled_dir / f'mlp_{index:02d}.safetensors'))
            distilled_manifest['layers'][index]['sha256'] = file_digest(distilled_dir / f'mlp_{index:02d}.safetensors')
            write_json(out / 'distillation_progress.json', training)
            print(f'DISTILL layer={index} {report}', flush=True)
            del train_data, val_data
            gc.collect()
            torch.cuda.empty_cache()
        write_json(distilled_dir / 'manifest.json', {**distilled_manifest, 'distillation_steps_per_layer': args.steps})
        model.eval().requires_grad_(False)
        distilled_results = [evaluate(model, processor, interface, clip) for clip in split['holdout']]
        write_json(out / 'distilled_holdout.json', distilled_results)
        protected_after = protected_digest(model)
        if protected_after != protected_before:
            raise RuntimeError('Protected attention/vision/expert weights changed')
        results = {'status': 'experiment_completed', 'protected_sha256_after': protected_after,
            'original_params': original_params, 'pruned_params': pruned_params,
            'removed_params': original_params - pruned_params, 'removed_fraction': 1 - pruned_params / original_params,
            'bf16_weight_gib_before': original_params * 2 / 2**30, 'bf16_weight_gib_after': pruned_params * 2 / 2**30,
            'dense_holdout': dense_results, 'pruned_holdout': pruned_results, 'distilled_holdout': distilled_results,
            'limitations': ['4 internal holdout clips from one official train chunk; no generalization claim',
                'block distillation uses teacher states; upstream student distribution shift remains',
                'shared GPU timing is diagnostic; no deployment feasibility claim']}
        write_json(out / 'result.json', results)
        print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
