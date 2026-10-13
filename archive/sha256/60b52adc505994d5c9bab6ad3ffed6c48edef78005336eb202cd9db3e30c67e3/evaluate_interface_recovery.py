"""Fresh-process parent/merged hidden and KV comparison, no trajectory inference."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from run_interface_recovery import (
    _enable_offline_mode, _block_network, _verify_model, _verify_dataset,
    _verify_config_dir, _verify_config_source, _load_local_model_and_processor,
    COSMOS_REVISION, QWEN_REVISION, MODEL_VLM_NAME, protected_digest, file_digest,
    load_overlay, write_json, forward_targets, text_logits,
)
from student_interface_distill import relative_mse, interface_loss, logits_kl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    root, exp = args.root.resolve(), args.experiment.resolve()
    prov = json.loads((exp / 'provenance.json').read_text())
    candidate = json.loads((exp / 'result.json').read_text())['candidates'][0]
    parent = json.loads((Path(prov['parent_experiment']) / 'result.json').read_text())['candidates'][0]
    configs = root / 'alp15-configs-delivery-20260919'
    cosmos = configs / 'dependencies/configs/Cosmos-Reason2-8B'
    qwen = configs / 'dependencies/configs/Qwen3-VL-2B-Instruct'
    _enable_offline_mode()
    torch.set_num_threads(8)
    result = {'selection_uses_this_report': False, 'semantics': 'Teacher-forced full student hidden/KV, actual merged BF16',
        'input_sha256': {}, 'models': {}}
    with _block_network():
        config = _verify_model(root / 'models')
        _verify_dataset(root / 'alp15-test-clip-delivery-20260917')
        for directory, label, repo, revision, name in (
            (cosmos, 'Cosmos', MODEL_VLM_NAME, COSMOS_REVISION, 'cosmos-reason2-source.json'),
            (qwen, 'Qwen', 'Qwen/Qwen3-VL-2B-Instruct', QWEN_REVISION, 'qwen3-vl-source.json')):
            _verify_config_dir(directory, label)
            _verify_config_source(directory, label, repo, revision, name, configs / 'manifests')
        model, _ = _load_local_model_and_processor(root / 'models', cosmos, qwen, config)
        model.eval().requires_grad_(False)
        parent_manifest = load_overlay(model, Path(parent['overlay_directory']))
        merged_dir = Path(candidate['overlay_directory'])
        merged_manifest = json.loads((merged_dir / 'manifest.json').read_text())
        if parent_manifest['layer_widths'] != merged_manifest['layer_widths']:
            raise RuntimeError('Recovery must not change the channel allocation')
        for label in ('parent', 'merged_bf16'):
            if label == 'merged_bf16':
                for index, layer in enumerate(model.vlm.model.language_model.layers):
                    old, new = parent_manifest['layers'][index], merged_manifest['layers'][index]
                    if old['keep_indices'] != new['keep_indices']:
                        raise RuntimeError('Recovery changed channel mapping')
                    path = merged_dir / f'mlp_{index:02d}.safetensors'
                    if file_digest(path) != new['sha256']:
                        raise RuntimeError('Merged weight SHA mismatch')
                    layer.mlp.load_state_dict(load_file(str(path)), strict=True)
                if protected_digest(model) != prov['protected_sha256_before']:
                    raise RuntimeError('Recovery changed protected weights')
            role_results = {}
            with torch.no_grad():
                for role in ('calibration', 'validation'):
                    records = []
                    for path in sorted((exp / 'teacher_cache').glob(f'{role}_*.pt')):
                        result['input_sha256'][str(path)] = file_digest(path)
                        example = torch.load(path, weights_only=True, map_location='cpu')
                        output = forward_targets(model, example)
                        loss, pieces = interface_loss(output, example['targets'])
                        total = loss.clone()
                        if 'teacher_logits' in example:
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                logits = text_logits(model, output['hidden'][:, example['logit_mask'].cuda()])
                            pieces['next_token_kl'] = logits_kl(logits, example['teacher_logits'])
                            total += prov.get('logit_weight', 0) * pieces['next_token_kl']
                        if 'expert_velocities' in output:
                            pieces['expert_velocity_nmse'] = torch.stack([relative_mse(s,t) for s,t in
                                zip(output['expert_velocities'], example['targets']['expert_velocities'], strict=True)]).mean()
                            total += prov.get('expert_weight', 0) * pieces['expert_velocity_nmse']
                        layers = []
                        for index, (key, value) in enumerate(zip(output['keys'], output['values'], strict=True)):
                            layers.append({'layer': index,
                                'key_nmse': relative_mse(key, example['targets']['keys'][index]).item(),
                                'value_nmse': relative_mse(value, example['targets']['values'][index]).item()})
                        records.append({'clip_id': example['clip_id'], 'loss': loss.item(), 'configured_total_loss': total.item(),
                            **{k: v.item() for k, v in pieces.items()}, 'layer_metrics': layers})
                    role_results[role] = {'mean_loss': sum(r['loss'] for r in records) / len(records),
                        'mean_configured_total_loss': sum(r['configured_total_loss'] for r in records) / len(records),
                        'records': records}
            result['models'][label] = role_results
            print('MERGED_INTERFACE_AUDIT', label, {r: v['mean_loss'] for r, v in role_results.items()}, flush=True)
        result['protected_sha256_after'] = protected_digest(model)
    result['status'] = 'completed'
    write_json(exp / 'merged_interface_audit.json', result)


if __name__ == '__main__':
    main()
