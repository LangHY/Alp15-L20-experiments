"""Full student MLP-adapter KD on frozen four training/two validation clips.

Teacher forcing reconstructs a CoC prefix from existing recorded reasoning;
these caches are not claimed to reproduce the original autoregressive rollout.
No trajectory teacher baseline is run, and no holdout is used for optimization.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from run_mlp_experiment import (
    _enable_offline_mode, _block_network, _verify_model, _verify_dataset,
    _verify_config_dir, _verify_config_source, _load_local_model_and_processor,
    _load_offline_dataset_interface, COSMOS_REVISION, QWEN_REVISION, MODEL_VLM_NAME,
    DATASET_REVISION, protected_digest, file_digest, load_overlay, evaluate, write_json,
)
from student_interface_distill import install_adapters, merge_adapters, interface_targets, interface_loss, logits_kl


def prepare(model, processor, interface, record, rollout_prefix=False):
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    data = load_physical_aiavdataset(record['clip_id'], t0_us=5_100_000, avdi=interface, maybe_stream=False)
    messages = helper.create_message(data['image_frames'].flatten(0, 1), camera_indices=data['camera_indices'])
    encoded = dict(processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors='pt'))
    # Use the known teacher reasoning text as a fixed prefix. Do not invent any
    # unrecorded meta-action or post-CoC token sequence.
    prompt_length = encoded['input_ids'].shape[1]
    if not rollout_prefix:
        reasoning = record['cot'][0][0][0]
        continuation = processor.tokenizer(reasoning, add_special_tokens=False, return_tensors='pt')['input_ids']
        encoded['input_ids'] = torch.cat([encoded['input_ids'], continuation], dim=1)
        if 'attention_mask' in encoded:
            encoded['attention_mask'] = torch.ones_like(encoded['input_ids'])
    encoded = helper.to_device(encoded, 'cuda')
    with torch.no_grad():
        encoded['input_ids'] = model.fuse_traj_tokens(encoded['input_ids'], {
            'ego_history_xyz': data['ego_history_xyz'].cuda(), 'ego_history_rot': data['ego_history_rot'].cuda()})
    if rollout_prefix:
        from alpamayo1_5.models.token_utils import StopAfterEOS
        from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
        from transformers import StoppingCriteriaList, LogitsProcessorList
        gen = copy.deepcopy(model.vlm.generation_config)
        gen.top_p, gen.temperature, gen.do_sample = 0.98, 0.6, True
        gen.top_k = None
        gen.num_return_sequences, gen.max_new_tokens = 1, 256
        gen.output_logits, gen.return_dict_in_generate = False, True
        gen.pad_token_id = model.tokenizer.pad_token_id
        future = model.tokenizer.convert_tokens_to_ids('<|traj_future_start|>')
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model.vlm.generate(**encoded, generation_config=gen,
                stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=future)]),
                logits_processor=LogitsProcessorList([ExpertLogitsProcessor(
                    traj_token_offset=model.config.traj_token_start_idx, traj_vocab_size=model.config.traj_vocab_size)]))
        sequence = output.sequences
        if not (sequence[0, prompt_length:] == future).any():
            raise RuntimeError('Dense teacher did not emit the trajectory-start boundary')
        # StopAfterEOS generates one extra token so the future-start token is
        # already cached. That last extra token is not part of the Expert prefix.
        encoded['input_ids'] = sequence[:, :-1].detach()
        if encoded['input_ids'][0, -1].item() != future:
            raise RuntimeError('Unexpected generated boundary; refusing to guess cache alignment')
        if 'attention_mask' in encoded:
            encoded['attention_mask'] = torch.ones_like(encoded['input_ids'])
        del output
    positions = torch.unique(torch.cat([
        torch.linspace(0, prompt_length - 1, min(128, prompt_length)).long(),
        torch.arange(prompt_length, encoded['input_ids'].shape[1])]))
    logit_mask = positions >= prompt_length - 1
    if rollout_prefix:
        logit_mask &= positions < encoded['input_ids'].shape[1] - 1
    return {k: v.cpu() for k, v in encoded.items()}, positions, logit_mask


def forward_targets(model, example):
    inputs = {k: v.cuda() for k, v in example['inputs'].items()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model.vlm.model(**inputs, use_cache=True, return_dict=True)
        return interface_targets(output, example['positions'].cuda())


def cpu_targets(targets):
    return {k: [x.detach().cpu() for x in v] if isinstance(v, list) else v.detach().cpu()
            for k, v in targets.items()}


def text_logits(model, hidden):
    logits = model.vlm.lm_head(hidden).float()
    start = model.config.traj_token_start_idx
    logits[..., start:start + model.config.traj_vocab_size] = -1e9
    return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--teacher-experiment', type=Path, required=True)
    parser.add_argument('--parent-experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--eval-every', type=int, default=12)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rollout-prefix', action='store_true', help='Collect actual dense CoC tokens through Expert boundary; no teacher trajectories')
    parser.add_argument('--logit-weight', type=float, default=0.0)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.rank < 1 or args.lr <= 0 or args.logit_weight < 0:
        parser.error('Training settings must be positive')
    root, out = args.root.resolve(), args.output.resolve()
    if out.exists():
        raise RuntimeError('Refusing to overwrite an experiment')
    out.mkdir(parents=True)
    (out / 'teacher_cache').mkdir()
    (out / 'code').mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, out / 'code' / path.name)
    split = json.loads((args.teacher_experiment / 'split.json').read_text())
    teacher = json.loads((args.teacher_experiment / 'teacher_calibration.json').read_text())
    parent = json.loads((args.parent_experiment / 'result.json').read_text())
    candidate = parent['candidates'][0]
    overlay = Path(candidate['overlay_directory'])
    manifest = json.loads((overlay / 'manifest.json').read_text())
    dense_mean = sum(r['minADE_m'] for r in teacher['validation']) / len(teacher['validation'])
    write_json(out / 'split.json', split)
    _enable_offline_mode()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    configs = root / 'alp15-configs-delivery-20260919'
    cosmos = configs / 'dependencies/configs/Cosmos-Reason2-8B'
    qwen = configs / 'dependencies/configs/Qwen3-VL-2B-Instruct'
    dataset = root / 'alp15-test-clip-delivery-20260917'
    with _block_network():
        config = _verify_model(root / 'models')
        _verify_dataset(dataset)
        for directory, label, repo, revision, name in (
            (cosmos, 'Cosmos', MODEL_VLM_NAME, COSMOS_REVISION, 'cosmos-reason2-source.json'),
            (qwen, 'Qwen', 'Qwen/Qwen3-VL-2B-Instruct', QWEN_REVISION, 'qwen3-vl-source.json')):
            _verify_config_dir(directory, label)
            _verify_config_source(directory, label, repo, revision, name, configs / 'manifests')
        interface = _load_offline_dataset_interface(dataset)
        model, processor = _load_local_model_and_processor(root / 'models', cosmos, qwen, config)
        model.eval().requires_grad_(False)
        protected = protected_digest(model)
        if protected != manifest['protected_sha256']:
            raise RuntimeError('Protected checkpoint identity mismatch')
        provenance = {
            'source_commit': '36aeb4c5938cbc2eb2aed33b22434773da4ab639',
            'checkpoint_revision': manifest['base_checkpoint_revision'], 'dataset_revision': DATASET_REVISION,
            'parent_experiment': str(args.parent_experiment), 'parent_overlay_sha256': file_digest(overlay / 'manifest.json'),
            'protected_sha256_before': protected, 'seed': 42, 'rank': args.rank, 'lr': args.lr,
            'steps': args.steps, 'eval_every': args.eval_every, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'argv': sys.argv, 'python': sys.version,
            'rollout_prefix': args.rollout_prefix, 'logit_weight': args.logit_weight,
            'gpu': torch.cuda.get_device_name(), 'code_sha256': {p.name: file_digest(p) for p in (out / 'code').glob('*.py')},
            'training_policy': 'Only four frozen calibration clips; two frozen validation clips select hidden/KV best checkpoint.',
            'objective': 'Final hidden NMSE + mean(all-layer key NMSE, all-layer value NMSE) + logit_weight * next-token KL. Full student graph, MLP LoRA only.',
            'teacher_forcing': ('Actual newly collected dense CoC tokens through trajectory-start boundary; no dense trajectory rollout' if args.rollout_prefix
                else 'Existing recorded CoC text appended to real fused-history multimodal prompt, without unrecorded suffix.'),
            'target_positions': '128 uniform prompt positions plus every recorded reasoning text token',
            'holdout_uses': 'Reporting only; never training, allocation, hyperparameter or checkpoint selection.',
            'limitations': ['Four training clips from one chunk; no external generalization claim',
                'Teacher-forced interface loss is not on-policy reasoning or differentiable trajectory loss',
                *([] if args.rollout_prefix else ['KV targets exclude the unrecorded trajectory-start suffix'])],
        }
        write_json(out / 'provenance.json', provenance)
        examples = {'calibration': [], 'validation': []}
        for role in examples:
            if {r['clip_id'] for r in teacher[role]} != set(split[role]):
                raise RuntimeError('Teacher and split identities differ')
            for index, record in enumerate(teacher[role]):
                inputs, positions, logit_mask = prepare(model, processor, interface, record, args.rollout_prefix)
                example = {'clip_id': record['clip_id'], 'inputs': inputs, 'positions': positions, 'logit_mask': logit_mask}
                with torch.no_grad():
                    example['targets'] = cpu_targets(forward_targets(model, example))
                    if args.logit_weight:
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            example['teacher_logits'] = text_logits(model, example['targets']['hidden'][:, logit_mask].cuda()).cpu()
                path = out / 'teacher_cache' / f'{role}_{index}.pt'
                torch.save(example, path)
                examples[role].append(example)
                print('TEACHER_INTERFACE', role, index, inputs['input_ids'].shape, flush=True)
        load_overlay(model, overlay)
        base_params = sum(p.numel() for p in model.parameters())
        adapters, originals = install_adapters(model, args.rank)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0)
        print('ADAPTER_PARAMS', sum(p.numel() for p in trainable), flush=True)
        start = time.perf_counter()
        rows = []
        def objective(example):
            output = forward_targets(model, example)
            loss, pieces = interface_loss(output, example['targets'])
            if args.logit_weight:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = text_logits(model, output['hidden'][:, example['logit_mask'].cuda()])
                kl = logits_kl(logits, example['teacher_logits'])
                loss = loss + args.logit_weight * kl
                pieces['next_token_kl'] = kl
            return loss, pieces
        def validate(step):
            losses = []
            metrics = []
            with torch.no_grad():
                for example in examples['validation']:
                    loss, pieces = objective(example)
                    losses.append(loss.item())
                    metrics.append({k: v.item() for k, v in pieces.items()})
            mean = sum(losses) / len(losses)
            if not math.isfinite(mean):
                raise RuntimeError('Nonfinite interface validation')
            row = {'phase': 'validation', 'step': step, 'loss': mean, 'per_clip_loss': losses,
                'metrics': metrics, 'elapsed_seconds': time.perf_counter() - start}
            append(row)
            print('INTERFACE_VALIDATION', step, mean, flush=True)
            return mean
        def append(row):
            rows.append(row)
            with (out / 'training_trace.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
        def snapshot():
            return [(m.a.detach().cpu().clone(), m.b.detach().cpu().clone()) for m in adapters]
        best_loss, best_step = validate(0), 0
        best = snapshot()
        for step in range(1, args.steps + 1):
            example = examples['calibration'][(step - 1) % len(examples['calibration'])]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            loss, pieces = objective(example)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            lr = args.lr * (0.1 + 0.9 * (1 + math.cos(math.pi * (step - 1) / args.steps)) / 2)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            row = {'phase': 'training', 'step': step, 'clip_id': example['clip_id'], 'loss': loss.item(),
                **{k: v.item() for k, v in pieces.items()}, 'learning_rate': lr, 'grad_norm': norm.item(),
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'elapsed_seconds': time.perf_counter() - start}
            append(row)
            print('INTERFACE_STEP', json.dumps(row), flush=True)
            del loss, pieces
            if step % args.eval_every == 0 or step == args.steps:
                val_loss = validate(step)
                if val_loss < best_loss:
                    best_loss, best_step, best = val_loss, step, snapshot()
                    torch.save(best, out / 'best_adapters.pt')
        with torch.no_grad():
            for adapter, (a, b) in zip(adapters, best, strict=True):
                adapter.a.copy_(a); adapter.b.copy_(b)
        merge_adapters(model, originals)
        model.requires_grad_(False)
        if sum(p.numel() for p in model.parameters()) != base_params or protected_digest(model) != protected:
            raise RuntimeError('Merged model changed protected weights or parameter count')
        width = candidate['keep_width']
        target = out / f'width_{width}'
        merged_dir = target / 'distilled'
        merged_dir.mkdir(parents=True)
        merged_manifest = copy.deepcopy(manifest)
        for index, layer in enumerate(model.vlm.model.language_model.layers):
            path = merged_dir / f'mlp_{index:02d}.safetensors'
            save_file({k: v.detach().cpu().contiguous() for k, v in layer.mlp.state_dict().items()}, str(path))
            merged_manifest['layers'][index]['sha256'] = file_digest(path)
        merged_manifest['recovery'] = {'method': 'full-student hidden/KV MLP LoRA KD, merged BF16',
            'rank': args.rank, 'selected_step': best_step, 'validation_interface_loss': best_loss}
        write_json(merged_dir / 'manifest.json', merged_manifest)
        validation = [evaluate(model, processor, interface, clip) for clip in split['validation']]
        mean = sum(r['minADE_m'] for r in validation) / len(validation)
        current = {**candidate, 'directory': str(target), 'overlay_directory': str(merged_dir),
            'validation': validation, 'validation_mean_minADE_m': mean, 'validation_pass': mean <= dense_mean * 1.15,
            'selected_interface_step': best_step, 'selected_interface_loss': best_loss,
            'parent_validation_mean_minADE_m': candidate['validation_mean_minADE_m']}
        current.pop('local_distillation_layer_reports', None)
        current['parent_local_distillation_reports'] = str(args.parent_experiment / 'result.json')
        current['elapsed_seconds'] = time.perf_counter() - start
        write_json(target / 'result.json', current)
        selection = {'status': 'selected_pending_holdout_and_reload' if current['validation_pass'] else 'no_validation_passing_candidate',
            'selection_uses_holdout': False, 'original_dense_params': parent['original_dense_params'],
            'previous_10B_stage_params': parent['previous_10B_stage_params'], 'dense_validation': teacher['validation'],
            'dense_validation_mean_minADE_m': dense_mean, 'validation_limit_minADE_m': dense_mean * 1.15,
            'selected': {k: current[k] for k in ('keep_width', 'params', 'vlm_params', 'layer_widths',
                'overlay_directory', 'validation_mean_minADE_m', 'validation_pass')} if current['validation_pass'] else None,
            'candidates': [current], 'selected_interface_step': best_step}
        write_json(out / 'selection.json', selection)
        write_json(out / 'result.json', selection)
        print('RECOVERY_FINISHED', json.dumps({'status': selection['status'],
            'params': base_params, 'validation_mean_minADE_m': mean,
            'best_interface_step': best_step, 'best_interface_loss': best_loss}), flush=True)


if __name__ == '__main__':
    main()
