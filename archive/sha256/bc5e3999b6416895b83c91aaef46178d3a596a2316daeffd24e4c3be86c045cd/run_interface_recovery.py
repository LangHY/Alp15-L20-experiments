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
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file
from run_mlp_experiment import (
    _enable_offline_mode, _block_network, _verify_model, _verify_dataset,
    _verify_config_dir, _verify_config_source, _load_local_model_and_processor,
    _load_offline_dataset_interface, COSMOS_REVISION, QWEN_REVISION, MODEL_VLM_NAME,
    DATASET_REVISION, protected_digest, file_digest, load_overlay, evaluate, write_json,
)
from student_interface_distill import install_adapters, merge_adapters, interface_targets, interface_loss, logits_kl, exported_weights, mlp_tensor_digest


def prepare(model, processor, interface, record, rollout_prefix=False):
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    data = load_physical_aiavdataset(record['clip_id'], t0_us=record.get('t0_us',5_100_000), avdi=interface, maybe_stream=False)
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


def forward_targets(model, example, probe_index=None):
    inputs = {k: v.cuda() for k, v in example['inputs'].items()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model.vlm.model(**inputs, use_cache=True, return_dict=True)
        targets = interface_targets(output, example['positions'].cuda())
        if example.get('expert_probes'):
            from expert_interface_distill import expert_velocity
            probes = example['expert_probes'] if probe_index is None else [example['expert_probes'][probe_index]]
            targets['expert_velocities'] = [expert_velocity(model, output, inputs, probe) for probe in probes]
        return targets


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
    parser.add_argument('--expert-weight', type=float, default=0.0, help='Frozen Expert velocity KD on fixed action probes; requires rollout-prefix')
    parser.add_argument('--evaluate-parent', action='store_true', help='Measure original two validation clips before recovery of a prepared overlay')
    parser.add_argument('--local-prefit-steps', type=int, default=0, help='Locally restore only narrowed MLPs before whole-student KD')
    parser.add_argument('--task-validation-dir', type=Path, help='Choose checkpoint by actual merged-BF16 trajectory ADE on the fixed 20 validation clips')
    parser.add_argument('--controlled-validation-dir', type=Path, help='Completed controlled20 diagnostic; independent paired seeds 42/43/44, never legacy references')
    parser.add_argument('--training-window-audit', type=Path, help='Completed offline audit of timestamps on original calibration clips only')
    parser.add_argument('--new-training-audit', type=Path, help='Completed input-only audit of new chunk0000 training/monitor clips')
    parser.add_argument('--shuffle-training', action='store_true', help='Deterministically reshuffle training clips at each epoch, seed=42+epoch')
    parser.add_argument('--teacher-interface-cache',type=Path,help='Reuse verified original-teacher full-prefix targets from a completed new-data recovery')
    parser.add_argument('--pruning-allocation',type=Path,help='Completed current-parent all-layer measured allocation, applied before whole-student KD')
    args = parser.parse_args()
    if args.task_validation_dir and args.controlled_validation_dir:
        parser.error('Choose one task-validation protocol')
    if args.training_window_audit and not args.rollout_prefix:
        parser.error('New timestamps require actual teacher-generated reasoning')
    if args.new_training_audit and (args.training_window_audit or not args.rollout_prefix or args.local_prefit_steps):
        parser.error('New-data recovery requires rollout-prefix, no local prefit and no old-window audit')
    if args.steps < 1 or args.eval_every < 1 or args.rank < 1 or args.lr <= 0 or args.logit_weight < 0 or args.expert_weight < 0:
        parser.error('Training settings must be positive')
    if args.expert_weight and not args.rollout_prefix:
        parser.error('Expert KD requires the actual verified trajectory-start prefix')
    if args.local_prefit_steps < 0:
        parser.error('local-prefit-steps cannot be negative')
    if args.teacher_interface_cache and not args.new_training_audit:
        parser.error('Teacher cache reuse requires the fixed new-data audit')
    if args.pruning_allocation and (not args.new_training_audit or not args.teacher_interface_cache or not args.controlled_validation_dir):
        parser.error('Measured pruning requires new-data audit, frozen teacher cache and controlled task validation')
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
    window_records = None
    new_data = None
    if args.new_training_audit:
        new_data = json.loads((args.new_training_audit/'result.json').read_text())
        if new_data['status']!='completed' or new_data['dataset_revision']!=DATASET_REVISION:
            raise RuntimeError('New data coverage audit incomplete or wrong revision')
        training_ids={r['clip_id'] for r in new_data['training_records']}
        monitor_ids={r['clip_id'] for r in new_data['monitor_records']}
        old_ids=set(sum((split[k] for k in ['calibration','validation','holdout']),[]))
        old_ids.add('030c760c-ae38-49aa-9ad8-f5650a545d26')
        if training_ids&monitor_ids or (training_ids|monitor_ids)&old_ids or len(training_ids)<40 or len(monitor_ids)<10:
            raise RuntimeError('New training/monitor identities invalid or overlap old splits')
        if args.controlled_validation_dir:
            extended=json.loads((args.controlled_validation_dir/'split.json').read_text())
            excluded=set(extended['expanded_validation']+extended['expanded_holdout'])
            if (training_ids|monitor_ids)&excluded:raise RuntimeError('New data overlaps old expanded evaluation clips')
        write_json(out/'new_data_audit.json',new_data)
        write_json(out/'optimization_split.json',{'training':sorted(training_ids),'monitor_validation':sorted(monitor_ids),
            'data_audit_sha256':file_digest(args.new_training_audit/'result.json'),'gradient_updates_on_monitor':False})
    if args.training_window_audit:
        audit = json.loads((args.training_window_audit/'result.json').read_text())
        if audit['status'] != 'completed' or audit['dataset_revision'] != DATASET_REVISION or audit['split_sha256'] != file_digest(args.parent_experiment/'split.json'):
            raise RuntimeError('Training timestamp audit incomplete or identity mismatch')
        window_records = audit['records']
        if {r['clip_id'] for r in window_records} != set(split['calibration']):
            raise RuntimeError('Temporal augmentation must only use frozen calibration clips')
        if len({(r['clip_id'],r['t0_us']) for r in window_records}) != len(window_records):
            raise RuntimeError('Duplicate training windows')
        write_json(out/'training_windows.json',audit)
    parent = json.loads((args.parent_experiment / 'result.json').read_text())
    candidate = parent['candidates'][0]
    candidate.setdefault('layer_widths', [candidate['keep_width']] * 36)
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
        training_interface = interface
        if new_data:
            training_dataset = Path(new_data['dataset_directory'])
            if file_digest(training_dataset/'delivery_receipt.json')!=new_data['receipt_sha256']:
                raise RuntimeError('New delivery receipt changed after coverage audit')
            receipt=json.loads((training_dataset/'delivery_receipt.json').read_text())
            for record in receipt['files']:
                path=training_dataset/'files'/record['path']
                if path.stat().st_size!=record['bytes'] or file_digest(path)!=record['sha256']:
                    raise RuntimeError('New dataset file differs from verified delivery')
            training_interface = _load_offline_dataset_interface(training_dataset)
        model, processor = _load_local_model_and_processor(root / 'models', cosmos, qwen, config)
        model.eval().requires_grad_(False)
        protected = protected_digest(model)
        if protected != manifest['protected_sha256']:
            raise RuntimeError('Protected checkpoint identity mismatch')
        provenance = {
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_commit': '36aeb4c5938cbc2eb2aed33b22434773da4ab639',
            'checkpoint_revision': manifest['base_checkpoint_revision'], 'dataset_revision': DATASET_REVISION,
            'parent_experiment': str(args.parent_experiment), 'parent_overlay_sha256': file_digest(overlay / 'manifest.json'),
            'protected_sha256_before': protected, 'seed': 42, 'rank': args.rank, 'lr': args.lr,
            'steps': args.steps, 'eval_every': args.eval_every, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'argv': sys.argv, 'python': sys.version,
            'rollout_prefix': args.rollout_prefix, 'logit_weight': args.logit_weight,
            'expert_weight': args.expert_weight,
            'local_prefit_steps': args.local_prefit_steps,
            'checkpoint_selection': ('Controlled 20 clips, independent paired trajectory seeds 42/43/44, merged BF16 ADE' if args.controlled_validation_dir else 'Actual merged BF16 mean ADE on fixed 20 expanded validation clips' if args.task_validation_dir else 'Hidden/KV/auxiliary loss on original two validation clips'),
            'expert_probe_policy': 'Four fixed noise/GT-action linear probes at t=0,.25,.5,.75; all four for validation, one rotating probe per training step',
            'gpu': torch.cuda.get_device_name(), 'code_sha256': {p.name: file_digest(p) for p in (out / 'code').glob('*.py')},
            'training_policy': ('New training clips only; new monitor split has no gradient updates; original controlled20 selects task checkpoint.' if new_data else 'Only four frozen calibration clips; original two monitor interface loss; controlled20 selects by trajectory ADE when enabled.'),
            'training_windows':len(new_data['training_records']) if new_data else len(window_records) if window_records else len(split['calibration']),
            'training_window_audit_sha256':file_digest(args.training_window_audit/'result.json') if args.training_window_audit else None,
            'new_data_audit_sha256':file_digest(args.new_training_audit/'result.json') if new_data else None,
            'training_order_policy':'Shuffle each epoch with seed=42+epoch' if args.shuffle_training else 'Fixed cyclic order',
            'effective_training_policy':('New chunk0000 training clips only; separate new monitor clips; old controlled20 selects task checkpoint; no old holdout' if new_data else 'Original four training clips, with temporal augmentation if configured'),
            'objective': 'Final hidden NMSE + mean(all-layer key NMSE, all-layer value NMSE) + logit_weight * next-token KL + expert_weight * frozen-Expert velocity NMSE. Full student graph, MLP LoRA only.',
            'teacher_forcing': ('Actual newly collected dense CoC tokens through trajectory-start boundary; no dense trajectory rollout' if args.rollout_prefix
                else 'Existing recorded CoC text appended to real fused-history multimodal prompt, without unrecorded suffix.'),
            'target_positions': '128 uniform prompt positions plus every recorded reasoning text token',
            'holdout_uses': 'Reporting only; never training, allocation, hyperparameter or checkpoint selection.',
            'limitations': [('New training/monitor clips from one chunk; no external generalization claim' if new_data else 'Four training clips from one chunk; no external generalization claim'),
                'Teacher-forced interface loss is not on-policy reasoning or differentiable trajectory loss',
                *([] if args.rollout_prefix else ['KV targets exclude the unrecorded trajectory-start suffix'])],
        }
        write_json(out / 'provenance.json', provenance)
        cached_examples=None
        if args.teacher_interface_cache:
            donor=args.teacher_interface_cache
            donor_provenance=json.loads((donor/'provenance.json').read_text())
            donor_split=json.loads((donor/'optimization_split.json').read_text())
            if set(donor_split['training'])!=training_ids or set(donor_split['monitor_validation'])!=monitor_ids:
                raise RuntimeError('Teacher-cache donor training/monitor roles differ')
            for key in ['source_commit','checkpoint_revision','dataset_revision','logit_weight','expert_weight']:
                if donor_provenance[key]!=provenance[key]:raise RuntimeError('Teacher-cache provenance differs: '+key)
            if not donor_provenance['rollout_prefix']:raise RuntimeError('Teacher-cache prefix not actual full rollout')
            cached_examples={}
            for role in ['calibration','validation']:
                role_ids=training_ids if role=='calibration' else monitor_ids
                for source in sorted((donor/'teacher_cache').glob(role+'_*.pt')):
                    e=torch.load(source,weights_only=True,map_location='cpu')
                    if e['clip_id'] not in role_ids or e.get('t0_us',5100000)!=5100000:raise RuntimeError('Teacher-cache identity mismatch')
                    if (role,e['clip_id']) in cached_examples:raise RuntimeError('Duplicate teacher cache')
                    cached_examples[(role,e['clip_id'])]=(source,e)
            if len(cached_examples)!=len(training_ids)+len(monitor_ids):raise RuntimeError('Teacher-cache donor incomplete')
            provenance['reused_teacher_cache']={'directory':str(donor),'source_sha256':{str(f):file_digest(f) for f,e in cached_examples.values()}}
            write_json(out/'provenance.json',provenance)
        examples = {'calibration': [], 'validation': []}
        for role in examples:
            if {r['clip_id'] for r in teacher[role]} != set(split[role]):
                raise RuntimeError('Teacher and split identities differ')
            records = window_records if role=='calibration' and window_records else teacher[role]
            if new_data:records=new_data['training_records'] if role=='calibration' else new_data['monitor_records']
            for index, record in enumerate(records):
                if cached_examples is not None:
                    source,example=cached_examples[(role,record['clip_id'])]
                    shutil.copy2(source,out/'teacher_cache'/f'{role}_{index}.pt')
                    examples[role].append(example)
                    print('REUSED_TEACHER_INTERFACE',role,index,record['clip_id'],flush=True)
                    continue
                inputs, positions, logit_mask = prepare(model, processor, training_interface, record, args.rollout_prefix)
                example = {'clip_id': record['clip_id'], 't0_us':record.get('t0_us',5_100_000), 'inputs': inputs, 'positions': positions, 'logit_mask': logit_mask}
                if args.expert_weight:
                    from expert_interface_distill import make_probes
                    example['expert_probes'] = make_probes(model, training_interface, record['clip_id'],t0_us=example['t0_us'])
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
        if args.pruning_allocation:
            from apply_measured_pruning import apply_plan
            plan=json.loads(args.pruning_allocation.read_text())
            profile=Path(plan['profile_directory'])
            if json.loads((profile/'importance.json').read_text())['status']!='completed':raise RuntimeError('All-layer importance profiling incomplete')
            if file_digest(profile/'provenance.json')!=plan['profile_provenance_sha256']:raise RuntimeError('Pruning profile provenance changed')
            if Path(plan['parent_experiment']).resolve()!=args.parent_experiment.resolve() or mlp_tensor_digest(model)!=plan['parent_MLP_SHA256']:
                raise RuntimeError('Measured pruning parent changed')
            manifest,candidate=apply_plan(model,manifest,candidate,plan)
            measurements=json.loads((profile/'measurement_progress.json').read_text())['curves']
            for i,record in enumerate(manifest['layers']):
                record['parent_calibration_token_count']=record.get('calibration_token_count')
                record['calibration_token_count']=measurements[i][0]['tokens']
            if protected_digest(model)!=protected:raise RuntimeError('Pruning changed protected weights')
            shutil.copy2(args.pruning_allocation,out/'as_applied_allocation.json')
            provenance['new_pruning_allocation_sha256']=file_digest(args.pruning_allocation)
            provenance['unpruned_parent_mean_ADE_m']=plan.get('parent_ADE_m',parent['candidates'][0]['validation_mean_minADE_m'])
            write_json(out/'provenance.json',provenance)
            print('MEASURED_PRUNING_APPLIED',candidate['params'],candidate['layer_widths'],flush=True)
        base_params = sum(p.numel() for p in model.parameters())
        if base_params != candidate['params'] or sum(p.numel() for p in model.vlm.parameters()) != candidate['vlm_params']:
            raise RuntimeError('Actual initialized student parameter count differs from recorded overlay')
        if args.evaluate_parent:
            parent_validation = [evaluate(model, processor, interface, clip) for clip in split['validation']]
            candidate['validation_mean_minADE_m'] = sum(r['minADE_m'] for r in parent_validation) / len(parent_validation)
            write_json(out / 'parent_initial_validation.json', parent_validation)
        if args.local_prefit_steps:
            from run_width_sweep import _cache_inputs
            from joint_mlp_distill import train_joint_mlp
            prefit = out / 'local_prefit'
            (prefit / 'training_steps').mkdir(parents=True)
            reports, artifacts, skipped = [], [], []
            for index, layer in enumerate(model.vlm.model.language_model.layers):
                if layer.mlp.intermediate_size == 5376:
                    skipped.append({'layer': index, 'reason': 'Width unchanged from accepted uniform 8B parent; skip local prefit'})
                    continue
                train, val = _cache_inputs(args.teacher_experiment, index)
                for prefix in ('train', 'val'):
                    path = args.teacher_experiment / 'teacher_cache' / f'{prefix}_{index:02d}.pt'
                    artifacts.append({'file': str(path), 'sha256': file_digest(path)})
                report = train_joint_mlp(layer.mlp, train['x'], train['y'], val['x'], val['y'],
                    steps=args.local_prefit_steps, batch_size=128, lr=3e-5, seed=42 + index, eval_every=20,
                    metrics_path=prefit / 'training_steps' / f'layer_{index:02d}.jsonl')
                reports.append({'layer': index, **report})
                write_json(prefit / 'distillation_progress.json', reports)
                print('LOCAL_PREFIT', index, report['final_val_nmse'], flush=True)
                del train, val
            write_json(prefit / 'teacher_artifacts.json', artifacts)
            write_json(prefit / 'skipped_unchanged_layers.json', skipped)
            if protected_digest(model) != protected:
                raise RuntimeError('Local prefit changed protected weights')
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        adapters, originals = install_adapters(model, args.rank)
        base_weights = [m.base.weight.detach().cpu().clone() for m in adapters] if (args.task_validation_dir or args.controlled_validation_dir) else None
        task_reference = None
        controlled_validator = None
        if args.controlled_validation_dir:
            from controlled_recovery_validation import ControlledValidation
            controlled_validator = ControlledValidation(args.controlled_validation_dir, out, provenance, split, parent_experiment=args.parent_experiment,check_parent_reproduction=not bool(args.pruning_allocation))
            task_reference = controlled_validator.reference
        if args.task_validation_dir:
            task_reference = json.loads((args.task_validation_dir / 'progress.json').read_text())
            identity = task_reference['identity']
            for key in ('source_commit', 'checkpoint_revision', 'dataset_revision'):
                if identity[key] != provenance[key]:
                    raise RuntimeError('Task validation reference identity mismatch')
            task_clips = task_reference['split']['expanded_validation']
            if len(task_clips) != 20 or len(set(task_clips)) != 20 or set(task_clips) & set(sum((split[k] for k in ('calibration','validation','holdout')),[])):
                raise RuntimeError('Task validation split overlaps original clips or has wrong size')
            for clip in task_clips:
                if task_reference['dense'][clip]['status'] != 'completed':
                    raise RuntimeError('Missing frozen dense task reference')
            write_json(out / 'task_validation_reference.json', {'clips': task_clips,
                'identity': {k: identity[k] for k in ('source_commit','checkpoint_revision','dataset_revision')},
                'dense': {c: task_reference['dense'][c] for c in task_clips},
                'selection_uses_holdout': False})
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0)
        print('ADAPTER_PARAMS', sum(p.numel() for p in trainable), flush=True)
        start = time.perf_counter()
        rows = []
        def objective(example, probe_index=None):
            output = forward_targets(model, example, probe_index)
            loss, pieces = interface_loss(output, example['targets'])
            if args.logit_weight:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = text_logits(model, output['hidden'][:, example['logit_mask'].cuda()])
                kl = logits_kl(logits, example['teacher_logits'])
                loss = loss + args.logit_weight * kl
                pieces['next_token_kl'] = kl
            if args.expert_weight:
                from student_interface_distill import relative_mse
                references = example['targets']['expert_velocities']
                if probe_index is not None:
                    references = [references[probe_index]]
                if torch.is_grad_enabled() and not all(v.requires_grad for v in output['expert_velocities']):
                    raise RuntimeError('Frozen Expert output detached from student KV gradient')
                expert_loss = torch.stack([relative_mse(s, t) for s, t in
                    zip(output['expert_velocities'], references, strict=True)]).mean()
                loss = loss + args.expert_weight * expert_loss
                pieces['expert_velocity_nmse'] = expert_loss
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
        def task_validate(step):
            if controlled_validator:
                return controlled_validator.evaluate(model, interface, adapters, base_weights, step)
            from evaluate_expanded import _run_phase
            directory = out / 'task_validation' / f'step_{step:06d}'
            directory.mkdir(parents=True)
            progress = {'split': {'expanded_validation': task_clips, 'expanded_holdout': []},
                'student': {}, 'dense': {c: task_reference['dense'][c] for c in task_clips}}
            with exported_weights(adapters, base_weights):
                signature = mlp_tensor_digest(model)
                _run_phase(model, processor, interface, task_clips, 'student', progress, directory)
            if any(r['status'] != 'completed' for r in progress['student'].values()) or len(progress['student']) != 20:
                raise RuntimeError('Task validation did not complete all 20 clips')
            records = [progress['student'][c]['result'] for c in task_clips]
            mean = sum(r['minADE_m'] for r in records) / 20
            missing = sum(r['trajectory_start_missing'] for r in records)
            result = {'step': step, 'mean_minADE_m': mean, 'missing_markers': missing,
                'mlp_tensor_sha256': signature, 'records': records, 'selection_uses_holdout': False,
                'evaluation_dtype': 'Actual merged BF16, temporary export state restored after validation'}
            write_json(directory / 'result.json', result)
            print('TASK_VALIDATION', step, mean, missing, flush=True)
            return (mean if missing == 0 else float('inf')), result
        best_loss, best_step = validate(0), 0
        best = snapshot()
        torch.save(best, out / 'best_adapters.pt')
        best_task, selected_task = task_validate(0) if task_reference else (None, None)
        for step in range(1, args.steps + 1):
            n_examples=len(examples['calibration'])
            if (step-1)%n_examples==0:
                training_order=list(range(n_examples))
                if args.shuffle_training:random.Random(42+(step-1)//n_examples).shuffle(training_order)
            example = examples['calibration'][training_order[(step - 1) % n_examples]]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            probe_index = ((step - 1) // len(examples['calibration'])) % 4 if args.expert_weight else None
            loss, pieces = objective(example, probe_index)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            lr = args.lr * (0.1 + 0.9 * (1 + math.cos(math.pi * (step - 1) / args.steps)) / 2)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            row = {'phase': 'training', 'step': step, 'clip_id': example['clip_id'], 'loss': loss.item(),
                't0_us':example.get('t0_us',5_100_000),'expert_probe_index':probe_index,
                **{k: v.item() for k, v in pieces.items()}, 'learning_rate': lr, 'grad_norm': norm.item(),
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'elapsed_seconds': time.perf_counter() - start}
            append(row)
            print('INTERFACE_STEP', json.dumps(row), flush=True)
            del loss, pieces
            if step % args.eval_every == 0 or step == args.steps:
                val_loss = validate(step)
                task_score, task_result = task_validate(step) if task_reference else (None, None)
                improved = task_score < best_task if task_reference else val_loss < best_loss
                if improved:
                    best_loss, best_step, best = val_loss, step, snapshot()
                    if task_reference:
                        best_task, selected_task = task_score, task_result
                    torch.save(best, out / 'best_adapters.pt')
        if task_reference and not math.isfinite(best_task):
            raise RuntimeError('No complete valid task checkpoint after recovery; no model selected')
        with torch.no_grad():
            for adapter, (a, b) in zip(adapters, best, strict=True):
                adapter.a.copy_(a); adapter.b.copy_(b)
        merge_adapters(model, originals)
        model.requires_grad_(False)
        if selected_task:
            if mlp_tensor_digest(model) != selected_task['mlp_tensor_sha256']:
                raise RuntimeError('Final merged weights differ from selected task-validation export')
            write_json(out / 'selected_task_validation.json', selected_task)
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
        if controlled_validator:
            validation = [{'clip_id':clip,'minADE_m':sum(r['ADE_m'] for r in selected_task['records'] if r['clip_id']==clip)/3,
                'metric_semantics':'Mean of three independent paired trajectory seeds'} for clip in controlled_validator.clips]
            teacher['validation'] = [{'clip_id':clip,'minADE_m':sum(r['ADE_m'] for r in controlled_validator.rows if r['clip_id']==clip)/3,
                'metric_semantics':'Mean of three independent paired trajectory seeds'} for clip in controlled_validator.clips]
            mean = selected_task['mean_ADE_m']
            dense_mean = selected_task['dense_mean_ADE_m']
        else:
            validation = [evaluate(model, processor, interface, clip) for clip in split['validation']]
            mean = sum(r['minADE_m'] for r in validation) / len(validation)
        current = {**candidate, 'status': 'completed', 'directory': str(target), 'overlay_directory': str(merged_dir),
            'validation': validation, 'validation_mean_minADE_m': mean, 'validation_pass': mean <= dense_mean * 1.15,
            'validation_dense_mean_minADE_m': dense_mean, 'validation_limit_minADE_m': dense_mean * 1.15,
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
        selection['evaluation_protocol'] = ('controlled20_mean_three_paired_seeds' if controlled_validator else 'legacy_original_two')
        write_json(out / 'selection.json', selection)
        write_json(out / 'result.json', selection)
        if controlled_validator:
            write_json(out / 'controlled_recovery_result.json', {
                'status':'completed','selected_step':best_step,'params':base_params,
                'parent_mean_ADE_m':json.loads((out/'task_validation/step_000000/result.json').read_text())['mean_ADE_m'],
                'selected_mean_ADE_m':mean,'dense_mean_ADE_m':dense_mean,
                'conditional_15pct_pass':mean<=dense_mean*1.15,
                'protected_sha256_before':protected,'protected_sha256_after':protected_digest(model),
                'protocol':'Controlled full-prefix KV; independent paired seeds 42/43/44; no holdout',
                'limitations':[('New training clips from one chunk' if new_data else 'Four training clips only'),'Validation used for checkpoint selection; not untouched test','Interface/velocity KD, not differentiable trajectory ADE training'],
                'overlay_directory':str(merged_dir)})
        print('RECOVERY_FINISHED', json.dumps({'status': selection['status'],
            'params': base_params, 'validation_mean_minADE_m': mean,
            'best_interface_step': best_step, 'best_interface_loss': best_loss}), flush=True)


if __name__ == '__main__':
    main()
