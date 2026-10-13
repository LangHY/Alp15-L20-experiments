"""训练侧真实驾驶 AWQ 的校准、导出与配对评估入口；不声称部署内核加速。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
import warnings
from pathlib import Path


def validate_calibration_manifest(manifest):
    """公开数据边界；整个 clip 隔离，禁止只按 token/时间点切开同一 clip。"""
    if manifest.get('schema') != 'alp15-awq-calibration-v1' or manifest.get('role') != 'training':
        raise ValueError('需要显式 training 角色与固定校准 manifest schema')
    examples, evaluation = manifest.get('examples'), manifest.get('evaluation_clips')
    if not isinstance(examples, list) or not examples or not isinstance(evaluation, list) or not evaluation:
        raise ValueError('校准样本和完整评测 clip 名单均不能为空')
    clips, paths = [], []
    for item in examples:
        if not isinstance(item, dict):
            raise ValueError('样本必须为 object')
        clip, path, digest = item.get('clip_id'), item.get('path'), item.get('sha256')
        if not isinstance(clip, str) or not clip or not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError('样本需要 clip_id 与绝对 path')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('样本需要真实 sha256')
        clips.append(clip)
        paths.append(path)
    if len(set(clips)) != len(clips) or len(set(paths)) != len(paths):
        raise ValueError('校准 clip/path 不得重复')
    if not all(isinstance(clip, str) and clip for clip in evaluation) or len(set(evaluation)) != len(evaluation):
        raise ValueError('评测 clip 名单非法或重复')
    if set(clips) & set(evaluation):
        raise ValueError('校准与评测存在 clip 泄漏')
    return {'calibration_clips': clips, 'evaluation_clips': evaluation,
            'clip_disjoint': True, 'independent_test_claim': False}


IDENTITIES = {'source_commit': '36aeb4c5938cbc2eb2aed33b22434773da4ab639',
              'checkpoint_revision': '7aba8293c09993f2e125c6819df05d7fa3e873ea',
              'dataset_revision': '33f9bf447ed3bcb7d545ce13f4226f824214fafb'}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_calibration_operation(recorder, name, operation):
    """在外层AWQ资源phase中计时，不嵌套phase/不重置外层CUDA峰值。"""
    started = time.monotonic()
    value = operation()
    timing = {'phase': name, 'elapsed_s': time.monotonic()-started,
              'resource_scope': 'current snapshots only; enclosing modelopt_awq owns peak counters'}
    if recorder.device is not None:
        import torch
        timing['cuda_allocated_after_bytes'] = torch.cuda.memory_allocated(recorder.device)
        timing['cuda_reserved_after_bytes'] = torch.cuda.memory_reserved(recorder.device)
    return value, timing


def verify_training_delivery(dataset, audit):
    """复用已完成覆盖审计，重新校验receipt及全部真实文件；禁止目录穿越。"""
    dataset = Path(dataset).resolve()
    expected_receipt = 'e502d4f307117d652cc748210431d1fda72931fb7c610d0cf5ea9ad8c6a6079c'
    if audit.get('status') != 'completed' or Path(audit['dataset_directory']).resolve() != dataset:
        raise ValueError('训练交付audit未完成或目录不符')
    if audit.get('receipt_sha256') != expected_receipt:
        raise ValueError('训练receipt来源与冻结覆盖audit不符')
    receipt_path = dataset/'delivery_receipt.json'
    if file_hash(receipt_path) != audit['receipt_sha256']:
        raise ValueError('训练receipt在覆盖审计后发生变化')
    receipt = json.loads(receipt_path.read_text())
    file_root = (dataset/'files').resolve()
    records = receipt.get('files')
    if not isinstance(records, list) or not records:
        raise ValueError('训练receipt没有实际文件清单')
    verified, total = [], 0
    for record in records:
        path = (file_root/record['path']).resolve()
        if not path.is_relative_to(file_root):
            raise ValueError('训练receipt路径越界')
        if path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
            raise ValueError('训练数据文件与已验证交付不符: '+record['path'])
        total += path.stat().st_size
        verified.append({'path': str(path), 'bytes': record['bytes'], 'sha256': record['sha256']})
        print('AWQ_TRAINING_FILE_VERIFIED', record['path'], record['bytes'], flush=True)
    return {'receipt_sha256': expected_receipt, 'files': verified,
            'verified_bytes': total, 'status': 'all_delivery_files_verified'}


def preflight(root, calibration_count):
    parent = root / 'experiments/mlp_newdata8_recovery_20261002_165238'
    reference = root / 'experiments/controlled8_expanded20_20261002_125000'
    split = json.loads((parent / 'optimization_split.json').read_text())
    provenance = json.loads((parent / 'provenance.json').read_text())
    dense = json.loads((reference / 'result.json').read_text())
    best = json.loads((parent / 'selected_task_validation.json').read_text())
    recovery = json.loads((parent / 'controlled_recovery_result.json').read_text())
    for key, expected in IDENTITIES.items():
        if provenance.get(key) != expected or dense.get(key) != expected:
            raise ValueError('冻结来源不符: ' + key)
    if provenance.get('rollout_prefix') is not True or dense.get('status') != 'completed':
        raise ValueError('训练缓存/评测参考未通过既有完成门禁')
    if recovery.get('status') != 'completed' or recovery.get('selected_step') != 400:
        raise ValueError('best8父模型未完成或不是step400')
    if best.get('step') != 400 or best.get('missing_markers') != 0 or best.get('trajectory_seeds') != [42, 43, 44]:
        raise ValueError('best8参考口径不符')
    refs = {'teacher': {(r['clip_id'], r['seed']): r for r in dense['records']
                        if r['model'] == 'dense' and r['prefix'] == 'teacher'},
            'best8': {(r['clip_id'], r['seed']): r for r in best['records']}}
    if set(refs['teacher']) != set(refs['best8']) or len(refs['teacher']) != 60:
        raise ValueError('需要冻结20clip×3seed配对参考')
    for identity in refs['teacher']:
        if refs['teacher'][identity]['initial_noise_sha256'] != refs['best8'][identity]['initial_noise_sha256']:
            raise ValueError('旧参考噪声不配对')
    evaluation = sorted({identity[0] for identity in refs['teacher']})
    training = split['training']
    if len(training) != 80 or len(set(training)) != 80 or set(training) & set(evaluation):
        raise ValueError('历史80训练clip名单不符或与开发评测交叠')
    examples = [{'clip_id': clip, 'path': str(parent / 'teacher_cache' / f'calibration_{index}.pt'),
                 'sha256': file_hash(parent / 'teacher_cache' / f'calibration_{index}.pt')}
                for index, clip in enumerate(training[:calibration_count])]
    manifest = {'schema': 'alp15-awq-calibration-v1', 'role': 'training',
                **IDENTITIES, 'examples': examples, 'evaluation_clips': evaluation,
                'prompt_source': 'Real training driving frames; cached teacher text removed by generate_prefix prompt truncation',
                'split_sha256': file_hash(parent / 'optimization_split.json'),
                'provenance_sha256': file_hash(parent / 'provenance.json')}
    validate_calibration_manifest(manifest)
    training_dataset = root/'datasets/physicalai_chunk0000_20261002_165238'
    audit = json.loads((parent/'new_data_audit.json').read_text())
    manifest['training_delivery_verification'] = verify_training_delivery(training_dataset, audit)
    return parent, reference, manifest, refs, recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('best8', 'teacher'), default='best8')
    parser.add_argument('--calibration-clips', type=int, choices=(2, 8, 80), default=2)
    parser.add_argument('--evaluation-clips', type=int, choices=(2, 20), default=2)
    parser.add_argument('--preflight', action='store_true', help='仅校验冻结来源、训练clip隔离与缓存hash，不加载模型/GPU')
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'argv': sys.argv, **IDENTITIES, 'variant': args.variant,
              'records': [], 'calibration_passes': [], 'packed_layers': [],
              'method': 'ModelOpt0.39 INT4_AWQ_CFG awq_lite alpha0.1; real training autonomous decode/full-prefill/diffusion',
              'limitations': ['Repeated-use20clip development set, not independent held-out test',
                              'Diagnostic dequantize F.linear, not optimized deployment INT4 kernel',
                              'Original BF16 loader still used; not packed-only low-peak deployment loader',
                              'No Orin/closed-loop simulation result, no speed ranking on shared GPU']}
    started = time.monotonic()

    def save():
        result['elapsed_seconds'] = time.monotonic() - started
        (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                   allow_nan=False, default=str) + '\n')

    try:
        parent, reference, manifest, refs, recovery = preflight(root, args.calibration_clips)
        (out / 'calibration_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        result['inputs_sha256'] = {str(parent/name): file_hash(parent/name) for name in
            ('optimization_split.json', 'provenance.json', 'selected_task_validation.json',
             'controlled_recovery_result.json', 'new_data_audit.json')}
        result['inputs_sha256'][str(reference/'result.json')] = file_hash(reference/'result.json')
        result['code_sha256'] = {str(path): file_hash(path) for path in
            (Path(__file__), Path(__file__).with_name('alp15_awq_full_packed.py'))}
        for path in (root/'scripts/alp15_packed_w4a16.py', root/'scripts/alp15_quant_route_pilot.py',
                     root/'scripts/test_alpamayo15_offline_compat.py',
                     root/'compression/controlled_8b_diagnostic.py', root/'compression/expert_interface_distill.py',
                     root/'compression/run_mlp_experiment.py', root/'compression/runtime_memory_record.py',
                     root/'compression/mlp_pruning.py'):
            result['code_sha256'][str(path)] = file_hash(path)
        result['calibration_split'] = validate_calibration_manifest(manifest)
        if args.preflight:
            result['status'] = 'passed_preflight_only'
            save()
            print('AWQ_PREFLIGHT_PASSED', str(out), flush=True)
            return 0
        sys.path.insert(0, str(root/'compression'))
        sys.path.insert(0, str(root/'scripts'))
        import copy
        import gc
        import importlib.metadata
        import subprocess
        import torch
        import modelopt.torch.quantization as mtq
        vendor_quantization = Path(mtq.__file__).resolve().parent
        result['modelopt_vendor_sha256'] = {str(path): file_hash(path) for path in
            (vendor_quantization/'config.py', vendor_quantization/'model_quant.py',
             vendor_quantization/'tensor_quant.py', vendor_quantization/'model_calib.py',
             vendor_quantization/'nn/modules/tensor_quantizer.py')}
        from alp15_awq_full_packed import PackedAWQLinear
        from controlled_8b_diagnostic import generate_prefix, sampled, digest
        from alp15_quant_route_pilot import summary
        from runtime_memory_record import MemoryRecorder, storage_inventory
        from run_mlp_experiment import (_enable_offline_mode, _block_network, _verify_model,
            _verify_dataset, _verify_config_dir, _verify_config_source,
            _load_local_model_and_processor, _load_offline_dataset_interface, load_overlay,
            MODEL_VLM_NAME, COSMOS_REVISION, QWEN_REVISION, DATASET_REVISION)
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        if importlib.metadata.version('nvidia-modelopt') != '0.39.0':
            raise RuntimeError('固定ModelOpt0.39.0不符')
        if importlib.metadata.version('physical_ai_av') != '0.2.0':
            raise RuntimeError('固定physical_ai_av0.2.0不符')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('需要BF16 CUDA服务器推理环境')
        _enable_offline_mode()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        head = subprocess.check_output(['git', '-C', str(root/'alpamayo1.5'), 'rev-parse', 'HEAD'], text=True).strip()
        if head != IDENTITIES['source_commit']:
            raise RuntimeError('官方源HEAD不符')
        result['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(), 'modelopt': '0.39.0', 'source_HEAD': head}
        recorder = MemoryRecorder(out/'memory_phases.jsonl', device=torch.device('cuda'))
        train_examples = {}
        for item in manifest['examples']:
            example = torch.load(item['path'], weights_only=True, map_location='cpu')
            if example.get('clip_id') != item['clip_id'] or example.get('t0_us', 5_100_000) != 5_100_000:
                raise RuntimeError('训练缓存clip/t0不符')
            train_examples[item['clip_id']] = {key: example[key] for key in ('clip_id', 'inputs', 'positions', 'logit_mask')}
            del example
        eval_clips = manifest['evaluation_clips'][:args.evaluation_clips]
        eval_examples = {}
        for path in sorted((reference/'teacher_prefix_cache').glob('*.pt')):
            example = torch.load(path, weights_only=True, map_location='cpu')
            if example['clip_id'] in eval_clips:
                eval_examples[example['clip_id']] = example
                result['inputs_sha256'][str(path)] = file_hash(path)
        if set(eval_examples) != set(eval_clips):
            raise RuntimeError('缺少评测prompt缓存')
        with _block_network(), torch.no_grad():
            config = _verify_model(root/'models')
            _verify_dataset(root/'alp15-test-clip-delivery-20260917')
            configs = root/'alp15-configs-delivery-20260919'
            cosmos = configs/'dependencies/configs/Cosmos-Reason2-8B'
            qwen = configs/'dependencies/configs/Qwen3-VL-2B-Instruct'
            for directory, label, repo, revision, source in (
                (cosmos, 'Cosmos', MODEL_VLM_NAME, COSMOS_REVISION, 'cosmos-reason2-source.json'),
                (qwen, 'Qwen', 'Qwen/Qwen3-VL-2B-Instruct', QWEN_REVISION, 'qwen3-vl-source.json')):
                _verify_config_dir(directory, label)
                _verify_config_source(directory, label, repo, revision, source, configs/'manifests')
            eval_interface = _load_offline_dataset_interface(root/'alp15-test-clip-delivery-20260917')
            training_dataset = root/'datasets/physicalai_chunk0000_20261002_165238'
            audit = json.loads((parent/'new_data_audit.json').read_text())
            if audit.get('status') != 'completed' or Path(audit['dataset_directory']).resolve() != training_dataset:
                raise RuntimeError('训练数据audit未完成或目录不符')
            # 原loader校验的是全局metadata映射；已由现场CPU证明可在chunk0000缓存读取训练clip。
            train_interface = _load_offline_dataset_interface(training_dataset)
            for clip in train_examples:
                if int(train_interface.get_clip_chunk(clip)) != 0:
                    raise RuntimeError('训练clip不是冻结chunk0000: ' + clip)
                data = load_physical_aiavdataset(clip, t0_us=5_100_000, avdi=train_interface, maybe_stream=False)
                if tuple(data['image_frames'].shape[:2]) != (4, 4) or tuple(data['ego_future_xyz'].shape) != (1, 1, 64, 3):
                    raise RuntimeError('训练4相机4帧/future schema不符')
                del data
            with recorder.phase('load_bf16'):
                model, processor = _load_local_model_and_processor(root/'models', cosmos, qwen, config)
                model.eval().requires_grad_(False)
                if args.variant == 'best8':
                    overlay = Path(recovery['overlay_directory'])
                    load_overlay(model, overlay)
                    result['inputs_sha256'][str(overlay/'manifest.json')] = file_hash(overlay/'manifest.json')
            result['dense_storage'] = storage_inventory(model)
            def rollout(example, interface, clip):
                generated, actual = generate_prefix(model, example)
                del actual
                inputs = {key: value.cuda() for key, value in generated['inputs'].items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model.vlm.model(**inputs, use_cache=True, return_dict=True)
                data = load_physical_aiavdataset(clip, t0_us=5_100_000, avdi=interface, maybe_stream=False)
                return generated, inputs, output, data
            def evaluate(label, selected):
                for clip in selected:
                    with recorder.phase(label+'/'+clip, model):
                        begin = time.monotonic()
                        generated, inputs, output, data = rollout(eval_examples[clip], eval_interface, clip)
                        for seed in (42, 43, 44):
                            measured = sampled(model, output, inputs, data, seed)
                            frozen = refs[args.variant][clip, seed]
                            if measured['initial_noise_sha256'] != frozen['initial_noise_sha256']:
                                raise RuntimeError('评测初始噪声不配对')
                            distances = torch.linalg.vector_norm(torch.tensor(measured['pred_xyz'])[:, :2] -
                                data['ego_future_xyz'][0, 0, :, :2].float().cpu(), dim=-1)
                            if not torch.isfinite(distances).all().item():
                                raise RuntimeError('轨迹非有限')
                            row = {'variant': label, 'clip_id': clip, 'seed': seed, **measured,
                                'FDE_m': distances[-1].item(), 'max_displacement_m': distances.max().item(),
                                'reference_ADE_m': frozen['ADE_m'],
                                'paired_ADE_difference_m': measured['ADE_m'] - frozen['ADE_m'],
                                'generated_tokens': generated['generated_tokens'],
                                'prefix_sha256': digest(inputs['input_ids']),
                                'diagnostic_elapsed_from_clip_start_s': time.monotonic() - begin}
                            if label.endswith('_bf16_reload') and abs(row['paired_ADE_difference_m']) > 1e-4:
                                raise RuntimeError('父模型复现差异超过1e-4m')
                            result['records'].append(row)
                            save()
                            print('AWQ_TASK', label, clip, seed, row['ADE_m'], row['FDE_m'], flush=True)
                        del generated, inputs, output, data
                        gc.collect()
                        torch.cuda.empty_cache()
            evaluate(args.variant+'_bf16_reload', eval_clips[:2])
            targets = {}
            for index, layer in enumerate(model.vlm.model.language_model.layers):
                for section, projections in (('mlp', ('gate_proj', 'up_proj', 'down_proj')),
                    ('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj'))):
                    for projection in projections:
                        name = f'language_model.layers.{index}.{section}.{projection}'
                        targets[name] = getattr(getattr(layer, section), projection)
            if len(targets) != 252 or any(not isinstance(value, torch.nn.Linear) for value in targets.values()):
                raise RuntimeError('预期252个语言Linear，不量化视觉/Expert/投影头')
            protected_keys = [key for key in model.state_dict() if not any(
                key.startswith('vlm.model.'+name+'.') for name in targets)]
            def guarded_digest():
                state = model.state_dict()
                hasher = hashlib.sha256()
                for key in protected_keys:
                    value = state[key]
                    hasher.update(key.encode())
                    hasher.update(str(tuple(value.shape)).encode())
                    hasher.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                return hasher.hexdigest()
            original_guard = guarded_digest()
            quant_config = copy.deepcopy(mtq.INT4_AWQ_CFG)
            if quant_config['algorithm'] != {'method': 'awq_lite', 'alpha_step': .1}:
                raise RuntimeError('固定官方AWQ算法不符')
            quant_config['quant_cfg'] = {'*': {'enable': False}, **{
                name+'.weight_quantizer': {'num_bits': 4, 'block_sizes': {-1: 128, 'type': 'static'}, 'enable': True}
                for name in targets}}
            result['actual_quant_config'] = quant_config
            callback_count = 0
            def calibration_loop(calibration_model):
                nonlocal callback_count
                if calibration_model is not model.vlm.model:
                    raise RuntimeError('ModelOpt没有原位校准VLM模型')
                callback_count += 1
                for clip, example in train_examples.items():
                    def calibrate_one():
                        generated, inputs, output, data = rollout(example, train_interface, clip)
                        sampled_record = sampled(model, output, inputs, data, 42)
                        row = {'pass': callback_count, 'clip_id': clip,
                            'generated_tokens': generated['generated_tokens'],
                            'prefix_sha256': digest(inputs['input_ids']),
                            'initial_noise_sha256': sampled_record['initial_noise_sha256'],
                            'sampled_ADE_m': sampled_record['ADE_m'], 'role': 'calibration_only_not_eval'}
                        del generated, inputs, output, data
                        gc.collect()
                        torch.cuda.empty_cache()
                        return row
                    row, timing = run_calibration_operation(recorder,
                        f'calibration/pass{callback_count}/{clip}', calibrate_one)
                    result['calibration_passes'].append({**row, 'diagnostic_timing': timing})
                    save()
                    print('AWQ_CALIBRATION_PASS', callback_count, clip, flush=True)
            with warnings.catch_warnings(record=True) as observed, recorder.phase('modelopt_awq', model):
                warnings.simplefilter('always')
                model.vlm.model = mtq.quantize(model.vlm.model, quant_config, forward_loop=calibration_loop)
            result['warnings'] = [str(item.message) for item in observed]
            if any('awq' in item.lower() and any(word in item.lower() for word in
                ('fallback', 'fall back', 'skipping', 'cannot perform', 'disabling', 'max calibration'))
                for item in result['warnings']):
                raise RuntimeError('AWQ回退/跳过警告，拒绝冒称成功')
            if callback_count < 2 or guarded_digest() != original_guard:
                raise RuntimeError('AWQ未执行两遍校准或改变保护模块')
            result['protected_sha256'] = original_guard
            probes = {}
            handles = []
            for name, linear in targets.items():
                def capture(module, inputs, name=name):
                    if name not in probes:
                        probes[name] = inputs[0].detach().reshape(-1, module.in_features)[:8].clone()
                handles.append(linear.register_forward_pre_hook(capture))
            try:
                first_clip = manifest['examples'][0]['clip_id']
                generated, inputs, output, data = rollout(train_examples[first_clip], train_interface, first_clip)
                del generated, inputs, output, data
            finally:
                for handle in handles:
                    handle.remove()
            if set(probes) != set(targets):
                raise RuntimeError('没有覆盖全部252量化层的真实驾驶输入')
            packed_directory = out/'packed'
            packed_directory.mkdir()
            with recorder.phase('pack_and_reload', model):
                for name, linear in targets.items():
                    iq, wq = linear.input_quantizer, linear.weight_quantizer
                    if iq.is_enabled or not wq.is_enabled or wq.num_bits != 4:
                        raise RuntimeError('激活/weight实际量化状态不符: '+name)
                    if wq.block_sizes.get(-1) != 128 or wq.block_sizes.get('type') != 'static':
                        raise RuntimeError('group128/static状态不符: '+name)
                    pre_scale, amax = iq.pre_quant_scale, wq.amax
                    if pre_scale is None or amax is None:
                        raise RuntimeError('缺失AWQ缩放或amax: '+name)
                    fake_weight = wq(linear.weight)
                    packed = PackedAWQLinear.from_quantized(fake_weight, amax, pre_scale, linear.bias)
                    path = packed_directory/(name.replace('.', '_')+'.safetensors')
                    packed.save(path)
                    reloaded = PackedAWQLinear.load(path, device=linear.weight.device)
                    if any(not torch.equal(value, reloaded.state_dict()[key]) for key, value in packed.state_dict().items()):
                        raise RuntimeError('packed重载不是逐字节一致')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        expected, actual = linear(probes[name]), reloaded(probes[name])
                    torch.testing.assert_close(actual, expected, rtol=.01, atol=.02)
                    result['packed_layers'].append({'name': name, 'shape': list(fake_weight.shape),
                        'file': str(path), 'sha256': file_hash(path), 'persistent_bytes': reloaded.storage_bytes(),
                        'bridge_max_abs_error': (actual.float()-expected.float()).abs().max().item(),
                        'amax_dtype': str(amax.dtype), 'pre_scale_dtype': str(pre_scale.dtype),
                        'pre_scale_nontrivial': not torch.equal(pre_scale, torch.ones_like(pre_scale))})
                    owner_name, attribute = name.rsplit('.', 1)
                    setattr(model.vlm.model.get_submodule(owner_name), attribute, reloaded)
                    targets[name] = None
                    del linear, fake_weight, packed, reloaded, expected, actual
                    torch.cuda.empty_cache()
            del targets, probes
            if guarded_digest() != original_guard:
                raise RuntimeError('导出改变保护模块')
            result['packed_storage'] = storage_inventory(model)
            (packed_directory/'manifest.json').write_text(json.dumps(result['packed_layers'], indent=2)+'\n')
            evaluate(args.variant+'_w4a16_awq', eval_clips)
            result['summary'] = summary([row for row in result['records'] if row['variant'].endswith('_w4a16_awq')])
            result['status'] = 'completed_awq_development_pilot'
            save()
            print('AWQ_FULL_PILOT_COMPLETED', str(out), flush=True)
            return 0
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        save()
        print('AWQ_FULL_PILOT_FAILED', type(error).__name__, str(error), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
