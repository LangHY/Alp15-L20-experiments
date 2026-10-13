"""隔离完整请求入口：冻结 fullprefill10，先同代码严格配对，再扩20clip。"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from alp15_awq_hybrid import (AWQ_RESULT_SHA256, DECISION_ID, TARGET_NAMES,
    build_packed_manifest, dispatch_inventory, frozen_source_receipt,
    install_hybrid_awq, load_bound_capture, sha256_file, validate_packed_manifest,
    validate_trajectory_row)

KV_SOURCE_SHA256 = '74886a867a8aae480dd848a6d068d2e3f817674d891774bcdabeb57f412292f8'
TEACHER_MEAN_ADE = 1.2323994789272548
SEEDS = (42, 43, 44)


def strict_pair_gate(reference, candidate):
    from alp15_kv_steps_probe import numerical_gate
    validate_trajectory_row(reference)
    validate_trajectory_row(candidate)
    gate = numerical_gate(reference, candidate)
    for key in ('prefix_sha256', 'generated_token_ids', 'future_start_token_id', 'prefix_length'):
        gate[key + '_paired'] = reference[key] == candidate[key]
    gate['passed'] = gate['passed'] and all(gate[k + '_paired'] for k in (
        'prefix_sha256', 'generated_token_ids', 'future_start_token_id', 'prefix_length'))
    gate['request_identity_paired'] = all(reference[key] == candidate[key] for key in ('clip_id', 'seed', 'decision_id', 'steps'))
    gate['passed'] = gate['passed'] and gate['request_identity_paired']
    gate.update(clip_id=candidate['clip_id'], seed=candidate['seed'], decision_id=DECISION_ID)
    return gate


def assert_same_provenance(reference, candidate):
    if reference.get('decision_id') != DECISION_ID or candidate.get('decision_id') != DECISION_ID:
        raise RuntimeError('reference/hybrid决策身份不符')
    for key in ('code_sha256', 'runtime_source_sha256', 'input_sha256', 'steps',
                'request_route', 'text_seed', 'trajectory_seeds'):
        if reference[key] != candidate[key]:
            raise RuntimeError('reference/hybrid 必须同代码/来源/资产/请求: ' + key)


def validate_smoke(smoke, current, all_clips):
    assert_same_provenance(smoke, current)
    expected = {(clip, seed) for clip in all_clips[:2] for seed in SEEDS}
    def complete_keys(rows):
        return len(rows) == 6 and {(r['clip_id'], r['seed']) for r in rows} == expected
    if (smoke.get('status') != 'completed_awq_hybrid_development_eval'
            or smoke.get('awq_forward') != 'hybrid_m1' or smoke.get('clips') != all_clips[:2]
            or not complete_keys(smoke.get('strict_pair_gates', []))
            or not all(g['passed'] for g in smoke['strict_pair_gates'])
            or not complete_keys(smoke.get('original_helper_gates', []))
            or not complete_keys(smoke.get('records', []))
            or any(r['status'] != 'completed' for r in smoke['records'])
            or not all(g['passed'] for g in smoke['original_helper_gates'])):
        raise RuntimeError('扩20需要新的同代码2clip×3seed严格 hybrid smoke；旧actualKV豁免无效')
    binding = smoke['paired_reference']
    path = Path(binding['result']).resolve(strict=True)
    if sha256_file(path) != binding['sha256']:
        raise RuntimeError('smoke paired reference结果变化')
    reference = json.loads(path.read_text())
    assert_same_provenance(reference, smoke)
    if (reference.get('status') != 'completed_awq_hybrid_development_eval' or reference.get('exit_code') != 0
            or reference.get('awq_forward') != 'reference' or not complete_keys(reference.get('records', []))):
        raise RuntimeError('smoke reference未完整成功')
    baseline = {(r['clip_id'], r['seed']): r for r in reference['records']}
    for row in smoke['records']:
        if not strict_pair_gate(baseline[row['clip_id'], row['seed']], row)['passed']:
            raise RuntimeError('smoke读取实际轨迹严格复核失败')
    probe_path = Path(smoke['matrix_probe_evidence']['result']).resolve(strict=True)
    if sha256_file(probe_path) != smoke['matrix_probe_evidence']['sha256']:
        raise RuntimeError('smoke矩阵探针终态变化')
    validate_probe_binding(json.loads(probe_path.read_text()), reference, smoke, path)


def validate_probe_binding(probe, paired, current, paired_result_path):
    if (probe.get('status') != 'passed_all252_cuda_m1_probe_with_actual'
            or probe.get('coverage') != '252/252' or probe.get('actual_activation_coverage') != '252/252'
            or probe.get('decision_id') != DECISION_ID or len(probe.get('matrices', [])) != 252
            or {r['name'] for r in probe['matrices']} != TARGET_NAMES
            or any(r['status'] != 'passed' or not any(a['kind'] == 'actual_decode_bf16' for a in r['activations'])
                   for r in probe['matrices'])):
        raise RuntimeError('正式候选需要全部252真实activation探针，random-only仅诊断')
    binding = probe['activation_binding']
    actual, hashes, manifest, captured_reference = load_bound_capture(binding['manifest'],
        paired_result_path if len(current['clips']) == 2 else None)
    if (binding.get('manifest_sha256') != hashes[binding['manifest']]
            or binding.get('reference_result') != manifest['reference_result']
            or binding.get('reference_result_sha256') != hashes[manifest['reference_result']]
            or binding.get('final_receipt_sha256') != hashes[manifest['final_receipt']]):
        raise RuntimeError('probe并非paired reference交付的capture')
    assert_same_provenance(captured_reference, paired)
    assert_same_provenance(captured_reference, current)
    for key in ('protected_sha256_before', 'protected_sha256_after'):
        if captured_reference[key] != paired[key]:
            raise RuntimeError('capture/probe/paired reference保护身份不符')
    for key in ('code_sha256', 'runtime_source_sha256', 'input_sha256', 'actual_import_sha256'):
        if probe['activation_provenance'].get(key) != captured_reference[key]:
            raise RuntimeError('probe capture来源身份漂移: ' + key)
    for path, digest in probe['source_sha256'].items():
        if current['code_sha256'].get(path) != digest:
            raise RuntimeError('probe源码与当前候选变化')
    for path, digest in hashes.items():
        if probe['input_sha256'].get(path) != digest:
            raise RuntimeError('probe捕获文件/终态绑定SHA不符')
    return {'manifest': binding['manifest'], 'manifest_sha256': binding['manifest_sha256'],
            'reference_result': manifest['reference_result'], 'coverage': '252/252'}


def imported_source_receipt():
    """记录真实 import 文件；公共来源门禁另冻结，候选独有 Triton 路径也留痕。"""
    receipt = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file():
            receipt[str(Path(path).resolve())] = sha256_file(path)
    return receipt


def capture_decode_activations(torch, model, processor, data, out, helpers, provenance):
    """单独未计时的真实生成；保存每个投影第一个M1，不改变正式请求。"""
    folder = out / 'decode_activations'
    folder.mkdir(exist_ok=False)
    rows, handles, captured = [], [], set()
    def hook(name):
        def save_activation(module, args):
            value = args[0]
            if name in captured or value.numel() != module.in_features:
                return
            if value.dtype != torch.bfloat16 or not value.is_contiguous() or value.requires_grad:
                raise RuntimeError('实际M1 activation不满足原 kernel 约束')
            path = folder / (name.replace('.', '_') + '.pt')
            torch.save(value.detach().cpu(), path)
            rows.append({'name': name, 'file': path.name, 'sha256': sha256_file(path),
                         'shape': list(value.shape), 'dtype': str(value.dtype)})
            captured.add(name)
        return save_activation
    try:
        for name in sorted(TARGET_NAMES):
            handles.append(model.get_submodule(name).register_forward_pre_hook(hook(name)))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prompt = helpers.raw_prompt(model, processor, data)
            inputs, actual = helpers.generate_actual_cache(torch, model, prompt)
        del inputs, actual, prompt
    finally:
        for handle in handles: handle.remove()
    if captured != TARGET_NAMES:
        raise RuntimeError('真实生成没有覆盖252个M1 activation；不补随机数据伪装真实')
    # 只留下tensor及内存描述；最终来源/保护复核成功前不生成可交付manifest。
    return {'activations': rows, 'folder': str(folder), 'clip_id': provenance['clips'][-1]}


def write_final_capture_manifest(capture, provenance, out):
    path = Path(capture['folder']) / 'manifest.json'
    manifest = {'schema': 'alp15-m1-activations-v2', 'activations': capture['activations'],
        'scope': 'unmeasured separate generation, first M1 per projection', 'decision_id': DECISION_ID,
        'awq_result_sha256': AWQ_RESULT_SHA256, 'protected_sha256': provenance['protected_sha256_before'],
        'clip_id': capture['clip_id'], 'text_seed': 42, 'reference_result': str(out / 'result.json'),
        'final_receipt': str(out / 'reference_final_receipt.json')}
    for key in ('code_sha256', 'runtime_source_sha256', 'input_sha256', 'actual_import_sha256'):
        manifest[key] = provenance[key]
    for row in capture['activations']:
        if sha256_file(path.parent / row['file']) != row['sha256']:
            raise RuntimeError('capture tensor在最终审核前变化')
    with path.open('x') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return {'manifest': str(path), 'sha256': sha256_file(path), 'coverage': '252/252',
            'final_receipt': manifest['final_receipt'], 'clip_id': capture['clip_id']}


def write_final_reference_receipt(result, out):
    """result保存完再写receipt；receipt失败则调用者将result改failed，旧receipt失效。"""
    capture = result['decode_activation_capture']
    result_path, manifest_path = out / 'result.json', Path(capture['manifest'])
    receipt = {'schema': 'alp15-reference-capture-final-v1', 'decision_id': DECISION_ID,
        'reference_result': str(result_path), 'reference_result_sha256': sha256_file(result_path),
        'activation_manifest': str(manifest_path), 'activation_manifest_sha256': sha256_file(manifest_path)}
    try:
        with Path(capture['final_receipt']).open('x') as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        load_bound_capture(manifest_path, result_path)
    except Exception as error:
        # 即使单独调用终态绑定器，失败也不能留下completed结果作为背书。
        result.update(status='failed', exit_code=1, finalization_error=str(error))
        temporary = out / 'result.tmp'
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(result_path)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--awq-artifact', type=Path)
    parser.add_argument('--awq-forward', choices=('reference', 'hybrid_m1'), default='reference')
    parser.add_argument('--candidate-decision', choices=(DECISION_ID,), required=True)
    parser.add_argument('--clips', type=int, choices=(2, 20), default=2)
    parser.add_argument('--steps', type=int, choices=(10,), default=10,
                        help='本轮只冻结10步；胜出少步/actualKV由后续独立验收纳入')
    parser.add_argument('--paired-reference', type=Path)
    parser.add_argument('--matrix-probe', type=Path)
    parser.add_argument('--approved-smoke', type=Path)
    parser.add_argument('--capture-decode-activations', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    if args.clips == 20 and not args.approved_smoke:
        parser.error('20clip需要本轮--approved-smoke')
    if args.awq_forward == 'hybrid_m1' and (not args.paired_reference or not args.matrix_probe):
        parser.error('hybrid_m1需要--paired-reference及--matrix-probe（252/252通过）')
    if args.capture_decode_activations and (args.awq_forward != 'reference' or args.clips != 2):
        parser.error('activation capture只在2clip reference额外未计时请求执行')
    root, out = args.root.resolve(), args.output.resolve()
    artifact = (args.awq_artifact or root / 'experiments/awq_full_20261008_best8_full20_v4').resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'alp15-awq-hybrid-eval-v1', 'status': 'running', 'argv': sys.argv,
        'decision_id': DECISION_ID, 'awq_forward': args.awq_forward, 'request_route': 'fullprefill',
        'provenance_version': 2,
        'steps': args.steps, 'text_seed': 42, 'trajectory_seeds': list(SEEDS), 'records': [],
        'strict_pair_gates': [], 'original_helper_gates': [], 'code_sha256': {},
        'runtime_source_sha256': {}, 'input_sha256': {}, 'gate_provenance': {},
        'variant_order': 'separate processes: all reference then all hybrid; same clip and seed order',
        'limitations': ['L20 operator/request diagnostics; not Orin deployment acceptance',
            'BF16-first cold loader retained, not the 16GB deployment loader',
            'Repeated-use development clips; not independent test quality',
            'actualKV and non10 steps disabled by this first hybrid decision']}
    started = time.perf_counter()
    def save():
        result['elapsed_seconds'] = time.perf_counter() - started
        temporary = out / 'result.tmp'
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(out / 'result.json')
    def freeze(path, destination):
        path = Path(path).resolve(strict=True)
        destination[str(path)] = sha256_file(path)
        return destination[str(path)]
    try:
        sys.path.insert(0, str(root / 'compression'))
        sys.path.insert(0, str(root / 'scripts'))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        helpers = importlib.import_module('alp15_kv_steps_probe')
        if sha256_file(helpers.__file__) != KV_SOURCE_SHA256:
            raise RuntimeError('复用的KV helper来源 SHA256 不符')
        result['code_sha256'].update(frozen_source_receipt())
        for path in (Path(__file__), Path(helpers.__file__), Path(__file__).parent / 'alp15_awq_hybrid.py',
                     Path(__file__).parent / 'probe_alp15_awq_hybrid.py'):
            freeze(path, result['code_sha256'])
        manifest = build_packed_manifest(artifact)
        packed = validate_packed_manifest(manifest)
        freeze(manifest['original_result'], result['input_sha256'])
        awq = json.loads(Path(manifest['original_result']).read_text())
        for row in packed: freeze(row['path'], result['input_sha256'])
        parent = root / 'experiments/mlp_newdata8_recovery_20261002_165238'
        reference = root / 'experiments/controlled8_expanded20_20261002_125000'
        dense_path, recovery_path = reference / 'result.json', parent / 'controlled_recovery_result.json'
        dense, recovery = json.loads(dense_path.read_text()), json.loads(recovery_path.read_text())
        if dense['status'] != 'completed' or recovery['status'] != 'completed' or recovery['selected_step'] != 400:
            raise RuntimeError('教师/父8B完成门禁不符')
        for path in (dense_path, recovery_path): freeze(path, result['input_sha256'])
        teacher_rows = [r for r in dense['records'] if r['model'] == 'dense' and r['prefix'] == 'teacher']
        awq_rows = [r for r in awq['records'] if r['variant'] == 'best8_w4a16_awq']
        all_clips, refs = helpers.validate_reference_pairs(teacher_rows, awq_rows)
        if abs(statistics.mean(r['ADE_m'] for r in teacher_rows) - TEACHER_MEAN_ADE) > 1e-12:
            raise RuntimeError('教师reference口径改变')
        clips = all_clips[:args.clips]
        result.update(clips=clips, all_frozen_clips=all_clips, independent_clips=len(clips))
        overlay = Path(recovery['overlay_directory']).resolve()
        freeze(overlay / 'manifest.json', result['input_sha256'])
        for path in sorted(overlay.glob('*.safetensors')): freeze(path, result['input_sha256'])
        for path in sorted((reference / 'teacher_prefix_cache').glob('*.pt')): freeze(path, result['input_sha256'])
        modules = [importlib.import_module(name) for name in ('controlled_8b_diagnostic',
            'expert_interface_distill', 'run_mlp_experiment', 'runtime_memory_record', 'mlp_pruning',
            'alp15_awq_full_packed', 'alp15_packed_w4a16', 'test_alpamayo15_offline_compat')]
        frozen_by_name = {Path(path).name: value for path, value in awq['code_sha256'].items()}
        for module in modules:
            if freeze(module.__file__, result['code_sha256']) != frozen_by_name.get(Path(module.__file__).name):
                raise RuntimeError('helper源码与AWQ冻结代码不等: ' + module.__name__)
        from alpamayo1_5.diffusion.flow_matching import FlowMatching
        from alpamayo1_5 import helper
        fixed_sources = {'base_model.py': '1f2e49a2ac62346cc61c3ea954982729da1a92674bc4ceed7bb43c87ba212cc2',
            'delta_tokenizer.py': '5c6c90f4c615bdd1d337810ccbd862ae0eb92b4e30331c665b229b654c19678e',
            'flow_matching.py': '57775891e6e0c047c3f203b7b141ffd6034b13969076926200f76404e48058d7',
            'alpamayo1_5.py': '87b23954570f4d466a6d0dd64090f779af635433b61106eb3b028d8d802ff1e8'}
        source_root = Path(inspect.getfile(helper)).resolve().parent
        for path in sorted(source_root.rglob('*.py')):
            digest = freeze(path, result['runtime_source_sha256'])
            if path.name in fixed_sources and digest != fixed_sources[path.name]:
                raise RuntimeError('官方现场源发生变化: ' + path.name)
        if not all(any(Path(path).name == name for path in result['runtime_source_sha256']) for name in fixed_sources):
            raise RuntimeError('官方源码覆盖不足')
        if 'inference_step' not in inspect.signature(FlowMatching.sample).parameters:
            raise RuntimeError('官方sample inference_step接口不符')
        for name in ('transformers.cache_utils', 'transformers.generation.utils', 'transformers.models.qwen3_vl.modeling_qwen3_vl'):
            freeze(importlib.import_module(name).__file__, result['runtime_source_sha256'])
        result['source'] = {'HEAD': subprocess.check_output(['git', '-C', str(root / 'alpamayo1.5'), 'rev-parse', 'HEAD'], text=True).strip(),
            'dirty_status': subprocess.check_output(['git', '-C', str(root / 'alpamayo1.5'), 'status', '--short'], text=True)}
        if result['source']['HEAD'] != awq['source_commit']: raise RuntimeError('source HEAD变化')
        for key in ('source_commit', 'checkpoint_revision', 'dataset_revision'):
            if awq[key] != dense[key]: raise RuntimeError('来源不配对: ' + key)
            result[key] = awq[key]
        result['actual_import_sha256_start'] = imported_source_receipt()
        paired, paired_result_path, pending_capture = None, None, None
        if args.paired_reference:
            path = args.paired_reference.resolve() / 'result.json'
            paired = json.loads(path.read_text())
            paired_result_path = path
            assert_same_provenance(paired, result)
            if (paired['status'] != 'completed_awq_hybrid_development_eval' or paired.get('exit_code') != 0
                    or paired['awq_forward'] != 'reference'
                    or paired['clips'] != clips or len(paired['records']) != len(clips) * 3):
                raise RuntimeError('配对reference必须同clip/seed完整完成')
            if ({(r['clip_id'], r['seed']) for r in paired['records']} != {(c, s) for c in clips for s in SEEDS}
                    or any(r['status'] != 'completed' for r in paired['records'])):
                raise RuntimeError('reference不得重复/丢失clip/seed或含失败')
            for row in paired['records']: validate_trajectory_row(row)
            for path_value, digest in paired['actual_import_sha256'].items():
                if sha256_file(path_value) != digest:
                    raise RuntimeError('reference实际import源码已变化: ' + path_value)
            result['gate_provenance'][str(path)] = sha256_file(path)
            result['paired_reference'] = {'result': str(path), 'sha256': sha256_file(path)}
        if args.matrix_probe:
            path = args.matrix_probe.resolve() / 'result.json'
            probe = json.loads(path.read_text())
            if paired is None: raise RuntimeError('正式probe门禁需要paired reference')
            result['verified_activation_binding'] = validate_probe_binding(probe, paired, result, paired_result_path)
            for row in packed:
                if probe['input_sha256'].get(str(row['path'])) != row['sha256']:
                    raise RuntimeError('矩阵探针资产变化')
            for path_value, digest in probe['source_sha256'].items():
                if result['code_sha256'].get(path_value) != digest: raise RuntimeError('矩阵探针代码变化')
            result['gate_provenance'][str(path)] = sha256_file(path)
            result['matrix_probe_evidence'] = {'result': str(path), 'sha256': sha256_file(path)}
        if args.approved_smoke:
            path = args.approved_smoke.resolve() / 'result.json'
            validate_smoke(json.loads(path.read_text()), result, all_clips)
            result['gate_provenance'][str(path)] = sha256_file(path)
        if args.preflight:
            result.update(status='passed_preflight_only', exit_code=0)
            result['actual_import_sha256'] = imported_source_receipt()
            save()
            return 0
        import gc
        import torch
        from controlled_8b_diagnostic import digest, sampled
        from alp15_awq_full_packed import PackedAWQLinear
        from runtime_memory_record import MemoryRecorder
        from run_mlp_experiment import (_enable_offline_mode, _block_network, _verify_model, _verify_dataset,
            _verify_config_dir, _verify_config_source, _load_local_model_and_processor,
            _load_offline_dataset_interface, load_overlay, MODEL_VLM_NAME, COSMOS_REVISION, QWEN_REVISION)
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        from probe_alp15_awq_hybrid import gpu_processes
        _enable_offline_mode()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('需要实际CUDA BF16环境')
        try:
            triton_version = importlib.metadata.version('triton')
        except importlib.metadata.PackageNotFoundError:
            triton_version = 'not installed; reference path does not require Triton'
        result['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(), 'gpu_processes_before': gpu_processes(),
            'python': sys.version, 'triton': triton_version}
        recorder = MemoryRecorder(out / 'memory_phases.jsonl', device=torch.device('cuda'))
        examples = {}
        for path in sorted((reference / 'teacher_prefix_cache').glob('*.pt')):
            example = torch.load(path, weights_only=True, map_location='cpu')
            if example['clip_id'] in clips: examples[example['clip_id']] = example
        if set(examples) != set(clips): raise RuntimeError('冻结prompt cache缺失')
        with _block_network(), torch.no_grad():
            with recorder.phase('asset_verification'):
                config = _verify_model(root / 'models')
                dataset = root / 'alp15-test-clip-delivery-20260917'
                _verify_dataset(dataset)
                configs = root / 'alp15-configs-delivery-20260919'
                cosmos, qwen = configs / 'dependencies/configs/Cosmos-Reason2-8B', configs / 'dependencies/configs/Qwen3-VL-2B-Instruct'
                for directory, label, repo, revision, source in ((cosmos, 'Cosmos', MODEL_VLM_NAME, COSMOS_REVISION, 'cosmos-reason2-source.json'),
                        (qwen, 'Qwen', 'Qwen/Qwen3-VL-2B-Instruct', QWEN_REVISION, 'qwen3-vl-source.json')):
                    _verify_config_dir(directory, label)
                    _verify_config_source(directory, label, repo, revision, source, configs / 'manifests')
                interface = _load_offline_dataset_interface(dataset)
            with recorder.phase('cold_bf16_load_and_overlay'):
                model, processor = _load_local_model_and_processor(root / 'models', cosmos, qwen, config)
                model.eval().requires_grad_(False)
                load_overlay(model, overlay)
            with recorder.phase('replace_existing252packed', model):
                for row in packed:
                    owner, attr = row['name'].rsplit('.', 1)
                    replacement = PackedAWQLinear.load(row['path'], device='cuda')
                    if [replacement.out_features, replacement.in_features] != row['shape']:
                        raise RuntimeError('packed实际shape不符')
                    setattr(model.get_submodule(owner), attr, replacement)
                    del replacement
                gc.collect()
                torch.cuda.empty_cache()
            result['protected_sha256_before'] = helpers.protected_model_digest(torch, model)
            if result['protected_sha256_before'] != awq['protected_sha256']:
                raise RuntimeError('冻结AWQ保护模块hash不符')
            result['storage_before'] = helpers.storage_inventory_by_component(model)
            if result['storage_before']['persistent_storage_bytes'] != sum(awq['packed_storage']['unique_storage_bytes_by_device'].values()):
                raise RuntimeError('冻结AWQ persistent storage变化')
            with recorder.phase('install_shared_hybrid', model):
                result['hybrid_install'] = install_hybrid_awq(model, manifest, enabled=args.awq_forward == 'hybrid_m1')
            result['storage_after'] = helpers.storage_inventory_by_component(model)
            if result['storage_after']['persistent_storage_bytes'] != result['storage_before']['persistent_storage_bytes']:
                raise RuntimeError('混合包装改变persistent storage')
            if model.diffusion.num_inference_steps != 10 or model.diffusion.int_method != 'euler':
                raise RuntimeError('冻结官方10step Euler默认不符')
            pair_rows = {(r['clip_id'], r['seed']): r for r in paired['records']} if paired else {}
            first_request = True
            for clip in clips:
                data = helpers.synchronized_stage(torch, separate := {}, 'disk_data_decode_ms', lambda:
                    load_physical_aiavdataset(clip, t0_us=5_100_000, avdi=interface, maybe_stream=False))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    raw = helpers.raw_prompt(model, processor, data)
                helpers.assert_prompt_equivalence(torch, raw, examples[clip])
                del raw
                for seed in SEEDS:
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        noise_hash = digest(torch.randn(1, *model.action_space.get_action_space_dims(), device='cuda'))
                    if noise_hash != refs['awq'][clip, seed]['initial_noise_sha256']:
                        raise RuntimeError('原seed/noise配对失败')
                    row = {'clip_id': clip, 'seed': seed, 'variant': args.awq_forward, 'steps': 10,
                        'decision_id': DECISION_ID,
                        'status': 'running', 'first_request': first_request, 'separate_io_ms': dict(separate)}
                    result['records'].append(row)
                    save()
                    dispatch_inventory(model, reset=True)
                    stage_times = {}
                    phase = f'{args.awq_forward}/{clip}/seed{seed}'
                    with recorder.phase(phase, model), torch.autocast('cuda', dtype=torch.bfloat16):
                        torch.cuda.synchronize()
                        began = time.perf_counter()
                        prompt = helpers.synchronized_stage(torch, stage_times, 'raw_processor_and_history_fuse_ms', lambda: helpers.raw_prompt(model, processor, data))
                        prompt_length = prompt['input_ids'].shape[1]
                        inputs, actual = helpers.synchronized_stage(torch, stage_times, 'generation_ms', lambda: helpers.generate_actual_cache(torch, model, prompt))
                        del actual
                        output = helpers.synchronized_stage(torch, stage_times, 'repeat_fullprefill_ms', lambda: model.vlm.model(**inputs, use_cache=True, return_dict=True))
                        output.rope_deltas = model.vlm.model.rope_deltas.detach().clone()
                        xyz, rot = helpers.synchronized_stage(torch, stage_times, 'expert_sample_and_action_to_traj_ms', lambda: helpers.sample_actions(torch, model, output, inputs, data, seed, 10))
                        torch.cuda.synchronize()
                        row['e2e_ms'] = (time.perf_counter() - began) * 1000
                    # token/noise/metric/hash回拷在计时外，不复用教师continuation。
                    row.update(helpers.measured_trajectory(torch, xyz, rot, data, noise_hash))
                    row.update(status='completed', stage_ms=stage_times, prefix_sha256=digest(inputs['input_ids']),
                        prefix_length=inputs['input_ids'].shape[1],
                        generated_token_ids=inputs['input_ids'][0, prompt_length:].detach().cpu().tolist(),
                        future_start_token_id=int(inputs['input_ids'][0, -1].item()),
                        teacher_ADE_m=refs['teacher'][clip, seed]['ADE_m'],
                        original_awq_ADE_m=refs['awq'][clip, seed]['ADE_m'], dispatch=dispatch_inventory(model))
                    row['e2e_scope'] = 'rawimages→processor/historyfuse→autonomous generate→fullprefill→official Expert Euler10→64point; disk I/O/load/audit excluded'
                    if args.awq_forward == 'hybrid_m1' and (row['dispatch']['module_count'] != 252
                            or any(c['m1_fused'] == 0 for c in row['dispatch']['modules'].values())):
                        raise RuntimeError('请求没有实际经过全部252个M1 fused投影')
                    original = sampled(model, output, inputs, data, seed)
                    gate = helpers.numerical_gate(row, original)
                    gate.update(clip_id=clip, seed=seed)
                    result['original_helper_gates'].append(gate)
                    if not gate['passed']: raise RuntimeError('新入口与官方原sample不一致')
                    if args.awq_forward == 'reference':
                        frozen = helpers.numerical_gate(refs['awq'][clip, seed], row)
                        row['frozen_awq_gate'] = frozen
                        if not frozen['passed']: raise RuntimeError('新reference与冻结AWQ不一致')
                    else:
                        gate = strict_pair_gate(pair_rows[clip, seed], row)
                        result['strict_pair_gates'].append(gate)
                        save()
                        if not gate['passed']:
                            result['status'] = 'strict_hybrid_gate_failed_requires_decision'
                            result['exit_code'] = 2
                            result['stop_reason'] = 'token/prefix/future-start/noise or ADE<=1e-4 NMSE<=1e-3 failed; actualKV exemption does not apply'
                            save()
                            return 2
                    first_request = False
                    save()
                    print('HYBRID_EVAL', clip, seed, args.awq_forward, row['ADE_m'], row['e2e_ms'], flush=True)
                    del output, inputs, prompt, xyz, rot, original
                    gc.collect()
                    torch.cuda.empty_cache()
                if args.capture_decode_activations and clip == clips[-1]:
                    pending_capture = capture_decode_activations(torch, model, processor, data, out, helpers, result)
                del data
            result['protected_sha256_after'] = helpers.protected_model_digest(torch, model)
            if result['protected_sha256_before'] != result['protected_sha256_after']:
                raise RuntimeError('混合安装/推理修改保护模块')
            result['actual_import_sha256'] = imported_source_receipt()
            for path_value, digest in result['actual_import_sha256_start'].items():
                if result['actual_import_sha256'].get(path_value) != digest:
                    raise RuntimeError('推理期间实际import来源变化: ' + path_value)
            for path, value in {**result['code_sha256'], **result['runtime_source_sha256'],
                                **result['input_sha256'], **result['gate_provenance'],
                                **result['actual_import_sha256']}.items():
                if sha256_file(path) != value: raise RuntimeError('实验中来源变化: ' + path)
            result['summary'] = helpers.summarize_complete(result['records'], clips)
            steady = [r['e2e_ms'] for r in result['records'] if not r['first_request']]
            result['first_request_ms'] = result['records'][0]['e2e_ms']
            result['steady_e2e'] = {'mean_ms': statistics.mean(steady), 'p50_ms': helpers.percentile(steady, .5),
                'p95_ms': helpers.percentile(steady, .95), 'p99_ms': helpers.percentile(steady, .99),
                'max_ms': max(steady), 'requests_over_300ms': sum(v > 300 for v in steady)}
            result['environment']['gpu_processes_after'] = gpu_processes()
            if pending_capture is not None:
                result['decode_activation_capture'] = write_final_capture_manifest(pending_capture, result, out)
            result.update(status='completed_awq_hybrid_development_eval', exit_code=0)
            save()
            if pending_capture is not None: write_final_reference_receipt(result, out)
            return 0
    except Exception as error:
        result.update(status='failed', exit_code=1, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        for row in result['records']:
            if row['status'] == 'running': row['status'] = 'failed'
        save()
        print('HYBRID_EVAL_FAILED', str(error), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
