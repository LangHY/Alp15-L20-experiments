"""冻结 AWQ 的实际 KV/官方少步配对诊断；不宣称 Orin 部署验收。"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

SEEDS = (42, 43, 44)
STEPS = (2, 4, 6, 10)
TEACHER_MEAN_ADE = 1.2323994789272548
AWQ_RESULT_SHA256 = '16aa6b20915bac68597614361e3932c89b955e7c05c024f1f5025b435ddd76cc'
CANDIDATE_DECISION = 'D-JETSON-KV-CANDIDATE-20261008'


def cache_candidate_allowed(gate, policy='strict', decision=None):
    """新候选只豁免数值一致性；严格默认、噪声和前缀配对保持不变。"""
    if policy not in ('strict', 'new_candidate'):
        raise ValueError('未知 cache policy')
    if not gate['noise_paired'] or not gate['prefix_paired']:
        return False
    if policy == 'new_candidate':
        if decision != CANDIDATE_DECISION:
            raise ValueError('新候选需要明确决策编号')
        return True
    return gate['passed']


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_reference_pairs(teacher, awq):
    """20 个 clip 为独立单元；缺失/重复/换 seed/换噪声均停止。"""
    refs = {}
    for label, rows in (('teacher', teacher), ('awq', awq)):
        mapping = {}
        for row in rows:
            identity = row['clip_id'], row['seed']
            if identity in mapping or row['seed'] not in SEEDS or not math.isfinite(row['ADE_m']):
                raise ValueError('重复、非法 seed 或非有限参考')
            noise = row['initial_noise_sha256']
            if not isinstance(noise, str) or len(noise) != 64 or any(c not in '0123456789abcdef' for c in noise):
                raise ValueError('参考缺少真实 noise SHA256')
            mapping[identity] = row
        clips = sorted({identity[0] for identity in mapping})
        if len(clips) != 20 or set(mapping) != {(clip, seed) for clip in clips for seed in SEEDS}:
            raise ValueError('需要完整固定20clip×seeds42/43/44')
        refs[label] = mapping
    if set(refs['teacher']) != set(refs['awq']):
        raise ValueError('教师与AWQ clip/seed 未配对')
    for key in refs['teacher']:
        if refs['teacher'][key]['initial_noise_sha256'] != refs['awq'][key]['initial_noise_sha256']:
            raise ValueError('教师与AWQ noise 未配对')
    return clips, refs


def numerical_gate(reference, candidate, nmse_limit=1e-3, ade_limit=1e-4):
    a, b = reference['pred_xyz'], candidate['pred_xyz']
    if len(a) != len(b) or not a or any(len(x) != 3 for x in a + b):
        raise ValueError('数值门禁轨迹形状不符')
    flat_a, flat_b = [v for row in a for v in row], [v for row in b for v in row]
    if not all(math.isfinite(v) for v in flat_a + flat_b + [reference['ADE_m'], candidate['ADE_m']]):
        raise ValueError('数值门禁出现非有限')
    nmse = sum((x-y)**2 for x, y in zip(flat_a, flat_b)) / max(sum(x*x for x in flat_a), 1e-12)
    delta = abs(candidate['ADE_m'] - reference['ADE_m'])
    noise_paired = reference['initial_noise_sha256'] == candidate['initial_noise_sha256']
    return {'trajectory_nmse': nmse, 'absolute_ADE_delta_m': delta,
            'max_abs_xyz_delta_m': max(abs(x-y) for x, y in zip(flat_a, flat_b)),
            'noise_paired': noise_paired, 'passed': noise_paired and nmse <= nmse_limit and delta <= ade_limit}


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered)-1)*q
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right]-ordered[left])*(position-left)


def summarize_complete(rows, clips):
    keys = [(r['clip_id'], r['seed']) for r in rows]
    expected = {(clip, seed) for clip in clips for seed in SEEDS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('summary不能丢弃/重复clip或seed')
    if any(r['status'] != 'completed' or not all(math.isfinite(r[k]) for k in ('ADE_m', 'FDE_m', 'e2e_ms', 'teacher_ADE_m')) for r in rows):
        raise ValueError('summary不能丢弃失败或非有限')
    means = [statistics.mean(r['ADE_m'] for r in rows if r['clip_id'] == clip) for clip in clips]
    mean = statistics.mean(means)
    teacher_means = [statistics.mean(r['teacher_ADE_m'] for r in rows if r['clip_id'] == clip) for clip in clips]
    paired_teacher = statistics.mean(teacher_means)
    if paired_teacher <= 0:
        raise ValueError('教师ADE必须为正，不能报告非法相对精度')
    latency = [r['e2e_ms'] for r in rows]
    return {'independent_clips': len(clips), 'trajectories': len(rows), 'mean_ADE_m': mean,
            'paired_teacher_mean_ADE_m': paired_teacher,
            'precision_reference_scope': 'same clip/seed rows, not global twenty-clip mean',
            'clip_ADE_p95_m': percentile(means, .95), 'clip_ADE_max_m': max(means),
            'clip_ADE_median_m': statistics.median(means),
            'clips_better_than_teacher': sum(a < b for a, b in zip(means, teacher_means)),
            'clips_worse_than_teacher': sum(a > b for a, b in zip(means, teacher_means)),
            'mean_FDE_m': statistics.mean(r['FDE_m'] for r in rows),
            'e2e_mean_ms': statistics.mean(latency), 'e2e_median_ms': statistics.median(latency),
            'e2e_p95_ms': percentile(latency, .95), 'e2e_p99_ms': percentile(latency, .99),
            'e2e_max_ms': max(latency), 'requests_over_300ms': sum(v > 300 for v in latency),
            'relative_teacher_ADE_loss': mean / paired_teacher - 1,
            'development_ADE_within_15pct': mean <= 1.15*paired_teacher,
            'Orin_gate_passed': False,
            'limitations': 'Repeated-use development clips; L20 shared-GPU timings do not verify Orin 300ms'}


def storage_inventory_by_component(model):
    """实测底层 storage，别名仅计一次；shared 信息保留，不能靠参数数猜容量。"""
    def component(name):
        if name.startswith('vlm.lm_head.'): return 'vlm_lm_head'
        if '.embed_tokens.' in name and name.startswith('vlm.'): return 'vlm_embedding'
        if name.startswith('vlm.model.visual.'): return 'vlm_visual'
        if name.startswith('vlm.model.language_model.'): return 'vlm_language'
        if name.startswith('expert.'): return 'expert'
        if name.startswith(('action_in_proj.', 'action_out_proj.')): return 'action_projection'
        if name.startswith(('action_space.', 'traj_tokenizer.', 'hist_traj_tokenizer.')): return 'action_and_trajectory'
        return 'other'
    seen, groups = {}, {}
    tensors = list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
    for name, tensor in tensors:
        storage = tensor.untyped_storage()
        identity = str(tensor.device), storage.data_ptr(), storage.nbytes()
        group = component(name)
        groups.setdefault(group, {'storage_bytes': 0, 'tensor_names': []})['tensor_names'].append(name)
        if identity in seen:
            seen[identity]['aliases'].append({'name': name, 'component': group})
            continue
        groups[group]['storage_bytes'] += storage.nbytes()
        seen[identity] = {'bytes': storage.nbytes(), 'assigned_component': group,
                          'aliases': [{'name': name, 'component': group}]}
    return {'persistent_storage_bytes': sum(row['bytes'] for row in seen.values()),
            'groups': groups, 'shared_storages': [row for row in seen.values() if len(row['aliases']) > 1],
            'assignment': 'first registered owner of shared storage; all aliases reported',
            'scope': 'parameters and persistent/runtime buffers only; excludes activations/CUDA allocator/RSS'}


def synchronized_stage(torch, timings, name, operation):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    timings[name] = (time.perf_counter()-start)*1000
    return value


def raw_prompt(model, processor, data):
    """现场 prepare 的 raw→processor→history fuse；没有复用 teacher continuation。"""
    from alpamayo1_5 import helper
    messages = helper.create_message(data['image_frames'].flatten(0, 1), camera_indices=data['camera_indices'])
    encoded = dict(processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors='pt'))
    encoded = helper.to_device(encoded, 'cuda')
    encoded['input_ids'] = model.fuse_traj_tokens(encoded['input_ids'], {
        'ego_history_xyz': data['ego_history_xyz'].cuda(), 'ego_history_rot': data['ego_history_rot'].cuda()})
    return encoded


def assert_prompt_equivalence(torch, raw, example):
    length = int(example['positions'][example['logit_mask']][0])+1
    expected = dict(example['inputs'])
    expected['input_ids'] = expected['input_ids'][:, :length]
    expected['attention_mask'] = torch.ones_like(expected['input_ids'])
    if set(raw) != set(expected):
        raise RuntimeError('raw processor 与冻结cache keys不等')
    for key in raw:
        if not torch.equal(raw[key].detach().cpu(), expected[key]):
            raise RuntimeError('raw processor/history fuse 与冻结prompt不等: '+key)


def generate_actual_cache(torch, model, prompt):
    """与冻结 generate_prefix 同参数；CPU回拷、hash、text decode 移至计时之外。"""
    import copy
    from transformers import StoppingCriteriaList, LogitsProcessorList
    from alpamayo1_5.models.token_utils import StopAfterEOS
    from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
    inputs = dict(prompt)
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    future = model.tokenizer.convert_tokens_to_ids('<|traj_future_start|>')
    gen = copy.deepcopy(model.vlm.generation_config)
    gen.top_p, gen.temperature, gen.top_k, gen.do_sample = .98, .6, None, True
    gen.num_return_sequences, gen.max_new_tokens = 1, 256
    gen.output_logits, gen.return_dict_in_generate = False, True
    gen.pad_token_id = model.tokenizer.pad_token_id
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    actual = model.vlm.generate(**inputs, generation_config=gen,
        stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=future)]),
        logits_processor=LogitsProcessorList([ExpertLogitsProcessor(
            model.config.traj_token_start_idx, model.config.traj_vocab_size)]))
    actual.rope_deltas = model.vlm.model.rope_deltas.detach().clone()
    inputs['input_ids'] = actual.sequences[:, :-1]
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    if inputs['input_ids'][0, -1].item() != future:
        raise RuntimeError('真实生成缺少future-start完整边界，拒绝剔除失败clip')
    if actual.past_key_values.get_seq_length() != inputs['input_ids'].shape[1]:
        raise RuntimeError('actual KV与sequence[:-1]长度不符')
    return inputs, actual


def sample_actions(torch, model, output, inputs, data, seed, steps):
    """复用原 expert_velocity，直接调用官方 sample inference_step。"""
    from expert_interface_distill import expert_velocity
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        action = model.diffusion.sample(batch_size=1,
            step_fn=lambda x, t: expert_velocity(model, output, inputs, {'x': x, 't': t}),
            device=torch.device('cuda'), return_all_steps=False, inference_step=steps)
        xyz, rot = model.action_space.action_to_traj(action, data['ego_history_xyz'][:, -1].cuda(),
            data['ego_history_rot'][:, -1].cuda())
    return xyz, rot


def measured_trajectory(torch, xyz, rot, data, noise_hash):
    from controlled_8b_diagnostic import digest
    if tuple(xyz.shape) != (1, 64, 3) or not torch.isfinite(xyz).all() or not torch.isfinite(rot).all():
        raise RuntimeError('轨迹形状错误/非finite；保留失败，不删除')
    distances = torch.linalg.vector_norm(xyz[0, :, :2].float().cpu()-data['ego_future_xyz'][0, 0, :, :2].float().cpu(), dim=-1)
    return {'ADE_m': distances.mean().item(), 'FDE_m': distances[-1].item(),
            'pred_xyz': xyz[0].float().cpu().tolist(), 'pred_xyz_sha256': digest(xyz),
            'initial_noise_sha256': noise_hash}


def protected_model_digest(torch, model):
    quantized = tuple(f'vlm.model.language_model.layers.{i}.{section}.{projection}.'
        for i in range(36) for section, projections in (('mlp', ('gate_proj', 'up_proj', 'down_proj')),
            ('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj'))) for projection in projections)
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(quantized): continue
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--awq-artifact', type=Path)
    parser.add_argument('--clips', type=int, choices=(2, 20), default=2)
    parser.add_argument('--steps', type=int, nargs='+', choices=STEPS, default=[10])
    parser.add_argument('--approved-smoke', type=Path, help='20clip须先通过同代码/资产2clip numerical gate')
    parser.add_argument('--preflight', action='store_true', help='只核验来源/资产，不加载模型')
    parser.add_argument('--cache-policy', choices=('strict', 'new_candidate'), default='strict')
    parser.add_argument('--candidate-decision', choices=(CANDIDATE_DECISION,))
    args = parser.parse_args()
    if len(args.steps) != len(set(args.steps)):
        parser.error('steps不能重复')
    if args.clips == 20 and not args.approved_smoke:
        parser.error('20clip需要--approved-smoke')
    if args.cache_policy == 'new_candidate' and args.candidate_decision != CANDIDATE_DECISION:
        parser.error('new_candidate需要--candidate-decision')
    root, out = args.root.resolve(), args.output.resolve()
    artifact = (args.awq_artifact or root/'experiments/awq_full_20261008_best8_full20_v4').resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'argv': sys.argv, 'decision': 'D-JETSON-KV-STEPS-20261008',
              'text_seed': 42, 'trajectory_seeds': list(SEEDS), 'steps': args.steps,
              'cache_policy': args.cache_policy, 'candidate_decision': args.candidate_decision,
              'records': [], 'numerical_gates': [], 'original_helper_gates': [],
              'teacher_mean_ADE_m': TEACHER_MEAN_ADE, 'target_ADE_m': 1.15*TEACHER_MEAN_ADE,
              'limitations': ['L20 shared GPU, not Orin real-device result',
                  'BF16-first loading, not packed-only deployment; cold load peak retained',
                  'Repeated-use20 development clips, independent n20 not n60',
                  'One rawimages→processor→generate→expert→trajectory per row; disk decode/hash/metric validation separate'],
              'code_sha256': {}, 'input_sha256': {}, 'runtime_source_sha256': {}}
    started = time.perf_counter()
    def save():
        result['elapsed_seconds'] = time.perf_counter()-started
        temporary = out/'result.tmp'
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temporary.replace(out/'result.json')
    def freeze(path, destination):
        path = Path(path).resolve()
        destination[str(path)] = sha256_file(path)
        return destination[str(path)]
    try:
        awq_path = artifact/'result.json'
        if freeze(awq_path, result['input_sha256']) != AWQ_RESULT_SHA256:
            raise RuntimeError('不是冻结full20 AWQ result')
        awq = json.loads(awq_path.read_text())
        if awq['status'] != 'completed_awq_development_pilot':
            raise RuntimeError('AWQ artifact未完成')
        parent = root/'experiments/mlp_newdata8_recovery_20261002_165238'
        reference = root/'experiments/controlled8_expanded20_20261002_125000'
        dense_path, recovery_path = reference/'result.json', parent/'controlled_recovery_result.json'
        dense, recovery = json.loads(dense_path.read_text()), json.loads(recovery_path.read_text())
        if dense['status'] != 'completed' or recovery['status'] != 'completed' or recovery['selected_step'] != 400:
            raise RuntimeError('教师/父8B完成门禁不符')
        for path in (dense_path, recovery_path): freeze(path, result['input_sha256'])
        teacher_rows = [r for r in dense['records'] if r['model'] == 'dense' and r['prefix'] == 'teacher']
        awq_rows = [r for r in awq['records'] if r['variant'] == 'best8_w4a16_awq']
        all_clips, refs = validate_reference_pairs(teacher_rows, awq_rows)
        actual_teacher = statistics.mean(r['ADE_m'] for r in teacher_rows)
        if abs(actual_teacher-TEACHER_MEAN_ADE) > 1e-12:
            raise RuntimeError('教师reference口径变更')
        result['clips'] = clips = all_clips[:args.clips]
        result['all_frozen_clips'] = all_clips
        result['independent_clips'] = len(clips)
        overlay = Path(recovery['overlay_directory']).resolve()
        freeze(overlay/'manifest.json', result['input_sha256'])
        for path in overlay.glob('*.safetensors'): freeze(path, result['input_sha256'])
        packed = awq['packed_layers']
        expected_names = {f'language_model.layers.{i}.{section}.{projection}' for i in range(36)
            for section, projections in (('mlp', ('gate_proj', 'up_proj', 'down_proj')),
                ('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj'))) for projection in projections}
        if len(packed) != 252 or {r['name'] for r in packed} != expected_names:
            raise RuntimeError('必须恰好252冻结量化层')
        for row in packed:
            path = artifact/'packed'/Path(row['file']).name
            if freeze(path, result['input_sha256']) != row['sha256']:
                raise RuntimeError('AWQ actual file SHA不符: '+str(path))
        sys.path.insert(0, str(root/'compression'))
        sys.path.insert(0, str(root/'scripts'))
        import importlib
        # 当前入口目录优先，root/scripts用于已有packed依赖，避免意外运行旧入口。
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        modules = [importlib.import_module(name) for name in ('controlled_8b_diagnostic',
            'expert_interface_distill', 'run_mlp_experiment', 'runtime_memory_record',
            'mlp_pruning', 'alp15_awq_full_packed', 'alp15_packed_w4a16', 'test_alpamayo15_offline_compat')]
        frozen_by_name = {Path(path).name: sha for path, sha in awq['code_sha256'].items()}
        freeze(Path(__file__), result['code_sha256'])
        for module in modules:
            digest = freeze(module.__file__, result['code_sha256'])
            if digest != frozen_by_name.get(Path(module.__file__).name):
                raise RuntimeError('helper源码与AWQ冻结代码不等: '+module.__name__)
        fixed_sources = {'base_model.py': '1f2e49a2ac62346cc61c3ea954982729da1a92674bc4ceed7bb43c87ba212cc2',
            'delta_tokenizer.py': '5c6c90f4c615bdd1d337810ccbd862ae0eb92b4e30331c665b229b654c19678e',
            'flow_matching.py': '57775891e6e0c047c3f203b7b141ffd6034b13969076926200f76404e48058d7',
            'alpamayo1_5.py': '87b23954570f4d466a6d0dd64090f779af635433b61106eb3b028d8d802ff1e8'}
        from alpamayo1_5.diffusion.flow_matching import FlowMatching
        from alpamayo1_5 import helper
        source_root = Path(inspect.getfile(helper)).resolve().parent
        for path in sorted(source_root.rglob('*.py')):
            digest = freeze(path, result['runtime_source_sha256'])
            if path.name in fixed_sources and digest != fixed_sources[path.name]:
                raise RuntimeError('现场源发生变化: '+path.name)
        if not all(any(Path(path).name == name for path in result['runtime_source_sha256']) for name in fixed_sources):
            raise RuntimeError('源码文件覆盖不足')
        if 'inference_step' not in inspect.signature(FlowMatching.sample).parameters:
            raise RuntimeError('不是已验证官方 FlowMatching inference_step接口')
        head = subprocess.check_output(['git', '-C', str(root/'alpamayo1.5'), 'rev-parse', 'HEAD'], text=True).strip()
        result['source'] = {'HEAD': head, 'dirty_status': subprocess.check_output([
            'git', '-C', str(root/'alpamayo1.5'), 'status', '--short'], text=True),
            'warning': 'HEAD identity does not mean unmodified official source; dirty hashes frozen and retained'}
        if head != awq['source_commit']: raise RuntimeError('source HEAD发生变化')
        for key in ('source_commit', 'checkpoint_revision', 'dataset_revision'):
            if awq[key] != dense[key]: raise RuntimeError('来源不配对: '+key)
            result[key] = awq[key]
        # 真正执行路径的 Transformers cache/generation/Qwen源码逐文件冻结；
        # smoke/expanded在同一检查时点比较，不能在扩评比较之后才补入。
        import transformers.cache_utils, transformers.generation.utils
        import transformers.models.qwen3_vl.modeling_qwen3_vl
        for module in (transformers.cache_utils, transformers.generation.utils,
                       transformers.models.qwen3_vl.modeling_qwen3_vl):
            freeze(module.__file__, result['runtime_source_sha256'])
        if args.approved_smoke:
            smoke_path = args.approved_smoke.resolve()/'result.json'
            smoke = json.loads(smoke_path.read_text())
            if (smoke['status'] != 'completed_kv_steps_development_probe' or smoke['clips'] != all_clips[:2]
                or smoke.get('cache_policy', 'strict') != args.cache_policy
                or smoke.get('candidate_decision') != args.candidate_decision
                or len(smoke['numerical_gates']) != 6
                or not all(cache_candidate_allowed(g, args.cache_policy, args.candidate_decision) for g in smoke['numerical_gates'])
                or not all(g['passed'] for g in smoke['original_helper_gates'])
                or smoke['code_sha256'] != result['code_sha256']
                or smoke['runtime_source_sha256'] != result['runtime_source_sha256']):
                raise RuntimeError('扩20必须同source/code/assets且2clip cache数学门禁通过')
            for path, digest in smoke['input_sha256'].items():
                if path in result['input_sha256'] and result['input_sha256'][path] != digest:
                    raise RuntimeError('2clip与20clip资产变化')
            freeze(smoke_path, result['input_sha256'])
        if args.preflight:
            result['status'] = 'passed_preflight_only'
            save()
            return 0
        import gc
        import torch
        from controlled_8b_diagnostic import digest, sampled
        from alp15_awq_full_packed import PackedAWQLinear
        from runtime_memory_record import MemoryRecorder
        from run_mlp_experiment import (_enable_offline_mode, _block_network, _verify_model,
            _verify_dataset, _verify_config_dir, _verify_config_source, _load_local_model_and_processor,
            _load_offline_dataset_interface, load_overlay, MODEL_VLM_NAME, COSMOS_REVISION, QWEN_REVISION)
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        _enable_offline_mode()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('需要L20原BF16 CUDA推理环境')
        result['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(), 'device': torch.cuda.current_device()}
        recorder = MemoryRecorder(out/'memory_phases.jsonl', device=torch.device('cuda'))
        examples = {}
        for path in sorted((reference/'teacher_prefix_cache').glob('*.pt')):
            example = torch.load(path, weights_only=True, map_location='cpu')
            if example['clip_id'] in clips:
                examples[example['clip_id']] = example
                freeze(path, result['input_sha256'])
        if set(examples) != set(clips): raise RuntimeError('缺少冻结prompt cache')
        with _block_network(), torch.no_grad():
            with recorder.phase('asset_verification'):
                config = _verify_model(root/'models')
                dataset = root/'alp15-test-clip-delivery-20260917'
                _verify_dataset(dataset)
                configs = root/'alp15-configs-delivery-20260919'
                cosmos, qwen = configs/'dependencies/configs/Cosmos-Reason2-8B', configs/'dependencies/configs/Qwen3-VL-2B-Instruct'
                for directory, label, repo, revision, source in ((cosmos, 'Cosmos', MODEL_VLM_NAME, COSMOS_REVISION, 'cosmos-reason2-source.json'),
                    (qwen, 'Qwen', 'Qwen/Qwen3-VL-2B-Instruct', QWEN_REVISION, 'qwen3-vl-source.json')):
                    _verify_config_dir(directory, label)
                    _verify_config_source(directory, label, repo, revision, source, configs/'manifests')
                interface = _load_offline_dataset_interface(dataset)
            with recorder.phase('cold_bf16_load_and_overlay'):
                model, processor = _load_local_model_and_processor(root/'models', cosmos, qwen, config)
                model.eval().requires_grad_(False)
                load_overlay(model, overlay)
            with recorder.phase('replace_existing_252packed', model):
                for row in packed:
                    owner, attr = row['name'].rsplit('.', 1)
                    replacement = PackedAWQLinear.load(artifact/'packed'/Path(row['file']).name, device='cuda')
                    if [replacement.out_features, replacement.in_features] != row['shape']:
                        raise RuntimeError('packed shape不符')
                    setattr(model.vlm.model.get_submodule(owner), attr, replacement)
                    del replacement
                gc.collect()
                torch.cuda.empty_cache()
            result['packed_storage_by_component'] = storage_inventory_by_component(model)
            expected_storage = sum(awq['packed_storage']['unique_storage_bytes_by_device'].values())
            if result['packed_storage_by_component']['persistent_storage_bytes'] != expected_storage:
                raise RuntimeError('冻结AWQ persistent storage不符')
            result['protected_sha256_before'] = protected_model_digest(torch, model)
            if result['protected_sha256_before'] != awq['protected_sha256']:
                raise RuntimeError('AWQ保护模块hash不符')
            if model.diffusion.num_inference_steps != 10 or model.diffusion.int_method != 'euler':
                raise RuntimeError('冻结官方10step Euler默认不符')
            for clip in clips:
                data = synchronized_stage(torch, times := {}, 'disk_data_decode_ms', lambda:
                    load_physical_aiavdataset(clip, t0_us=5_100_000, avdi=interface, maybe_stream=False))
                # raw prompt与旧prefix来源核验在一次真实E2E之外，避免误归属时间。
                with recorder.phase('raw_prompt_validation/'+clip, model):
                    validation_start = time.perf_counter()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        raw = raw_prompt(model, processor, data)
                    assert_prompt_equivalence(torch, raw, examples[clip])
                    del raw
                    times['raw_prompt_validation_ms'] = (time.perf_counter()-validation_start)*1000
                variants = [('fullprefill', 10), ('actualKV', 10)] + [('actualKV', s) for s in args.steps if s != 10]
                for seed in SEEDS:
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        noise_hash = digest(torch.randn(1, *model.action_space.get_action_space_dims(), device='cuda'))
                    if noise_hash != refs['awq'][clip, seed]['initial_noise_sha256']:
                        raise RuntimeError('原seed/noise配对失败')
                    baseline = None
                    cache_gate_passed = False
                    for route, steps in variants:
                        label = f'{route}_{steps}steps'
                        phase = f'{label}/{clip}/seed{seed}'
                        row = {'clip_id': clip, 'seed': seed, 'variant': label, 'steps': steps,
                            'status': 'running', 'separate_io_and_validation_ms': dict(times)}
                        result['records'].append(row)
                        save()
                        stage_times = {}
                        with recorder.phase(phase, model), torch.autocast('cuda', dtype=torch.bfloat16):
                            torch.cuda.synchronize()
                            begin = time.perf_counter()
                            prompt = synchronized_stage(torch, stage_times, 'raw_processor_and_history_fuse_ms', lambda: raw_prompt(model, processor, data))
                            prompt_length = prompt['input_ids'].shape[1]
                            inputs, actual = synchronized_stage(torch, stage_times, 'generation_ms', lambda: generate_actual_cache(torch, model, prompt))
                            output = actual
                            if route == 'fullprefill':
                                # 原AWQ路径在重prefill前释放actual cache，保持内存比较公平。
                                del output, actual
                                actual = None
                                output = synchronized_stage(torch, stage_times, 'repeat_fullprefill_ms', lambda: model.vlm.model(**inputs, use_cache=True, return_dict=True))
                                output.rope_deltas = model.vlm.model.rope_deltas.detach().clone()
                            xyz, rot = synchronized_stage(torch, stage_times, 'expert_sample_and_action_to_traj_ms', lambda:
                                sample_actions(torch, model, output, inputs, data, seed, steps))
                            torch.cuda.synchronize()
                            row['e2e_ms'] = (time.perf_counter()-begin)*1000
                            row['e2e_scope'] = 'one rawimages request→processor+historyfuse→generation→expert→trajectory; excludes diskdecode/load/audit'
                        audit_start = time.perf_counter()
                        row.update(measured_trajectory(torch, xyz, rot, data, noise_hash))
                        row.update(status='completed', stage_ms=stage_times,
                            generated_tokens=inputs['input_ids'].shape[1]-prompt_length,
                            prefix_sha256=digest(inputs['input_ids']),
                            teacher_ADE_m=refs['teacher'][clip, seed]['ADE_m'],
                            original_awq_ADE_m=refs['awq'][clip, seed]['ADE_m'])
                        if route == 'fullprefill':
                            baseline = dict(row)
                            # 原 frozen helper10steps 数学对拍，在计时外；保留官方默认行为。
                            original = sampled(model, output, inputs, data, seed)
                            original_gate = numerical_gate(row, original, ade_limit=1e-4)
                            original_gate.update(clip_id=clip, seed=seed)
                            result['original_helper_gates'].append(original_gate)
                            frozen = numerical_gate(refs['awq'][clip, seed], row, ade_limit=1e-4)
                            if not original_gate['passed'] or not frozen['passed']:
                                row['original_helper_gate'] = original_gate
                                row['frozen_awq_gate'] = frozen
                                raise RuntimeError('fullprefill10 与官方旧sample/frozenAWQ不一致')
                        elif steps == 10:
                            gate = numerical_gate(baseline, row)
                            gate.update(clip_id=clip, seed=seed,
                                prefix_paired=baseline['prefix_sha256'] == row['prefix_sha256'])
                            gate['passed'] = gate['passed'] and gate['prefix_paired']
                            result['numerical_gates'].append(gate)
                            cache_gate_passed = cache_candidate_allowed(gate, args.cache_policy, args.candidate_decision)
                            gate['numerical_gate_passed'] = gate['passed']
                            gate['approved_as_new_candidate'] = args.cache_policy == 'new_candidate' and cache_gate_passed
                            gate['decision_id'] = args.candidate_decision
                            if not cache_gate_passed:
                                result['status'] = 'cache_equivalence_requires_decision'
                                result['stop_reason'] = 'actualKV10 ADE differs >1e-4m/NMSE>1e-3/prefix mismatch; preserve candidate, no low-step rollout until decision'
                                save()
                                return 2
                        elif not cache_gate_passed:
                            raise RuntimeError('少步前必须actualKV10门禁通过')
                        row['separate_metric_and_math_audit_ms'] = (time.perf_counter()-audit_start)*1000
                        save()
                        print('KV_STEPS', clip, seed, label, row['ADE_m'], row['e2e_ms'], flush=True)
                        del output, actual, inputs, prompt, xyz, rot
                        gc.collect()
                        torch.cuda.empty_cache()
                del data
            result['protected_sha256_after'] = protected_model_digest(torch, model)
            if result['protected_sha256_after'] != result['protected_sha256_before']:
                raise RuntimeError('推理修改了保护模块')
            for path, digest_value in {**result['code_sha256'], **result['runtime_source_sha256'], **result['input_sha256']}.items():
                if sha256_file(path) != digest_value: raise RuntimeError('实验中来源变化: '+path)
            labels = sorted({r['variant'] for r in result['records']})
            result['summary'] = {label: summarize_complete([r for r in result['records'] if r['variant'] == label], clips) for label in labels}
            result['status'] = 'completed_kv_steps_development_probe'
            save()
            return 0
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        for row in result['records']:
            if row['status'] == 'running': row['status'] = 'failed'
        save()
        print('KV_STEPS_FAILED', type(exc).__name__, str(exc), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
