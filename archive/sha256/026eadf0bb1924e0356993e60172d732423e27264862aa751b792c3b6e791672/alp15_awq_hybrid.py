"""冻结 AWQ 的 M=1 混合分派；不重新 pack，不建立持久 dense 权重。"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

from alp15_awq_full_packed import FORMAT, PackedAWQLinear

DECISION_ID = 'D-ALP15-HYBRID-M1-20261008'
AWQ_RESULT_SHA256 = '16aa6b20915bac68597614361e3932c89b955e7c05c024f1f5025b435ddd76cc'
TORCH_REMOTE_NAME = '_remote_module_non_scriptable'
TORCH_REMOTE_SHA256 = '8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8'
TORCH_REMOTE_GENERATORS = {
    'torch.distributed.nn.jit.instantiator': '567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615',
    'torch.distributed.nn.jit.templates.remote_module_template': '0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc',
}
TORCH_VIRTUAL_NAMESPACES = {'torch.ops': ('torch._ops', '_Ops', '_ops.py'),
                          'torch.classes': ('torch._classes', '_Classes', '_classes.py')}
FROZEN_SOURCE = {
    'alp15_awq_full_packed.py': '150e1310ad3dbcb9f82966146f9627855e87bd12a29f45f410d0b94f1a96de0a',
    'alp15_awq_kernel.py': '3f09aea862990733b9bb1db536c450c79fd6e2ac93c7015eed6b1eef4f191025',
    'alp15_packed_w4a16.py': '241edf75d838ad7cba1250374119237864780b95ec81cd8555a7d7c6d15fab12',
}
TARGET_NAMES = frozenset(
    f'vlm.model.language_model.layers.{i}.{section}.{projection}'
    for i in range(36)
    for section, projections in (('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
                                 ('mlp', ('gate_proj', 'up_proj', 'down_proj')))
    for projection in projections)


def validate_trajectory_row(row):
    """读入结果也执行64点门禁，不能只相信预先保存的passed标志。"""
    points = row.get('pred_xyz')
    if (not isinstance(points, list) or len(points) != 64
            or any(not isinstance(point, list) or len(point) != 3 for point in points)):
        raise ValueError('轨迹必须为64×3')
    numbers = [value for point in points for value in point] + [row.get('ADE_m')]
    if 'FDE_m' in row: numbers.append(row['FDE_m'])
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers):
        raise ValueError('轨迹/ADE/FDE必须有限')
    if row.get('decision_id') != DECISION_ID or row.get('steps') != 10:
        raise ValueError('轨迹decision/10step身份不符')
    if not isinstance(row.get('clip_id'), str) or not row['clip_id'] or row.get('seed') not in (42, 43, 44):
        raise ValueError('轨迹clip/seed身份不符')
    for key in ('initial_noise_sha256', 'prefix_sha256'):
        value = row.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('轨迹noise/prefix SHA256不符')
    tokens = row.get('generated_token_ids')
    future, length = row.get('future_start_token_id'), row.get('prefix_length')
    if (not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens)
            or type(future) is not int or future < 0 or tokens[-1] != future
            or type(length) is not int or length <= len(tokens)):
        raise ValueError('轨迹token/future-start/prefix边界不合法')


def verify_hash_inventory(inventory, label):
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError(label + '来源hash缺失')
    for path, digest in inventory.items():
        if sha256_file(path) != digest:
            raise ValueError(label + '来源漂移: ' + path)


def _dynamic_identity(metadata):
    stable = {key: metadata[key] for key in ('module_name', 'source_sha256', 'torch_version', 'generator_entry')}
    stable['generator_sha256'] = {name: row['sha256'] for name, row in metadata['generators'].items()}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def snapshot_torch_remote_module(module, snapshot_directory):
    """只冻结已核实Torch2.8生成器的原字节；不重建已消失的来源。"""
    name = module.__name__
    if name != TORCH_REMOTE_NAME or str(torch.__version__).split('+')[0] != '2.8.0':
        raise ValueError('未知动态模块/torch版本，不放宽来源校验')
    source = Path(module.__file__).resolve(strict=True)
    spec = getattr(module, '__spec__', None)
    if spec is None or spec.name != name or Path(spec.origin).resolve(strict=True) != source:
        raise ValueError('动态module/spec来源身份不符')
    factory = sys.modules.get('torch.distributed.nn.jit.instantiator')
    if (factory is None or source.name != name + '.py'
            or source.parent != Path(factory.INSTANTIATED_TEMPLATE_DIR_PATH).resolve(strict=True)):
        raise ValueError('动态源码不属于已核实Torch生成器目录')
    generators = {}
    for module_name, expected in TORCH_REMOTE_GENERATORS.items():
        generator = sys.modules.get(module_name)
        if generator is None: raise ValueError('动态源码生成器/模板未实际import')
        path = Path(generator.__file__).resolve(strict=True)
        digest = sha256_file(path)
        if digest != expected: raise ValueError('Torch生成器/模板源码漂移')
        generators[module_name] = {'path': str(path), 'sha256': digest}
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != TORCH_REMOTE_SHA256:
        raise ValueError('动态源码字节不是冻结的Torch2.8 non_scriptable模板实例')
    directory = Path(snapshot_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / (name + '-' + digest + '.py')
    if snapshot.exists():
        if sha256_file(snapshot) != digest: raise ValueError('动态源码快照被修改；拒绝覆盖')
    else:
        with snapshot.open('xb') as handle: handle.write(payload)
    metadata = {'module_name': name, 'original_path': str(source), 'spec_origin': str(source),
        'generator_temp_directory': str(source.parent), 'snapshot_path': str(snapshot),
        'source_sha256': digest, 'torch_version': str(torch.__version__), 'generators': generators,
        'generator_entry': 'instantiate_non_scriptable_remote_module_template',
        'generation_mode': 'non_scriptable, enable_moving_cpu_tensors_to_cuda=True',
        'snapshot_policy': 'original bytes read while producing process is alive, not regenerated'}
    metadata['stable_identity_sha256'] = _dynamic_identity(metadata)
    return metadata


def verify_import_provenance(provenance, require_live=False):
    verify_hash_inventory(provenance['actual_import_sha256'], 'actual_import_sha256')
    directory = Path(provenance['source_snapshot_directory']).resolve()
    dynamic = provenance['dynamic_import_sources']
    if not isinstance(dynamic, dict) or set(dynamic) - {TORCH_REMOTE_NAME}:
        raise ValueError('未知动态源码映射')
    snapshots = set()
    for name, metadata in dynamic.items():
        original = Path(metadata['original_path']).resolve()
        snapshot = Path(metadata['snapshot_path']).resolve(strict=True)
        digest = metadata['source_sha256']
        if (name != metadata['module_name'] or name != TORCH_REMOTE_NAME or digest != TORCH_REMOTE_SHA256
                or str(metadata['torch_version']).split('+')[0] != '2.8.0'
                or metadata['generator_entry'] != 'instantiate_non_scriptable_remote_module_template'
                or metadata['stable_identity_sha256'] != _dynamic_identity(metadata)
                or snapshot.parent != directory or snapshot.name != name + '-' + digest + '.py'
                or original.name != name + '.py' or metadata['spec_origin'] != str(original)
                or metadata['generator_temp_directory'] != str(original.parent)
                or provenance['actual_import_sha256'].get(str(snapshot)) != digest):
            raise ValueError('动态源码稳定身份/快照映射不符')
        if set(metadata['generators']) != set(TORCH_REMOTE_GENERATORS):
            raise ValueError('动态源码生成器库存遗漏')
        for module_name, expected in TORCH_REMOTE_GENERATORS.items():
            row = metadata['generators'][module_name]
            if (row['sha256'] != expected or sha256_file(row['path']) != expected
                    or provenance['actual_import_sha256'].get(row['path']) != expected):
                raise ValueError('动态源码生成器/模板漂移或未在import库存')
        if require_live or original.exists():
            if sha256_file(original) != digest: raise ValueError('生成进程原动态源码漂移')
        snapshots.add(str(snapshot))
    actual_snapshots = {path for path in provenance['actual_import_sha256'] if Path(path).parent.resolve() == directory}
    if snapshots != actual_snapshots:
        raise ValueError('动态快照缺少完整映射，禁止以路径快照掩盖来源')
    virtual = provenance['virtual_import_sources']
    if not isinstance(virtual, dict) or set(virtual) - set(TORCH_VIRTUAL_NAMESPACES):
        raise ValueError('未知虚拟module映射')
    for name, row in virtual.items():
        owner, class_name, marker = TORCH_VIRTUAL_NAMESPACES[name]
        if (row['module_name'] != name or row['implementation_module'] != owner or row['class_name'] != class_name
                or row['original_file_marker'] != marker or Path(row['implementation_path']).name != marker
                or provenance['actual_import_sha256'].get(row['implementation_path']) != row['implementation_sha256']):
            raise ValueError('虚拟module实现来源映射不符')


def assert_dynamic_sources_equal(reference, current):
    a = {name: row['stable_identity_sha256'] for name, row in reference['dynamic_import_sources'].items()}
    b = {name: row['stable_identity_sha256'] for name, row in current['dynamic_import_sources'].items()}
    if a != b: raise ValueError('两进程动态源码/生成器稳定身份不等')
    if reference['virtual_import_sources'] != current['virtual_import_sources']:
        raise ValueError('两进程虚拟module实现身份不等')


def load_bound_capture(manifest_path, paired_reference=None):
    """manifest→终态receipt→result，result只含receipt路径，避免hash循环。"""
    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'alp15-m1-activations-v3' or manifest.get('decision_id') != DECISION_ID
            or manifest.get('awq_result_sha256') != AWQ_RESULT_SHA256):
        raise ValueError('真实activation需要v3完整来源及稳定动态源码；旧结果不可补造来源')
    result_path = Path(manifest['reference_result']).resolve(strict=True)
    receipt_path = Path(manifest['final_receipt']).resolve(strict=True)
    if (path != result_path.parent / 'decode_activations/manifest.json'
            or receipt_path != result_path.parent / 'reference_final_receipt.json'):
        raise ValueError('孤立capture或终态receipt路径不符')
    if paired_reference is not None:
        expected = Path(paired_reference).resolve(strict=True)
        if expected.is_dir(): expected = expected / 'result.json'
        if result_path != expected:
            raise ValueError('capture绑定了错误paired reference')
    result, receipt = json.loads(result_path.read_text()), json.loads(receipt_path.read_text())
    if (receipt.get('schema') != 'alp15-reference-capture-final-v1' or receipt.get('decision_id') != DECISION_ID
            or receipt.get('reference_result') != str(result_path)
            or receipt.get('reference_result_sha256') != sha256_file(result_path)
            or receipt.get('activation_manifest') != str(path)
            or receipt.get('activation_manifest_sha256') != sha256_file(path)):
        raise ValueError('capture终态receipt SHA256/身份不符')
    if (result.get('status') != 'completed_awq_hybrid_development_eval' or result.get('exit_code') != 0
            or result.get('awq_forward') != 'reference' or result.get('decision_id') != DECISION_ID):
        raise ValueError('capture必须绑定成功reference终态exit0')
    delivered = result.get('decode_activation_capture', {})
    if (delivered.get('manifest') != str(path) or delivered.get('sha256') != sha256_file(path)
            or delivered.get('final_receipt') != str(receipt_path) or delivered.get('coverage') != '252/252'):
        raise ValueError('capture不是成功reference实际交付')
    for key in ('code_sha256', 'runtime_source_sha256', 'input_sha256', 'actual_import_sha256'):
        if manifest.get(key) != result.get(key):
            raise ValueError('capture/reference来源不配对: ' + key)
        verify_hash_inventory(manifest[key], key)
    for key in ('source_snapshot_directory', 'dynamic_import_sources', 'virtual_import_sources'):
        if manifest.get(key) != result.get(key): raise ValueError('capture/reference动态源码映射不等')
    verify_import_provenance(result)
    if (manifest.get('protected_sha256') != result.get('protected_sha256_before')
            or result.get('protected_sha256_before') != result.get('protected_sha256_after')
            or manifest.get('text_seed') != result.get('text_seed') or result.get('text_seed') != 42
            or manifest.get('clip_id') not in result.get('clips', [])):
        raise ValueError('capture/reference保护模块或clip/text seed身份不符')
    records = result.get('records', [])
    expected = {(clip, seed) for clip in result.get('clips', []) for seed in (42, 43, 44)}
    if (len(result.get('clips', [])) != 2 or len(records) != 6
            or {(row['clip_id'], row['seed']) for row in records} != expected):
        raise ValueError('capture reference需要完整2clip×3seed')
    for row in records:
        validate_trajectory_row(row)
        if row.get('status') != 'completed': raise ValueError('capture reference含失败请求')
    activations, files, hashes = {}, set(), {str(path): sha256_file(path), str(result_path): sha256_file(result_path),
                               str(receipt_path): sha256_file(receipt_path)}
    for row in manifest['activations']:
        name = row['name']
        file = (path.parent / row['file']).resolve(strict=True)
        if (name not in TARGET_NAMES or name in activations or file in files or file.parent != path.parent
                or sha256_file(file) != row['sha256']):
            raise ValueError('真实activation名字/文件/SHA256不符')
        activations[name] = file
        files.add(file)
        hashes[str(file)] = row['sha256']
    if len(activations) != 252 or set(activations) != TARGET_NAMES:
        raise ValueError('正式真实capture必须完整252/252；不得随机补层')
    return activations, hashes, manifest, result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frozen_source_receipt(directory=None):
    directory = Path(directory or Path(__file__).resolve().parent)
    receipt = {}
    for name, expected in FROZEN_SOURCE.items():
        path = (directory / name).resolve(strict=True)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError('冻结 packed/kernel 源码 SHA256 变化: ' + name)
        receipt[str(path)] = actual
    return receipt


def _make_fused(reference):
    # 禁用模式和 M>1 连候选模块都无需导入；首次 M1 才建立共享 buffer 包装。
    from alp15_awq_kernel import FusedAWQLinear
    return FusedAWQLinear.from_reference(reference)


class HybridAWQLinear(nn.Module):
    def __init__(self, reference, enabled=True):
        super().__init__()
        if type(reference) is not PackedAWQLinear:
            raise TypeError('必须提供已经验真的原 PackedAWQLinear')
        if not isinstance(enabled, bool):
            raise TypeError('enabled 须为 bool')
        self.reference = reference
        self.fused = None
        self.enabled = enabled
        self.in_features, self.out_features = reference.in_features, reference.out_features
        self.reset_dispatch()

    @classmethod
    def from_reference(cls, reference, enabled=True):
        return cls(reference, enabled)

    @property
    def qweight(self): return self.reference.qweight

    @property
    def amax(self): return self.reference.amax

    @property
    def pre_scale(self): return self.reference.pre_scale

    @property
    def bias(self): return self.reference.bias

    def reset_dispatch(self):
        self.dispatch = {'m1_fused': 0, 'm_gt1_reference': 0, 'disabled_reference': 0}

    def forward(self, inputs):
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim < 1
                or inputs.shape[-1] != self.in_features or inputs.device != self.qweight.device):
            raise ValueError('输入 shape/device 不符')
        if inputs.numel() == 0:
            raise ValueError('输入不得为空')
        if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('输入 dtype 须为浮点 BF16/FP16/FP32')
        m = inputs.numel() // self.in_features
        if self.enabled and m == 1:
            if self.fused is None:
                self.fused = _make_fused(self.reference)
            # 原 fused 自行检查 CUDA、连续 BF16、无梯度；禁止转换或 fallback。
            result = self.fused(inputs)
            self.dispatch['m1_fused'] += 1
            return result
        result = self.reference(inputs)
        self.dispatch['disabled_reference' if not self.enabled else 'm_gt1_reference'] += 1
        return result

    def storage_bytes(self):
        seen = {}
        for value in self.buffers():
            storage = value.untyped_storage()
            seen[(str(value.device), storage.data_ptr(), storage.nbytes())] = storage.nbytes()
        return sum(seen.values())


def build_packed_manifest(artifact):
    """原 result 是唯一资产权威；本 envelope 不允许用户替换校准结果。"""
    artifact = Path(artifact).resolve(strict=True)
    manifest = {'schema': 'alp15-awq-hybrid-manifest-v1', 'format': FORMAT,
                'original_result': str(artifact / 'result.json'),
                'original_result_sha256': AWQ_RESULT_SHA256,
                'packed_directory': str(artifact / 'packed')}
    source = json.loads((artifact / 'result.json').read_text())
    manifest['layers'] = source['packed_layers']
    validate_packed_manifest(manifest)
    return manifest


def validate_packed_manifest(manifest):
    if (manifest.get('schema') != 'alp15-awq-hybrid-manifest-v1'
            or manifest.get('format') != FORMAT
            or manifest.get('original_result_sha256') != AWQ_RESULT_SHA256):
        raise ValueError('需要冻结原 AWQ artifact SHA256/格式')
    original = Path(manifest['original_result']).resolve(strict=True)
    if sha256_file(original) != AWQ_RESULT_SHA256:
        raise ValueError('原 AWQ result SHA256 不符')
    source = json.loads(original.read_text())
    if source.get('status') != 'completed_awq_development_pilot' or manifest.get('layers') != source.get('packed_layers'):
        raise ValueError('manifest 必须逐条等于原冻结 artifact')
    directory = Path(manifest['packed_directory']).resolve(strict=True)
    if directory != original.parent / 'packed':
        raise ValueError('packed 目录必须属于冻结 artifact')
    rows, names, files = [], set(), set()
    for item in manifest['layers']:
        name = 'vlm.model.' + item['name']
        if name not in TARGET_NAMES or name in names:
            raise ValueError('量化边界/重复名字不符: ' + name)
        shape = item.get('shape')
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(v) is not int or v <= 0 for v in shape) or shape[1] % 128):
            raise ValueError('冻结 shape/group128 不符: ' + name)
        if item.get('amax_dtype') != 'torch.bfloat16' or item.get('pre_scale_dtype') != 'torch.bfloat16':
            raise ValueError('候选只能使用实际 BF16 amax/pre_scale: ' + name)
        path = (directory / Path(item['file']).name).resolve(strict=True)
        if path.parent != directory or path in files or sha256_file(path) != item['sha256']:
            raise ValueError('packed 文件重复/越界/SHA256 不符: ' + name)
        names.add(name)
        files.add(path)
        rows.append({'name': name, 'path': path, 'shape': shape, 'sha256': item['sha256']})
    if len(rows) != 252 or names != TARGET_NAMES:
        raise ValueError('需要恰好252个冻结投影；不得遗漏/增加保护模块')
    return rows


def verify_reference_asset(reference, row):
    """逐矩阵比较原 buffer；临时 CPU 副本释放后才处理下一矩阵。"""
    if type(reference) is not PackedAWQLinear:
        raise TypeError('目标不是原 PackedAWQLinear: ' + row['name'])
    if [reference.out_features, reference.in_features] != row['shape']:
        raise ValueError('实际模块 shape 不符: ' + row['name'])
    loaded = PackedAWQLinear.load(row['path'], device='cpu')
    try:
        expected = loaded.state_dict()
        actual = reference.state_dict()
        if set(actual) != set(expected):
            raise ValueError('buffer keys 不符: ' + row['name'])
        for name, value in actual.items():
            if value.dtype != expected[name].dtype or not torch.equal(value.detach().cpu(), expected[name]):
                raise ValueError('实际 packed buffer 与冻结资产不等: ' + row['name'] + '.' + name)
    finally:
        del loaded


def install_hybrid_awq(model, packed_manifest, enabled=True):
    if not isinstance(enabled, bool):
        raise TypeError('enabled 须为 bool')
    frozen_source_receipt()
    rows = validate_packed_manifest(packed_manifest)
    # 完整预检后再改模型；不在错误出现前留下部分替换。
    originals = []
    for row in rows:
        reference = model.get_submodule(row['name'])
        verify_reference_asset(reference, row)
        originals.append((row, reference))
    before = sum(reference.storage_bytes() for _, reference in originals)
    for row, reference in originals:
        owner, attribute = row['name'].rsplit('.', 1)
        setattr(model.get_submodule(owner), attribute, HybridAWQLinear.from_reference(reference, enabled))
    after = sum(model.get_submodule(row['name']).storage_bytes() for row in rows)
    if before != after:
        raise RuntimeError('混合包装增加了 packed storage')
    return {'coverage': '252/252', 'enabled': enabled, 'decision_id': DECISION_ID,
            'target_names': sorted(TARGET_NAMES), 'packed_bytes_before': before,
            'packed_bytes_after': after, 'persistent_dense': False}


def dispatch_inventory(model, reset=False):
    modules = {name: dict(module.dispatch) for name, module in model.named_modules()
               if isinstance(module, HybridAWQLinear)}
    totals = {key: sum(row[key] for row in modules.values())
              for key in ('m1_fused', 'm_gt1_reference', 'disabled_reference')}
    if reset:
        for module in model.modules():
            if isinstance(module, HybridAWQLinear): module.reset_dispatch()
    return {'module_count': len(modules), 'totals': totals, 'modules': modules}
