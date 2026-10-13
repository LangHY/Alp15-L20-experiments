"""252 个真实 AWQ 矩阵的逐层 CUDA M1 桥接；任何失败停止候选。"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

from alp15_awq_full_packed import PackedAWQLinear
from alp15_awq_hybrid import (AWQ_RESULT_SHA256, DECISION_ID, HybridAWQLinear, build_packed_manifest,
                              frozen_source_receipt, load_bound_capture, sha256_file, validate_packed_manifest)


def measure(operation, device, repeats, warmup=0):
    stream = torch.cuda.current_stream(device)
    for _ in range(warmup):
        value = operation()
        stream.synchronize()
        del value
    stream.synchronize()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    wall, events = [], []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream.synchronize()
        began = time.perf_counter()
        start.record(stream)
        value = operation()
        stop.record(stream)
        stream.synchronize()
        wall.append((time.perf_counter() - began) * 1000)
        events.append(start.elapsed_time(stop))
        del value
    return {'wall_ms': {'samples': wall, 'median': statistics.median(wall), 'max': max(wall)},
            'cuda_event_ms': {'samples': events, 'median': statistics.median(events), 'max': max(events)},
            'base_allocated_bytes': base, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_extra_allocated_bytes': torch.cuda.max_memory_allocated(device) - base,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'warmup': warmup, 'repeats': repeats}


def tensor_sha256(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_outputs(reference, candidate):
    if not torch.isfinite(reference).all().item() or not torch.isfinite(candidate).all().item():
        raise RuntimeError('M1 输出出现非有限')
    error = (reference.float() - candidate.float()).abs()
    relative = error / reference.float().abs().clamp_min(1e-12)
    flat = error.flatten()
    return {'exact_equal': torch.equal(reference, candidate), 'max_abs_error': error.max().item(),
            'mean_abs_error': error.mean().item(), 'max_relative_error': relative.max().item(),
            'abs_error_p50_p95_p99': torch.quantile(flat, torch.tensor([.5, .95, .99], device=flat.device)).tolist(),
            'rtol': .01, 'atol': .02,
            'outside_tolerance_count': int((error > .02 + .01 * reference.float().abs()).sum().item())}


def load_actual_activations(path, paired_reference=None):
    if path is None:
        return {}, {}, None
    result, hashes, data, reference = load_bound_capture(path, paired_reference)
    for key in ('code_sha256', 'runtime_source_sha256', 'input_sha256', 'actual_import_sha256'):
        hashes.update(data[key])
    data['reference_result_sha256'] = sha256_file(data['reference_result'])
    return result, hashes, data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--awq-artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-decision', choices=(DECISION_ID,), required=True)
    parser.add_argument('--activation-manifest', type=Path)
    parser.add_argument('--paired-reference', type=Path, help='真实capture必须属于已成功的reference输出目录')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261008)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error('warmup 须非负，repeats 须正')
    if args.activation_manifest is not None and args.paired_reference is None:
        parser.error('真实activation需要--paired-reference；random-only省略activation仅作诊断')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'alp15-awq-hybrid252-probe-v1', 'status': 'running', 'decision_id': DECISION_ID,
              'argv': sys.argv, 'matrices': [], 'source_sha256': {}, 'input_sha256': {},
              'limitations': ['CUDA operator evidence only; not trajectory or Orin acceptance',
                             'Each matrix tested separately; equal shapes never stand in for other weights']}
    def save():
        temporary = out / 'result.tmp'
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(out / 'result.json')
    try:
        result['source_sha256'].update(frozen_source_receipt())
        for path in (Path(__file__), Path(__file__).parent / 'alp15_awq_hybrid.py'):
            result['source_sha256'][str(path.resolve())] = sha256_file(path)
        manifest = build_packed_manifest(args.awq_artifact)
        rows = validate_packed_manifest(manifest)
        result['input_sha256'][manifest['original_result']] = manifest['original_result_sha256']
        result['input_sha256'].update({str(r['path']): r['sha256'] for r in rows})
        actual, hashes, activation_metadata = load_actual_activations(args.activation_manifest, args.paired_reference)
        if set(actual) - {row['name'] for row in rows}:
            raise ValueError('真实 activation 超出252边界')
        result['input_sha256'].update(hashes)
        if activation_metadata:
            for row in rows:
                if activation_metadata['input_sha256'].get(str(row['path'])) != row['sha256']:
                    raise ValueError('真实 activation 与当前 packed 资产不配对')
            original = json.loads(Path(manifest['original_result']).read_text())
            if activation_metadata['protected_sha256'] != original['protected_sha256']:
                raise ValueError('真实 activation 保护模块来源不符')
            result['activation_provenance'] = {key: activation_metadata[key] for key in (
                'awq_result_sha256', 'code_sha256', 'runtime_source_sha256', 'input_sha256',
                'actual_import_sha256', 'dynamic_import_sources', 'source_snapshot_directory', 'virtual_import_sources',
                'protected_sha256', 'clip_id', 'text_seed', 'scope')}
            result['activation_binding'] = {'manifest': str(args.activation_manifest.resolve()),
                'manifest_sha256': sha256_file(args.activation_manifest),
                'reference_result': activation_metadata['reference_result'],
                'reference_result_sha256': activation_metadata['reference_result_sha256'],
                'final_receipt': activation_metadata['final_receipt'],
                'final_receipt_sha256': sha256_file(activation_metadata['final_receipt'])}
        device = torch.device(args.device)
        if device.type != 'cuda' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('必须实际 CUDA BF16；CPU/skipped 不算验收')
        result['environment'] = {'python': sys.version, 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'triton': importlib.metadata.version('triton'),
            'gpu': torch.cuda.get_device_name(device), 'capability': list(torch.cuda.get_device_capability(device)),
            'device': str(device), 'gpu_processes_before': gpu_processes()}
        result['actual_activation_coverage'] = f'{len(actual)}/252'
        with torch.cuda.device(device), torch.inference_mode():
            for index, row in enumerate(rows):
                report = {'name': row['name'], 'shape': row['shape'], 'sha256': row['sha256'], 'status': 'running'}
                result['matrices'].append(report)
                save()
                matrix = PackedAWQLinear.load(row['path'], device=device)
                if [matrix.out_features, matrix.in_features] != row['shape']:
                    raise ValueError('实际矩阵 shape 不符')
                hybrid = HybridAWQLinear.from_reference(matrix)
                report['buffers'] = {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                    'sha256': tensor_sha256(value)} for name, value in matrix.named_buffers()}
                report['persistent_bytes_before'] = hybrid.storage_bytes()
                generator = torch.Generator().manual_seed(args.seed + index)
                inputs = [('fixed_random_bf16', torch.randn(1, matrix.in_features, generator=generator).bfloat16())]
                if row['name'] in actual:
                    captured = torch.load(actual[row['name']], map_location='cpu', weights_only=True)
                    if (not isinstance(captured, torch.Tensor) or captured.ndim < 1
                            or captured.shape[-1] != matrix.in_features or captured.numel() != matrix.in_features
                            or captured.dtype != torch.bfloat16 or not captured.is_contiguous() or captured.requires_grad):
                        raise ValueError('真实 decode activation 不支持；不得 cast/contiguous/fallback')
                    inputs.append(('actual_decode_bf16', captured))
                report['activation_note'] = 'actual included' if len(inputs) == 2 else 'No captured decode activation; random only'
                report['activations'] = []
                for kind, cpu_input in inputs:
                    value = cpu_input.to(device)
                    receipt = {'kind': kind, 'shape': list(value.shape), 'dtype': str(value.dtype),
                               'sha256': tensor_sha256(cpu_input), 'seed': args.seed + index if kind.startswith('fixed') else None}
                    report['activations'].append(receipt)
                    # first M1 包括 fused 工厂和 GEMV 首次 JIT；桥接放在首次之后。
                    receipt['reference_first'] = measure(lambda: matrix(value), device, 1)
                    receipt['first_for_this_activation_includes_factory'] = hybrid.fused is None
                    receipt['hybrid_first_for_this_activation'] = measure(lambda: hybrid(value), device, 1)
                    receipt['jit_note'] = 'kernel compilation/cache may be shared by equal shapes; actual input measured after random input'
                    a, b = matrix(value), hybrid(value)
                    receipt['numerics'] = compare_outputs(a, b)
                    save()
                    torch.testing.assert_close(b, a, rtol=.01, atol=.02)
                    del a, b
                    receipt['reference_warm'] = measure(lambda: matrix(value), device, args.repeats, args.warmup)
                    receipt['hybrid_warm'] = measure(lambda: hybrid(value), device, args.repeats, args.warmup)
                    del value
                # 此处只持有当前矩阵两个验证 dense，立即释放，不跨层保留。
                torch.cuda.synchronize(device)
                bridge_base = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                dense = matrix.dequantize(torch.bfloat16)
                bridge = hybrid.fused.dequantize_for_validation()
                report['dequant_bridge'] = {'exact_equal': torch.equal(dense, bridge),
                    'max_abs_error': (dense.float() - bridge.float()).abs().max().item(),
                    'peak_extra_allocated_bytes': torch.cuda.max_memory_allocated(device) - bridge_base}
                del dense, bridge
                if not report['dequant_bridge']['exact_equal']:
                    raise RuntimeError('BF16 dequant bridge 非逐元素一致')
                disabled = HybridAWQLinear.from_reference(matrix, enabled=False)
                value = inputs[0][1].to(device)
                report['disabled_bitwise_equal'] = torch.equal(matrix(value), disabled(value))
                if not report['disabled_bitwise_equal'] or disabled.fused is not None:
                    raise RuntimeError('禁用路径不是原 reference')
                # M>1 逐元素桥接，单矩阵 M2 即可；大M分派由CPU mock覆盖。
                multi = value.expand(2, -1).contiguous()
                report['m2_bitwise_equal'] = torch.equal(matrix(multi), hybrid(multi))
                if not report['m2_bitwise_equal']:
                    raise RuntimeError('M>1 非原 reference')
                report['dispatch'] = dict(hybrid.dispatch)
                report['disabled_dispatch'] = dict(disabled.dispatch)
                report['persistent_bytes_after'] = hybrid.storage_bytes()
                if report['persistent_bytes_after'] != report['persistent_bytes_before']:
                    raise RuntimeError('fused 包装增加持久 storage')
                report['status'] = 'passed'
                del matrix, hybrid, disabled, value, multi, inputs, cpu_input
                if row['name'] in actual: del captured
                gc.collect()
                torch.cuda.empty_cache()
                save()
                print('HYBRID252', index + 1, row['name'], 'passed', flush=True)
        for path, digest in {**result['source_sha256'], **result['input_sha256']}.items():
            if sha256_file(path) != digest: raise RuntimeError('探针期间来源变化: ' + path)
        result['coverage'] = '252/252'
        result['environment']['gpu_processes_after'] = gpu_processes()
        result['status'] = ('passed_all252_cuda_m1_probe_with_actual' if actual
                            else 'passed_all252_random_only_cuda_diagnostic')
        result['exit_code'] = 0
        save()
        return 0
    except Exception as error:
        result.update(status='failed', exit_code=1, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        for row in result['matrices']:
            if row['status'] == 'running': row['status'] = 'failed'
        save()
        print('HYBRID252_FAILED', str(error), flush=True)
        return 1


def gpu_processes():
    import subprocess
    try:
        return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader'], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        return 'unavailable: ' + str(error)


if __name__ == '__main__':
    raise SystemExit(main())
