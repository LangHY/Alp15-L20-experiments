"""既有真实 AWQ packed 矩阵的隔离基线；不加载或改变完整模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

import alp15_awq_full_packed
import alp15_packed_w4a16
from alp15_awq_full_packed import PackedAWQLinear


def load_reference_matrix(path, *, expected_sha256=None):
    """从既有 safetensors 原样重载；不重算 amax、pre_scale 或整数。"""
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError("packed 矩阵 SHA256 与冻结记录不符")
    matrix = PackedAWQLinear.load(path, device="cpu")
    return matrix, {"path": str(path), "sha256": actual_sha,
                    "file_bytes": path.stat().st_size,
                    "shape": [matrix.out_features, matrix.in_features],
                    "persistent_tensor_bytes": matrix.storage_bytes(),
                    "buffers": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                for name, value in matrix.named_buffers()}}


def _summary(samples):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return {"median": statistics.median(ordered),
            "p95": ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower),
            "min": ordered[0], "max": ordered[-1], "samples": samples}


@torch.inference_mode()
def _measure_cuda(operation, *, device, warmup, repeats):
    """wall 包含 Python 和 CUDA 同步，event 单独报告设备时间。"""
    stream = torch.cuda.current_stream(device)
    for _ in range(warmup):
        value = operation()
        stream.synchronize()
        del value
    stream.synchronize()
    base_allocated = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    events, walls = [], []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream.synchronize()
        started = time.perf_counter()
        start.record(stream)
        value = operation()
        stop.record(stream)
        stream.synchronize()
        walls.append((time.perf_counter() - started) * 1000)
        events.append(start.elapsed_time(stop))
        del value
    return {"wall_ms": _summary(walls), "cuda_event_ms": _summary(events),
            "memory": {"base_allocated_bytes": base_allocated, "base_reserved_bytes": base_reserved,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                       "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - base_allocated},
            "warmup": warmup, "repeats": repeats}


@torch.inference_mode()
def benchmark_matrix(matrix, *, device, m_values, dtype, warmup, repeats, seed, candidate=False):
    """真实权重+固定随机激活的算子成本分解，不替代驾驶评测。"""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("需要实际 CUDA 设备；禁止把 CPU 或 skipped 检查当 CUDA 性能验收")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("诊断 dtype 限定 BF16/FP16")
    if warmup < 0 or repeats <= 0 or not m_values or any(m <= 0 for m in m_values):
        raise ValueError("M/repeats 须为正，warmup 须非负")
    with torch.cuda.device(device):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备不支持 BF16")
        matrix = matrix.to(device)
        fused = None
        weight_bridge = None
        if candidate:
            from alp15_awq_kernel import FusedAWQLinear
            fused = FusedAWQLinear.from_reference(matrix)
            reference_weight = matrix.dequantize(torch.bfloat16)
            candidate_weight = fused.dequantize_for_validation()
            weight_bridge = {"exact_equal": torch.equal(reference_weight, candidate_weight),
                             "max_abs_error": (reference_weight.float() - candidate_weight.float()).abs().max().item()}
            del reference_weight, candidate_weight
            if not weight_bridge["exact_equal"]:
                raise RuntimeError("候选 BF16 反量化权重与冻结 reference 不逐字节一致")
        reports = []
        for m in m_values:
            generator = torch.Generator(device="cpu").manual_seed(seed + m)
            cpu_inputs = torch.randn(m, matrix.in_features, generator=generator).to(dtype)
            activation_sha = hashlib.sha256(cpu_inputs.view(torch.uint8).numpy().tobytes()).hexdigest()
            inputs = cpu_inputs.to(device)
            first = _measure_cuda(lambda: matrix(inputs), device=device, warmup=0, repeats=1)
            actual = matrix(inputs)
            if not torch.isfinite(actual).all().item():
                raise RuntimeError("原 reference forward 输出非有限")
            output_shape, output_dtype = list(actual.shape), str(actual.dtype)
            candidate_metrics = None
            if fused is not None:
                # 含 JIT 的首次时间独立报告，后续warmup才可用于速度比较。
                candidate_first = _measure_cuda(lambda: fused(inputs), device=device, warmup=0, repeats=1)
                candidate_output = fused(inputs)
                if not torch.isfinite(candidate_output).all().item():
                    raise RuntimeError("候选输出非有限")
                difference = (candidate_output.float() - actual.float()).abs()
                tolerance = 0.02 + 0.01 * actual.float().abs()
                candidate_metrics = {"first_including_jit": candidate_first,
                                     "numerics": {"exact_equal": torch.equal(candidate_output, actual),
                                                  "max_abs_error": difference.max().item(),
                                                  "mean_abs_error": difference.mean().item(),
                                                  "rtol": 0.01, "atol": 0.02,
                                                  "outside_tolerance_count": int((difference > tolerance).sum().item())}}
                torch.testing.assert_close(candidate_output, actual, rtol=0.01, atol=0.02)
                del candidate_output, difference, tolerance
                candidate_metrics["steady"] = _measure_cuda(lambda: fused(inputs), device=device,
                                                              warmup=warmup, repeats=repeats)
            del actual
            full = _measure_cuda(lambda: matrix(inputs), device=device, warmup=warmup, repeats=repeats)
            scaled_dtype = torch.promote_types(inputs.dtype, matrix.pre_scale.dtype)
            dequant = _measure_cuda(lambda: matrix.dequantize(scaled_dtype),
                                    device=device, warmup=warmup, repeats=repeats)
            prescale = _measure_cuda(lambda: inputs * matrix.pre_scale,
                                     device=device, warmup=warmup, repeats=repeats)
            # 仅此阶段持有单矩阵 dense，报告其额外驻留字节；不修改 packed 模块。
            scaled = inputs * matrix.pre_scale
            dense = matrix.dequantize(scaled.dtype)
            bias = None if matrix.bias is None else matrix.bias.to(scaled.dtype)
            dense_only = _measure_cuda(lambda: F.linear(scaled, dense, bias),
                                       device=device, warmup=warmup, repeats=repeats)
            reports.append({"m": m, "seed": seed + m, "activation_sha256": activation_sha,
                            "input_dtype": str(dtype), "scaled_dtype": str(scaled.dtype),
                            "output_shape": output_shape, "output_dtype": output_dtype,
                            "first_reference_forward": first, "full_reference": full,
                            "dequant_only": dequant, "prescale_only": prescale,
                            "dense_linear_only": dense_only,
                            "dense_linear_diagnostic_resident_bytes": dense.numel() * dense.element_size(),
                            "candidate": candidate_metrics})
            del dense, bias, scaled, inputs, cpu_inputs
        properties = torch.cuda.get_device_properties(device)
        return {"status": "completed_candidate_cuda_operator_probe" if candidate else "completed_reference_cuda_operator_probe",
                "reports": reports, "candidate_weight_bridge": weight_bridge,
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                                "gpu": properties.name, "capability": list(torch.cuda.get_device_capability(device)),
                                "gpu_total_bytes": properties.total_memory, "device": str(device),
                                "stream": "torch.cuda.current_stream(device)"},
                "limits": ["isolated operator with fixed random activations; not trajectory quality",
                           "GPU operator results are not Jetson16GB end-to-end latency or unified memory",
                           "dense_linear_only holds one dense matrix strictly for cost decomposition",
                           "stages are separate measurements; times and memory peaks must not be added",
                           "reference retains original unpack nibble validation and host synchronization"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--candidate", action="store_true", help="显式运行 Triton BF16 候选及数值门禁")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--m", type=int, nargs="+", default=[1, 17, 256])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    matrix, receipt = load_reference_matrix(args.matrix, expected_sha256=args.sha256)
    result = {"schema": "alp15-awq-kernel-probe-v1", "matrix": receipt,
              "started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
              "status": "passed_reference_format_inspection_only",
              "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), Path(alp15_awq_full_packed.__file__),
                                             Path(alp15_packed_w4a16.__file__))}}
    if not args.inspect_only:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        result.update(benchmark_matrix(matrix, device=args.device, m_values=args.m, dtype=dtype,
                                       warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                                       candidate=args.candidate))
        if args.candidate:
            import alp15_awq_kernel
            candidate_source = Path(alp15_awq_kernel.__file__).resolve()
            result["source_sha256"][str(candidate_source)] = hashlib.sha256(candidate_source.read_bytes()).hexdigest()
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
