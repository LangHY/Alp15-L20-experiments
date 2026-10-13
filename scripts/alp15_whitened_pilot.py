"""全协方差白化与闭式重建混合 pilot；不是 SVD-LLM 官方仓库复现。"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import alp15_lowrank_pilot as base


LAYERS = (6, 35)
RANKS = (896, 1024)
METHODS = ("whitening", "whitening_refit_parent", "whitening_refit_compressed")
REGULARIZATION_RATIO = 1e-4
BASE_CODE_SHA256 = "c2975113b04a0a5e295ab14f0ba09db4152568fdb751ec957b45c87b3693ad69"


def regularized_cholesky(gram):
    """固定 lambda=1e-4*trace(G)/d；非有限或 Cholesky 失败直接终止。"""
    import torch
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or len(gram) == 0:
        raise ValueError("Gram 必须是非空方阵")
    base._finite(gram, "Gram")
    gram = gram.double()
    if not torch.allclose(gram, gram.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Gram 不对称")
    regularizer = REGULARIZATION_RATIO * gram.trace() / len(gram)
    if not torch.isfinite(regularizer).item() or regularizer.item() <= 0:
        raise ValueError("Gram 正则量必须有限且为正")
    lower = torch.linalg.cholesky(gram + regularizer * torch.eye(len(gram), dtype=torch.float64, device=gram.device))
    base._finite(lower, "Cholesky L")
    return lower, regularizer.item()


def whitened_svd(weight, lower, driver=None):
    """FP32 分解 W@L；rank 共用该分解，不构造 inverse。"""
    if weight.ndim != 2 or lower.ndim != 2 or tuple(lower.shape) != (weight.shape[1], weight.shape[1]):
        raise ValueError("W/L 尺寸不符")
    base._finite(weight, "W")
    base._finite(lower, "L")
    matrix = weight.float() @ lower.float()
    u, singular, vh, _ = base.compute_svd(matrix, driver=driver)
    return u, singular, vh, lower


def balanced_factors(decomposition, rank, dtype=None):
    """B=U sqrt(S)，A=sqrt(S) Vh L^-1，仅以三角求解实现。"""
    import torch
    u, singular, vh, lower = decomposition
    base.positive_integer(rank, "rank")
    if rank > len(singular):
        raise ValueError("rank 超出矩阵最大秩")
    root = singular[:rank].sqrt()
    b = (u[:, :rank] * root.unsqueeze(0)).contiguous()
    unscaled = torch.linalg.solve_triangular(lower.T, vh[:rank].double().T, upper=True).T
    a = (root.double().unsqueeze(1) * unscaled).float().contiguous()
    if dtype is not None:
        b, a = b.to(dtype), a.to(dtype)
    base._finite(b, "白化 B")
    base._finite(a, "白化 A")
    return b, a


def ridge_closed_form(ztz_sum, zty_sum, tokens):
    """闭式 B.T=(Z.T Z/N + mu I)^-1 Z.T Y/N；使用 solve，mu 固定。"""
    import torch
    base.positive_integer(tokens, "tokens")
    if ztz_sum.ndim != 2 or ztz_sum.shape[0] != ztz_sum.shape[1] or len(ztz_sum) == 0:
        raise ValueError("ZTZ 尺寸非法")
    if zty_sum.ndim != 2 or zty_sum.shape[0] != len(ztz_sum):
        raise ValueError("ZTY 尺寸非法")
    base._finite(ztz_sum, "ZTZ")
    base._finite(zty_sum, "ZTY")
    gram, cross = ztz_sum.double() / tokens, zty_sum.double() / tokens
    mu = REGULARIZATION_RATIO * gram.trace() / len(gram)
    if not torch.isfinite(mu).item() or mu.item() <= 0:
        raise ValueError("闭式重建正则必须为正有限值")
    solution = torch.linalg.solve(gram + mu * torch.eye(len(gram), dtype=torch.float64, device=gram.device), cross)
    base._finite(solution, "闭式重建解")
    return solution, mu.item()


def _fit_grams(x, fit_indices, weights):
    import torch
    hidden, width = weights["gate_proj"].shape[1], weights["gate_proj"].shape[0]
    input_sum = torch.zeros((hidden, hidden), dtype=torch.float64, device="cuda")
    feature_sum = torch.zeros((width, width), dtype=torch.float64, device="cuda")
    feature_square_sum = torch.zeros(width, dtype=torch.float64, device="cuda")
    for inputs in base._chunks(x, fit_indices):
        features = base.ffn_features(inputs, weights)
        base._finite(features, "父 gated features")
        xd, fd = inputs.double(), features.double()
        input_sum += xd.T @ xd
        feature_sum += fd.T @ fd
        # 历史通道评分继续使用 FP32 平方/求和、FP64 累积。
        feature_square_sum += features.float().square().sum(0).double()
    tokens = len(fit_indices)
    input_lower, input_lambda = regularized_cholesky(input_sum / tokens)
    feature_lower, feature_lambda = regularized_cholesky(feature_sum / tokens)
    scores = base.channel_scores(feature_square_sum / tokens, weights["down_proj"])
    return input_lower, feature_lower, scores, {"gate_up_lambda": input_lambda, "down_lambda": feature_lambda}


def _refit_down(x, fit_indices, weights, factors, use_compressed):
    import torch
    from torch.nn import functional as F
    down_a = factors["down_proj"][1]
    rank, hidden = down_a.shape[0], weights["down_proj"].shape[0]
    ztz = torch.zeros((rank, rank), dtype=torch.float64, device="cuda")
    zty = torch.zeros((rank, hidden), dtype=torch.float64, device="cuda")
    for inputs in base._chunks(x, fit_indices):
        if use_compressed:
            features = F.silu(base.factor_linear(inputs, factors["gate_proj"])) * base.factor_linear(inputs, factors["up_proj"])
        else:
            features = base.ffn_features(inputs, weights)
        # Z 使用真实 BF16 因子线性；Y 始终是 BF16 父完整 FFN 输出。
        z = F.linear(features, down_a)
        y = base.dense_ffn(inputs, weights)
        base._finite(z, "闭式拟合 Z")
        base._finite(y, "闭式拟合父 Y")
        zd, yd = z.double(), y.double()
        ztz += zd.T @ zd
        zty += zd.T @ yd
    solution, mu = ridge_closed_form(ztz, zty, len(fit_indices))
    b = solution.T.contiguous().to(torch.bfloat16)
    base._finite(b, "闭式 BF16 B")
    return b, mu


def _plot(output, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return [f"matplotlib unavailable; JSON/CSV preserved; no dependency installed: {exc}"]
    for layer in LAYERS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for method in ("linked_channel", *METHODS):
            selected = sorted((row for row in rows if row["layer"] == layer and row["method"] == method), key=lambda row: row["rank"])
            for axis, metric in zip(axes, ("train_nmse", "calibration_eval_nmse")):
                axis.plot([row["parameter_ratio"] for row in selected], [row[metric] for row in selected], ".-", label=method)
                axis.set_xlabel("FFN parameter ratio")
                axis.set_ylabel(metric)
                axis.grid(alpha=.3)
                axis.legend(fontsize=8)
        fig.suptitle(f"Layer {layer}: fixed full-covariance whitening + ridge refit pilot")
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(output / f"layer_{layer:02d}_curves.{suffix}", dpi=160)
        plt.close(fig)
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-fit-tokens", type=int, default=8192)
    parser.add_argument("--svd-driver", choices=("gesvdj", "gesvd"), default="gesvdj")
    args = parser.parse_args()
    base.positive_integer(args.max_fit_tokens, "max-fit-tokens")
    root, output = args.root.resolve(strict=True), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    manifest = {"status": "running", "argv": sys.argv, "started_unix": time.time(),
        "layers": LAYERS, "ranks": RANKS, "seed": base.SEED, "max_fit_tokens": args.max_fit_tokens,
        "svd_driver": args.svd_driver, "code_sha256": base.sha256_file(Path(__file__)),
        "phase1_code_sha256": base.sha256_file(Path(base.__file__)),
        "algorithm": "Hybrid full covariance regularized whitening plus optional ridge down-B refit; not official SVD-LLM repository reproduction",
        "gram_policy": "FP64 XTX/N; lambda=1e-4*trace(G)/d; gate/up share X Gram; down Gram uses BF16 parent full gated features",
        "factor_policy": "FP32 SVD W@L; balanced U*sqrt(S), sqrt(S)*Vh*L^-1 by triangular solve; BF16 real two-linear projections",
        "refit_policy": "Fixed BF16 down-A; FP64 normal equations; mu=1e-4*trace(ZTZ/N)/rank; BF16 B; Y always parent BF16 full FFN output",
        "metric_target": "Parent full BF16 FFN output; all candidate evaluations use compressed gate/up complete FFN",
        "channel_score_formula": "Historical mean(feature^2)*squared down-column L2 norm; FP32 square/sum, FP64 accumulation",
        "method_selection_rule": "Lowest all-fit train NMSE among whitening, whitening_refit_parent, whitening_refit_compressed; fixed listed order breaks ties; linked_channel comparator only; calibration_eval never participates",
        "limitations": ["Local FFN proxy, not trajectory ADE, 6B success or Jetson acceptance",
                         "Both 64 fit and 16 calibration_eval are prior training clips; no independent test claim"],
        "input_sha256": {}, "results_completed": 0}
    base._persist(output, rows, manifest)
    weights = factors = fitted = pruned = decompositions = input_lower = feature_lower = None
    scores = ranking = prediction = target = None
    try:
        import torch
        if manifest["phase1_code_sha256"] != BASE_CODE_SHA256:
            raise ValueError("阶段1模块不再是冻结版本")
        if not torch.cuda.is_available() or "L20" not in torch.cuda.get_device_name():
            raise RuntimeError("仅允许在已授权 L20 CUDA 环境执行 pilot")
        torch.set_num_threads(8)
        torch.set_grad_enabled(False)
        torch.manual_seed(base.SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        manifest["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "python": sys.version, "platform": platform.platform(), "cpu_threads": torch.get_num_threads()}
        records, fit_clips, eval_clips, hashes, provenance = base.preflight(root)
        manifest.update({"input_sha256": hashes, "parent_MLP_SHA256": provenance["parent_MLP_SHA256"],
            "fit_clips": fit_clips, "calibration_eval_clips": eval_clips})
        base._persist(output, rows, manifest)
        for layer in LAYERS:
            layer_start = time.perf_counter()
            path = root / base.PROFILE / "cache" / f"train_{layer:02d}.pt"
            hashes[str(path)] = base.sha256_file(path)
            cache = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            if set(cache) != {"x", "y"} or cache["x"].ndim != 2 or cache["x"].shape != cache["y"].shape:
                raise ValueError("profile cache 必须为同形二维 x/y")
            x, y = cache["x"], cache["y"]
            if x.shape[1] != 4096 or x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16:
                raise ValueError("cache hidden/dtype 不符")
            base._finite(x, "cache x")
            base._finite(y, "cache y")
            boundaries = base.clip_boundaries(records, len(x))
            fit_indices, eval_indices = base._row_indices(boundaries, fit_clips), base._row_indices(boundaries, eval_clips)
            generator = torch.Generator().manual_seed(base.SEED)
            sampled = fit_indices[torch.randperm(len(fit_indices), generator=generator)[:args.max_fit_tokens]]
            manifest.setdefault("layer_inputs", {})[str(layer)] = {"boundaries": boundaries,
                "fit_tokens": len(fit_indices), "calibration_eval_tokens": len(eval_indices),
                "fit_statistics_tokens": len(sampled), "fit_statistics_row_indices": sampled.tolist()}
            weights = {name: tensor.cuda() for name, tensor in base._load_weights(root / base.PARENT / "width_5376/distilled" / f"mlp_{layer:02d}.safetensors").items()}
            torch.cuda.reset_peak_memory_stats()
            error, energy = 0., 0.
            for start in range(0, len(x), 64):
                target = y[start:start + 64].cuda().float()
                prediction = base.dense_ffn(x[start:start + 64].cuda(), weights).float()
                base._finite(prediction, "父 cache 复现")
                error += (prediction - target).double().square().sum().item()
                energy += target.double().square().sum().item()
            if energy <= 0 or error / energy > 1e-3:
                raise ValueError(f"父 cache 复现失败: layer={layer}, nmse={error / max(energy, 1e-300)}")
            manifest["layer_inputs"][str(layer)]["parent_cache_nmse"] = error / energy
            stats_start = time.perf_counter()
            input_lower, feature_lower, scores, regularizers = _fit_grams(x, sampled, weights)
            decompositions = {name: whitened_svd(weights[name], feature_lower if name == "down_proj" else input_lower, args.svd_driver) for name in base.PROJECTIONS}
            torch.cuda.synchronize()
            shared_seconds = time.perf_counter() - stats_start
            shared_peak, shared_reserved_peak = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            manifest["layer_inputs"][str(layer)].update({**regularizers, "shared_gram_svd_seconds": shared_seconds,
                "shared_peak_allocated_bytes": shared_peak, "shared_peak_reserved_bytes": shared_reserved_peak})
            ranking = torch.argsort(scores, descending=True, stable=True)
            dense_params = sum(tensor.numel() for tensor in weights.values())

            def record(method, rank, candidate, params, width, started, mu=None):
                train = base._nmse(x, fit_indices, weights, candidate)
                by_clip = {row["clip_id"]: row for row in boundaries}
                per_clip = []
                for clip in eval_clips:
                    boundary = by_clip[clip]
                    metric = base._nmse(x, torch.arange(boundary["start"], boundary["stop"]), weights, candidate)
                    per_clip.append({"clip_id": clip, **metric})
                eval_nmse = sum(row["squared_error"] for row in per_clip) / sum(row["target_energy"] for row in per_clip)
                filename = f"layer_{layer:02d}_{method}_rank_{rank}_calibration_eval.json"
                base._write_json(output / filename, per_clip)
                manifest.setdefault("per_clip_outputs", {})[filename] = base.sha256_file(output / filename)
                torch.cuda.synchronize()
                rows.append({"layer": layer, "method": method, "rank": rank, "channel_width": width,
                    "actual_params": params, "parent_params": dense_params, "parameter_ratio": params / dense_params,
                    "train_nmse": train["nmse"], "calibration_eval_nmse": eval_nmse,
                    "train_tokens": len(fit_indices), "calibration_eval_tokens": len(eval_indices),
                    "fit_statistics_tokens": len(sampled), "bf16_finite": True, "ridge_mu": mu,
                    "elapsed_seconds": time.perf_counter() - started, "shared_gram_svd_seconds": shared_seconds,
                    "per_clip_file": filename,
                    "peak_allocated_bytes": max(shared_peak, torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": max(shared_reserved_peak, torch.cuda.max_memory_reserved())})
                manifest["results_completed"] = len(rows)
                base._persist(output, rows, manifest)
                print("WHITENED_PILOT_RESULT", json.dumps(rows[-1], allow_nan=False), flush=True)

            record("parent", 0, lambda inputs: base.dense_ffn(inputs, weights), dense_params, 5376, layer_start)
            for rank in RANKS:
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                width = base.matched_channel_width(4096, 5376, rank)
                ids = ranking[:width].sort().values
                pruned = {"gate_proj": weights["gate_proj"].index_select(0, ids), "up_proj": weights["up_proj"].index_select(0, ids), "down_proj": weights["down_proj"].index_select(1, ids)}
                params = sum(tensor.numel() for tensor in pruned.values())
                if params > base.lowrank_parameter_count(4096, 5376, rank):
                    raise ValueError("通道基线超出参数预算")
                record("linked_channel", rank, lambda inputs: base.dense_ffn(inputs, pruned), params, width, started)
                pruned = None
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                factors = {name: balanced_factors(decompositions[name], rank, torch.bfloat16) for name in base.PROJECTIONS}
                params = base.verify_factor_parameter_count(factors, 4096, 5376, rank)
                record("whitening", rank, lambda inputs: base.factor_ffn(inputs, factors), params, None, started)
                for method, compressed in (("whitening_refit_parent", False), ("whitening_refit_compressed", True)):
                    started = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    b, mu = _refit_down(x, sampled, weights, factors, compressed)
                    fitted = {**factors, "down_proj": (b, factors["down_proj"][1])}
                    refit_params = base.verify_factor_parameter_count(fitted, 4096, 5376, rank)
                    record(method, rank, lambda inputs: base.factor_ffn(inputs, fitted), refit_params, None, started, mu)
                    fitted = b = None
                factors = None
                gc.collect()
                torch.cuda.empty_cache()
            selections = {}
            for rank in RANKS:
                compared = {row["method"]: row for row in rows if row["layer"] == layer and row["rank"] == rank}
                selected = min((compared[method] for method in METHODS), key=lambda row: (row["train_nmse"], METHODS.index(row["method"])))
                selections[str(rank)] = {"selected_method": selected["method"],
                    "fit_nmse_by_method": {method: row["train_nmse"] for method, row in compared.items()},
                    "selected_calibration_eval_nmse": selected["calibration_eval_nmse"],
                    "linked_channel_comparator_calibration_eval_nmse": compared["linked_channel"]["calibration_eval_nmse"]}
            manifest.setdefault("fit_selection", {})[str(layer)] = selections
            base._write_json(output / "method_selection.json", {"rule": manifest["method_selection_rule"], "layers": manifest["fit_selection"]})
            manifest["method_selection_sha256"] = base.sha256_file(output / "method_selection.json")
            manifest["layer_inputs"][str(layer)]["elapsed_seconds"] = time.perf_counter() - layer_start
            weights = decompositions = input_lower = feature_lower = scores = ranking = prediction = target = None
            del cache, x, y
            gc.collect()
            torch.cuda.empty_cache()
            base._persist(output, rows, manifest)
        manifest["limitations"].extend(_plot(output, rows))
        manifest.update({"status": "completed", "ended_unix": time.time()})
        base._persist(output, rows, manifest)
        manifest["artifact_sha256"] = {path.name: base.sha256_file(path) for path in output.iterdir()
            if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".tmp")}
        base._write_json(output / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update({"status": "failed", "ended_unix": time.time(), "error": {"type": type(exc).__name__, "message": str(exc)}})
        base._persist(output, rows, manifest)
        raise
    finally:
        weights = factors = fitted = pruned = decompositions = input_lower = feature_lower = None
        scores = ranking = prediction = target = None
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
