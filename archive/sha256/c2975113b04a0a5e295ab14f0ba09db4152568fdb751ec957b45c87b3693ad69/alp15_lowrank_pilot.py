"""L20 探索性 FFN 低秩校准；不训练、不导出模型、不代表 Jetson 验收。"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path


PARENT = "experiments/mlp_newdata8_recovery_20261002_165238"
PROFILE = "experiments/recovered8_importance_20261002_175834"
# 已完整审阅的脚本按文件名字典序、每 clip positions 选行后 torch.cat。
PROFILE_SOURCE_SHA256 = "904f2553a2ee9167f40808cc9a977875688f178d2d0e711b21490ce49c5a01c9"
SEED = 20261007
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
FROZEN_REFERENCES = {
    "source_commit": "36aeb4c5938cbc2eb2aed33b22434773da4ab639",
    "checkpoint_revision": "7aba8293c09993f2e125c6819df05d7fa3e873ea",
    "dataset_revision": "33f9bf447ed3bcb7d545ce13f4226f824214fafb",
    "parent_MLP_SHA256": "be6239cbbd8c6cfd01317a33233486c1a422469ee0c323bdfe37767c75d0ab27",
}


def positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return value


def lowrank_parameter_count(hidden: int, width: int, rank: int) -> int:
    for name, value in (("hidden", hidden), ("width", width), ("rank", rank)):
        positive_integer(value, name)
    if rank > min(hidden, width):
        raise ValueError("rank 大于矩阵的最大秩")
    return 3 * rank * (hidden + width)


def matched_channel_width(hidden: int, width: int, rank: int) -> int:
    budget = lowrank_parameter_count(hidden, width, rank)
    result = min(width, budget // (3 * hidden))
    if result < 1:
        raise ValueError("预算不足以保留一个通道")
    return result


def verify_factor_parameter_count(factors: dict, hidden: int, width: int, rank: int) -> int:
    """只接受三组独立因子，不允许隐含 dense base 或伪报参数。"""
    if set(factors) != set(PROJECTIONS):
        raise ValueError("必须且只能含 gate/up/down 三组因子")
    total = 0
    for name in PROJECTIONS:
        pair = factors[name]
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("因子必须为 (B, A)，不能附加 dense base")
        b, a = pair
        out_dim, in_dim = (hidden, width) if name == "down_proj" else (width, hidden)
        if tuple(b.shape) != (out_dim, rank) or tuple(a.shape) != (rank, in_dim):
            raise ValueError(f"{name} 因子尺寸不符")
        total += b.numel() + a.numel()
    expected = lowrank_parameter_count(hidden, width, rank)
    if total != expected:
        raise ValueError("实际因子元素数与参数公式不一致")
    return total


def clip_boundaries(records: list[dict], total_rows: int) -> list[dict]:
    positive_integer(total_rows, "total_rows")
    if not records or len({r["clip_id"] for r in records}) != len(records):
        raise ValueError("clip 为空或重复")
    cursor, result = 0, []
    for record in records:
        count = positive_integer(record["rows"], "clip rows")
        result.append({**record, "start": cursor, "stop": cursor + count})
        cursor += count
    if cursor != total_rows:
        raise ValueError(f"positions 行数之和 {cursor} 与 cache 行数 {total_rows} 不一致")
    return result


def split_clips(clip_ids: list[str], training: list[str], excluded: list[str],
                seed: int = SEED, fit_count: int = 64) -> tuple[list[str], list[str]]:
    if len(clip_ids) != 80 or len(set(clip_ids)) != 80:
        raise ValueError("必须是互不重复的 80 个训练 clip")
    if len(training) != 80 or len(set(training)) != 80 or set(clip_ids) != set(training):
        raise ValueError("缓存 clips 与父 training split 不一致")
    if set(clip_ids) & set(excluded):
        raise ValueError("训练缓存与 validation/monitor/holdout 重叠")
    if fit_count != 64:
        raise ValueError("预注册划分必须为 64 fit / 16 calibration_eval")
    shuffled = sorted(clip_ids)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:fit_count], shuffled[fit_count:]


def _finite(tensor, name: str) -> None:
    import torch
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} 包含非有限值")


def activation_scale(inputs, minimum: float = 1e-6):
    """ASVD 对角缩放：拟合输入 mean(abs(x)) 的平方根。"""
    if inputs.ndim != 2 or inputs.shape[0] == 0 or not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("缩放输入或 clamp 非法")
    _finite(inputs, "拟合输入")
    return inputs.float().abs().mean(dim=0).sqrt().clamp_min(minimum)


def compute_svd(weight, scale=None, driver: str | None = None):
    """每个投影每种方法仅分解一次，所有 rank 共用 economy SVD。"""
    import torch
    if weight.ndim != 2 or min(weight.shape) < 1:
        raise ValueError("weight 必须为非空二维矩阵")
    _finite(weight, "weight")
    matrix = weight.float()
    if scale is None:
        scale = torch.ones(weight.shape[1], device=weight.device, dtype=torch.float32)
    if scale.ndim != 1 or scale.shape[0] != weight.shape[1]:
        raise ValueError("scale 尺寸不符")
    _finite(scale, "scale")
    if not (scale > 0).all().item():
        raise ValueError("scale 必须严格为正")
    matrix = matrix * scale.float().unsqueeze(0)
    _finite(matrix, "加权矩阵")
    kwargs = {"full_matrices": False}
    if driver is not None:
        if not weight.is_cuda or driver not in ("gesvdj", "gesvd"):
            raise ValueError("driver 仅支持 CUDA gesvdj/gesvd")
        kwargs["driver"] = driver
    u, s, vh = torch.linalg.svd(matrix, **kwargs)
    for name, value in (("U", u), ("S", s), ("Vh", vh)):
        _finite(value, name)
    return u, s, vh, scale.float()


def factors_from_svd(decomposition, rank: int, dtype=None):
    u, s, vh, scale = decomposition
    positive_integer(rank, "rank")
    if rank > len(s):
        raise ValueError("rank 超出分解尺寸")
    b = (u[:, :rank] * s[:rank].unsqueeze(0)).contiguous()
    a = (vh[:rank] / scale.unsqueeze(0)).contiguous()
    if dtype is not None:
        b, a = b.to(dtype), a.to(dtype)
    _finite(b, "B 因子")
    _finite(a, "A 因子")
    return b, a


def factor_linear(inputs, factors):
    from torch.nn import functional as F
    b, a = factors
    return F.linear(F.linear(inputs, a), b)


def channel_scores(feature_square_mean, down_weight):
    """历史联动评分：激活二阶矩乘 down 列平方范数，与 ASVD 缩放分开。"""
    if feature_square_mean.ndim != 1 or down_weight.ndim != 2 or down_weight.shape[1] != len(feature_square_mean):
        raise ValueError("通道二阶矩与 down 尺寸不符")
    _finite(feature_square_mean, "激活二阶矩")
    _finite(down_weight, "down 权重")
    if not (feature_square_mean >= 0).all().item():
        raise ValueError("激活二阶矩不能为负")
    scores = feature_square_mean.double() * down_weight.float().square().sum(dim=0).double()
    _finite(scores, "历史联动剪枝分数")
    return scores


def ffn_features(inputs, weights):
    from torch.nn import functional as F
    return F.silu(F.linear(inputs, weights["gate_proj"])) * F.linear(inputs, weights["up_proj"])


def dense_ffn(inputs, weights):
    from torch.nn import functional as F
    return F.linear(ffn_features(inputs, weights), weights["down_proj"])


def factor_ffn(inputs, factors):
    from torch.nn import functional as F
    return factor_linear(F.silu(factor_linear(inputs, factors["gate_proj"])) *
                         factor_linear(inputs, factors["up_proj"]), factors["down_proj"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    # 原子替换保证中断后上一份进度仍能读取；只写新实验目录。
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_hash(path: Path, expected: str, hashes: dict) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"哈希不符: {path}")
    hashes[str(path)] = actual


def _load_weights(path: Path):
    import torch
    from safetensors.torch import load_file
    state = load_file(str(path), device="cpu")
    if set(state) != {name + ".weight" for name in PROJECTIONS}:
        raise ValueError(f"父 overlay 必须且只能含三投影: {path}")
    weights = {name: state[name + ".weight"] for name in PROJECTIONS}
    gate, up, down = (weights[name] for name in PROJECTIONS)
    if gate.ndim != 2 or up.shape != gate.shape or down.shape != gate.T.shape:
        raise ValueError(f"父三投影尺寸不符: {path}")
    if gate.shape[1] != 4096 or gate.shape[0] != 5376:
        raise ValueError(f"当前 pilot 限定父 hidden4096 / width5376: {path}")
    for name, tensor in weights.items():
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"父权重必须为 BF16: {path}: {name}")
        _finite(tensor, str(path) + name)
    return weights


def preflight(root: Path):
    import torch
    parent, profile = root / PARENT, root / PROFILE
    provenance = _json(profile / "provenance.json")
    split = _json(parent / "optimization_split.json")
    reference_split = _json(parent / "split.json")
    parent_prov = _json(parent / "provenance.json")
    selected = _json(parent / "selected_task_validation.json")
    recovery = _json(parent / "controlled_recovery_result.json")
    reload = _json(parent / "controlled_reload_check.json")
    importance = _json(profile / "importance.json")
    hashes = {}
    for key, expected in FROZEN_REFERENCES.items():
        if provenance.get(key) != expected:
            raise ValueError(f"profile 与预注册冻结来源 {key} 不一致")
    if Path(provenance["parent"]).resolve() != parent:
        raise ValueError("profile parent 路径不符")
    if Path(recovery["overlay_directory"]).resolve() != parent / "width_5376/distilled":
        raise ValueError("父 selected overlay 路径不符")
    if reload.get("status") != "completed" or importance.get("status") != "completed":
        raise ValueError("父独立 reload 或 profile 未完成")
    if len(importance.get("curves", [])) != 36 or len(importance.get("layers", [])) != 36:
        raise ValueError("profile 未包含完整 36 层测量")
    if provenance.get("monitor_used") is not False or provenance.get("holdout_used") is not False:
        raise ValueError("profile 使用了 monitor/holdout 或缺少明确排除证据")
    for key in ("source_commit", "checkpoint_revision", "dataset_revision"):
        if provenance[key] != parent_prov[key]:
            raise ValueError(f"profile 与父来源 {key} 不符")
    if provenance["parent_MLP_SHA256"] != selected["mlp_tensor_sha256"]:
        raise ValueError("profile 与 selected 父 MLP 哈希不符")
    source = profile / "code/profile_recovered8_pruning.py"
    if provenance["source_sha256"] != PROFILE_SOURCE_SHA256:
        raise ValueError("profile 源版本未被审阅；不能推断 clip 拼接逻辑")
    _verify_hash(source, PROFILE_SOURCE_SHA256, hashes)
    _verify_hash(parent / "optimization_split.json", provenance["parent_split_sha256"], hashes)
    manifest_path = parent / "width_5376/distilled/manifest.json"
    _verify_hash(manifest_path, provenance["parent_overlay_manifest_sha256"], hashes)
    overlay_manifest = _json(manifest_path)
    if overlay_manifest["base_checkpoint_revision"] != provenance["checkpoint_revision"]:
        raise ValueError("overlay checkpoint revision 不符")
    # 顺序与既有 mlp_tensor_digest 完全一致；逐文件读取，不实例化整模型。
    digest = hashlib.sha256()
    overlay_records = {record["layer"]: record for record in overlay_manifest["layers"]}
    if set(overlay_records) != set(range(36)):
        raise ValueError("overlay 必须记录完整 36 层")
    for layer in range(36):
        path = parent / "width_5376/distilled" / f"mlp_{layer:02d}.safetensors"
        _verify_hash(path, overlay_records[layer]["sha256"], hashes)
        weights = _load_weights(path)
        for name in PROJECTIONS:
            tensor = weights[name]
            digest.update(f"{layer}.{name}".encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        del weights, tensor
    if digest.hexdigest() != provenance["parent_MLP_SHA256"]:
        raise ValueError("当前父模型实际张量哈希与 profile/selected 不一致")
    teacher_paths = sorted((parent / "teacher_cache").glob("calibration_*.pt"))
    expected_teacher = provenance["teacher_input_cache_sha256"]
    if {str(path) for path in teacher_paths} != set(expected_teacher):
        raise ValueError("当前 teacher cache 文件集合与 profile 不一致")
    records = []
    for path in teacher_paths:
        _verify_hash(path, expected_teacher[str(path)], hashes)
        example = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        positions = example["positions"]
        inputs = example["inputs"]["input_ids"]
        if positions.ndim != 1 or positions.dtype != torch.int64 or len(positions) == 0:
            raise ValueError(f"positions 必须为非空 int64 一维: {path}")
        if inputs.ndim != 2 or inputs.shape[0] != 1:
            raise ValueError(f"teacher 输入必须为单 clip batch: {path}")
        if positions.min().item() < 0 or positions.max().item() >= inputs.shape[1]:
            raise ValueError(f"positions 越界: {path}")
        if len(torch.unique(positions)) != len(positions):
            raise ValueError(f"positions 重复: {path}")
        records.append({"clip_id": example["clip_id"], "rows": len(positions),
                        "teacher_file": str(path), "input_sequence_length": inputs.shape[1]})
        del example, inputs, positions
    training = split["training"]
    if sorted(training) != sorted(provenance["training_clips"]):
        raise ValueError("profile training_clips 与父 split 不一致")
    excluded = [clip for role, values in split.items() if role != "training"
                and isinstance(values, list) for clip in values if isinstance(clip, str)]
    excluded += reference_split["validation"] + reference_split["holdout"]
    excluded += [row["clip_id"] for row in selected["records"]]
    fit, evaluation = split_clips([r["clip_id"] for r in records], training, excluded)
    for path in (profile / "provenance.json", profile / "importance.json", parent / "provenance.json",
                 parent / "split.json",
                 parent / "selected_task_validation.json", parent / "controlled_recovery_result.json",
                 parent / "controlled_reload_check.json"):
        hashes[str(path)] = sha256_file(path)
    return records, fit, evaluation, hashes, provenance


def _row_indices(boundaries: list[dict], clip_ids: list[str]):
    import torch
    by_clip = {row["clip_id"]: row for row in boundaries}
    return torch.cat([torch.arange(by_clip[clip]["start"], by_clip[clip]["stop"]) for clip in clip_ids])


def _chunks(x, indices, batch: int = 64):
    import torch
    for start in range(0, len(indices), batch):
        yield x.index_select(0, indices[start:start + batch]).to(device="cuda", dtype=torch.bfloat16)


def _fit_statistics(x, indices, weights):
    import torch
    hidden, width = weights["gate_proj"].shape[1], weights["gate_proj"].shape[0]
    x_abs = torch.zeros(hidden, device="cuda", dtype=torch.float64)
    feature_abs = torch.zeros(width, device="cuda", dtype=torch.float64)
    feature_square = torch.zeros(width, device="cuda", dtype=torch.float64)
    for inputs in _chunks(x, indices):
        features = ffn_features(inputs, weights)
        _finite(features, "父 FFN 拟合激活")
        x_abs += inputs.double().abs().sum(0)
        feature_abs += features.double().abs().sum(0)
        # 与既有 compute_activation_scores 保持相同的 FP32 平方/求和、FP64 累加。
        feature_square += features.float().square().sum(0).double()
    x_mean, feature_mean = (x_abs / len(indices)).float(), (feature_abs / len(indices)).float()
    scores = channel_scores(feature_square / len(indices), weights["down_proj"])
    _finite(scores, "联动剪枝分数")
    return x_mean.sqrt().clamp_min(1e-6), feature_mean.sqrt().clamp_min(1e-6), scores


def _nmse(x, indices, weights, candidate):
    error, energy = 0.0, 0.0
    for inputs in _chunks(x, indices):
        target, prediction = dense_ffn(inputs, weights), candidate(inputs)
        _finite(target, "父 FFN 输出")
        _finite(prediction, "BF16 候选输出")
        error += (prediction.float() - target.float()).double().square().sum().item()
        energy += target.double().square().sum().item()
    if energy <= 0 or not math.isfinite(energy) or not math.isfinite(error):
        raise ValueError("NMSE 分母为零或非有限")
    return {"nmse": error / energy, "squared_error": error, "target_energy": energy,
            "tokens": len(indices)}


def _persist(output: Path, rows: list[dict], manifest: dict) -> None:
    _write_json(output / "results.json", rows)
    if rows:
        temporary = output / "results.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / "results.csv")
    _write_json(output / "manifest.json", manifest)


def _plot(output: Path, rows: list[dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return [f"matplotlib 不可用；保留 JSON/CSV，未安装依赖: {exc}"]
    for layer in sorted({row["layer"] for row in rows}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for method in ("linked_channel", "svd", "asvd"):
            selected = sorted([r for r in rows if r["layer"] == layer and r["method"] == method], key=lambda r: r["rank"])
            if selected:
                for ax, metric in zip(axes, ("train_nmse", "calibration_eval_nmse")):
                    ax.plot([r["parameter_ratio"] for r in selected], [r[metric] for r in selected], ".-", label=method)
                    ax.set_xlabel("FFN parameter ratio")
                    ax.set_ylabel(metric)
                    ax.grid(alpha=.3)
                    ax.legend()
        fig.suptitle(f"Layer {layer}: existing training clips, 64 fit / 16 calibration eval")
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(output / f"layer_{layer:02d}_curves.{suffix}", dpi=160)
        plt.close(fig)
    return []


def _parse_ints(raw: str) -> list[int]:
    values = [int(value) for value in raw.split(",")]
    if not values or len(set(values)) != len(values):
        raise ValueError("参数列表为空或重复")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="6,35")
    parser.add_argument("--ranks", default="320,896,1024,1216")
    parser.add_argument("--max-fit-tokens", type=int, default=8192)
    parser.add_argument("--svd-driver", choices=("gesvdj", "gesvd"), default="gesvdj")
    args = parser.parse_args()
    layers, ranks = _parse_ints(args.layers), _parse_ints(args.ranks)
    if any(layer < 0 or layer >= 36 for layer in layers):
        raise ValueError("层编号必须在 0..35")
    positive_integer(args.max_fit_tokens, "max-fit-tokens")
    for rank in ranks:
        if lowrank_parameter_count(4096, 5376, rank) >= 3 * 4096 * 5376:
            raise ValueError("pilot rank 必须实际减少父 FFN 参数")
    root, output = args.root.resolve(strict=True), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, manifest = [], {"status": "running", "argv": sys.argv, "seed": SEED,
        "code_sha256": sha256_file(Path(__file__)), "layers": layers, "ranks": ranks,
        "max_fit_tokens": args.max_fit_tokens, "svd_driver": args.svd_driver,
        "scope": "L20 single-layer calibration pilot; no optimizer/model export/Jetson acceptance",
        "metric_target": "Recomputed BF16 parent complete gated FFN output",
        "split_policy": "64 fit / 16 calibration_eval clips from existing 80 training clips only",
        "fit_policy": "Fixed random token subset of fit clips for scaling and channel scores only; train NMSE uses all 64 fit clips",
        "channel_score_formula": "mean(parent BF16 gated activation squared) * FP32 squared L2 norm of parent down column; historical compute_activation_scores formula; FP64 accumulation; stable descending rank then original channel order",
        "asvd_scale_formula": "sqrt(mean(abs(parent BF16 projection inputs))).clamp_min(1e-6); down inputs are parent gated activations",
        "method_selection_rule": "Per layer/rank select minimum all-fit train_nmse among svd/asvd only; ties choose svd; linked_channel is comparator only; calibration_eval never participates; measurement only, no automatic training approval",
        "limitations": ["Local FFN NMSE does not establish trajectory ADE or 6B task accuracy",
                         "calibration_eval belongs to prior training data; not independent validation/test"],
        "input_sha256": {}, "results_completed": 0, "started_unix": time.time()}
    _persist(output, rows, manifest)
    weights = factors = pruned = gpu_svd = decompositions = None
    scores = ranking = x_scale = down_scale = prediction = target = None
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("pilot 模型操作仅支持已授权 L20 CUDA 环境")
        if "L20" not in torch.cuda.get_device_name():
            raise RuntimeError("pilot 仅允许 NVIDIA L20")
        torch.set_grad_enabled(False)
        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        manifest["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                   "gpu": torch.cuda.get_device_name(), "python": sys.version,
                                   "platform": platform.platform()}
        records, fit_clips, eval_clips, hashes, provenance = preflight(root)
        manifest.update({"input_sha256": hashes, "parent_MLP_SHA256": provenance["parent_MLP_SHA256"],
                         "fit_clips": fit_clips, "calibration_eval_clips": eval_clips,
                         "profile_source_sha256": PROFILE_SOURCE_SHA256})
        _persist(output, rows, manifest)
        for layer in layers:
            layer_start = time.perf_counter()
            path = root / PROFILE / "cache" / f"train_{layer:02d}.pt"
            hashes[str(path)] = sha256_file(path)
            cache = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            if set(cache) != {"x", "y"} or cache["x"].ndim != 2 or cache["x"].shape != cache["y"].shape:
                raise ValueError("profile cache 必须为同形状二维 x/y")
            x, y = cache["x"], cache["y"]
            if x.shape[1] != 4096 or x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16:
                raise ValueError("cache hidden/dtype 与 BF16 父模型不符")
            _finite(x, "cache x")
            _finite(y, "cache y")
            boundaries = clip_boundaries(records, len(x))
            fit_indices, eval_indices = _row_indices(boundaries, fit_clips), _row_indices(boundaries, eval_clips)
            generator = torch.Generator().manual_seed(SEED)
            selected_rows = fit_indices[torch.randperm(len(fit_indices), generator=generator)[:args.max_fit_tokens]]
            manifest.setdefault("layer_inputs", {})[str(layer)] = {
                "boundaries": boundaries, "fit_tokens": len(fit_indices), "calibration_eval_tokens": len(eval_indices),
                "fit_statistics_tokens": len(selected_rows), "fit_statistics_row_indices": selected_rows.tolist()}
            weights = {name: value.cuda() for name, value in _load_weights(root / PARENT / "width_5376/distilled" / f"mlp_{layer:02d}.safetensors").items()}
            torch.cuda.reset_peak_memory_stats()
            # 验证实际 cache 来自当前父模型；后续所有方法都以重新计算的父输出为目标。
            error, energy = 0., 0.
            all_indices = torch.arange(len(x))
            for start in range(0, len(x), 64):
                target = y[start:start + 64].cuda().float()
                prediction = dense_ffn(x[start:start + 64].cuda(), weights).float()
                _finite(prediction, "父 cache 复现输出")
                error += (prediction - target).double().square().sum().item()
                energy += target.double().square().sum().item()
            if energy <= 0 or error / energy > 1e-3:
                raise ValueError(f"父 cache 复现失败: layer={layer}, nmse={error / max(energy, 1e-300)}")
            manifest["layer_inputs"][str(layer)]["parent_cache_nmse"] = error / energy
            x_scale, down_scale, scores = _fit_statistics(x, selected_rows, weights)
            ranking = torch.argsort(scores, descending=True, stable=True)
            dense_params = sum(weight.numel() for weight in weights.values())

            def record(method, rank, candidate, actual_params, width, start_time, decomposition_seconds=0.):
                train = _nmse(x, fit_indices, weights, candidate)
                by_clip = {row["clip_id"]: row for row in boundaries}
                per_clip = []
                for clip in eval_clips:
                    boundary = by_clip[clip]
                    metric = _nmse(x, torch.arange(boundary["start"], boundary["stop"]), weights, candidate)
                    per_clip.append({"clip_id": clip, **metric})
                evaluation_error = sum(row["squared_error"] for row in per_clip)
                evaluation_energy = sum(row["target_energy"] for row in per_clip)
                evaluation = {"nmse": evaluation_error / evaluation_energy, "tokens": len(eval_indices)}
                per_clip_file = f"layer_{layer:02d}_{method}_rank_{rank}_calibration_eval.json"
                _write_json(output / per_clip_file, per_clip)
                manifest.setdefault("per_clip_outputs", {})[per_clip_file] = sha256_file(output / per_clip_file)
                torch.cuda.synchronize()
                rows.append({"layer": layer, "method": method, "rank": rank, "channel_width": width,
                    "actual_params": actual_params, "parent_params": dense_params,
                    "parameter_ratio": actual_params / dense_params,
                    "train_nmse": train["nmse"], "calibration_eval_nmse": evaluation["nmse"],
                    "train_tokens": train["tokens"], "calibration_eval_tokens": evaluation["tokens"],
                    "fit_statistics_tokens": len(selected_rows), "bf16_finite": True,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "shared_decomposition_seconds": decomposition_seconds,
                    "per_clip_file": per_clip_file,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
                manifest["results_completed"] = len(rows)
                _persist(output, rows, manifest)
                print("PILOT_RESULT", json.dumps(rows[-1], allow_nan=False), flush=True)

            record("parent", 0, lambda inputs: dense_ffn(inputs, weights), dense_params, 5376, layer_start)
            for rank in ranks:
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                width = matched_channel_width(4096, 5376, rank)
                ids = ranking[:width].sort().values
                pruned = {"gate_proj": weights["gate_proj"].index_select(0, ids),
                          "up_proj": weights["up_proj"].index_select(0, ids),
                          "down_proj": weights["down_proj"].index_select(1, ids)}
                params = sum(weight.numel() for weight in pruned.values())
                if params > lowrank_parameter_count(4096, 5376, rank):
                    raise ValueError("联动剪枝超出低秩参数预算")
                record("linked_channel", rank, lambda inputs: dense_ffn(inputs, pruned), params, width, started)
                del pruned, ids
            for method in ("svd", "asvd"):
                torch.cuda.reset_peak_memory_stats()
                decomposition_start = time.perf_counter()
                decompositions = {}
                for name in PROJECTIONS:
                    scale = (down_scale if name == "down_proj" else x_scale) if method == "asvd" else None
                    gpu_svd = compute_svd(weights[name], scale, args.svd_driver)
                    decompositions[name] = tuple(tensor.cpu() for tensor in gpu_svd)
                    del gpu_svd
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                decomposition_seconds = time.perf_counter() - decomposition_start
                decomposition_peak = torch.cuda.max_memory_allocated()
                decomposition_reserved_peak = torch.cuda.max_memory_reserved()
                for rank in ranks:
                    started = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    factors = {name: tuple(tensor.cuda() for tensor in factors_from_svd(decompositions[name], rank, torch.bfloat16)) for name in PROJECTIONS}
                    params = verify_factor_parameter_count(factors, 4096, 5376, rank)
                    record(method, rank, lambda inputs: factor_ffn(inputs, factors), params, None, started, decomposition_seconds)
                    rows[-1]["peak_allocated_bytes"] = max(rows[-1]["peak_allocated_bytes"], decomposition_peak)
                    rows[-1]["peak_reserved_bytes"] = max(rows[-1]["peak_reserved_bytes"], decomposition_reserved_peak)
                    _persist(output, rows, manifest)
                    del factors
                del decompositions
                gc.collect()
                torch.cuda.empty_cache()
            method_order = {"svd": 0, "asvd": 1}
            selections = {}
            for rank in ranks:
                compared = {row["method"]: row for row in rows if row["layer"] == layer and row["rank"] == rank}
                selected = min((compared[method] for method in method_order),
                               key=lambda row: (row["train_nmse"], method_order[row["method"]]))
                selections[str(rank)] = {"selected_method": selected["method"],
                    "fit_nmse_by_method": {method: row["train_nmse"] for method, row in compared.items()},
                    "selected_calibration_eval_nmse": selected["calibration_eval_nmse"],
                    "linked_channel_comparator_calibration_eval_nmse": compared["linked_channel"]["calibration_eval_nmse"]}
            manifest.setdefault("fit_selection", {})[str(layer)] = selections
            _write_json(output / "method_selection.json", {"rule": manifest["method_selection_rule"], "layers": manifest["fit_selection"]})
            manifest["method_selection_sha256"] = sha256_file(output / "method_selection.json")
            manifest["layer_inputs"][str(layer)]["elapsed_seconds"] = time.perf_counter() - layer_start
            del weights, scores, ranking, x_scale, down_scale, cache, x, y, all_indices
            gc.collect()
            torch.cuda.empty_cache()
        manifest["limitations"].extend(_plot(output, rows))
        manifest["artifact_sha256"] = {path.name: sha256_file(path) for path in output.iterdir()
            if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".tmp")}
        manifest.update({"status": "completed", "ended_unix": time.time()})
        _persist(output, rows, manifest)
    except BaseException as exc:
        manifest.update({"status": "failed", "ended_unix": time.time(),
                         "error": {"type": type(exc).__name__, "message": str(exc)}})
        _persist(output, rows, manifest)
        raise
    finally:
        # 进程失败也释放所有本地 GPU 引用；不留下后台作业。
        weights = factors = pruned = gpu_svd = decompositions = None
        scores = ranking = x_scale = down_scale = prediction = target = None
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
