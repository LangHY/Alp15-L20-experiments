"""纯标准库汇总阶段2/3低秩证据；只读输入，不进行模型或 GPU 测量。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path


HIDDEN = 4096
WIDTH = 5376
METHODS = ("whitening", "whitening_refit_parent", "whitening_refit_compressed")
RANKS = (896, 1024)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"JSON 包含非有限常量: {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def finite_number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} 必须为有限且不小于 {minimum} 的数")
    return value


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须为正整数")
    return value


def close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-14):
        raise ValueError(f"{name} 重算不一致: {actual} != {expected}")


def artifact_path(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError(f"artifact 路径越界或不是文件: {relative}")
    return path


def percentile(values, probability):
    """采用排序后 (n-1)*p 的线性插值分位数，避免隐含不同 p90 定义。"""
    if not values or not 0 <= probability <= 1:
        raise ValueError("分位数输入非法")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def validate_input(root):
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError(f"实验尚未 completed: {root}")
    hashes = {"manifest.json": sha256_file(manifest_path)}
    declared = manifest["artifact_sha256"]
    if not {"results.json", "results.csv", "method_selection.json"} <= set(declared):
        raise ValueError("manifest 缺少主结果 artifact 哈希")
    for relative, expected in declared.items():
        actual = sha256_file(artifact_path(root, relative))
        if actual != expected:
            raise ValueError(f"artifact SHA-256 不一致: {root / relative}")
        hashes[relative] = actual
    if hashes["method_selection.json"] != manifest["method_selection_sha256"]:
        raise ValueError("方法选择文件哈希不一致")
    selection = read_json(root / "method_selection.json")
    if selection["rule"] != manifest["method_selection_rule"] or selection["layers"] != manifest["fit_selection"]:
        raise ValueError("方法选择记录与 manifest 不一致")
    fit, evaluation = manifest["fit_clips"], manifest["calibration_eval_clips"]
    if len(fit) != 64 or len(evaluation) != 16 or len(set(fit)) != 64 or len(set(evaluation)) != 16 or set(fit) & set(evaluation):
        raise ValueError("clip 划分不满足互斥 64/16")
    if tuple(manifest["ranks"]) != RANKS or manifest["seed"] != 20261007:
        raise ValueError("rank/seed 与固定阶段2/3协议不一致")
    layers = manifest["layers"]
    if len(layers) != 2 or len(set(layers)) != 2:
        raise ValueError("阶段2/3每个输入须含两个不同层")
    if manifest.get("phase", 2) == 3 and manifest["confirmation_layers"] != layers:
        raise ValueError("阶段3实际层号与入口记录不一致")
    rows = read_json(root / "results.json")
    expected_keys = {(layer, "parent", 0) for layer in layers} | {
        (layer, method, rank) for layer in layers for rank in RANKS for method in ("linked_channel", *METHODS)}
    actual_keys = [(row["layer"], row["method"], row["rank"]) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys or manifest["results_completed"] != len(rows):
        raise ValueError("实验配置集合不完整、重复或 completed 计数不符")
    clips_by_key = {}
    for row in rows:
        key = (row["layer"], row["method"], row["rank"])
        relative = row["per_clip_file"]
        if relative not in declared or manifest["per_clip_outputs"][relative] != hashes[relative]:
            raise ValueError("逐 clip 文件未被两个 manifest 哈希记录覆盖")
        clips = read_json(artifact_path(root, relative))
        if len(clips) != 16 or {clip["clip_id"] for clip in clips} != set(evaluation):
            raise ValueError("逐 clip 文件不含完整16个校准评估 clips")
        for clip in clips:
            error = finite_number(clip["squared_error"], "clip SSE")
            energy = finite_number(clip["target_energy"], "clip target energy", minimum=1e-300)
            positive_int(clip["tokens"], "clip tokens")
            close(finite_number(clip["nmse"], "clip NMSE"), error / energy, "clip NMSE")
        if sum(clip["tokens"] for clip in clips) != row["calibration_eval_tokens"]:
            raise ValueError("逐 clip tokens 与总计不符")
        total_error = math.fsum(clip["squared_error"] for clip in clips)
        total_energy = math.fsum(clip["target_energy"] for clip in clips)
        close(finite_number(row["calibration_eval_nmse"], "eval NMSE"), total_error / total_energy, "eval 能量 NMSE")
        for field in ("train_nmse", "elapsed_seconds", "shared_gram_svd_seconds", "parameter_ratio"):
            finite_number(row[field], field)
        for field in ("train_tokens", "fit_statistics_tokens", "actual_params", "parent_params", "peak_allocated_bytes", "peak_reserved_bytes"):
            positive_int(row[field], field)
        if row["bf16_finite"] is not True:
            raise ValueError("来源报告 BF16 非有限，不能作成功结果汇总")
        if row["parent_params"] != 3 * HIDDEN * WIDTH:
            raise ValueError("父参数不满足固定 hidden4096/width5376 协议")
        close(row["parameter_ratio"], row["actual_params"] / row["parent_params"], "参数比")
        budget = row["parent_params"] if row["rank"] == 0 else 3 * row["rank"] * (HIDDEN + WIDTH)
        if row["actual_params"] > budget or (row["method"] in METHODS and row["actual_params"] != budget):
            raise ValueError("实际参数不满足低秩预算")
        if row["method"] in ("parent", "linked_channel") and row["actual_params"] != row["channel_width"] * 3 * HIDDEN:
            raise ValueError("联动通道参数核算不符")
        layer_info = manifest["layer_inputs"][str(row["layer"])]
        if row["train_tokens"] != layer_info["fit_tokens"] or row["fit_statistics_tokens"] != layer_info["fit_statistics_tokens"]:
            raise ValueError("fit token 数与 manifest 不一致")
        if row["ridge_mu"] is not None:
            finite_number(row["ridge_mu"], "ridge mu", minimum=1e-300)
        clips_by_key[key] = {clip["clip_id"]: clip for clip in clips}
    rows_by_key = {(row["layer"], row["method"], row["rank"]): row for row in rows}
    for layer in layers:
        for rank in RANKS:
            selected = min(METHODS, key=lambda method: (rows_by_key[(layer, method, rank)]["train_nmse"], METHODS.index(method)))
            item = selection["layers"][str(layer)][str(rank)]
            if item["selected_method"] != selected:
                raise ValueError("已保存方法未按 fit-only 规则选择")
            for method in ("linked_channel", *METHODS):
                close(item["fit_nmse_by_method"][method], rows_by_key[(layer, method, rank)]["train_nmse"], "方法选择 fit NMSE")
            close(item["selected_calibration_eval_nmse"], rows_by_key[(layer, selected, rank)]["calibration_eval_nmse"], "已选方法关联 eval")
    return manifest, rows, clips_by_key, hashes


def describe_row(experiment, manifest, row, clips):
    errors = [clip["nmse"] for clip in clips.values()]
    sse = math.fsum(clip["squared_error"] for clip in clips.values())
    energy = math.fsum(clip["target_energy"] for clip in clips.values())
    tokens = sum(clip["tokens"] for clip in clips.values())
    selection = manifest["fit_selection"][str(row["layer"])].get(str(row["rank"]), {})
    return {"experiment": experiment, "phase": manifest.get("phase", 2), "layer": row["layer"],
        "rank": row["rank"], "method": row["method"], "actual_params": row["actual_params"],
        "parameter_budget": row["parent_params"] if row["rank"] == 0 else 3 * row["rank"] * (HIDDEN + WIDTH),
        "parent_params": row["parent_params"], "parameter_ratio": row["parameter_ratio"],
        "train_energy_nmse_reported": row["train_nmse"], "train_squared_error_unavailable": None,
        "train_target_energy_unavailable": None, "eval_energy_nmse_recomputed": sse / energy,
        "eval_clip_equal_mean_nmse": statistics.fmean(errors), "eval_clip_median_nmse": statistics.median(errors),
        "eval_clip_p90_nmse": percentile(errors, .9), "eval_clip_max_nmse": max(errors),
        "eval_clip_p95_nmse": percentile(errors, .95),
        "eval_squared_error_sum": sse, "eval_target_squared_energy": energy,
        "output_dimension_fixed": HIDDEN, "eval_elements": tokens * HIDDEN, "eval_element_mse": sse / (tokens * HIDDEN),
        "eval_feature_rmse": math.sqrt(sse / (tokens * HIDDEN)),
        "train_tokens": row["train_tokens"], "eval_tokens": tokens, "fit_statistics_tokens": row["fit_statistics_tokens"],
        "eval_clips": len(clips), "bf16_finite_reported": row["bf16_finite"], "saved_numbers_finite_verified": True,
        "fit_selected_method": selection.get("selected_method"), "is_fit_selected_lowrank_method": row["method"] == selection.get("selected_method"),
        "method_elapsed_seconds_not_inference_benchmark": row["elapsed_seconds"],
        "shared_gram_svd_seconds_not_additive": row["shared_gram_svd_seconds"],
        "peak_allocated_bytes_including_shared_work": row["peak_allocated_bytes"],
        "peak_reserved_bytes_including_shared_work": row["peak_reserved_bytes"], "ridge_mu": row["ridge_mu"]}


def paired_summary(experiment, layer, rank, baseline_row, candidate_row, baseline, candidate):
    if set(baseline) != set(candidate) or baseline_row["actual_params"] != candidate_row["actual_params"]:
        raise ValueError("配对比较 clip 集或实际参数数不相等")
    paired = []
    for clip_id in sorted(baseline):
        before, after = baseline[clip_id], candidate[clip_id]
        if before["tokens"] != after["tokens"]:
            raise ValueError("配对 clip token 数不同")
        close(before["target_energy"], after["target_energy"], "配对父 target energy")
        if before["nmse"] <= 0:
            raise ValueError("通道基线误差为零，无法定义相对改善")
        paired.append({"experiment": experiment, "layer": layer, "rank": rank, "clip_id": clip_id,
            "linked_channel_nmse": before["nmse"], "compressed_refit_nmse": after["nmse"],
            "nmse_reduction": before["nmse"] - after["nmse"], "relative_reduction_fraction": 1 - after["nmse"] / before["nmse"],
            "baseline_squared_error": before["squared_error"], "candidate_squared_error": after["squared_error"],
            "target_energy": before["target_energy"], "tokens": before["tokens"]})
    baseline_sse = math.fsum(row["baseline_squared_error"] for row in paired)
    candidate_sse = math.fsum(row["candidate_squared_error"] for row in paired)
    energy = math.fsum(row["target_energy"] for row in paired)
    baseline_sum = math.fsum(row["linked_channel_nmse"] for row in paired)
    candidate_sum = math.fsum(row["compressed_refit_nmse"] for row in paired)
    loo_energy, loo_equal, loo_absolute = [], [], []
    for row in paired:
        loo_energy.append(1 - (candidate_sse - row["candidate_squared_error"]) / (baseline_sse - row["baseline_squared_error"]))
        loo_equal.append(1 - (candidate_sum - row["compressed_refit_nmse"]) / (baseline_sum - row["linked_channel_nmse"]))
        loo_absolute.append(((baseline_sse - row["baseline_squared_error"]) - (candidate_sse - row["candidate_squared_error"])) / (energy - row["target_energy"]))
    reductions = [row["relative_reduction_fraction"] for row in paired]
    before_nmse, after_nmse = [row["linked_channel_nmse"] for row in paired], [row["compressed_refit_nmse"] for row in paired]
    before_p95, after_p95 = percentile(before_nmse, .95), percentile(after_nmse, .95)
    return {"experiment": experiment, "layer": layer, "rank": rank, "actual_params_each": candidate_row["actual_params"],
        "paired_clips": len(paired), "improved_clips": sum(row["nmse_reduction"] > 0 for row in paired),
        "tied_clips": sum(row["nmse_reduction"] == 0 for row in paired), "worse_clips": sum(row["nmse_reduction"] < 0 for row in paired),
        "energy_nmse_reduction_fraction": 1 - candidate_sse / baseline_sse,
        "clip_equal_mean_nmse_reduction_fraction": 1 - candidate_sum / baseline_sum,
        "minimum_clip_relative_reduction_fraction": min(reductions), "median_clip_relative_reduction_fraction": statistics.median(reductions),
        "minimum_clip_absolute_nmse_reduction": min(row["nmse_reduction"] for row in paired),
        "baseline_clip_p95_nmse": before_p95, "candidate_clip_p95_nmse": after_p95,
        "clip_p95_reduction_fraction": 1 - after_p95 / before_p95,
        "baseline_clip_max_nmse": max(before_nmse), "candidate_clip_max_nmse": max(after_nmse),
        "clip_max_reduction_fraction": 1 - max(after_nmse) / max(before_nmse),
        "p95_worsened": after_p95 > before_p95, "max_worsened": max(after_nmse) > max(before_nmse),
        "median_clip_absolute_nmse_reduction": statistics.median(row["nmse_reduction"] for row in paired),
        "loo_energy_nmse_reduction_min_fraction": min(loo_energy), "loo_energy_nmse_reduction_max_fraction": max(loo_energy),
        "loo_clip_equal_mean_reduction_min_fraction": min(loo_equal), "loo_clip_equal_mean_reduction_max_fraction": max(loo_equal),
        "loo_absolute_energy_nmse_reduction_min": min(loo_absolute), "loo_absolute_energy_nmse_reduction_max": max(loo_absolute)}, paired


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def report_markdown(metrics, pairs, experiments):
    lines = ["# 阶段2/3低秩校准证据完整指标", "", f"共 {len(metrics)} 个完整配置，{len(pairs)} 组 compressed-refit 与等实际参数通道基线配对。全部候选保留；方法选择仅重核 fit 指标，没有按 calibration_eval 重新选方法。", "",
        "能量 NMSE=总 SSE/总父输出平方能量；clip 等权指标先按各 clip 自身能量归一化，再取均值/中位数/p90/p95/最大值。p90/p95 使用排序后 (n−1)×p 的线性插值。元素 MSE=总 SSE/(token 数×4096)，feature RMSE=元素 MSE 的平方根；4096 是冻结实验的 FFN 输出维度，不能将 MSE/RMSE 解读为米、轨迹误差或 ADE。", "",
        "训练侧仅保存了原始能量 NMSE 与 token 数，没有训练 SSE/target energy/逐 clip 文件，故不能独立重算训练 NMSE 或训练元素 MSE；CSV 对缺失项留空。评估侧 SSE、能量、token 与全部 NMSE 已逐 clip 独立重算。BF16 有限值是执行结果记录；这里只校验保存数值有限，未重新测量模型。", "",
        "## 全部候选", "", "|层|rank|方法|实际参数|预算|参数比|fit NMSE（原记）|eval 能量 NMSE|clip均值|中位数|p90|p95|最大值|fit所选|", "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for row in metrics:
        lines.append(f"|{row['layer']}|{row['rank']}|{row['method']}|{row['actual_params']:,}|{row['parameter_budget']:,}|{row['parameter_ratio']:.6f}|{row['train_energy_nmse_reported']:.8g}|{row['eval_energy_nmse_recomputed']:.8g}|{row['eval_clip_equal_mean_nmse']:.8g}|{row['eval_clip_median_nmse']:.8g}|{row['eval_clip_p90_nmse']:.8g}|{row['eval_clip_p95_nmse']:.8g}|{row['eval_clip_max_nmse']:.8g}|{'是' if row['is_fit_selected_lowrank_method'] else ''}|")
    lines += ["", "## 配对描述：compressed-refit 对通道基线", "", "下表相对下降为 1−候选误差/基线误差，负数代表恶化。改善数按逐 clip NMSE 严格下降计数。留一范围逐次去掉一个 clip 后重算，衡量对单个 clip 的敏感性；这是描述性检查，不是置信区间或显著性检验。", "", "|层|rank|改善/相等/恶化|能量误差下降|clip均值误差下降|最小clip下降|中位clip下降|留一能量下降范围|留一clip均值下降范围|", "|---:|---:|---|---:|---:|---:|---:|---|---|"]
    for row in pairs:
        lines.append(f"|{row['layer']}|{row['rank']}|{row['improved_clips']}/{row['tied_clips']}/{row['worse_clips']}|{row['energy_nmse_reduction_fraction']:.2%}|{row['clip_equal_mean_nmse_reduction_fraction']:.2%}|{row['minimum_clip_relative_reduction_fraction']:.2%}|{row['median_clip_relative_reduction_fraction']:.2%}|{row['loo_energy_nmse_reduction_min_fraction']:.2%}–{row['loo_energy_nmse_reduction_max_fraction']:.2%}|{row['loo_clip_equal_mean_reduction_min_fraction']:.2%}–{row['loo_clip_equal_mean_reduction_max_fraction']:.2%}|")
    lines += ["", "## 尾部诊断", "", "总体/中位误差改善不保证最差场景改善。p95 与最大值分别对同一16个clip的分布计算，数值对应的clip可能不同；这些附加描述指标不参与方法选择。", "", "|层|rank|基线p95|候选p95|基线max|候选max|尾部状态|", "|---:|---:|---:|---:|---:|---:|---|"]
    for row in pairs:
        warning = "/".join(label for label, flag in (("p95恶化", row["p95_worsened"]), ("最大值恶化", row["max_worsened"])) if flag) or "两项改善"
        lines.append(f"|{row['layer']}|{row['rank']}|{row['baseline_clip_p95_nmse']:.8g}|{row['candidate_clip_p95_nmse']:.8g}|{row['baseline_clip_max_nmse']:.8g}|{row['candidate_clip_max_nmse']:.8g}|{warning}|")
    worsened = [row for row in pairs if row["p95_worsened"] or row["max_worsened"]]
    if worsened:
        lines += ["", "需特别关注：" + "；".join(f"第{row['layer']}层 rank{row['rank']} 改善{row['improved_clips']}/{row['paired_clips']}个clip，但" + "/".join(label for label, flag in (("p95", row["p95_worsened"]), ("最大值", row["max_worsened"])) if flag) + "高于通道基线" for row in worsened) + "。不能只据平均误差改善宣称稳健胜出。"]
    lines += ["", "## 资源与证据边界", "", "以下为校准脚本耗时和 PyTorch 显存分配/保留峰值，不是端到端推理 benchmark，不含轨迹 ADE、整模型参数量、真实推理延迟、吞吐或 Jetson 资源结论。各配置显存包含共享 Gram/SVD 工作；共享秒数不可对各配置累加。完整SSE、元素MSE、feature RMSE、tokens和资源数值见 configurations.csv。", "", "|实验|记录总耗时(s)|最大allocated(GiB)|最大reserved(GiB)|环境|", "|---|---:|---:|---:|---|"]
    for item in experiments:
        lines.append(f"|{item['directory']}|{item['recorded_total_elapsed_seconds']:.3f}|{item['maximum_reported_allocated_bytes']/2**30:.3f}|{item['maximum_reported_reserved_bytes']/2**30:.3f}|{item['environment'].get('gpu', '')}; torch {item['environment'].get('torch', '')}|")
    lines += ["", "四层共用既有训练侧的同一16个 calibration_eval clips，不能把不同层结果当作64个独立样本，也不能声称独立测试集泛化。局部 NMSE 改善不能证明6B轨迹精度成功；仍需新的完整模型/真实轨迹验证。", "", "已校验输入 manifest 声明的全部 artifacts、每个逐 clip 文件与方法选择文件的哈希和内部统计一致性。输入 manifest 自身哈希另存 input_hashes.json；它们记录的服务器上游模型/缓存路径未在本地重新读取，不能冒称本轮再次验证模型权重。", "", "[阶段决策与执行记录](../../ALP15-6B-LR-PILOT-20261007.md)；[完整服务器源码、CPU测试与运行日志](../lowrank_pilot_20261007_code/)。", "", "产物：configurations.csv 全部配置及MSE/资源；clip_metrics.csv 全部逐clip；paired_comparisons.csv 配对聚合；paired_clips.csv 配对明细；summary.json 机器可读聚合；input_hashes.json 输入与输出留痕。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [path.resolve(strict=True) for path in args.inputs]
    output = args.output.resolve()
    if output.exists() or len(set(roots)) != len(roots) or any(output.is_relative_to(root) for root in roots):
        raise ValueError("output 必须尚不存在且位于输入目录之外；inputs 不得重复")
    metrics, pairs, paired_clips, clip_metrics, experiments, input_hashes = [], [], [], [], [], {}
    seen_layers, reference_split = set(), None
    for root in roots:
        manifest, rows, clips_by_key, hashes = validate_input(root)
        split = (manifest["fit_clips"], manifest["calibration_eval_clips"], manifest["parent_MLP_SHA256"])
        if reference_split is not None and split != reference_split:
            raise ValueError("不同实验使用了不同 split 或父模型，不能联合成同口径确认报告")
        reference_split = split
        if seen_layers & set(manifest["layers"]):
            raise ValueError("输入包含重复层，不能冒称独立层确认")
        seen_layers.update(manifest["layers"])
        input_hashes[str(root)] = hashes
        rows_by_key = {(row["layer"], row["method"], row["rank"]): row for row in rows}
        for row in rows:
            key = (row["layer"], row["method"], row["rank"])
            clips = clips_by_key[key]
            metrics.append(describe_row(root.name, manifest, row, clips))
            for clip_id, clip in clips.items():
                clip_metrics.append({"experiment": root.name, "layer": row["layer"], "rank": row["rank"], "method": row["method"],
                    "clip_id": clip_id, **{field: clip[field] for field in ("nmse", "squared_error", "target_energy", "tokens")},
                    "output_dimension_fixed": HIDDEN, "elements": clip["tokens"] * HIDDEN,
                    "element_mse": clip["squared_error"] / (clip["tokens"] * HIDDEN)})
                clip_metrics[-1]["feature_rmse"] = math.sqrt(clip_metrics[-1]["element_mse"])
        for layer in manifest["layers"]:
            for rank in RANKS:
                baseline_key, candidate_key = (layer, "linked_channel", rank), (layer, "whitening_refit_compressed", rank)
                pair, details = paired_summary(root.name, layer, rank, rows_by_key[baseline_key], rows_by_key[candidate_key], clips_by_key[baseline_key], clips_by_key[candidate_key])
                pairs.append(pair)
                paired_clips.extend(details)
        experiments.append({"directory": str(root), "manifest_sha256": hashes["manifest.json"], "phase": manifest.get("phase", 2),
            "recorded_total_elapsed_seconds": manifest["ended_unix"] - manifest["started_unix"],
            "maximum_reported_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
            "maximum_reported_reserved_bytes": max(row["peak_reserved_bytes"] for row in rows), "environment": manifest["environment"]})
    metrics.sort(key=lambda row: (row["layer"], row["rank"], ("parent", "linked_channel", *METHODS).index(row["method"])))
    pairs.sort(key=lambda row: (row["layer"], row["rank"]))
    output.mkdir(parents=True, exist_ok=False)
    for name, records in (("configurations.csv", metrics), ("clip_metrics.csv", clip_metrics), ("paired_comparisons.csv", pairs), ("paired_clips.csv", paired_clips)):
        write_csv(output / name, records)
    write_json(output / "summary.json", {"output_dimension_fixed": HIDDEN, "configurations": metrics,
        "paired_comparisons": pairs, "experiments": experiments, "statistical_status": "Descriptive only; no significance test or confidence interval"})
    (output / "REPORT.md").write_text(report_markdown(metrics, pairs, experiments), encoding="utf-8")
    write_json(output / "input_hashes.json", {"status": "completed", "argv": sys.argv, "created_unix": time.time(),
        "script_sha256": sha256_file(Path(__file__)), "inputs": input_hashes,
        "output_sha256": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()},
        "configuration_rows": len(metrics), "paired_groups": len(pairs), "clip_rows": len(clip_metrics), "paired_clip_rows": len(paired_clips)})
    print(json.dumps({"output": str(output), "configuration_rows": len(metrics), "paired_groups": len(pairs),
                      "clip_rows": len(clip_metrics), "paired_clip_rows": len(paired_clips)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
