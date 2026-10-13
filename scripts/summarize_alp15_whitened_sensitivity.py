"""纯标准库独立阶段4汇总；验证全部36层，不分配6B预算或按eval选型。"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from pathlib import Path

import alp15_whitened_sensitivity as runner
import summarize_alp15_lowrank_evidence as stats


def resource_metrics(path):
    with path.open(newline="", encoding="utf-8") as stream:
        observations = list(csv.DictReader(stream))
    if not observations:
        raise ValueError("资源观测为空")
    rss = [int(row["rss_kib"]) for row in observations if row["rss_kib"]]
    gpu = [float(row["child_gpu_used_mib"]) for row in observations]
    if any(not math.isfinite(value) or value < 0 for value in gpu) or any(value < 0 for value in rss):
        raise ValueError("资源观测非法")
    if any(row["other_compute_pids"] for row in observations):
        raise ValueError("completed层仍有GPU竞争记录")
    return {"resource_samples": len(observations), "observed_child_gpu_peak_mib": max(gpu),
            "observed_child_rss_peak_kib": max(rss) if rss else None}


def old_results(paths, reference):
    result, hashes = {}, {}
    for path in paths:
        directory = path.resolve(strict=True)
        manifest, rows, _, files = stats.validate_input(directory)
        if manifest.get("phase", 2) not in (2, 3):
            raise ValueError("重复一致性只能输入冻结阶段2/3证据")
        if manifest.get("code_sha256") != runner.ENGINE_SHA256 or manifest.get("phase1_code_sha256") != runner.BASE_SHA256:
            raise ValueError("旧重复测量引擎/基础源码与阶段4不一致")
        if "engine_code_sha256" in manifest and manifest["engine_code_sha256"] != runner.ENGINE_SHA256:
            raise ValueError("旧入口记录的engine源码与阶段4不一致")
        if manifest["parent_MLP_SHA256"] != reference["parent_MLP_SHA256"] or manifest["fit_clips"] != reference["fit_clips"] or manifest["calibration_eval_clips"] != reference["calibration_eval_clips"]:
            raise ValueError("旧重复测量父模型/fit/eval split与阶段4不一致")
        shared = {name: digest for name, digest in manifest["input_sha256"].items() if "/cache/train_" not in name}
        if shared != reference["input_sha256_shared"]:
            raise ValueError("旧重复测量共享来源/输入哈希与阶段4不一致")
        for layer in manifest["layers"]:
            if manifest["layer_inputs"][str(layer)]["fit_statistics_row_indices"] != reference["fit_statistics_row_indices"]:
                raise ValueError("旧重复测量拟合行与阶段4不一致")
        hashes[str(directory)] = files
        for row in rows:
            key = (row["layer"], row["rank"], row["method"])
            if key in result:
                raise ValueError("旧重复证据含相同配置")
            result[key] = {"row": row, "manifest": manifest}
    if result and ({key[0] for key in result} != {6, 12, 24, 35} or len(result) != 36):
        raise ValueError("重复证据必须完整覆盖6/12/24/35层")
    return result, hashes


def markdown(metrics, pairs, ranking, repeated, state):
    lines = ["# 阶段4：全36层FFN敏感性证据", "", "36个唯一层、324个配置、72组fit所选低秩方法与等参数通道对照。全层报告统一采用阶段4；历史重复测量仅检查一致性，不选择两次中较好者。", "",
        "这是同一父模型的局部完整FFN校准，不是完整6B模型、ADE/FDE、自主生成、稳态延迟、Q/O采集或Jetson验收。全部36层共用历史训练侧同一64 fit/16 calibration_eval clips，层/rank/token不是额外独立样本。", "",
        "NMSE能量口径=总SSE/总父输出平方能量；等权口径=每clip NMSE均值。p90/p95采用排序后(n−1)×p线性插值。feature RMSE=sqrt(SSE/(tokens×4096))，不是米或ADE。fit侧仅有原记聚合NMSE/token，没有SSE、目标能量或逐clip记录，无法独立重算，缺失项保留空值。", "",
        "## 每层每rank：fit所选方法与通道配对", "", "方法只由fit NMSE最小选择，平局沿冻结方法顺序；此处eval/尾部附加统计不改变选择。", "",
        "|层|rank|fit所选方法|候选eval能量NMSE|通道eval能量NMSE|改善/相等/恶化clip|能量下降|等权下降|最小clip相对下降|p95状态|max状态|",
        "|---:|---:|---|---:|---:|---|---:|---:|---:|---|---|"]
    keyed = {(row["layer"], row["rank"], row["method"]): row for row in metrics}
    for pair in pairs:
        candidate = keyed[(pair["layer"], pair["rank"], pair["selected_method"])]
        baseline = keyed[(pair["layer"], pair["rank"], "linked_channel")]
        lines.append(f"|{pair['layer']}|{pair['rank']}|{pair['selected_method']}|{candidate['eval_energy_nmse_recomputed']:.8g}|{baseline['eval_energy_nmse_recomputed']:.8g}|{pair['improved_clips']}/{pair['tied_clips']}/{pair['worse_clips']}|{pair['energy_nmse_reduction_fraction']:.2%}|{pair['clip_equal_mean_nmse_reduction_fraction']:.2%}|{pair['minimum_clip_relative_reduction_fraction']:.2%}|{'恶化' if pair['p95_worsened'] else '改善/相等'}|{'恶化' if pair['max_worsened'] else '改善/相等'}|")
    tail = [pair for pair in pairs if pair["p95_worsened"] or pair["max_worsened"]]
    lines += ["", f"{len(tail)}/{len(pairs)}组出现p95或最大值恶化。不能凭平均局部误差改善宣称稳健胜出，也不由此自动决定层/rank或完整模型预算。具体数值、最坏clip及逐次留一范围见 paired_comparisons.csv、paired_clips.csv 和 worst_clips.csv。", "",
        "## 训练侧敏感层排序（仅探索性诊断）", "", "各rank按fit所选方法的原记train NMSE从高到低排列；不使用eval、不分配6B预算、不将局部代理称作因果轨迹重要性。完整72项见training_only_sensitivity.csv。", "",
        "|rank|前10层：层号(fit NMSE)|", "|---:|---|"]
    for rank in runner.RANKS:
        selected = [item for item in ranking if item["rank"] == rank][:10]
        lines.append(f"|{rank}|" + ", ".join(f"{item['layer']}({item['selected_train_nmse']:.6g})" for item in selected) + "|")
    lines += ["", "## 重复一致性", "", "旧6/12/24/35层所有9配置与本轮逐一比较，原始两次都保留，不按优势择优。repeat_consistency.csv保存fit/eval差值与相对差值；数值差异只作描述。"]
    if repeated:
        lines += ["", f"已比较{len(repeated)}配置；最大fit绝对差={max(abs(item['train_nmse_difference']) for item in repeated):.8g}，最大eval绝对差={max(abs(item['eval_nmse_difference']) for item in repeated):.8g}。"]
    else:
        lines += ["", "未提供旧证据，重复一致性尚未复核；报告不能宣称该项已通过。"]
    lines += ["", "## 资源、来源与缺失", "", "每层资源.csv持续记录RSS和NVML子进程显存，每5秒采样，可能错过瞬时峰值。配置同时记录CUDA allocated/reserved峰值，包含共享Gram/SVD工作，不可作为端到端VLA推理benchmark。共享分解秒数不可跨配置重复相加。未设置凭历史2GiB推断的RSS阈值。", "",
        f"root记录耗时={state['ended_unix']-state['started_unix']:.3f}s；分层退出码、父cache重现NMSE、lambda/mu、分解/方法耗时、资源峰值与样本数详见 configurations.csv。root/child状态、源码与采前决策、副本和原始artifact/16clip哈希、324配置、fit-only选择全部已复核。服务器上游权重/缓存哈希为原引擎preflight留痕，本地未再次读取服务器模型。", "",
        "[采前决策](../../ALP15-FFN-SENSITIVITY-20261007.md)；[先前局部指标](../lowrank_metrics_20261007_full/REPORT.md)。汇总输出包含当前汇总器、冻结统计助手和controller源码副本，input_hashes.json记录所有读入证据及输出哈希。未测项没有补造或插值。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-inputs", nargs="+", type=Path, default=[])
    args = parser.parse_args()
    root, output = args.input.resolve(strict=True), args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("输出须为输入目录之外的新目录，禁止覆盖")
    if any(output.is_relative_to(path.resolve(strict=True)) for path in args.prior_inputs):
        raise ValueError("输出不得位于旧输入目录内")
    state = runner.read_json(root / "manifest.json")
    if state.get("status") != "completed" or state.get("phase") != 4 or state.get("completed_rows") != 324:
        raise ValueError("root未completed324配置，禁止全层结论")
    if state["layer_order"] != runner.layer_order() or len(state["completed_layers"]) != 36 or set(state["completed_layers"]) != set(range(36)) or set(state["layers"]) != {str(layer) for layer in range(36)}:
        raise ValueError("root缺层或重复层")
    if runner.sha256_file(Path(runner.__file__)) != state["source_sha256"][Path(runner.__file__).name]:
        raise ValueError("汇总校验controller必须与执行冻结版本相同")
    runner.verify_artifacts(root, state["artifact_sha256"])
    old, old_hashes = old_results(args.prior_inputs, state["reference_signature"])
    metrics, pairs, paired_clips, clip_metrics, worst_clips, ranking, repeated = [], [], [], [], [], [], []
    input_hashes = {str(root): {"manifest.json": runner.sha256_file(root / "manifest.json"), **state["artifact_sha256"]}, **old_hashes}
    reference = None
    for layer in range(36):
        child = root / f"layer_{layer:02d}"
        manifest, rows = runner.validate_child(child, layer, state["decision_sha256"], state["source_sha256"])
        record = state["layers"][str(layer)]
        if record["status"] != "completed" or record["exit_code"] != 0 or record["rows"] != 9 or record["manifest_sha256"] != runner.sha256_file(child / "manifest.json"):
            raise ValueError("root与child完成记录不符")
        for suffix, key in ((".log", "log_sha256"), (".resources.csv", "resource_csv_sha256")):
            path = root / "logs" / f"layer_{layer:02d}{suffix}"
            if runner.sha256_file(path) != record[key]:
                raise ValueError("root日志/资源哈希不符")
            input_hashes[str(root)][str(path.relative_to(root))] = record[key]
        signature = runner.reference_signature(manifest)
        if reference is not None and signature != reference:
            raise ValueError("跨层模型/split/拟合行/共享输入不一致")
        reference = signature
        if signature != state["reference_signature"]:
            raise ValueError("root参考签名不符")
        input_hashes[str(child)] = {"manifest.json": runner.sha256_file(child / "manifest.json"), **manifest["artifact_sha256"]}
        observed = resource_metrics(child / "resources.csv")
        by_key, clips_by_key = {}, {}
        for row in rows:
            key = (row["layer"], row["rank"], row["method"])
            clips = {clip["clip_id"]: clip for clip in runner.read_json(child / row["per_clip_file"])}
            by_key[key], clips_by_key[key] = row, clips
            info = manifest["layer_inputs"][str(layer)]
            description = stats.describe_row(child.name, manifest, row, clips)
            description.update({**observed, "parent_cache_nmse": info["parent_cache_nmse"],
                "gate_up_lambda": info["gate_up_lambda"], "down_lambda": info["down_lambda"],
                "child_total_elapsed_seconds": manifest["ended_unix"] - manifest["started_unix"], "child_exit_code": manifest["exit_code"]})
            metrics.append(description)
            for clip_id, clip in clips.items():
                clip_metrics.append({"layer": layer, "rank": row["rank"], "method": row["method"], "clip_id": clip_id,
                    **{field: clip[field] for field in ("nmse", "squared_error", "target_energy", "tokens")},
                    "feature_rmse": math.sqrt(clip["squared_error"] / (clip["tokens"] * 4096))})
            worst = max(clips.values(), key=lambda clip: clip["nmse"])
            worst_clips.append({"layer": layer, "rank": row["rank"], "method": row["method"],
                "worst_clip_id": worst["clip_id"], "worst_clip_nmse": worst["nmse"]})
            if key in old:
                previous = old[key]["row"]
                old_cache = {path: digest for path, digest in old[key]["manifest"]["input_sha256"].items()
                             if path.endswith(f"/cache/train_{layer:02d}.pt")}
                if len(old_cache) != 1 or any(manifest["input_sha256"].get(path) != digest for path, digest in old_cache.items()):
                    raise ValueError("旧重复测量对应层cache哈希与阶段4不一致")
                repeated.append({"layer": layer, "rank": row["rank"], "method": row["method"],
                    "old_train_nmse": previous["train_nmse"], "stage4_train_nmse": row["train_nmse"],
                    "train_nmse_difference": row["train_nmse"] - previous["train_nmse"],
                    "train_nmse_relative_difference": row["train_nmse"] / previous["train_nmse"] - 1 if previous["train_nmse"] else None,
                    "old_eval_nmse": previous["calibration_eval_nmse"], "stage4_eval_nmse": row["calibration_eval_nmse"],
                    "eval_nmse_difference": row["calibration_eval_nmse"] - previous["calibration_eval_nmse"],
                    "eval_nmse_relative_difference": row["calibration_eval_nmse"] / previous["calibration_eval_nmse"] - 1 if previous["calibration_eval_nmse"] else None})
        for rank in runner.RANKS:
            method = manifest["fit_selection"][str(layer)][str(rank)]["selected_method"]
            baseline_key, candidate_key = (layer, rank, "linked_channel"), (layer, rank, method)
            pair, details = stats.paired_summary(child.name, layer, rank, by_key[baseline_key], by_key[candidate_key], clips_by_key[baseline_key], clips_by_key[candidate_key])
            pair["selected_method"] = method
            pairs.append(pair)
            for detail in details:
                detail["selected_method"] = method
                detail["fit_selected_method"] = method
                detail["selected_method_eval_nmse"] = detail.pop("compressed_refit_nmse")
            paired_clips.extend(details)
            ranking.append({"rank": rank, "layer": layer, "selected_method": method,
                "selected_train_nmse": by_key[candidate_key]["train_nmse"], "channel_train_nmse": by_key[baseline_key]["train_nmse"]})
    if len(metrics) != 324 or len(pairs) != 72 or len(clip_metrics) != 5184 or len(paired_clips) != 1152:
        raise ValueError("总指标数量不符")
    ranking.sort(key=lambda item: (item["rank"], -item["selected_train_nmse"], item["layer"]))
    for rank in runner.RANKS:
        for position, item in enumerate((row for row in ranking if row["rank"] == rank), 1):
            item["training_only_rank"] = position
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("configurations.csv", metrics), ("paired_comparisons.csv", pairs), ("paired_clips.csv", paired_clips),
                         ("clip_metrics.csv", clip_metrics), ("worst_clips.csv", worst_clips), ("training_only_sensitivity.csv", ranking)):
        stats.write_csv(output / name, values)
    if repeated:
        stats.write_csv(output / "repeat_consistency.csv", repeated)
    stats.write_json(output / "summary.json", {"configuration_rows": len(metrics), "paired_groups": len(pairs),
        "configurations": metrics, "paired_comparisons": pairs, "training_only_sensitivity": ranking, "repeat_consistency": repeated,
        "scope": "Descriptive local FFN calibration only; no automatic 6B allocation or model/trajectory claim"})
    (output / "REPORT.md").write_text(markdown(metrics, pairs, ranking, repeated, state), encoding="utf-8")
    (output / "code").mkdir()
    for source in (Path(__file__), Path(runner.__file__), Path(stats.__file__)):
        shutil.copy2(source, output / "code" / source.name)
    stats.write_json(output / "input_hashes.json", {"status": "completed", "argv": sys.argv, "created_unix": time.time(),
        "script_sha256": runner.sha256_file(Path(__file__)), "inputs": input_hashes,
        "output_sha256": {str(path.relative_to(output)): runner.sha256_file(path) for path in output.rglob("*") if path.is_file()},
        "configuration_rows": 324, "paired_groups": 72, "clip_rows": 5184, "paired_clip_rows": 1152})
    print(f"SENSITIVITY_SUMMARY_COMPLETE {output} configurations=324 pairs=72 clips=5184")


if __name__ == "__main__":
    main()
