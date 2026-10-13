"""阶段4逐层独立测量；固定协议，严格恢复，不自动重跑失败层。"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


ENGINE_SHA256 = "fa2e18bf86a768e724ec08536db1a0e6a9ed88125fe199fcebf04317ef86b007"
BASE_SHA256 = "c2975113b04a0a5e295ab14f0ba09db4152568fdb751ec957b45c87b3693ad69"
SEED = 20261007
METHODS = ("whitening", "whitening_refit_parent", "whitening_refit_compressed")
RANKS = (896, 1024)
CHILD_CODE = "import sys; sys.path.insert(0,sys.argv.pop(1)); import alp15_whitened_pilot as engine; engine.LAYERS=(int(sys.argv.pop(1)),); engine.main()"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"JSON 非有限值: {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def layer_order():
    order = list(range(36))
    random.Random(SEED).shuffle(order)
    return order


def verify_sources():
    directory = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve(), directory / "alp15_whitened_pilot.py", directory / "alp15_lowrank_pilot.py"]
    if sha256_file(paths[1]) != ENGINE_SHA256 or sha256_file(paths[2]) != BASE_SHA256:
        raise ValueError("冻结引擎或基础模块哈希不符")
    return paths, {path.name: sha256_file(path) for path in paths}


def verify_artifacts(directory, hashes):
    for name, expected in hashes.items():
        path = (directory / name).resolve(strict=True)
        if Path(name).is_absolute() or not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"artifact 路径越界: {name}")
        if sha256_file(path) != expected:
            raise ValueError(f"artifact SHA-256 不符: {path}")


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(f"{name} 非法")
    return value


def same(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-14):
        raise ValueError(f"{name} 不一致")


def validate_child(directory, layer, decision_sha, source_hashes):
    """恢复/汇总统一门禁；验证真实9行、16clip、来源及 fit-only 选择。"""
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") != "completed" or manifest.get("exit_code") != 0:
        raise ValueError(f"层{layer}没有 completed/exit0；保持原现场，禁止自动重跑")
    expected = {"phase": 4, "layers": [layer], "ranks": list(RANKS), "seed": SEED,
        "max_fit_tokens": 8192, "svd_driver": "gesvdj", "decision_sha256": decision_sha,
        "engine_code_sha256": ENGINE_SHA256, "phase1_code_sha256": BASE_SHA256}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"层{layer} {key} 不符")
    if manifest["code_sha256"] != ENGINE_SHA256 or manifest["copied_source_sha256"] != source_hashes:
        raise ValueError("child 执行源码版本不符")
    if manifest["parent_MLP_SHA256"] != "be6239cbbd8c6cfd01317a33233486c1a422469ee0c323bdfe37767c75d0ab27":
        raise ValueError("父模型哈希不符")
    if not manifest.get("input_sha256"):
        raise ValueError("缺少逐输入哈希")
    fit, evaluation = manifest["fit_clips"], manifest["calibration_eval_clips"]
    if len(fit) != 64 or len(set(fit)) != 64 or len(evaluation) != 16 or len(set(evaluation)) != 16 or set(fit) & set(evaluation):
        raise ValueError("64/16 clip split 不符")
    artifacts = manifest["artifact_sha256"]
    if not {"results.json", "results.csv", "method_selection.json", "execution.log", "resources.csv", "code/decision.md"} <= set(artifacts):
        raise ValueError("child 缺少必要 artifact")
    verify_artifacts(directory, artifacts)
    if artifacts["code/decision.md"] != decision_sha:
        raise ValueError("采前决策副本哈希不符")
    for name, digest in source_hashes.items():
        if artifacts.get("code/" + name) != digest:
            raise ValueError("child 源码副本哈希不符")
    selection = read_json(directory / "method_selection.json")
    if artifacts["method_selection.json"] != manifest["method_selection_sha256"] or selection["layers"] != manifest["fit_selection"] or selection["rule"] != manifest["method_selection_rule"]:
        raise ValueError("fit-only 选择文件不符")
    rows = read_json(directory / "results.json")
    keys = [(row["layer"], row["method"], row["rank"]) for row in rows]
    expected_keys = {(layer, "parent", 0)} | {(layer, method, rank) for rank in RANKS for method in ("linked_channel", *METHODS)}
    if len(rows) != 9 or len(set(keys)) != 9 or set(keys) != expected_keys or manifest["results_completed"] != 9:
        raise ValueError("child 必须有9条唯一完整配置")
    info = manifest["layer_inputs"][str(layer)]
    if (info["fit_tokens"], info["calibration_eval_tokens"], info["fit_statistics_tokens"]) != (9185, 2319, 8192):
        raise ValueError("实际 fit/eval/统计 token 数与采前决策不符")
    if len(info["fit_statistics_row_indices"]) != 8192 or len(set(info["fit_statistics_row_indices"])) != 8192:
        raise ValueError("拟合行重复或数量不符")
    if number(info["parent_cache_nmse"], "父 cache NMSE") > 1e-3:
        raise ValueError("父 cache 复现失败")
    for key in ("gate_up_lambda", "down_lambda"):
        number(info[key], key, positive=True)
    by_method = {}
    for row in rows:
        rank, method = row["rank"], row["method"]
        if (row["train_tokens"], row["calibration_eval_tokens"], row["fit_statistics_tokens"]) != (9185, 2319, 8192) or row["bf16_finite"] is not True:
            raise ValueError("配置 tokens/有限性不符")
        for key in ("train_nmse", "calibration_eval_nmse", "elapsed_seconds", "shared_gram_svd_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
            number(row[key], key)
        if row["ridge_mu"] is not None:
            number(row["ridge_mu"], "ridge_mu", positive=True)
        params = 3 * 4096 * 5376 if rank == 0 else 3 * rank * (4096 + 5376)
        if row["actual_params"] != params or row["parent_params"] != 66060288:
            raise ValueError("实际参数预算不符")
        same(row["parameter_ratio"], params / 66060288, "参数比")
        filename = row["per_clip_file"]
        if filename not in artifacts or manifest["per_clip_outputs"].get(filename) != artifacts[filename]:
            raise ValueError("逐 clip 哈希不符")
        clips = read_json(directory / filename)
        if len(clips) != 16 or {item["clip_id"] for item in clips} != set(evaluation):
            raise ValueError("逐 clip 集合不符")
        for item in clips:
            number(item["squared_error"], "SSE")
            number(item["target_energy"], "target energy", positive=True)
            number(item["tokens"], "tokens", positive=True)
            same(number(item["nmse"], "clip NMSE"), item["squared_error"] / item["target_energy"], "clip NMSE")
        if sum(item["tokens"] for item in clips) != 2319:
            raise ValueError("逐 clip token 和不符")
        same(row["calibration_eval_nmse"], math.fsum(item["squared_error"] for item in clips) / math.fsum(item["target_energy"] for item in clips), "eval NMSE")
        by_method[(method, rank)] = row
    for rank in RANKS:
        chosen = min(METHODS, key=lambda method: (by_method[(method, rank)]["train_nmse"], METHODS.index(method)))
        saved = selection["layers"][str(layer)][str(rank)]
        if saved["selected_method"] != chosen:
            raise ValueError("方法选择不是 fit-only")
        for method in ("linked_channel", *METHODS):
            same(saved["fit_nmse_by_method"][method], by_method[(method, rank)]["train_nmse"], "方法 fit 指标")
        same(saved["selected_calibration_eval_nmse"], by_method[(chosen, rank)]["calibration_eval_nmse"], "已选 eval 关联")
    return manifest, rows


def reference_signature(manifest):
    """跨层只接受同一父模型/clip split/拟合行及共享输入哈希。"""
    layer = manifest["layers"][0]
    shared = {path: digest for path, digest in manifest["input_sha256"].items() if "/cache/train_" not in path}
    return {"parent_MLP_SHA256": manifest["parent_MLP_SHA256"], "fit_clips": manifest["fit_clips"],
        "calibration_eval_clips": manifest["calibration_eval_clips"], "input_sha256_shared": shared,
        "fit_statistics_row_indices": manifest["layer_inputs"][str(layer)]["fit_statistics_row_indices"]}


def resource_sample(pid):
    ps = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=10)
    rss = int(ps.stdout.strip()) if ps.stdout.strip() else None
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
    gpu_mib, competitors = 0., []
    for record in csv.reader(gpu.stdout.splitlines()):
        if not record:
            continue
        process = int(record[0].strip())
        if process == pid:
            gpu_mib += float(record[1].strip())
        else:
            competitors.append(process)
    return {"unix_time": time.time(), "child_pid": pid, "rss_kib": rss,
            "child_gpu_used_mib": gpu_mib, "other_compute_pids": ",".join(str(value) for value in sorted(set(competitors)))}


def run_child(command, log_path, resource_path):
    """只终止本入口启动的 child；其他GPU进程只记录，不发送信号。"""
    with log_path.open("w", encoding="utf-8") as log, resource_path.open("w", newline="", encoding="utf-8") as resource:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        writer = csv.DictWriter(resource, fieldnames=("unix_time", "child_pid", "rss_kib", "child_gpu_used_mib", "other_compute_pids"))
        writer.writeheader()
        failure = None
        try:
            while process.poll() is None:
                observation = resource_sample(process.pid)
                writer.writerow(observation)
                resource.flush()
                if observation["other_compute_pids"]:
                    raise RuntimeError("检测到其他GPU计算进程竞争，只停止本child")
                time.sleep(5)
            return process.returncode
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if failure is not None:
                # 异常不会吞掉真实退出码；由主循环在 finally 留痕后再停止。
                failure.child_exit_code = process.returncode
                failure.child_stop_reason = str(failure)
                log.write(f"\nCONTROLLER_STOP exit_code={process.returncode} reason={failure}\n")
                log.flush()


def finalize_child(child, layer, output, decision_sha, source_hashes, exit_code):
    """成功/失败均保存新child原始日志和采前源码，不修改旧实验。"""
    manifest_path = child / "manifest.json"
    if not manifest_path.is_file():
        return
    code = child / "code"
    code.mkdir(exist_ok=True)
    for source in (output / "code").iterdir():
        shutil.copy2(source, code / source.name)
    for name, suffix in (("execution.log", ".log"), ("resources.csv", ".resources.csv")):
        shutil.copy2(output / "logs" / f"layer_{layer:02d}{suffix}", child / name)
    manifest = read_json(manifest_path)
    manifest.update({"phase": 4, "phase_id": "ALP15-WHITENED-ALL36-MEASURE-20261007",
        "decision_sha256": decision_sha, "entrypoint_code_sha256": source_hashes[Path(__file__).name],
        "engine_code_sha256": ENGINE_SHA256, "copied_source_sha256": source_hashes, "exit_code": exit_code})
    artifacts = manifest.setdefault("artifact_sha256", {})
    for path in code.iterdir():
        artifacts[str(path.relative_to(child))] = sha256_file(path)
    for name in ("execution.log", "resources.csv"):
        artifacts[name] = sha256_file(child / name)
    write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, output, decision = args.root.resolve(strict=True), args.output.resolve(), args.decision.resolve(strict=True)
    sources, source_hashes = verify_sources()
    decision_sha = sha256_file(decision)
    fresh = not output.exists()
    if not fresh and not args.resume:
        raise FileExistsError("既存 output 只允许显式 --resume；禁止覆盖")
    if fresh and args.resume:
        raise FileNotFoundError("--resume 需要既存 root output")
    if fresh:
        output.mkdir(parents=True, exist_ok=False)
    with (output / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("已有controller持锁，禁止并行写入") from exc
        manifest_path = output / "manifest.json"
        if fresh:
            (output / "code").mkdir()
            (output / "logs").mkdir()
            for source in sources:
                shutil.copy2(source, output / "code" / source.name)
            shutil.copy2(decision, output / "code/decision.md")
            state = {"status": "running", "phase": 4, "argv": sys.argv, "root": str(root),
                "started_unix": time.time(), "decision_sha256": decision_sha, "source_sha256": source_hashes,
                "artifact_sha256": {"code/" + name: digest for name, digest in source_hashes.items()},
                "layer_order": layer_order(), "seed": SEED, "ranks": list(RANKS), "max_fit_tokens": 8192,
                "svd_driver": "gesvdj", "regularization_ratio": 1e-4,
                "completed_layers": [], "layers": {}, "total_expected_layers": 36, "total_expected_rows": 324,
                "resource_policy": "5s RSS/GPU samples; GPU competitor or monitor failure stops only own child; no guessed RSS threshold",
                "limitations": ["Local FFN calibration only; no model export/training/ADE/QO/Jetson", "Fit SSE/energy/perclip unavailable in frozen engine"]}
            state["artifact_sha256"]["code/decision.md"] = decision_sha
            write_json(manifest_path, state)
        else:
            state = read_json(manifest_path)
            for key, value in (("phase", 4), ("root", str(root)), ("decision_sha256", decision_sha), ("source_sha256", source_hashes),
                               ("layer_order", layer_order()), ("seed", SEED), ("ranks", list(RANKS)), ("max_fit_tokens", 8192),
                               ("svd_driver", "gesvdj"), ("regularization_ratio", 1e-4)):
                if state.get(key) != value:
                    raise ValueError(f"resume 冻结记录不符: {key}")
            verify_artifacts(output, state["artifact_sha256"])
        try:
            completed = []
            for layer in state["layer_order"]:
                if verify_sources()[1] != source_hashes:
                    raise ValueError("controller运行期间源码发生变化")
                child = output / f"layer_{layer:02d}"
                record = state["layers"].get(str(layer))
                if child.exists():
                    if not record or record.get("status") != "completed" or record.get("exit_code") != 0:
                        raise ValueError(f"层{layer}既存 running/failed/不完整；禁止自动attempt")
                    child_manifest, rows = validate_child(child, layer, decision_sha, source_hashes)
                    if record["manifest_sha256"] != sha256_file(child / "manifest.json"):
                        raise ValueError("child manifest 与 root 完成记录哈希不符")
                else:
                    if record:
                        raise ValueError("root有child记录但子目录缺失；禁止自动重建")
                    command = [sys.executable, "-c", CHILD_CODE, str(Path(__file__).resolve().parent), str(layer),
                               "--root", str(root), "--output", str(child), "--max-fit-tokens", "8192", "--svd-driver", "gesvdj"]
                    state["layers"][str(layer)] = {"status": "running", "command": command, "started_unix": time.time()}
                    state.update({"status": "running", "current_layer": layer, "completed_layers": completed})
                    write_json(manifest_path, state)
                    exit_code = None
                    try:
                        exit_code = run_child(command, output / "logs" / f"layer_{layer:02d}.log", output / "logs" / f"layer_{layer:02d}.resources.csv")
                    except BaseException as exc:
                        exit_code = getattr(exc, "child_exit_code", None)
                        state["layers"][str(layer)]["stop_reason"] = getattr(exc, "child_stop_reason", str(exc))
                        raise
                    finally:
                        # 先保全已知退出码；归档自身失败也不能抹掉执行留痕。
                        state["layers"][str(layer)].update({"exit_code": exit_code, "ended_unix": time.time()})
                        for suffix, key in ((".log", "log_sha256"), (".resources.csv", "resource_csv_sha256")):
                            log_file = output / "logs" / f"layer_{layer:02d}{suffix}"
                            if log_file.is_file():
                                state["layers"][str(layer)][key] = sha256_file(log_file)
                        write_json(manifest_path, state)
                        finalize_child(child, layer, output, decision_sha, source_hashes, exit_code)
                    if exit_code != 0:
                        raise RuntimeError(f"层{layer}退出码{exit_code}，停止且不自动attempt")
                    child_manifest, rows = validate_child(child, layer, decision_sha, source_hashes)
                signature = reference_signature(child_manifest)
                if "reference_signature" in state and signature != state["reference_signature"]:
                    raise ValueError("跨层父模型/split/拟合行/共享输入哈希不一致")
                state["reference_signature"] = signature
                completed.append(layer)
                state["layers"][str(layer)].update({"status": "completed", "exit_code": 0, "rows": len(rows),
                    "manifest_sha256": sha256_file(child / "manifest.json")})
                state.update({"completed_layers": completed, "completed_rows": len(completed) * 9})
                write_json(manifest_path, state)
                print("SENSITIVITY_PROGRESS", layer, len(completed), "/36", flush=True)
            if len(completed) != 36 or set(completed) != set(range(36)):
                raise ValueError("未完成全部36层，不能标completed")
            # 最后再次完整验证所有child，不能仅依靠前面的内存计数。
            for layer in range(36):
                validate_child(output / f"layer_{layer:02d}", layer, decision_sha, source_hashes)
            state.update({"status": "completed", "ended_unix": time.time(), "current_layer": None, "completed_rows": 324})
            write_json(manifest_path, state)
        except BaseException as exc:
            current = state.get("current_layer")
            if current is not None and str(current) in state["layers"] and state["layers"][str(current)].get("status") != "completed":
                state["layers"][str(current)]["status"] = "failed"
            state.update({"status": "failed", "ended_unix": time.time(), "error": {"type": type(exc).__name__, "message": str(exc)}})
            write_json(manifest_path, state)
            raise


if __name__ == "__main__":
    main()
