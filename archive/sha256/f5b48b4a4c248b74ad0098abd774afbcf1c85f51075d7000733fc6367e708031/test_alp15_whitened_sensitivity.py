"""纯标准库模拟公开 CLI；不运行真实 child、模型、GPU 或网络。"""

from __future__ import annotations

import copy
import fcntl
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/alp15_whitened_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("sensitivity", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fake_child(command, log_path, resource_path):
    """生成独立数学一致的9条配置；这里只模拟执行协议，不伪装模型结果。"""
    layer = int(command[4])
    child = Path(command[command.index("--output") + 1])
    child.mkdir()
    log_path.write_text("CPU MOCK ONLY; no model/GPU\n")
    resource_path.write_text("unix_time,child_pid,rss_kib,child_gpu_used_mib,other_compute_pids\n0,0,0,0,\n")
    fit = [f"fit_{index}" for index in range(64)]
    evaluation = [f"eval_{index}" for index in range(16)]
    artifacts, per_clip_hashes, rows, selections = {}, {}, [], {}
    configurations = [("parent", 0)] + [(method, rank) for rank in runner.RANKS for method in ("linked_channel", *runner.METHODS)]
    for method, rank in configurations:
        nmse = {"parent": 0., "linked_channel": .1, "whitening": .08, "whitening_refit_parent": .07, "whitening_refit_compressed": .06}[method]
        params = 66060288 if rank == 0 else 3 * rank * 9472
        clips = [{"clip_id": clip, "nmse": nmse, "squared_error": nmse * 10., "target_energy": 10.,
                  "tokens": 159 if index == 0 else 144} for index, clip in enumerate(evaluation)]
        filename = f"layer_{layer:02d}_{method}_rank_{rank}_calibration_eval.json"
        runner.write_json(child / filename, clips)
        artifacts[filename] = per_clip_hashes[filename] = runner.sha256_file(child / filename)
        rows.append({"layer": layer, "method": method, "rank": rank, "channel_width": params // 12288 if method in ("parent", "linked_channel") else None,
            "actual_params": params, "parent_params": 66060288, "parameter_ratio": params / 66060288,
            "train_nmse": nmse, "calibration_eval_nmse": nmse, "train_tokens": 9185,
            "calibration_eval_tokens": 2319, "fit_statistics_tokens": 8192, "bf16_finite": True,
            "ridge_mu": .0001 if "refit" in method else None, "elapsed_seconds": 1.,
            "shared_gram_svd_seconds": 1., "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
            "per_clip_file": filename})
    for rank in runner.RANKS:
        selections[str(rank)] = {"selected_method": "whitening_refit_compressed", "fit_nmse_by_method": {
            row["method"]: row["train_nmse"] for row in rows if row["rank"] == rank},
            "selected_calibration_eval_nmse": .06, "linked_channel_comparator_calibration_eval_nmse": .1}
    selection = {"rule": "CPU MOCK fit-only", "layers": {str(layer): selections}}
    runner.write_json(child / "results.json", rows)
    (child / "results.csv").write_text("CPU MOCK CSV\n")
    runner.write_json(child / "method_selection.json", selection)
    for name in ("results.json", "results.csv", "method_selection.json"):
        artifacts[name] = runner.sha256_file(child / name)
    manifest = {"status": "completed", "code_sha256": runner.ENGINE_SHA256,
        "phase1_code_sha256": runner.BASE_SHA256, "layers": [layer], "ranks": list(runner.RANKS),
        "seed": runner.SEED, "max_fit_tokens": 8192, "svd_driver": "gesvdj", "results_completed": 9,
        "parent_MLP_SHA256": "be6239cbbd8c6cfd01317a33233486c1a422469ee0c323bdfe37767c75d0ab27",
        "input_sha256": {"parent": "a" * 64, f"/cache/train_{layer:02d}.pt": "b" * 64},
        "fit_clips": fit, "calibration_eval_clips": evaluation,
        "layer_inputs": {str(layer): {"fit_tokens": 9185, "calibration_eval_tokens": 2319, "fit_statistics_tokens": 8192,
            "fit_statistics_row_indices": list(range(8192)), "parent_cache_nmse": 0., "gate_up_lambda": .001, "down_lambda": .001}},
        "artifact_sha256": artifacts, "per_clip_outputs": per_clip_hashes,
        "method_selection_rule": selection["rule"], "fit_selection": selection["layers"],
        "method_selection_sha256": artifacts["method_selection.json"]}
    runner.write_json(child / "manifest.json", manifest)
    return 0


class SensitivityProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "experiment"
        self.decision = self.root / "approved.md"
        self.decision.write_text("CPU MOCK approved protocol fixture; no actual experiment\n")

    def invoke(self, resume=False, implementation=fake_child):
        argv = [str(SCRIPT), "--root", str(self.root), "--output", str(self.output), "--decision", str(self.decision)]
        if resume:
            argv.append("--resume")
        with patch.object(runner.sys, "argv", argv), patch.object(runner, "run_child", side_effect=implementation) as execute, patch("builtins.print"):
            runner.main()
            return execute.call_count

    def test_public_cli_all36_and_safe_completed_resume(self):
        self.assertEqual(self.invoke(), 36)
        state = runner.read_json(self.output / "manifest.json")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["completed_rows"], 324)
        self.assertEqual(set(state["completed_layers"]), set(range(36)))
        self.assertEqual(state["layer_order"], runner.layer_order())
        self.assertEqual(self.invoke(resume=True), 0)
        with self.assertRaises(FileExistsError):
            self.invoke()

    def test_failed_child_preserved_and_not_retried(self):
        def fail(command, log, resource):
            fake_child(command, log, resource)
            return 7
        with self.assertRaises(RuntimeError):
            self.invoke(implementation=fail)
        layer = runner.layer_order()[0]
        child = self.output / f"layer_{layer:02d}"
        manifest_hash = runner.sha256_file(child / "manifest.json")
        with self.assertRaises(ValueError):
            self.invoke(resume=True)
        self.assertEqual(runner.sha256_file(child / "manifest.json"), manifest_hash)
        self.assertEqual(runner.read_json(self.output / "manifest.json")["status"], "failed")

    def test_missing_decision_or_source_change_rejected(self):
        self.invoke()
        self.decision.write_text("Changed protocol\n")
        with self.assertRaises(ValueError):
            self.invoke(resume=True)
        with patch.object(runner, "ENGINE_SHA256", "0" * 64):
            with self.assertRaises(ValueError):
                runner.verify_sources()

    def test_tampered_artifact_and_incomplete_rows_rejected(self):
        self.invoke()
        layer = runner.layer_order()[0]
        child = self.output / f"layer_{layer:02d}"
        (child / "results.json").write_text("[]\n")
        state = runner.read_json(self.output / "manifest.json")
        with self.assertRaises(ValueError):
            runner.validate_child(child, layer, state["decision_sha256"], state["source_sha256"])
        with self.assertRaises(ValueError):
            self.invoke(resume=True)

    def test_incomplete_running_child_blocks_without_attempt(self):
        def interrupted(command, log, resource):
            child = Path(command[command.index("--output") + 1])
            child.mkdir()
            log.write_text("Interrupted CPU mock\n")
            resource.write_text("header\n")
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(implementation=interrupted)
        with self.assertRaises(ValueError):
            self.invoke(resume=True)

    def test_lock_and_deterministic_order(self):
        self.invoke()
        with (self.output / "controller.lock").open("a+") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(RuntimeError):
                self.invoke(resume=True)
        self.assertEqual(len(set(runner.layer_order())), 36)
        self.assertNotEqual(runner.layer_order(), list(range(36)))

    def test_competition_stop_records_actual_child_exit_code(self):
        class MockProcess:
            pid = 123
            returncode = None
            def poll(self):
                return self.returncode
            def terminate(self):
                self.returncode = -15
            def wait(self, timeout=None):
                return self.returncode
        observation = {"unix_time": 0, "child_pid": 123, "rss_kib": 1,
                       "child_gpu_used_mib": 1., "other_compute_pids": "456"}
        with patch.object(runner.subprocess, "Popen", return_value=MockProcess()), patch.object(runner, "resource_sample", return_value=observation):
            with self.assertRaises(RuntimeError) as stopped:
                runner.run_child(["mock"], self.root / "mock.log", self.root / "resources.csv")
        self.assertEqual(stopped.exception.child_exit_code, -15)
        self.assertIn("exit_code=-15", (self.root / "mock.log").read_text())

    def test_monitor_exception_records_actual_exit_code(self):
        class MockProcess:
            pid = 321
            returncode = None
            def poll(self):
                return self.returncode
            def terminate(self):
                self.returncode = -15
            def wait(self, timeout=None):
                return self.returncode
        with patch.object(runner.subprocess, "Popen", return_value=MockProcess()), patch.object(runner, "resource_sample", side_effect=RuntimeError("monitor failed")):
            with self.assertRaises(RuntimeError) as stopped:
                runner.run_child(["mock"], self.root / "monitor.log", self.root / "monitor.resources.csv")
        self.assertEqual(stopped.exception.child_exit_code, -15)
        self.assertEqual(stopped.exception.child_stop_reason, "monitor failed")

    def test_archive_failure_retains_known_exit_and_end_time(self):
        def invalid_manifest(command, log, resource):
            fake_child(command, log, resource)
            child = Path(command[command.index("--output") + 1])
            (child / "manifest.json").write_text("invalid JSON\n")
            return 0
        with self.assertRaises(json.JSONDecodeError):
            self.invoke(implementation=invalid_manifest)
        state = runner.read_json(self.output / "manifest.json")
        record = state["layers"][str(runner.layer_order()[0])]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["exit_code"], 0)
        self.assertIn("ended_unix", record)
        self.assertIn("log_sha256", record)
        self.assertIn("resource_csv_sha256", record)


if __name__ == "__main__":
    unittest.main()
