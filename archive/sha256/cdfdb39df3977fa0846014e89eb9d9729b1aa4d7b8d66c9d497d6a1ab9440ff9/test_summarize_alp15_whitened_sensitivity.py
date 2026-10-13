"""阶段4汇总纯标准库夹具；不加载真实模型，不运行GPU。"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY / "scripts"))
import summarize_alp15_whitened_sensitivity as summary

SPEC = importlib.util.spec_from_file_location("sensitivity_fixtures", DIRECTORY / "tests/test_alp15_whitened_sensitivity.py")
assert SPEC and SPEC.loader
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
runner = summary.runner


def make_child(command, log, resource):
    result = fixtures.fake_child(command, log, resource)
    child = Path(command[command.index("--output") + 1])
    manifest = runner.read_json(child / "manifest.json")
    manifest.update({"started_unix": 100., "ended_unix": 101., "environment": {"gpu": "CPU MOCK; no GPU"}})
    # 特意令某配置fit选whitening，以捕获汇总器默认compressed-refit的错误。
    if manifest["layers"] == [7]:
        rows = runner.read_json(child / "results.json")
        for row in rows:
            if row["method"] == "whitening" and row["rank"] == 896:
                row["train_nmse"] = .01
        saved = manifest["fit_selection"]["7"]["896"]
        saved.update({"selected_method": "whitening", "selected_calibration_eval_nmse": .08})
        saved["fit_nmse_by_method"]["whitening"] = .01
        runner.write_json(child / "results.json", rows)
        runner.write_json(child / "method_selection.json", {"rule": manifest["method_selection_rule"], "layers": manifest["fit_selection"]})
        for name in ("results.json", "method_selection.json"):
            manifest["artifact_sha256"][name] = runner.sha256_file(child / name)
        manifest["method_selection_sha256"] = manifest["artifact_sha256"]["method_selection.json"]
    runner.write_json(child / "manifest.json", manifest)
    return result


def make_prior(root, destination, layers, phase):
    """把模拟阶段4层组合成旧两层证据，用于来源一致性的独立边界测试。"""
    destination.mkdir()
    combined, artifacts, rows, info, selected, per_clip = None, {}, [], {}, {}, {}
    for layer in layers:
        child = root / f"layer_{layer:02d}"
        manifest = runner.read_json(child / "manifest.json")
        if combined is None:
            combined = copy.deepcopy(manifest)
        for name in manifest["artifact_sha256"]:
            if name not in ("results.json", "results.csv", "method_selection.json"):
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child / name, target)
                artifacts[name] = runner.sha256_file(target)
        rows.extend(runner.read_json(child / "results.json"))
        info.update(manifest["layer_inputs"])
        selected.update(manifest["fit_selection"])
        per_clip.update(manifest["per_clip_outputs"])
        combined["input_sha256"].update(manifest["input_sha256"])
    combined.update({"phase": phase, "layers": list(layers), "results_completed": 18,
        "layer_inputs": info, "fit_selection": selected, "per_clip_outputs": per_clip})
    if phase == 3:
        combined["confirmation_layers"] = list(layers)
    runner.write_json(destination / "results.json", rows)
    (destination / "results.csv").write_text("CPU MOCK legacy CSV\n")
    runner.write_json(destination / "method_selection.json", {"rule": combined["method_selection_rule"], "layers": selected})
    for name in ("results.json", "results.csv", "method_selection.json"):
        artifacts[name] = runner.sha256_file(destination / name)
    combined["artifact_sha256"] = artifacts
    combined["method_selection_sha256"] = artifacts["method_selection.json"]
    runner.write_json(destination / "manifest.json", combined)


class SensitivitySummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.root = cls.directory / "stage4"
        decision = cls.directory / "decision.md"
        decision.write_text("CPU MOCK approved protocol\n")
        argv = ["controller", "--root", str(cls.directory), "--output", str(cls.root), "--decision", str(decision)]
        with patch.object(runner.sys, "argv", argv), patch.object(runner, "run_child", side_effect=make_child), patch("builtins.print"):
            runner.main()
        cls.prior2, cls.prior3 = cls.directory / "stage2", cls.directory / "stage3"
        make_prior(cls.root, cls.prior2, (6, 35), 2)
        make_prior(cls.root, cls.prior3, (12, 24), 3)

    def invoke(self, output, root=None, priors=True):
        argv = ["summary", "--input", str(root or self.root), "--output", str(output)]
        if priors:
            argv += ["--prior-inputs", str(self.prior2), str(self.prior3)]
        with patch.object(summary.sys, "argv", argv), patch("builtins.print"):
            summary.main()

    def test_all36_324rows_72_fit_selected_pairs_and_repeat_checks(self):
        output = self.directory / "complete_summary"
        self.invoke(output)
        result = runner.read_json(output / "summary.json")
        self.assertEqual((result["configuration_rows"], result["paired_groups"]), (324, 72))
        self.assertEqual(len(result["repeat_consistency"]), 36)
        pair = next(row for row in result["paired_comparisons"] if row["layer"] == 7 and row["rank"] == 896)
        self.assertEqual(pair["selected_method"], "whitening")
        with (output / "paired_clips.csv").open(encoding="utf-8-sig", newline="") as stream:
            details = list(csv.DictReader(stream))
        self.assertEqual(len(details), 1152)
        self.assertIn("selected_method_eval_nmse", details[0])
        self.assertNotIn("fit_selected_nmse", details[0])
        self.assertIn("fit_selected_method", details[0])
        receipt = runner.read_json(output / "input_hashes.json")
        for name, digest in receipt["output_sha256"].items():
            self.assertEqual(runner.sha256_file(output / name), digest)
        with self.assertRaises(ValueError):
            self.invoke(output)

    def test_old_parent_split_sample_source_and_cache_changes_rejected(self):
        original = runner.read_json(self.prior2 / "manifest.json")
        reference = runner.read_json(self.root / "manifest.json")["reference_signature"]
        variants = []
        changed = copy.deepcopy(original)
        changed["parent_MLP_SHA256"] = "0" * 64
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["fit_clips"] = list(reversed(changed["fit_clips"]))
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["calibration_eval_clips"] = list(reversed(changed["calibration_eval_clips"]))
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["layer_inputs"]["6"]["fit_statistics_row_indices"] = list(reversed(changed["layer_inputs"]["6"]["fit_statistics_row_indices"]))
        variants.append(changed)
        for field in ("code_sha256", "phase1_code_sha256"):
            changed = copy.deepcopy(original)
            changed[field] = "0" * 64
            variants.append(changed)
        changed = copy.deepcopy(original)
        changed["input_sha256"]["parent"] = "0" * 64
        variants.append(changed)
        try:
            for changed in variants:
                runner.write_json(self.prior2 / "manifest.json", changed)
                with self.subTest(changed=changed.keys()), self.assertRaises(ValueError):
                    summary.old_results([self.prior2, self.prior3], reference)
            changed = copy.deepcopy(original)
            changed["input_sha256"]["/cache/train_06.pt"] = "0" * 64
            runner.write_json(self.prior2 / "manifest.json", changed)
            output = self.directory / "bad_cache"
            with self.assertRaises(ValueError):
                self.invoke(output)
            self.assertFalse(output.exists())
        finally:
            runner.write_json(self.prior2 / "manifest.json", original)

    def test_partial_root_is_not_reported_completed(self):
        partial = self.directory / "partial"
        partial.mkdir()
        state = runner.read_json(self.root / "manifest.json")
        state.update({"status": "running", "completed_rows": 9})
        runner.write_json(partial / "manifest.json", state)
        output = self.directory / "partial_report"
        with self.assertRaises(ValueError):
            self.invoke(output, root=partial)
        self.assertFalse(output.exists())

    def test_changed_child_artifact_blocks_report(self):
        path = self.root / "layer_00/results.json"
        original = path.read_bytes()
        try:
            path.write_text("[]\n")
            output = self.directory / "tamper_report"
            with self.assertRaises(ValueError):
                self.invoke(output)
            self.assertFalse(output.exists())
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
