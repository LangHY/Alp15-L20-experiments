"""隔离 packed-only headers/meta 探针。失败产物与退出码不伪装为 loaded。"""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from alp15_packed_loader import (DECISION, GateError, audit_loader_assets,
    build_meta_skeleton, load_packed_only_model, sha256_file)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage", choices=("headers", "meta", "load"), default="headers")
    value.add_argument("--root", required=True)
    value.add_argument("--awq-artifact", required=True)
    value.add_argument("--overlay", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--source-root")
    value.add_argument("--model-dir")
    value.add_argument("--config-bundle")
    value.add_argument("--helper")
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--chunk-bytes", type=int, default=33554432)
    value.add_argument("--meta-audit-dir")
    return value


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
        stream.write("\n")


def main(argv=None):
    args = parser().parse_args(argv)
    # 存在即拒绝；不覆写失败目录和旧结果。
    output = Path(args.output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    report = {"decision": DECISION, "stage": args.stage, "status": "started",
              "pid": os.getpid(),
              "argv": sys.argv if argv is None else argv, "start_unix_s": start,
              "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "code_sha256": {str(Path(__file__).resolve()): sha256_file(__file__),
                  str(Path(__file__).with_name("alp15_packed_loader.py").resolve()):
                      sha256_file(Path(__file__).with_name("alp15_packed_loader.py"))}}
    print(f"PACKED_LOADER_PHASEA_PID {os.getpid()} STAGE {args.stage}", flush=True)
    try:
        paths = {key: value for key, value in vars(args).items() if key in (
            "root", "awq_artifact", "overlay", "source_root", "model_dir", "config_bundle", "helper")
            and value is not None}
        plan = audit_loader_assets(paths)
        write_json(output / "LoaderPlan.json", plan.to_dict())
        report["plan_sha256"] = plan.plan_sha256
        report["counts"] = {"original": len(plan.original), "protected": len(plan.protected),
                            "packed_modules": len(plan.packed), "target": len(plan.target)}
        report["original_payload_read_bytes"] = 0
        report["original_payload_hash_verified"] = False
        if args.stage == "meta":
            model, processor, audit = build_meta_skeleton(plan)
            write_json(output / "MetaAudit.json", audit)
            report["status"] = "meta_passed"
        elif args.stage == "load":
            if not args.meta_audit_dir:
                raise GateError("load必须提供--meta-audit-dir且实际同来源MetaAudit通过")
            load_packed_only_model(plan, device=args.device, chunk_bytes=args.chunk_bytes,
                                   audit_dir=args.meta_audit_dir)
            raise GateError("PhaseB未实现，禁止意外返回loaded")
        else:
            report["status"] = "headers_only"
        report["exit_code"] = 0
    except Exception as error:
        report.update(status="failed", exit_code=1, error_type=type(error).__name__, error=str(error))
        if hasattr(error, "phase_evidence"):
            report["phase_evidence"] = error.phase_evidence
        if hasattr(error, "cpu_meta_candidate_audit"):
            write_json(output / "MetaAuditCandidate.json", error.cpu_meta_candidate_audit)
            report["cpu_meta_candidate_audit"] = "MetaAuditCandidate.json"
        trace = traceback.format_exc()
        with (output / "traceback.txt").open("x") as stream:
            stream.write(trace)
        print(trace, file=sys.stderr, flush=True)
    finally:
        report["end_unix_s"] = time.time()
        report["elapsed_s"] = report["end_unix_s"] - start
        report["entity_loaded"] = False
        write_json(output / "result.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
