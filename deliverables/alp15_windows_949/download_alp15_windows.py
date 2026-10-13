"""冻结的949片段候选池下载器；默认仅离线检查计划。"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
REVISION = "33f9bf447ed3bcb7d545ce13f4226f824214fafb"
ENDPOINT = "https://huggingface.co"
PLAN_SHA = "7a2356ea86921d535a96ef84e49f1018f8209dd31f7011423c6574372703471f"
CSV_SHA = "9fbf80b944f8b810557838da52e7318d9cc7d3855746bc7f39b183dd539e76af"
TOTAL = 54382445102
HUB_VERSION = "1.31.0"
BASE = Path(__file__).absolute().parent
DEFAULT_OUTPUT = r"E:\alp15-small-test-949"
# 必须在import huggingface_hub之前设置；只使用公开HTTP下载与续传。
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_ENDPOINT"] = ENDPOINT
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class DownloadError(ValueError):
    """仅此类型的受控错误可直接显示给用户。"""


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def reject_links(path):
    """lstat检查自身及祖先，不跟随Windows junction/reparse point。"""
    path = Path(path)
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise DownloadError("Link/reparse point is forbidden: " + str(part))


def regular_file(path):
    reject_links(path)
    if path.exists() and not path.is_file():
        raise DownloadError("File path conflict; preserved: " + str(path))


def load_plan():
    plan_path, csv_path = BASE / "small-download-plan.json", BASE / "test-candidates.csv"
    regular_file(plan_path)
    regular_file(csv_path)
    if digest(plan_path) != PLAN_SHA or digest(csv_path) != CSV_SHA:
        raise DownloadError("Bundled plan/CSV SHA256 mismatch; stop.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan["dataset"] != REPO or plan["revision"] != REVISION
            or plan["authority_source"] != ENDPOINT or plan["transport_source"] != ENDPOINT
            or plan["clip_count"] != 949 or plan["chunk_count"] != 10
            or len(plan["files"]) != 50 or plan["total_bytes"] != TOTAL):
        raise DownloadError("Frozen subset identity mismatch.")
    paths = set()
    for row in plan["files"]:
        name = row["path"]
        parts = PurePosixPath(name).parts
        if (not name.endswith(".zip") or not parts or name.startswith("/")
                or ".." in parts or "\\" in name or ":" in name
                or any(part in (".", "") or part.endswith((".", " ")) for part in name.split("/"))
                or name.casefold() in paths):
            raise DownloadError("Unsafe or duplicate plan path.")
        paths.add(name.casefold())
        if type(row["size"]) is not int or row["size"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise DownloadError("Missing size/SHA256 in plan.")
    if sum(row["size"] for row in plan["files"]) != TOTAL:
        raise DownloadError("Frozen total mismatch.")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if (len(rows) != 949 or len({row["clip_id"] for row in rows}) != 949
            or {int(row["chunk"]) for row in rows} != set(plan["chunks"])):
        raise DownloadError("949 candidate CSV mismatch.")
    return plan


def validate_output(output):
    out = Path(output)
    if not out.is_absolute() or ".." in out.parts:
        raise DownloadError("Output must be an absolute dedicated directory without '..'.")
    if out == Path(out.anchor) or out == Path.home().absolute():
        raise DownloadError("Drive/filesystem root and user home cannot be output directories.")
    if os.name == "nt":
        win = PureWindowsPath(output)
        if not re.fullmatch(r"[A-Za-z]:", win.drive):
            raise DownloadError("Output must use a local Windows drive, for example E:\\alp15-small-test-949.")
        if any(part.endswith((".", " ")) or ":" in part for part in win.parts[1:]):
            raise DownloadError("Unsafe Windows output path.")
    reject_links(out)
    for parent in (out, *out.parents):
        if parent.exists() and not parent.is_dir():
            raise DownloadError("Directory path conflict; preserved: " + str(parent))
    return out


def inspect_cache(out):
    cache = out / ".cache"
    reject_links(cache)
    if cache.exists() and not cache.is_dir():
        raise DownloadError("Cache path conflict; preserved.")
    if cache.exists():
        for parent, folders, files in os.walk(cache, followlinks=False):
            reject_links(Path(parent))
            for name in folders + files:
                reject_links(Path(parent) / name)
    hf_cache = cache / "huggingface"
    if hf_cache.exists() and not hf_cache.is_dir():
        raise DownloadError("Hugging Face cache path conflict; preserved.")


def identity():
    return {"plan_sha256": PLAN_SHA, "candidates_sha256": CSV_SHA,
            "dataset": REPO, "revision": REVISION}


def inspect_records(out, plan):
    expected = {row["path"]: row for row in plan["files"]}
    for name in ("verified-files.jsonl", "download-complete.jsonl"):
        path = out / name
        regular_file(path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if any(row.get(key) != value for key, value in identity().items()):
                    raise DownloadError("Receipt belongs to another plan; preserved: " + name)
                if name == "verified-files.jsonl":
                    item = expected.get(row.get("path"))
                    if item is None or any(row.get(key) != item[key] for key in ("size", "sha256")):
                        raise DownloadError("Receipt file identity mismatch; preserved.")
                elif row.get("files") != 50 or row.get("bytes_verified") != TOTAL or row.get("clip_count") != 949:
                    raise DownloadError("Completion identity mismatch; preserved.")


def inspect_files(out, plan):
    verified = set()
    for row in plan["files"]:
        target = out / row["path"]
        regular_file(target)
        if target.exists():
            if target.stat().st_size != row["size"] or digest(target) != row["sha256"]:
                raise DownloadError("Existing ZIP mismatch; preserved, never overwritten: " + row["path"])
            verified.add(row["path"])
    return verified


def inspect_metadata(out):
    for name in ("small-download-plan.json", "test-candidates.csv"):
        target = out / name
        regular_file(target)
        if target.exists() and digest(target) != digest(BASE / name):
            raise DownloadError("Existing metadata differs; preserved: " + name)


def check_space(out, needed):
    existing = out
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    minimum = math.ceil(needed * 1.05)
    print("Free bytes: %d; required bytes (remaining ZIPs + 5%%): %d" % (free, minimum), flush=True)
    if free < minimum:
        raise DownloadError("Insufficient disk space; download not started.")


def append_record(path, value):
    regular_file(path)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def download(plan, output):
    out = validate_output(output)
    inspect_cache(out)
    inspect_records(out, plan)
    inspect_metadata(out)
    verified = inspect_files(out, plan)
    remaining = sum(row["size"] for row in plan["files"] if row["path"] not in verified)
    check_space(out, remaining)
    if importlib.metadata.version("huggingface_hub") != HUB_VERSION:
        raise DownloadError("Use the supplied PowerShell venv with huggingface_hub==" + HUB_VERSION)
    from huggingface_hub import HfApi, get_token, hf_hub_download
    token = get_token()
    if not token:
        raise DownloadError("No saved Hugging Face login. Run the venv hf auth login command in README.")
    try:
        HfApi(endpoint=ENDPOINT).whoami(token=token)
    except Exception:
        raise DownloadError("Saved Hugging Face authentication is invalid/unreachable; check login and network.") from None
    out.mkdir(parents=True, exist_ok=True)
    for name in ("small-download-plan.json", "test-candidates.csv"):
        target = out / name
        regular_file(target)
        if not target.exists():
            with target.open("xb") as handle:
                handle.write((BASE / name).read_bytes())
    for index, row in enumerate(plan["files"], 1):
        target = out / row["path"]
        regular_file(target)
        inspect_cache(out)
        if row["path"] not in verified:
            if target.exists():
                raise DownloadError("Target appeared during download; preserved: " + row["path"])
            check_space(out, remaining)
            # local_dir缓存由官方客户端管理，失败保留.incomplete以便续传。
            actual = Path(hf_hub_download(REPO, row["path"], repo_type="dataset",
                revision=REVISION, endpoint=ENDPOINT, local_dir=str(out), token=token))
            if actual.absolute() != target.absolute():
                raise DownloadError("Client returned an unexpected target path; files preserved.")
        regular_file(target)
        if target.stat().st_size != row["size"] or digest(target) != row["sha256"]:
            raise DownloadError("Downloaded ZIP failed size/SHA256; files and cache preserved: " + row["path"])
        append_record(out / "verified-files.jsonl", dict(identity(), **row,
            verified_at=datetime.now(timezone.utc).isoformat()))
        if row["path"] not in verified:
            remaining -= row["size"]
        print("Verified %d / 50: %s" % (index, row["path"]), flush=True)
    append_record(out / "download-complete.jsonl", dict(identity(), files=50,
        bytes_verified=TOTAL, clip_count=949, independence_verified=False, input_qc_verified=False,
        completed_at=datetime.now(timezone.utc).isoformat()))
    print("COMPLETE: 50 ZIPs verified for 949 candidate clips. No extraction/model run performed.")


def main():
    if sys.version_info < (3, 10) or struct.calcsize("P") != 8:
        raise DownloadError("64-bit Python 3.10 or newer is required; Python 3.12 is recommended.")
    parser = argparse.ArgumentParser(description="Frozen 949-clip / 50-ZIP download. Default: offline plan only.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--confirm-bytes", type=int)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan = load_plan()
    print(json.dumps(dict(identity(), files=50, clip_count=949, total_bytes=TOTAL,
        GB=TOTAL / 10**9, GiB=TOTAL / 2**30, output=args.output,
        scope=plan["scope"], source=ENDPOINT), indent=2), flush=True)
    if not args.download:
        print("PLAN ONLY: no network, dependency installation, output writes, or ZIP download.")
        return 0
    if args.confirm_bytes != TOTAL:
        raise DownloadError("Download requires --confirm-bytes 54382445102.")
    download(plan, args.output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Files/cache preserved; rerun the same explicit command to resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        # 不输出第三方异常内容，避免URL/header中可能存在认证数据。
        message = str(exc) if isinstance(exc, DownloadError) else type(exc).__name__
        print("STOP: " + message + "\nFiles/cache preserved. No automatic deletion.", file=sys.stderr)
        sys.exit(1)
