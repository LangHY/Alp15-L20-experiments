"""归档本机结果与早期交付包；不联网、不运行模型。"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = {".json", ".csv", ".md", ".jsonl", ".txt"}
SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")


def sanitize(data):
    text = data.decode("utf-8-sig")
    if SECRET.search(text):
        raise ValueError("检测到疑似凭据，拒绝发布")
    # 保留指标，公开副本不包含个人账户、主机地址与认证命令。
    text = re.sub(r"/Users/[^/\s\"']+", "/Users/USER", text)
    text = re.sub(r"/home/[^/\s\"']+", "/home/USER", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "REDACTED_EMAIL", text)
    text = re.sub(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", "REDACTED_IP", text)
    text = re.sub(r"(?i)(?:sshpass\s+-p\s+|(?:password|passwd|api_key|access_token)\s*[=:]\s*)[^\s,;]+", "REDACTED_CREDENTIAL", text)
    return text.encode("utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--handoff", type=Path, required=True)
    args = p.parse_args()
    output = ROOT / "results/packed_evidence"
    if output.exists():
        raise FileExistsError(output)
    sources = []
    for directory in ("docs/baselines", "results/baselines"):
        for file in sorted((args.source_root / directory).rglob("*")):
            if not file.is_file() or file.suffix not in EXT:
                continue
            relative = file.relative_to(args.source_root).as_posix()
            if "/full_pool_download_" in relative or "/code/" in relative:
                continue
            sources.append(("local", relative, file.read_bytes()))
    with zipfile.ZipFile(args.handoff) as archive:
        for item in archive.infolist():
            if not item.filename.startswith("Lighten_VLA/compression/results/"):
                continue
            relative = item.filename.removeprefix("Lighten_VLA/")
            if Path(relative).suffix in EXT and "/code/" not in relative:
                sources.append(("handoff_20261004", relative, archive.read(item)))
    blobs, entries, excluded = {}, [], []
    for origin, path, original in sources:
        try:
            data = sanitize(original)
        except (UnicodeError, ValueError) as error:
            excluded.append({"origin": origin, "path": path, "reason": type(error).__name__})
            continue
        sha = hashlib.sha256(data).hexdigest()
        blobs[sha] = data
        entries.append({"origin": origin, "path": path, "original_sha256": hashlib.sha256(original).hexdigest(), "published_sha256": sha, "bytes": len(data), "sanitized": data != original})
        if origin == "local" and "/evidence/" not in path and Path(path).suffix == ".md":
            destination = ROOT / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for sha, data in sorted(blobs.items()):
            item = tarfile.TarInfo(sha)
            item.size = len(data)
            item.mtime = 0
            tar.addfile(item, io.BytesIO(data))
    packed = gzip.compress(buffer.getvalue(), compresslevel=9, mtime=0)
    encoded = base64.b64encode(packed).decode()
    output.mkdir(parents=True)
    parts = []
    for offset in range(0, len(encoded), 90000):
        name = f"part-{len(parts):04d}.b64"
        (output / name).write_text(encoded[offset:offset + 90000] + "\n")
        parts.append(name)
    index = {"scope": "local text metrics/results and 20261004 handoff compression/results; no live server collection", "packed_sha256": hashlib.sha256(packed).hexdigest(), "parts": parts, "entries": entries, "excluded": excluded, "unique_blobs": len(blobs)}
    # 索引本身压缩，避免较大的网络提交载荷。
    index_bytes = json.dumps(index, ensure_ascii=False, indent=2).encode()
    encoded_index = base64.b64encode(gzip.compress(index_bytes, mtime=0)).decode()
    (output / "index.b64").write_text(encoded_index + "\n")
    (ROOT / "results/EXPORT_STATS.json").write_text(json.dumps({"source_records": len(entries), "unique_blobs": len(blobs), "published_bytes": sum(len(b) for b in blobs.values()), "compressed_bytes": len(packed), "parts": len(parts), "excluded": excluded}, ensure_ascii=False, indent=2) + "\n")
    print((ROOT / "results/EXPORT_STATS.json").read_text())


if __name__ == "__main__":
    main()
