#!/usr/bin/env python3
"""预取一个 Physical AI AV clip。

这不是 NVIDIA 官方脚本；它仅调用 physical_ai_av==0.2.0 的官方公开 API，
用于准备 NVIDIA 官方入口所需的一个离线 clip。默认 dry-run，不下载或写入文件。
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import operator
import re
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True

DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
DATASET_REV = "33f9bf447ed3bcb7d545ce13f4226f824214fafb"
PHYSICAL_AI_AV_COMMIT = "59d578b53ef186ffc3f1de1d1b2598717a982d64"
REQUIRED_VERSION = "0.2.0"
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000
CAMERA_ATTRIBUTES = (
    "CAMERA_CROSS_LEFT_120FOV",
    "CAMERA_FRONT_WIDE_120FOV",
    "CAMERA_CROSS_RIGHT_120FOV",
    "CAMERA_FRONT_TELE_30FOV",
)
METADATA_PATHS = (
    "features.csv",
    "clip_index.parquet",
    "metadata/feature_presence.parquet",
    "metadata/data_collection.parquet",
)
CACHE_REPO_DIR = "datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
SNAPSHOT_PREFIX = f"{CACHE_REPO_DIR}/snapshots/{DATASET_REV}"
BLOB_NAME = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
FEATURE_TEMPLATES = {
    "LABELS.EGOMOTION": "labels/egomotion/egomotion.chunk_{chunk_id:04d}.zip",
    "CAMERA.CAMERA_CROSS_LEFT_120FOV": "camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_{chunk_id:04d}.zip",
    "CAMERA.CAMERA_FRONT_WIDE_120FOV": "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{chunk_id:04d}.zip",
    "CAMERA.CAMERA_CROSS_RIGHT_120FOV": "camera/camera_cross_right_120fov/camera_cross_right_120fov.chunk_{chunk_id:04d}.zip",
    "CAMERA.CAMERA_FRONT_TELE_30FOV": "camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_{chunk_id:04d}.zip",
}


def fail(message: str) -> None:
    print(f"错误：{message}", file=sys.stderr)
    raise SystemExit(2)


def blob_content_info(path: Path, blob_name: str) -> tuple[int, str]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    git_digest = hashlib.sha1() if len(blob_name) == 40 else None
    if git_digest is not None:
        git_digest.update(f"blob {size}\0".encode("ascii"))
    bytes_read = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            if git_digest is not None:
                git_digest.update(block)
            bytes_read += len(block)
    actual_identity = git_digest.hexdigest() if git_digest is not None else digest.hexdigest()
    if bytes_read != size or path.stat().st_size != size or actual_identity != blob_name:
        fail(f"HF blob 文件名与内容摘要不匹配：{path}")
    return size, digest.hexdigest()


def require_exact_api() -> tuple[Any, Any]:
    try:
        installed_version = version("physical-ai-av")
    except PackageNotFoundError as error:
        fail(f"未安装 physical-ai-av=={REQUIRED_VERSION}：{error}")
    if installed_version != REQUIRED_VERSION:
        fail(f"physical-ai-av 版本必须为 {REQUIRED_VERSION}，实际为 {installed_version!r}")
    try:
        import physical_ai_av as avdi
    except ImportError as error:
        fail(f"physical-ai-av 发行包存在但无法导入 physical_ai_av：{error}")
    interface_cls = getattr(avdi, "PhysicalAIAVDatasetInterface", None)
    if interface_cls is None:
        fail("physical_ai_av 顶层缺少 PhysicalAIAVDatasetInterface；API 与冻结版本不一致，请升级 Sol")
    if not {"revision", "cache_dir"}.issubset(inspect.signature(interface_cls).parameters):
        fail("PhysicalAIAVDatasetInterface 构造器不支持冻结的 revision/cache_dir 参数；请升级 Sol")
    return avdi, interface_cls


def resolve_path(filename: Any) -> str:
    if not isinstance(filename, (str, os.PathLike)):
        fail("get_chunk_feature_filename 未返回路径字符串；不猜测 API 返回结构，请升级 Sol")
    raw = os.fspath(filename)
    if not isinstance(raw, str):
        fail("官方 feature 路径不是文本字符串；停止")
    relative = PurePosixPath(raw)
    if (not raw or raw != relative.as_posix() or raw.startswith("/")
            or "\\" in raw or any(part in {".", ".."} for part in raw.split("/"))):
        fail("官方 feature 路径不安全；停止")
    return raw


def source_blob(cache_dir: Path, relative_path: str) -> tuple[Path, str]:
    snapshot = cache_dir / SNAPSHOT_PREFIX / relative_path
    if not snapshot.is_symlink():
        fail(f"冻结 snapshot 链接缺失：{relative_path}")
    target = os.readlink(snapshot)
    blob = snapshot.parent / target
    expected_blobs = cache_dir / CACHE_REPO_DIR / "blobs"
    if (os.path.isabs(target) or blob.is_symlink() or not blob.is_file()
            or blob.resolve().parent != expected_blobs.resolve()
            or not BLOB_NAME.fullmatch(blob.name)):
        fail(f"冻结 snapshot 链接目标不安全或 blob 缺失：{relative_path}")
    return blob, blob.name


def copy_record(output_dir: Path, relative_path: str,
                source: tuple[Path, str, int, str],
                chunk_id: int | None = None) -> dict[str, Any]:
    blob, blob_name, source_size, source_sha = source
    dest_blob = output_dir / CACHE_REPO_DIR / "blobs" / blob_name
    dest_snapshot = output_dir / SNAPSHOT_PREFIX / relative_path
    dest_blob.parent.mkdir(parents=True, exist_ok=True)
    dest_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not dest_blob.exists():
        shutil.copy2(blob, dest_blob)
    if blob_content_info(dest_blob, blob_name) != (source_size, source_sha):
        fail(f"交付 blob 与缓存不一致：{blob_name}")
    link_target = os.path.relpath(dest_blob, dest_snapshot.parent)
    dest_snapshot.symlink_to(link_target)
    if dest_snapshot.resolve() != dest_blob.resolve():
        fail(f"交付 snapshot 链接无效：{relative_path}")
    record: dict[str, Any] = {
        "path": f"{SNAPSHOT_PREFIX}/{relative_path}",
        "size": source_size,
        "sha256": source_sha,
    }
    if chunk_id is not None:
        record["chunk"] = chunk_id
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="预取一个固定 Physical AI AV clip（默认 dry-run）")
    parser.add_argument("--output-dir", required=True, type=Path, help="空的离线包 datasets/physical-ai-av 目录")
    parser.add_argument("--cache-dir", type=Path, help="持久 HF 下载缓存；默认是 output-dir 同级的 <output-dir-name>-hf-cache")
    parser.add_argument("--execute", action="store_true", help="显式允许调用官方 API 下载")
    args = parser.parse_args()
    _, interface_cls = require_exact_api()
    output_dir = args.output_dir.resolve()
    cache_dir = (args.cache_dir or args.output_dir.with_name(args.output_dir.name + "-hf-cache")).resolve()
    if output_dir == cache_dir or output_dir in cache_dir.parents or cache_dir in output_dir.parents:
        fail("cache-dir 与 output-dir 不能相同或互相嵌套")

    print("非 NVIDIA 官方脚本；仅使用 physical_ai_av==0.2.0 顶层官方 API。")
    print(f"dataset={DATASET_REPO}@{DATASET_REV}")
    print(f"physical_ai_av_version={REQUIRED_VERSION}")
    print(f"physical_ai_av_source_commit={PHYSICAL_AI_AV_COMMIT}")
    print(f"clip_id={CLIP_ID}; t0_us={T0_US}")
    print(f"cache_dir={cache_dir}")
    print(f"output_dir={output_dir}")
    if not args.execute:
        print("dry-run：未实例化数据集接口、未创建目录、未调用 get_clip_chunk 或 download_file。")
        print("dry-run 不声称已解析 chunk、计划文件或大小。")
        return 0

    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        fail(f"输出目录必须不存在或为空目录，拒绝混入/覆盖已有资产：{output_dir}")
    if cache_dir.exists() and not cache_dir.is_dir():
        fail(f"cache-dir 不是目录：{cache_dir}")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    interface = interface_cls(revision=DATASET_REV, cache_dir=str(cache_dir))

    features_namespace = getattr(interface, "features", None)
    labels = getattr(features_namespace, "LABELS", None)
    camera = getattr(features_namespace, "CAMERA", None)
    egomotion = getattr(labels, "EGOMOTION", None)
    if features_namespace is None or egomotion is None or camera is None:
        fail("interface.features 缺少 LABELS.EGOMOTION 或 CAMERA；不猜测兼容 API，请升级 Sol")
    feature_specs = [("LABELS.EGOMOTION", egomotion)]
    for attribute in CAMERA_ATTRIBUTES:
        value = getattr(camera, attribute, None)
        if value is None:
            fail(f"interface.features.CAMERA 缺少冻结属性 {attribute}；请升级 Sol")
        feature_specs.append((f"CAMERA.{attribute}", value))
    get_chunk = getattr(interface, "get_clip_chunk", None)
    download_file = getattr(interface, "download_file", None)
    get_filename = getattr(features_namespace, "get_chunk_feature_filename", None)
    if get_chunk is None or download_file is None or get_filename is None:
        fail("interface 缺少 get_clip_chunk/download_file 或 features.get_chunk_feature_filename；请升级 Sol")
    if "filename" not in inspect.signature(download_file).parameters:
        fail("download_file 不支持冻结的 filename 参数；不猜测兼容调用，请升级 Sol")
    if "clip_id" not in inspect.signature(get_chunk).parameters:
        fail("get_clip_chunk 不支持冻结的 clip_id 参数；不猜测兼容调用，请升级 Sol")
    if len(inspect.signature(get_filename).parameters) != 2:
        fail("get_chunk_feature_filename 不是冻结的双参数 API；不猜测兼容调用，请升级 Sol")

    try:
        normalized_chunk = operator.index(chunk_id := get_chunk(CLIP_ID))
    except TypeError:
        fail(f"get_clip_chunk 必须返回非负整数，实际为 {chunk_id!r}")
    if isinstance(chunk_id, bool) or normalized_chunk < 0:
        fail(f"get_clip_chunk 必须返回非负整数，实际为 {chunk_id!r}")
    chunk_id = normalized_chunk
    feature_plan = []
    for feature_key, feature in feature_specs:
        relative = resolve_path(get_filename(chunk_id, feature))
        if relative != FEATURE_TEMPLATES[feature_key].format(chunk_id=chunk_id):
            fail(f"冻结 feature 路径与官方当前返回不一致：{feature_key}={relative}")
        feature_plan.append((feature_key, relative))
    if len({relative for _, relative in feature_plan}) != 5 or set(METADATA_PATHS) & {relative for _, relative in feature_plan}:
        fail("冻结 feature 路径重复或与 metadata 冲突")
    print(f"chunk_id={chunk_id}")
    print("features=" + ", ".join(relative for _, relative in feature_plan))

    # 逐个下载官方分块 ZIP，复用持久缓存；本脚本不制作 .pt 或自定义 ZIP。
    for _, relative in feature_plan:
        download_file(relative)
    sources = {}
    for relative in [name for _, name in feature_plan] + list(METADATA_PATHS):
        blob, blob_name = source_blob(cache_dir, relative)
        size, digest = blob_content_info(blob, blob_name)
        sources[relative] = (blob, blob_name, size, digest)
    feature_records = []
    for feature_key, relative in feature_plan:
        record = copy_record(output_dir, relative, sources[relative], chunk_id)
        record["feature"] = feature_key
        feature_records.append(record)
    metadata_records = [copy_record(output_dir, relative, sources[relative]) for relative in METADATA_PATHS]
    manifest = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REV,
        "physical_ai_av_version": REQUIRED_VERSION,
        "physical_ai_av_source_commit": PHYSICAL_AI_AV_COMMIT,
        "clip_id": CLIP_ID,
        "t0_us": T0_US,
        "chunk_id": chunk_id,
        "features": feature_records,
        "metadata": metadata_records,
    }
    manifest_path = output_dir / "dataset-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"已生成 dataset manifest：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
