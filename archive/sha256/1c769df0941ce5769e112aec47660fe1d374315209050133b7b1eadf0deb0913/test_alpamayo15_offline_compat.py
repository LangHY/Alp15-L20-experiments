"""Alpamayo 1.5 单 clip 离线兼容性测试；不替代 NVIDIA 官方原始入口。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import operator
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000
CHUNK_ID = 3119
DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
DATASET_REVISION = "33f9bf447ed3bcb7d545ce13f4226f824214fafb"
PHYSICAL_AI_AV_COMMIT = "59d578b53ef186ffc3f1de1d1b2598717a982d64"
MODEL_VLM_NAME = "nvidia/Cosmos-Reason2-8B"
COSMOS_REVISION = "a9fae2cf89dc64db96b12860417f0eb403013bb9"
QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
DATASET_CACHE_NAME = "datasets--nvidia--PhysicalAI-Autonomous-Vehicles"

# 冻结值来自 docs/baselines/ALP15-OFFLINE-ASSET-001.md 与已验收的单 clip manifest。
MODEL_FILES = {
    "config.json": (
        3058,
        "824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e",
    ),
    "model.safetensors.index.json": (
        104778,
        "b899e51816e15a96edc7cd57a01cd36c5e39fe1bcbf1537c4ac914f0cad43a0d",
    ),
    "model-00001-of-00005.safetensors": (
        4928204944,
        "537259bb56815f9dfdcb028d5606e84dff5ab0c8e0e559782d63d91b6671d15c",
    ),
    "model-00002-of-00005.safetensors": (
        4915963032,
        "18841e5049b7836c16c034123ae6d40b2ff0414f9f77008bc7ca72490302a47b",
    ),
    "model-00003-of-00005.safetensors": (
        4983071160,
        "e8953b27fe827a60604a006b07ab1216fe766375f9926d1c3bd5ecee84ecef78",
    ),
    "model-00004-of-00005.safetensors": (
        4980341192,
        "604f0c0f19986f363a23a1e0b5736eedadd231ca285beab28d5db968a1d37602",
    ),
    "model-00005-of-00005.safetensors": (
        2349614196,
        "9d889c09634e5a21b4c957957ede2cdab4418c268ee1d870c1f942d04242adf6",
    ),
}

FEATURES = {
    "LABELS.EGOMOTION": (
        "labels/egomotion/egomotion.chunk_3119.zip",
        37429314,
        "0ebffb3ffc643ea0d8ad49385d5482badf57b0a2dc56a480dca48e3ef403ee05",
    ),
    "CAMERA.CAMERA_CROSS_LEFT_120FOV": (
        "camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_3119.zip",
        1843721004,
        "fc52a97e895ff4ec14b17fe16be57ac7de748e6ca97a551281a3afa0190b2c94",
    ),
    "CAMERA.CAMERA_FRONT_WIDE_120FOV": (
        "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_3119.zip",
        1529028333,
        "433cf9f1d73026392c5924f837890999cca181c9ed92edc1e590426978f8649b",
    ),
    "CAMERA.CAMERA_CROSS_RIGHT_120FOV": (
        "camera/camera_cross_right_120fov/camera_cross_right_120fov.chunk_3119.zip",
        1873594402,
        "51838bcb7e8446fcfb16a31f4a59cf3be0da4fd74237cd60f5c5cba9887c1556",
    ),
    "CAMERA.CAMERA_FRONT_TELE_30FOV": (
        "camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_3119.zip",
        1245382715,
        "efcd155958fdbe79523fd9a3bf080598ce5b16f602350c554b4bee3f87abc140",
    ),
}

METADATA = {
    "features.csv": (
        7076,
        "61105260827f5cf193c6c3ecbfed13b185432d04899d52bff3a7430ab4c0f304",
    ),
    "clip_index.parquet": (
        11132300,
        "9630a3de5ee057ded194deaa94d3b9b42af6f18c301785706e50141314a6935c",
    ),
    "metadata/feature_presence.parquet": (
        11209157,
        "a9946feb269124e3a7cbfe22d97669585735134f634c2ca97332e7abc4314790",
    ),
    "metadata/data_collection.parquet": (
        11339400,
        "9624750cadabf8028264cc1404e75e81777654be9def06b00e35c9f834b173d5",
    ),
}

CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.json",
    "merges.txt",
    "vocab.json",
)


def _directory(raw: str, label: str) -> Path:
    path = Path(raw).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{label} 不是目录: {path}")
    return path


def _verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件: {path}")
    size = path.stat().st_size
    if size != expected_size:
        raise ValueError(f"大小不匹配: {path}: {size} != {expected_size}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"SHA-256 不匹配: {path}")


def _verify_model(model_dir: Path) -> dict:
    for filename, (size, digest) in MODEL_FILES.items():
        _verify_file(model_dir / filename, size, digest)
    expected_shards = {name for name in MODEL_FILES if name.endswith(".safetensors")}
    actual_shards = {path.name for path in model_dir.glob("*.safetensors")}
    if actual_shards != expected_shards:
        raise ValueError(f"模型分片集合不匹配: {sorted(actual_shards)}")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != expected_shards:
        raise ValueError("checkpoint 索引引用了非冻结模型分片")
    payload = json.loads((model_dir / "config.json").read_text())
    if payload.get("vlm_name_or_path") != MODEL_VLM_NAME:
        raise ValueError("模型 config.json 的 VLM 来源不是冻结的 Cosmos 仓库")
    return payload


def _verify_dataset(dataset_dir: Path) -> None:
    manifest_path = dataset_dir / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_headers = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "clip_id": CLIP_ID,
        "t0_us": T0_US,
        "chunk_id": CHUNK_ID,
        "physical_ai_av_version": "0.2.0",
        "physical_ai_av_source_commit": PHYSICAL_AI_AV_COMMIT,
    }
    for key, expected in expected_headers.items():
        if manifest.get(key) != expected:
            raise ValueError(f"dataset manifest {key} 不匹配")

    prefix = f"{DATASET_CACHE_NAME}/snapshots/{DATASET_REVISION}/"
    blob_dir = dataset_dir / DATASET_CACHE_NAME / "blobs"
    records = manifest.get("features", []) + manifest.get("metadata", [])
    if len(manifest.get("features", [])) != len(FEATURES) or len(
        manifest.get("metadata", [])
    ) != len(METADATA):
        raise ValueError("dataset manifest 条目数量不匹配")
    expected_records = {
        prefix + path: (size, digest, feature)
        for feature, (path, size, digest) in FEATURES.items()
    }
    expected_records.update(
        {
            prefix + path: (size, digest, None)
            for path, (size, digest) in METADATA.items()
        }
    )
    if {record.get("path") for record in records} != set(expected_records):
        raise ValueError("dataset manifest 文件路径集合不匹配")
    for record in records:
        relative = record["path"]
        size, digest, feature = expected_records[relative]
        if record.get("size") != size or record.get("sha256") != digest:
            raise ValueError(f"dataset manifest 大小或哈希不匹配: {relative}")
        if feature is not None and (
            record.get("feature") != feature or record.get("chunk") != CHUNK_ID
        ):
            raise ValueError(f"dataset manifest feature/chunk 不匹配: {relative}")
        link = dataset_dir / relative
        if not link.is_symlink():
            raise ValueError(f"dataset snapshot 不是符号链接: {link}")
        blob = link.resolve(strict=True)
        if blob.parent != blob_dir:
            raise ValueError(f"dataset 符号链接越过交付 blob 目录: {link}")
        _verify_file(blob, size, digest)


def _verify_config_dir(config_dir: Path, label: str) -> None:
    for filename in CONFIG_FILES:
        if not (config_dir / filename).is_file():
            raise FileNotFoundError(f"{label} 缺少非权重配置: {filename}")
    if any(
        any(config_dir.rglob(pattern)) for pattern in ("*.safetensors", "*.bin", "*.pt")
    ):
        raise ValueError(f"{label} 配置目录含有权重文件")


def _verify_config_source(
    config_dir: Path,
    label: str,
    repo: str,
    revision: str,
    manifest_name: str,
    manifest_dir: Path | None,
) -> None:
    """用 bundle 来源清单或固定 HF snapshot+blob 身份证明配置来源。"""
    if manifest_dir is not None:
        manifest = json.loads((manifest_dir / manifest_name).read_text())
        if manifest.get("repo") != repo or manifest.get("revision") != revision:
            raise ValueError(f"{label} 配置来源 revision 不匹配")
        files = manifest.get("files")
        if not isinstance(files, list) or {item.get("path") for item in files} != set(
            CONFIG_FILES
        ):
            raise ValueError(f"{label} 来源清单文件集合不匹配")
        if len(files) != len(CONFIG_FILES):
            raise ValueError(f"{label} 来源清单有重复文件")
        for item in files:
            _verify_file(config_dir / item["path"], item["size"], item["sha256"])
        return

    expected_cache_name = "models--" + repo.replace("/", "--")
    if (
        config_dir.name != revision
        or config_dir.parent.name != "snapshots"
        or config_dir.parent.parent.name != expected_cache_name
    ):
        raise ValueError(
            f"{label} 配置目录不是冻结 HF snapshot；请传 --config-manifest-dir"
        )
    blob_dir = config_dir.parent.parent / "blobs"
    for filename in CONFIG_FILES:
        link = config_dir / filename
        if not link.is_symlink():
            raise ValueError(f"{label} HF snapshot 文件不是 blob 符号链接: {filename}")
        blob = link.resolve(strict=True)
        if blob.parent != blob_dir:
            raise ValueError(f"{label} HF snapshot 链接越过 blob 目录: {filename}")
        size = blob.stat().st_size
        if len(blob.name) == 40:
            digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
        elif len(blob.name) == 64:
            digest = hashlib.sha256()
        else:
            raise ValueError(f"{label} HF blob 文件名无效: {filename}")
        with blob.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != blob.name:
            raise ValueError(f"{label} HF blob 身份不匹配: {filename}")


def _require_physical_ai_av_version() -> None:
    version = importlib.metadata.version("physical-ai-av")
    if version != "0.2.0":
        raise RuntimeError(f"physical-ai-av 版本应为 0.2.0，实际为 {version}")


def _enable_offline_mode() -> None:
    # 必须在首次导入 huggingface_hub/transformers 前设置。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


@contextmanager
def _block_network():
    """阻止 Python socket 外联；允许 CUDA/本地依赖可能使用的 Unix 域 IPC。"""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock, address):
        if sock.family != socket.AF_UNIX:
            raise RuntimeError(f"离线兼容性测试阻止外部 socket.connect: {address!r}")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family != socket.AF_UNIX:
            raise RuntimeError(f"离线兼容性测试阻止外部 socket.connect_ex: {address!r}")
        return original_connect_ex(sock, address)

    def create_connection(address, *args, **kwargs):
        raise RuntimeError(f"离线兼容性测试阻止 socket.create_connection: {address!r}")

    with (
        patch.object(socket.socket, "connect", connect),
        patch.object(socket.socket, "connect_ex", connect_ex),
        patch.object(socket, "create_connection", create_connection),
    ):
        yield


def _load_offline_dataset_interface(dataset_dir: Path):
    import huggingface_hub
    import physical_ai_av

    expected_call = {
        "repo_id": DATASET_REPO,
        "repo_type": "dataset",
        "revision": DATASET_REVISION,
        "filename": "metadata/feature_presence.parquet",
    }
    calls = []

    def cached_metadata_exists(_api, *args, **kwargs):
        # physical_ai_av 0.2.0 在初始化时无条件远程查询此单个元数据文件。
        # 已由 _verify_dataset 验证其存在与哈希，只允许回答这一精确查询。
        if args or kwargs != expected_call:
            raise RuntimeError(
                f"出现未批准的 Hugging Face 文件探测: {args!r}, {kwargs!r}"
            )
        calls.append(kwargs)
        return True

    with patch.object(huggingface_hub.HfApi, "file_exists", cached_metadata_exists):
        interface = physical_ai_av.PhysicalAIAVDatasetInterface(
            revision=DATASET_REVISION,
            cache_dir=str(dataset_dir),
            token=False,
        )
    if calls != [expected_call]:
        raise RuntimeError("physical_ai_av 元数据检查行为与 0.2.0 冻结实现不符")
    if operator.index(interface.get_clip_chunk(CLIP_ID)) != CHUNK_ID:
        raise ValueError("官方数据接口返回的 chunk_id 不匹配")
    for feature, (relative, _, _) in FEATURES.items():
        category, name = feature.split(".", 1)
        official_feature = getattr(getattr(interface.features, category), name)
        if (
            interface.features.get_chunk_feature_filename(CHUNK_ID, official_feature)
            != relative
        ):
            raise ValueError(f"官方数据接口返回的 feature 路径不匹配: {feature}")
    return interface


def _local_model_config(model_config: dict, cosmos_dir: Path) -> dict:
    """只在内存副本中重映射冻结 config 的 VLM 非权重配置来源。"""
    local_config = dict(model_config)
    local_config["vlm_name_or_path"] = str(cosmos_dir)
    return local_config


def _load_local_model_and_processor(
    model_dir: Path, cosmos_dir: Path, qwen_dir: Path, model_config: dict
):
    import torch
    from transformers import AutoProcessor

    from alpamayo1_5 import helper
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    # 原始 config.json 始终只读；重映射仅作用于本次内存中的配置对象。
    config = Alpamayo1_5Config(**_local_model_config(model_config, cosmos_dir))
    model = Alpamayo1_5.from_pretrained(
        str(model_dir), config=config, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(
        str(qwen_dir),
        min_pixels=helper.MIN_PIXELS,
        max_pixels=helper.MAX_PIXELS,
        local_files_only=True,
    )
    processor.tokenizer = model.tokenizer
    return model, processor


def _run_inference(
    model_dir: Path, cosmos_dir: Path, qwen_dir: Path, model_config: dict, interface
) -> None:
    import numpy as np
    import torch

    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    if not torch.cuda.is_available():
        raise RuntimeError("离线兼容性测试需要 NVIDIA CUDA GPU")
    data = load_physical_aiavdataset(
        CLIP_ID, t0_us=T0_US, avdi=interface, maybe_stream=False
    )
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    model, processor = _load_local_model_and_processor(
        model_dir, cosmos_dir, qwen_dir, model_config
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = (
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
        )

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = float(diff.min())
    if not math.isfinite(min_ade) or min_ade < 0:
        raise ValueError(f"minADE 无效: {min_ade}")
    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])
    print("minADE:", min_ade, "meters")
    print("pred_xyz shape:", tuple(pred_xyz.shape))
    print("pred_rot shape:", tuple(pred_rot.shape))
    if min_ade >= 1.0:
        print(
            f"WARNING: minADE ({min_ade:.2f}m) is above 1.0m. Model sampling can be stochastic."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--cosmos-config-dir", required=True)
    parser.add_argument("--qwen-config-dir", required=True)
    parser.add_argument("--config-manifest-dir")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    model_dir = _directory(args.model_dir, "model-dir")
    dataset_dir = _directory(args.dataset_dir, "dataset-dir")
    cosmos_dir = _directory(args.cosmos_config_dir, "cosmos-config-dir")
    qwen_dir = _directory(args.qwen_config_dir, "qwen-config-dir")
    manifest_dir = (
        _directory(args.config_manifest_dir, "config-manifest-dir")
        if args.config_manifest_dir is not None
        else None
    )
    _enable_offline_mode()
    with _block_network():
        _require_physical_ai_av_version()
        model_config = _verify_model(model_dir)
        _verify_dataset(dataset_dir)
        _verify_config_dir(cosmos_dir, "Cosmos")
        _verify_config_dir(qwen_dir, "Qwen")
        _verify_config_source(
            cosmos_dir,
            "Cosmos",
            MODEL_VLM_NAME,
            COSMOS_REVISION,
            "cosmos-reason2-source.json",
            manifest_dir,
        )
        _verify_config_source(
            qwen_dir,
            "Qwen",
            "Qwen/Qwen3-VL-2B-Instruct",
            QWEN_REVISION,
            "qwen3-vl-source.json",
            manifest_dir,
        )
        interface = _load_offline_dataset_interface(dataset_dir)
        print("ALP15 离线兼容性测试：本地资产预检通过")
        print(f"clip_id={CLIP_ID}; t0_us={T0_US}; chunk_id={CHUNK_ID}")
        if args.preflight_only:
            return 0
        _run_inference(model_dir, cosmos_dir, qwen_dir, model_config, interface)
        print("ALP15 离线兼容性测试：端到端执行完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
