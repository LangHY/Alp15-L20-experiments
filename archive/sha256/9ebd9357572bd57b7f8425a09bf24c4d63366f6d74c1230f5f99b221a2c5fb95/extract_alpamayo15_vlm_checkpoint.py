"""从 Alpamayo 1.5 checkpoint 提取可独立加载的 Qwen3-VL VLM。

脚本只读取源 checkpoint，不修改源权重；输出通过同目录临时目录构建，完成结构
校验后再原子改名。默认核对本项目冻结的 Alpamayo 1.5 10B 文件 SHA-256。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


MODEL_INDEX = "model.safetensors.index.json"
SOURCE_PREFIX = "vlm."
EXPECTED_ENVIRONMENT = {
    "torch": "2.8.0",
    "transformers": "4.57.1",
    "safetensors": "0.7.0",
}
COSMOS_REPO = "nvidia/Cosmos-Reason2-8B"
COSMOS_REVISION = "a9fae2cf89dc64db96b12860417f0eb403013bb9"

# 冻结值来自已验收的 Alpamayo 1.5 10B 本地交付物。
FROZEN_MODEL_FILES = {
    "config.json": (
        3058,
        "824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e",
    ),
    MODEL_INDEX: (
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

REQUIRED_VLM_KEYS = {
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.visual.patch_embed.proj.weight",
}

TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}
SPECIAL_TOKENS_KEYS = (
    "prompt_start",
    "prompt_end",
    "image_start",
    "_padding_0",
    "image_end",
    "traj_history_start",
    "_padding_1",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "_padding_2",
    "_padding_3",
    "traj_future_start",
    "_padding_4",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "_padding_5",
    "_padding_6",
    "_padding_7",
    "_padding_8",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
)

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorSpec:
    source_key: str
    target_key: str
    source_shard: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    nbytes: int


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层不是对象: {path}")
    return payload


def _json_dump(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"大小不匹配: {path}: {actual_size} != {expected_size}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"SHA-256 不匹配: {path}: {actual_sha256}")


def _installed_versions() -> dict[str, str]:
    versions = {}
    for distribution in EXPECTED_ENVIRONMENT:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"推理环境缺少依赖: {distribution}") from error
    return versions


def _verify_environment(allow_mismatch: bool) -> dict[str, str]:
    versions = _installed_versions()
    mismatches = {
        name: (EXPECTED_ENVIRONMENT[name], actual)
        for name, actual in versions.items()
        if actual != EXPECTED_ENVIRONMENT[name]
    }
    if mismatches and not allow_mismatch:
        detail = ", ".join(
            f"{name}={actual}（期望 {expected}）"
            for name, (expected, actual) in mismatches.items()
        )
        raise RuntimeError(
            f"当前不是冻结的 L20 推理环境: {detail}。"
            "如确需在其他环境检查，显式传入 --allow-version-mismatch。"
        )
    return versions


def _verify_frozen_model(model_dir: Path, skip_hash_verification: bool) -> None:
    for filename, (size, digest) in FROZEN_MODEL_FILES.items():
        path = model_dir / filename
        if skip_hash_verification:
            if not path.is_file():
                raise FileNotFoundError(f"缺少文件: {path}")
        else:
            print(f"校验源模型: {filename}", flush=True)
            _verify_file(path, size, digest)


def _verify_processor_manifest(
    processor_dir: Path, manifest_path: Path
) -> dict[str, Any]:
    manifest = _json_load(manifest_path)
    if manifest.get("repo") != COSMOS_REPO:
        raise ValueError(f"Processor 来源仓库不匹配: {manifest.get('repo')}")
    if manifest.get("revision") != COSMOS_REVISION:
        raise ValueError(f"Processor revision 不匹配: {manifest.get('revision')}")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Processor 来源清单没有 files")
    root = processor_dir.resolve()
    for record in records:
        relative = Path(record["path"])
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Processor 清单包含越界路径: {relative}")
        _verify_file(path, int(record["size"]), str(record["sha256"]))
    return manifest


def _select_vlm_tensors(index: dict[str, Any]) -> list[tuple[str, str, str]]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint 索引缺少 weight_map")
    selected: list[tuple[str, str, str]] = []
    target_keys: set[str] = set()
    for source_key, raw_shard in weight_map.items():
        if not source_key.startswith(SOURCE_PREFIX):
            continue
        target_key = source_key.removeprefix(SOURCE_PREFIX)
        shard = str(raw_shard)
        if Path(shard).name != shard:
            raise ValueError(f"索引分片路径必须是文件名: {shard}")
        if target_key in target_keys:
            raise ValueError(f"移除 {SOURCE_PREFIX!r} 后键冲突: {target_key}")
        target_keys.add(target_key)
        selected.append((source_key, target_key, shard))
    missing = REQUIRED_VLM_KEYS - target_keys
    if missing:
        raise ValueError(f"VLM 缺少关键权重: {sorted(missing)}")
    if not selected:
        raise ValueError(f"没有找到 {SOURCE_PREFIX!r} 权重")
    return selected


def _require_safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("需要推理环境中的 torch 与 safetensors") from error
    return safe_open, save_file


def _inspect_tensors(
    model_dir: Path, selected: Sequence[tuple[str, str, str]]
) -> list[TensorSpec]:
    safe_open, _ = _require_safetensors()
    by_shard: dict[str, list[tuple[str, str, str]]] = {}
    for item in selected:
        by_shard.setdefault(item[2], []).append(item)

    specs: list[TensorSpec] = []
    for shard, items in by_shard.items():
        source = model_dir / shard
        if not source.is_file():
            raise FileNotFoundError(f"索引引用的分片不存在: {source}")
        with safe_open(source, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for source_key, target_key, _ in items:
                if source_key not in available:
                    raise ValueError(f"分片缺少索引权重: {shard}:{source_key}")
                view = handle.get_slice(source_key)
                shape = tuple(int(value) for value in view.get_shape())
                dtype = str(view.get_dtype())
                if dtype not in DTYPE_BYTES:
                    raise ValueError(f"不支持的 safetensors dtype: {dtype}")
                numel = math.prod(shape)
                specs.append(
                    TensorSpec(
                        source_key=source_key,
                        target_key=target_key,
                        source_shard=shard,
                        shape=shape,
                        dtype=dtype,
                        numel=numel,
                        nbytes=numel * DTYPE_BYTES[dtype],
                    )
                )
    return specs


def _plan_output_shards(
    specs: Sequence[TensorSpec], max_shard_bytes: int
) -> list[list[TensorSpec]]:
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes 必须大于 0")
    by_source: dict[str, list[TensorSpec]] = {}
    for spec in specs:
        by_source.setdefault(spec.source_shard, []).append(spec)

    groups: list[list[TensorSpec]] = []
    for source_specs in by_source.values():
        current: list[TensorSpec] = []
        current_size = 0
        for spec in source_specs:
            if current and current_size + spec.nbytes > max_shard_bytes:
                groups.append(current)
                current = []
                current_size = 0
            current.append(spec)
            current_size += spec.nbytes
        if current:
            groups.append(current)
    return groups


def _output_shard_name(index: int, total: int) -> str:
    return f"model-{index:05d}-of-{total:05d}.safetensors"


def _write_weight_shards(
    model_dir: Path, output_dir: Path, groups: Sequence[Sequence[TensorSpec]]
) -> dict[str, str]:
    safe_open, save_file = _require_safetensors()
    weight_map: dict[str, str] = {}
    total = len(groups)
    for index, group in enumerate(groups, start=1):
        source_shards = {spec.source_shard for spec in group}
        if len(source_shards) != 1:
            raise AssertionError("一个输出分片不能跨越多个源分片")
        source_shard = next(iter(source_shards))
        output_name = _output_shard_name(index, total)
        print(
            f"写入 {output_name}: tensors={len(group)} "
            f"size={sum(spec.nbytes for spec in group) / 2**30:.2f} GiB",
            flush=True,
        )
        with safe_open(
            model_dir / source_shard, framework="pt", device="cpu"
        ) as handle:
            tensors = {
                spec.target_key: handle.get_tensor(spec.source_key).contiguous()
                for spec in group
            }
        save_file(tensors, output_dir / output_name, metadata={"format": "pt"})
        for spec in group:
            weight_map[spec.target_key] = output_name
        del tensors
    return weight_map


def _verify_output_shards(
    model_dir: Path,
    output_dir: Path,
    groups: Sequence[Sequence[TensorSpec]],
    weight_map: dict[str, str],
) -> None:
    safe_open, _ = _require_safetensors()
    expected_specs = {spec.target_key: spec for group in groups for spec in group}
    expected_files = set(weight_map.values())
    actual_files = {path.name for path in output_dir.glob("*.safetensors")}
    if actual_files != expected_files:
        raise ValueError(
            f"输出分片集合不匹配: actual={sorted(actual_files)} "
            f"expected={sorted(expected_files)}"
        )
    for filename in sorted(expected_files):
        expected_keys = {key for key, shard in weight_map.items() if shard == filename}
        source_shards = {expected_specs[key].source_shard for key in expected_keys}
        if len(source_shards) != 1:
            raise AssertionError("一个输出分片不能跨越多个源分片")
        source_shard = next(iter(source_shards))
        with (
            safe_open(
                output_dir / filename, framework="pt", device="cpu"
            ) as output_handle,
            safe_open(
                model_dir / source_shard, framework="pt", device="cpu"
            ) as source_handle,
        ):
            actual_keys = set(output_handle.keys())
            if actual_keys != expected_keys:
                raise ValueError(f"输出分片键集合不匹配: {filename}")
            for key in expected_keys:
                view = output_handle.get_slice(key)
                spec = expected_specs[key]
                if (
                    tuple(view.get_shape()) != spec.shape
                    or str(view.get_dtype()) != spec.dtype
                ):
                    raise ValueError(f"输出 tensor 元数据不匹配: {filename}:{key}")
                source_tensor = source_handle.get_tensor(spec.source_key)
                output_tensor = output_handle.get_tensor(key)
                if not source_tensor.equal(output_tensor):
                    raise ValueError(f"输出 tensor 数值不匹配: {filename}:{key}")
                del source_tensor, output_tensor


def _build_vlm_config(
    processor_source_dir: Path, alpamayo_config: dict[str, Any], vocab_size: int
) -> dict[str, Any]:
    config = _json_load(processor_source_dir / "config.json")
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(
            f"VLM config.model_type 不是 qwen3_vl: {config.get('model_type')}"
        )
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("VLM config 缺少 text_config")
    text_config["vocab_size"] = vocab_size
    config["vocab_size"] = vocab_size
    config["dtype"] = alpamayo_config.get(
        "model_dtype", config.get("dtype", "bfloat16")
    )
    return config


def _prepare_processor(
    processor_source_dir: Path,
    output_dir: Path,
    alpamayo_config: dict[str, Any],
    expected_vocab_size: int,
) -> dict[str, Any]:
    try:
        from transformers import AutoProcessor
    except ImportError as error:
        raise RuntimeError("推理环境缺少 transformers") from error

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    processor = AutoProcessor.from_pretrained(
        processor_source_dir,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer
    traj_vocab_size = int(alpamayo_config["traj_vocab_size"])
    discrete_tokens = [f"<i{value}>" for value in range(traj_vocab_size)]
    tokenizer.add_tokens(discrete_tokens)
    if alpamayo_config.get("add_special_tokens"):
        special_tokens = [f"<|{key}|>" for key in SPECIAL_TOKENS_KEYS]
    else:
        special_tokens = list(TRAJ_TOKEN.values())
    tokenizer.add_tokens(special_tokens, special_tokens=True)

    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != expected_vocab_size:
        raise ValueError(
            f"扩展 tokenizer 大小与 VLM embedding 不匹配: "
            f"{actual_vocab_size} != {expected_vocab_size}"
        )
    expected_start = int(alpamayo_config["traj_token_start_idx"])
    actual_start = int(tokenizer.convert_tokens_to_ids("<i0>"))
    if actual_start != expected_start:
        raise ValueError(f"<i0> token id 不匹配: {actual_start} != {expected_start}")
    for name, expected_id in alpamayo_config["traj_token_ids"].items():
        actual_id = int(tokenizer.convert_tokens_to_ids(TRAJ_TOKEN[name]))
        if actual_id != int(expected_id):
            raise ValueError(
                f"轨迹 token id 不匹配: {name}: {actual_id} != {expected_id}"
            )

    processor.save_pretrained(output_dir)
    for source_file in processor_source_dir.iterdir():
        if not source_file.is_file() or source_file.name == "config.json":
            continue
        if source_file.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
            raise ValueError(f"Processor 配置目录不应包含权重: {source_file}")
        destination = output_dir / source_file.name
        if not destination.exists():
            shutil.copy2(source_file, destination)
    return {
        "vocab_size": actual_vocab_size,
        "traj_token_start_idx": actual_start,
        "traj_token_ids": {
            name: int(tokenizer.convert_tokens_to_ids(token))
            for name, token in TRAJ_TOKEN.items()
        },
    }


def _verify_full_load(output_dir: Path) -> None:
    try:
        import torch
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise RuntimeError("完整加载复验需要 torch 与 transformers") from error
    print("开始 CPU 完整加载复验；不会使用 GPU。", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        output_dir,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    del model


def _file_records(directory: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    records = []
    for name in sorted(set(names)):
        path = directory / name
        records.append(
            {"path": name, "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    return records


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", required=True, help="Alpamayo 1.5 checkpoint 目录"
    )
    parser.add_argument(
        "--processor-source-dir",
        required=True,
        help="已验收的 Cosmos-Reason2-8B 非权重配置目录",
    )
    parser.add_argument(
        "--processor-source-manifest",
        required=True,
        help="cosmos-reason2-source.json 来源清单",
    )
    parser.add_argument("--output-dir", required=True, help="不存在的 VLM 输出目录")
    parser.add_argument(
        "--max-shard-size-gib",
        type=float,
        default=2.0,
        help="单个输出分片目标上限；单 tensor 大于上限时除外（默认 2 GiB）",
    )
    parser.add_argument(
        "--skip-model-hash-verification",
        action="store_true",
        help="跳过约 22 GB 源模型 SHA-256；不建议用于正式实验",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="允许偏离冻结的 L20 推理环境版本",
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        help="完成后在 CPU 上用 Qwen3VLForConditionalGeneration 做完整加载复验",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验、统计并打印分片计划，不写输出",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    model_dir = Path(args.model_dir).expanduser().resolve(strict=True)
    processor_dir = Path(args.processor_source_dir).expanduser().resolve(strict=True)
    processor_manifest = (
        Path(args.processor_source_manifest).expanduser().resolve(strict=True)
    )
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    if not model_dir.is_dir() or not processor_dir.is_dir():
        raise ValueError("--model-dir 与 --processor-source-dir 必须是目录")
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_dir}")
    if output_dir.is_relative_to(model_dir) or output_dir.is_relative_to(processor_dir):
        raise ValueError("输出目录不能位于源模型或 Processor 配置目录内部")
    if args.max_shard_size_gib <= 0:
        raise ValueError("--max-shard-size-gib 必须大于 0")

    versions = _verify_environment(args.allow_version_mismatch)
    _verify_frozen_model(model_dir, args.skip_model_hash_verification)
    processor_source = _verify_processor_manifest(processor_dir, processor_manifest)
    alpamayo_config = _json_load(model_dir / "config.json")
    index = _json_load(model_dir / MODEL_INDEX)
    selected = _select_vlm_tensors(index)
    specs = _inspect_tensors(model_dir, selected)
    groups = _plan_output_shards(
        specs, int(args.max_shard_size_gib * 1024 * 1024 * 1024)
    )
    total_size = sum(spec.nbytes for spec in specs)
    total_parameters = sum(spec.numel for spec in specs)
    embedding = next(
        spec
        for spec in specs
        if spec.target_key == "model.language_model.embed_tokens.weight"
    )
    vocab_size = embedding.shape[0]

    print(
        json.dumps(
            {
                "selected_tensors": len(specs),
                "parameters": total_parameters,
                "weight_bytes": total_size,
                "weight_gib": round(total_size / 2**30, 4),
                "vocab_size": vocab_size,
                "output_shards": len(groups),
                "environment": versions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        print("dry_run=PASS；未写入任何输出。")
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_dir.parent).free
    required_bytes = total_size + 1024**3
    if free_bytes < required_bytes:
        raise OSError(
            f"空间不足: free={free_bytes / 2**30:.2f} GiB, "
            f"required={required_bytes / 2**30:.2f} GiB"
        )
    staging = output_dir.parent / f".{output_dir.name}.partial-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"临时目录已存在，拒绝覆盖: {staging}")
    staging.mkdir()
    print(f"临时输出目录: {staging}", flush=True)

    processor_result = _prepare_processor(
        processor_dir, staging, alpamayo_config, vocab_size
    )
    vlm_config = _build_vlm_config(processor_dir, alpamayo_config, vocab_size)
    _json_dump(vlm_config, staging / "config.json")
    weight_map = _write_weight_shards(model_dir, staging, groups)
    output_index = {
        "metadata": {
            "total_parameters": total_parameters,
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }
    _json_dump(output_index, staging / MODEL_INDEX)
    _verify_output_shards(model_dir, staging, groups, weight_map)

    if args.verify_load:
        _verify_full_load(staging)

    output_names = list(weight_map.values()) + [
        path.name
        for path in staging.iterdir()
        if path.is_file()
        and path.name not in set(weight_map.values())
        and path.name != "extraction-manifest.json"
    ]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "extract_alpamayo15_vlm_checkpoint",
        "source": {
            "model_index_sha256": _sha256(model_dir / MODEL_INDEX),
            "model_config_sha256": _sha256(model_dir / "config.json"),
            "processor_repo": processor_source["repo"],
            "processor_revision": processor_source["revision"],
            "processor_manifest_sha256": _sha256(processor_manifest),
            "prefix_removed": SOURCE_PREFIX,
        },
        "environment": versions,
        "result": {
            "tensor_count": len(specs),
            "parameters": total_parameters,
            "weight_bytes": total_size,
            "vocab_size": vocab_size,
            "output_shards": len(groups),
            "processor": processor_result,
            "full_load_verified": bool(args.verify_load),
        },
        "files": _file_records(staging, output_names),
    }
    _json_dump(manifest, staging / "extraction-manifest.json")
    os.replace(staging, output_dir)
    print(f"VLM 独立 checkpoint 已生成: {output_dir}")
    print("source_checkpoint_modified=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
