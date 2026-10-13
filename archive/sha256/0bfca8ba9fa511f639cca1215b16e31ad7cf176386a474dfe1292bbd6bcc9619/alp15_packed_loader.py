"""D-ALP15-PACKED-LOADER-20261008 的 Phase A；实体加载尚未通过门禁。

模块导入只用标准库。headers 不读取原 checkpoint payload，不导入 CUDA、
ModelOpt 或 Triton；meta 必须由 L20 的真实官方构造器验证。
"""
from __future__ import annotations

import builtins
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import inspect
import json
import math
import os
import resource
import struct
import sys
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

DECISION = "D-ALP15-PACKED-LOADER-20261008"
CAPABILITY_DECISION = "D-ALP15-PACKED-META-CAPABILITY-20261009"
SOURCE_COMMIT = "36aeb4c5938cbc2eb2aed33b22434773da4ab639"
CHECKPOINT_REVISION = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
AWQ_RESULT_SHA = "16aa6b20915bac68597614361e3932c89b955e7c05c024f1f5025b435ddd76cc"
OVERLAY_SHA = "d0133763c495781933cf9e5f6e32726b3c2577cb00319a27eb7c0c57bcc27d5c"
HELPER_SHA = "1c769df0941ce5769e112aec47660fe1d374315209050133b7b1eadf0deb0913"
CONFIG_SHA = "824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e"
INDEX_SHA = "b899e51816e15a96edc7cd57a01cd36c5e39fe1bcbf1537c4ac914f0cad43a0d"
FORMAT = "alp15-modelopt039-awq-signed-int4-group128-v1"
PROTECTED_SHA = "94b2256c7e277915c3b570c6ba941aaf49b56dfa4094b2adac06b70a394093bb"
PARENT_GUARD_SHA = "77b2770ec860d12e3c69db459a4439d4fa7cc66cbb3079c45a240b35a016f094"
MAX_CHUNK_BYTES = 33554432
SOURCE_HASHES = {
    "alpamayo1_5.models.base_model": "1f2e49a2ac62346cc61c3ea954982729da1a92674bc4ceed7bb43c87ba212cc2",
    "alpamayo1_5.models.alpamayo1_5": "87b23954570f4d466a6d0dd64090f779af635433b61106eb3b028d8d802ff1e8",
    "alpamayo1_5.config": "224ba379806886f3e4b122dc72fb527eaa3e11f9518576555eb0e540581dc6be",
    "alpamayo1_5.models.action_in_proj": "11a21d317c9b8df1097a00d32c8f93cc96e46eeda73f4c16948a4be84205f240",
    "alpamayo1_5.action_space.unicycle_accel_curvature": "1e8035a3f2456b82219e96f7058fe2c0d94410a60367fe7395df3f22b4bedf9e",
    "alpamayo1_5.action_space.discrete_action_space": "d8e7dfbdc0b45fd64c24182fea6f0dbc436a8afa5a719c053c0f905a3cc8ccdb",
    "transformers.models.qwen3_vl.modeling_qwen3_vl": "dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da",
}
HEADER_SHAS = (
    "49df741ba611dbe3ca4dbba7cff0b35dd73de4cb0131178f374acd89d3d6058a",
    "89c487d89c1de9be5f75f7adc5842acd6fb283a1f8c78e8db7c64bf2b655bb2b",
    "422596c5eae572fa8a8c4e66f711622bc47eacf8d162838c84c3f9dc63b2c48d",
    "1330aad43ef7dc0a6624ed9f345a2f535dbfad936a818896faa262d7e1a54031",
    "d767e19b5997d6cc48aa1aed85a7f09bbaf862dedcd0e4609328345a74ae99c8",
)
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "I64": 8}
HF_CAPABILITY_SOURCES = {
    "utils/import_utils.py": "66b2ac40be2c538dbd9409370c1f6bf769934b8b232edac365e16299c5ec1108",
    "modeling_utils.py": "bf1c6b2a43cf7c36fb79f37c981424dd6ae78eb863fcaa5d2a37e76c9828611d",
    "modeling_flash_attention_utils.py": "293fe81c6bd38aac8a3bc6ae086ff21b9695b349e97c9650b1fe61e42a36d710",
    "integrations/flash_attention.py": "850aa63f9473391188c357454cfb6a89339394d26f6024ccbe246f00f6344144",
}
TORCH_CAPABILITY_FUNCTION_SHAS = {
    "is_available": "b91a896857c103611c58b84e1455f789393531efa0c56cf7971e931b99fdffce",
    "device_count": "16443ec7027f24db4f8c744133c12ee22d148ceeeb29b55d479f6b095b72e3b3",
}


class GateError(RuntimeError):
    """明确的停止门禁，不回退到完整 CPU/CUDA 模型。"""


def _require(condition, message):
    if not condition:
        raise GateError(message)


def sha256_file(path, block_bytes=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "JSON 出现重复键: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)


def within(path, parent):
    resolved, parent = Path(path).resolve(strict=True), Path(parent).resolve(strict=True)
    _require(resolved.is_relative_to(parent), f"路径越过 allowlist: {resolved}")
    return resolved


def fingerprint(path, *, digest=None):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
              "inode": stat.st_ino}
    if digest is not None:
        result["sha256"] = digest
    return result


def verify_hash(path, expected):
    actual = sha256_file(path)
    _require(actual == expected, f"SHA256 不符: {path}: {actual}")
    return fingerprint(path, digest=actual)


def read_safetensors_header(path):
    """仅两次 bounded read: 首 8 bytes + JSON；从不读取 payload。"""
    path = Path(path)
    stat_before = fingerprint(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        _require(len(prefix) == 8, f"safetensors 首部截断: {path}")
        length = struct.unpack("<Q", prefix)[0]
        _require(2 <= length <= 16 * 1024 * 1024, f"header 长度越界: {path}")
        _require(length + 8 <= stat_before["size"], f"header 超过文件: {path}")
        raw = stream.read(length)
        _require(len(raw) == length, f"header 截断: {path}")
    header = json.loads(raw, object_pairs_hook=_unique_object)
    _require(isinstance(header, dict), "header 必须是对象")
    metadata = header.pop("__metadata__", {})
    _require(isinstance(metadata, dict), "metadata 必须是对象")
    cursor, tensors = 0, {}
    for key, item in sorted(header.items(), key=lambda pair: pair[1]["data_offsets"][0]):
        dtype, shape, offsets = item.get("dtype"), item.get("shape"), item.get("data_offsets")
        _require(dtype in DTYPE_BYTES, f"未知 dtype: {key}: {dtype}")
        _require(isinstance(shape, list) and all(type(v) is int and v >= 0 for v in shape),
                 "非法 shape: " + key)
        _require(isinstance(offsets, list) and len(offsets) == 2 and
                 all(type(v) is int for v in offsets), "非法 offsets: " + key)
        size = math.prod(shape) * DTYPE_BYTES[dtype]
        _require(offsets == [cursor, cursor + size], f"重叠、空洞或 size 不符: {key}")
        tensors[key] = {"dtype": dtype, "shape": shape, "offset": 8 + length + cursor,
                        "bytes": size}
        cursor += size
    _require(cursor + 8 + length == stat_before["size"], "payload 长度与 header 不符")
    _require(fingerprint(path) == stat_before, f"读 header 时文件变化: {path}")
    return {"file": stat_before, "header_bytes": length,
            "header_sha256": hashlib.sha256(raw).hexdigest(), "tensors": tensors,
            "metadata": metadata, "payload_read_bytes": 0}


def quantized_modules():
    return [f"vlm.model.language_model.layers.{layer}.{section}.{projection}"
            for layer in range(36)
            for section, projections in (("mlp", ("gate_proj", "up_proj", "down_proj")),
                                         ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")))
            for projection in projections]


def row_chunks(shape, dtype, chunk_bytes=MAX_CHUNK_BYTES):
    _require(type(chunk_bytes) is int and 0 < chunk_bytes <= MAX_CHUNK_BYTES,
             "chunk_bytes 必须在 (0, 32MiB] 内")
    item_bytes = DTYPE_BYTES[dtype]
    if not shape:
        _require(item_bytes <= chunk_bytes, "scalar 超过块上限")
        return [(0, 1, item_bytes)]
    row_bytes = math.prod(shape[1:]) * item_bytes
    _require(row_bytes > 0 and row_bytes <= chunk_bytes, "单行超过块上限或空 tensor")
    step = chunk_bytes // row_bytes
    return [(start, min(start + step, shape[0]),
             (min(start + step, shape[0]) - start) * row_bytes)
            for start in range(0, shape[0], step)]


@dataclass(frozen=True)
class LoaderPlan:
    schema: str
    decision: str
    paths: dict
    assets: dict
    original: dict
    protected: dict
    packed: dict
    target: dict
    model_config: dict
    plan_sha256: str

    def to_dict(self):
        return asdict(self)


def _freeze_plan(**kwargs):
    return LoaderPlan(**kwargs, plan_sha256=canonical_sha(kwargs))


def validate_plan(plan):
    values = plan.to_dict()
    expected = values.pop("plan_sha256")
    _require(canonical_sha(values) == expected, "LoaderPlan 内容变化")
    for record in plan.assets.values():
        if "path" not in record:
            continue
        current = fingerprint(record["path"])
        _require(all(current[key] == record[key] for key in current),
                 "来源文件在 plan 后变化: " + record["path"])
        if "sha256" in record:
            _require(sha256_file(record["path"]) == record["sha256"],
                     "来源 hash 在 plan 后变化: " + record["path"])


def _load_frozen_helper(path):
    verify_hash(path, HELPER_SHA)
    spec = importlib.util.spec_from_file_location("_alp15_packed_loader_frozen_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_overlay_header(header):
    tensors = header["tensors"]
    _require(set(tensors) == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"},
             "overlay 不得覆盖任何保护键")
    for name, tensor in tensors.items():
        _require(tensor["dtype"] == "BF16" and tensor["shape"] ==
                 ([4096, 5376] if name == "down_proj.weight" else [5376, 4096]), "overlay shape/dtype不符")


def audit_loader_assets(paths) -> LoaderPlan:
    """审核 headers；packed/overlay 是小文件逐个 hash，原分片绝不做 payload hash。"""
    root = Path(paths["root"]).resolve(strict=True)
    model_dir = within(paths.get("model_dir", root / "models"), root)
    awq_dir = within(paths["awq_artifact"], root)
    overlay_dir = within(paths["overlay"], root)
    bundle = within(paths.get("config_bundle", root / "alp15-configs-delivery-20260919"), root)
    helper_path = within(paths.get("helper", root / "scripts/test_alpamayo15_offline_compat.py"), root)
    source_root = within(paths.get("source_root", root / "alpamayo1.5/src"), root)
    helper = _load_frozen_helper(helper_path)
    assets = {"helper": verify_hash(helper_path, HELPER_SHA),
              "model_config": verify_hash(model_dir / "config.json", CONFIG_SHA),
              "model_index": verify_hash(model_dir / "model.safetensors.index.json", INDEX_SHA),
              "awq_result": verify_hash(awq_dir / "result.json", AWQ_RESULT_SHA),
              "overlay_manifest": verify_hash(overlay_dir / "manifest.json", OVERLAY_SHA)}
    cosmos = within(bundle / "dependencies/configs/Cosmos-Reason2-8B", bundle)
    qwen = within(bundle / "dependencies/configs/Qwen3-VL-2B-Instruct", bundle)
    for label, directory, repo, revision, manifest in (
        ("Cosmos", cosmos, helper.MODEL_VLM_NAME, helper.COSMOS_REVISION, "cosmos-reason2-source.json"),
        ("Qwen", qwen, "Qwen/Qwen3-VL-2B-Instruct", helper.QWEN_REVISION, "qwen3-vl-source.json"),
    ):
        helper._verify_config_dir(directory, label)
        helper._verify_config_source(directory, label, repo, revision, manifest, bundle / "manifests")
        assets[f"{label}_manifest"] = fingerprint(bundle / "manifests" / manifest,
                                                        digest=sha256_file(bundle / "manifests" / manifest))
        for name in helper.CONFIG_FILES:
            path = within(directory / name, directory)
            assets[f"{label}/{name}"] = fingerprint(path, digest=sha256_file(path))
    config = read_json(model_dir / "config.json")
    _require(config["vlm_name_or_path"] == helper.MODEL_VLM_NAME, "VLM 来源变化")
    index = read_json(model_dir / "model.safetensors.index.json")["weight_map"]
    expected_shards = [f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)]
    _require(set(index.values()) == set(expected_shards), "index 分片集合不符")
    _require({p.name for p in model_dir.glob("*.safetensors")} == set(expected_shards),
             "checkpoint 目录分片集合不符")
    original = {}
    for shard, header_sha in zip(expected_shards, HEADER_SHAS):
        path = within(model_dir / shard, model_dir)
        _require(path.stat().st_size == helper.MODEL_FILES[shard][0], "冻结 shard size 不符")
        header = read_safetensors_header(path)
        _require(header["header_sha256"] == header_sha, "冻结 shard header hash 不符")
        assets[shard] = {**header["file"], "header_sha256": header_sha,
                         "header_bytes": header["header_bytes"], "payload_sha256_verified": False,
                         "frozen_payload_sha256": helper.MODEL_FILES[shard][1]}
        for key, tensor in header["tensors"].items():
            _require(key not in original and index.get(key) == shard, "header/index 键冲突: " + key)
            _require(tensor["dtype"] == "BF16", "checkpoint 非 BF16: " + key)
            row_chunks(tensor["shape"], tensor["dtype"])
            original[key] = {**tensor, "file": str(path), "source_key": key, "kind": "parameter"}
    _require(set(original) == set(index) and len(original) == 1159, "原1159键不完整")
    modules = quantized_modules()
    qkeys = {name + ".weight" for name in modules}
    _require(qkeys <= original.keys(), "缺少252 dense projection keys")
    protected = {key: value for key, value in original.items() if key not in qkeys}
    _require(len(protected) == 907 and sum(v["bytes"] for v in protected.values()) == 8265517540,
             "907保护键或字节数不符")
    parent = read_json(overlay_dir / "manifest.json")
    _require(parent["base_checkpoint_revision"] == CHECKPOINT_REVISION and
             parent["protected_sha256"] == PARENT_GUARD_SHA, "父来源/guard不符")
    _require(parent["keep_width"] == 5376 and parent["recovery"]["rank"] == 32 and
             parent["recovery"]["selected_step"] == 400 and
             len(parent["layers"]) == 36, "父恢复配置不符")
    for layer, record in enumerate(parent["layers"]):
        keep = record["keep_indices"]
        _require(record["layer"] == layer and record["original_width"] == 12288 and
                 record["keep_width"] == 5376 and len(keep) == 5376 and
                 all(type(v) is int and 0 <= v < 12288 for v in keep) and
                 keep == sorted(set(keep)), "父keep_indices不符")
        path = within(overlay_dir / f"mlp_{layer:02d}.safetensors", overlay_dir)
        assets[f"overlay/{layer}"] = verify_hash(path, record["sha256"])
        header = read_safetensors_header(path)
        _validate_overlay_header(header)
    awq = read_json(awq_dir / "result.json")
    _require(awq["source_commit"] == SOURCE_COMMIT and awq["checkpoint_revision"] == CHECKPOINT_REVISION
             and awq["protected_sha256"] == PROTECTED_SHA, "AWQ 来源或907 guard不符")
    rows = awq["packed_layers"]
    _require(len(rows) == 252, "packed 不是252")
    packed, target = {}, dict(protected)
    for row in rows:
        name = "vlm.model." + row["name"]
        _require(name in modules and name not in packed, "packed 名称重复或非允许名: " + name)
        filename = row["name"].replace(".", "_") + ".safetensors"
        _require(Path(row["file"]).name == filename, "packed文件名和module不符")
        path = within(awq_dir / "packed" / filename, awq_dir)
        assets["packed/" + name] = verify_hash(path, row["sha256"])
        header = read_safetensors_header(path)
        shape = ([4096, 5376] if name.endswith("mlp.down_proj") else
                 [5376, 4096] if ".mlp." in name else
                 [1024, 4096] if name.endswith(("k_proj", "v_proj")) else [4096, 4096])
        _require(row["shape"] == shape, "packed row shape不符")
        out_dim, in_dim = shape
        expected = {"qweight": ("U8", [out_dim, in_dim // 2]),
                    "amax": ("BF16", [out_dim, in_dim // 128]), "pre_scale": ("BF16", [in_dim])}
        metadata = header["metadata"]
        _require(metadata.get("format") == FORMAT and metadata.get("group_size") == "128" and
                 metadata.get("in_features") == str(in_dim) and metadata.get("out_features") == str(out_dim),
                 "packed metadata格式不符")
        _require(set(header["tensors"]) == set(expected), "packed必须恰好3 buffers，无bias")
        _require(row["amax_dtype"] == row["pre_scale_dtype"] == "torch.bfloat16", "AWQ row dtype不符")
        for key, tensor in header["tensors"].items():
            _require((tensor["dtype"], tensor["shape"]) == expected[key], "packed header shape/dtype不符")
            target[name + "." + key] = {**tensor, "file": str(path), "source_key": key, "kind": "buffer"}
        _require(sum(v["bytes"] for v in header["tensors"].values()) == row["persistent_bytes"],
                 "packed row字节数不符")
        packed[name] = {"file": str(path), "sha256": row["sha256"], "shape": shape,
                        "tensors": header["tensors"]}
    _require(set(packed) == set(modules) and len(target) == 1663, "目标1663 keys不符")
    _require(sum(v["bytes"] for v in target.values()) == 10272485860, "目标持久字节数不符")
    # source snapshot用于证据绑定；实际导入文件在meta阶段单独核验，Mac旧树不能替代。
    snapshot = root / "docs/baselines/evidence/jetson_300ms_20261008/source_snapshot"
    for module in ("alpamayo1_5.models.base_model", "alpamayo1_5.models.alpamayo1_5"):
        path = snapshot / (module.rsplit(".", 1)[1] + ".py")
        if path.exists():
            assets["snapshot/" + module] = verify_hash(path, SOURCE_HASHES[module])
    actual_paths = {"root": str(root), "model_dir": str(model_dir), "awq_artifact": str(awq_dir),
                    "overlay": str(overlay_dir), "config_bundle": str(bundle), "helper": str(helper_path),
                    "source_root": str(source_root), "cosmos": str(cosmos), "qwen": str(qwen)}
    return _freeze_plan(schema="alp15-loader-plan-v1", decision=DECISION, paths=actual_paths,
                        assets=assets, original=original, protected=protected, packed=packed,
                        target=target, model_config=config)


def _tensor_row(tensor, *, kind, persistent=None):
    row = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
           "bytes": tensor.numel() * tensor.element_size(), "kind": kind}
    if persistent is not None:
        row["persistent"] = persistent
    return row


def audit_state(model, expected, *, require_meta=True):
    """审查真实参数/buffer种类、完整集合及对象alias；不用meta data_ptr。"""
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    state = model.state_dict()
    _require(set(state) == set(expected),
             f"state集合不符 missing={sorted(set(expected)-set(state))[:10]} extra={sorted(set(state)-set(expected))[:10]}")
    dtype_map = {"BF16": "torch.bfloat16", "U8": "torch.uint8"}
    rows, groups = {}, {}
    for key, spec in expected.items():
        tensor = parameters.get(key) if spec["kind"] == "parameter" else buffers.get(key)
        _require(tensor is not None, "参数/buffer种类不符: " + key)
        _require(list(tensor.shape) == spec["shape"] and str(tensor.dtype) == dtype_map[spec["dtype"]],
                 "实际shape/dtype不符: " + key)
        if require_meta:
            _require(tensor.is_meta, "发现实体持久tensor: " + key)
        rows[key] = _tensor_row(tensor, kind=spec["kind"], persistent=True)
        groups.setdefault(id(tensor), []).append(key)
    aliases = [names for names in groups.values() if len(names) > 1]
    _require(not aliases, "持久tensor对象alias不符合独立权重: " + repr(aliases))
    return rows


def nonpersistent_buffers(model):
    return {f"{prefix}.{name}".lstrip("."): _tensor_row(value, kind="buffer", persistent=False)
            for prefix, module in model.named_modules()
            for name, value in module._buffers.items()
            if value is not None and name in module._non_persistent_buffers_set}


def unregistered_tensors(model):
    """遍历模块普通属性及trajectory tokenizer对象；保留跨属性同对象关系。"""
    import torch
    rows, seen = {}, set()
    def walk(value, name, depth=0):
        if isinstance(value, torch.Tensor):
            rows[name] = {**_tensor_row(value, kind="ordinary_attribute"), "object_id": id(value)}
            return
        if depth > 12 or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{name}.{key}", depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{name}.{index}", depth + 1)
        elif isinstance(value, torch.nn.Module):
            # Python对象内的nn.Module（traj_tokenizer.action_space）不属于模型树。
            for key, item in value.__dict__.items():
                if key in ("_parameters", "_buffers", "_modules"):
                    walk(item, f"{name}.{key}", depth + 1)
                elif not key.startswith("_"):
                    walk(item, f"{name}.{key}", depth + 1)
        elif hasattr(value, "__dict__") and value.__class__.__module__.startswith("alpamayo1_5"):
            for key, item in vars(value).items():
                walk(item, f"{name}.{key}", depth + 1)
    for prefix, module in model.named_modules():
        for key, value in vars(module).items():
            if key not in ("_parameters", "_buffers", "_modules") and not key.startswith("_"):
                walk(value, f"{prefix}.{key}".lstrip("."))
    return rows


@contextmanager
def default_bf16():
    import torch
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        yield str(original)
    finally:
        torch.set_default_dtype(original)


def _frame_identity(frame):
    path = Path(frame.f_code.co_filename).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path), "function": frame.f_code.co_name,
            "line": frame.f_lineno}


def _matches_frame(identity, allowed):
    return any(all(identity[key] == record[key] for key in ("path", "sha256", "function"))
               for record in allowed)


def _frozen_hf_capability_callers():
    # 只查package spec，不执行HF或flash-attn导入；actual source SHA先于import核验。
    spec = importlib.util.find_spec("transformers")
    _require(spec is not None and spec.submodule_search_locations, "缺少实际Transformers源码位置")
    package = Path(next(iter(spec.submodule_search_locations))).resolve(strict=True)
    sources = {name: verify_hash(within(package / name, package), digest)
               for name, digest in HF_CAPABILITY_SOURCES.items()}
    callers = []
    for name, functions in (("utils/import_utils.py", ("is_flash_attn_2_available", "is_flash_attn_3_available")),
                            ("modeling_utils.py", ("_flash_attn_2_can_dispatch",))):
        for function in functions:
            callers.append({"path": sources[name]["path"], "sha256": sources[name]["sha256"],
                            "function": function})
    return callers, sources


def _validate_torch_capability_source(torch):
    _require(torch.__version__.split("+")[0] == "2.8.0", "能力查询实现只批准Torch2.8.0")
    path = Path(torch.cuda.__file__).resolve(strict=True)
    result = fingerprint(path, digest=sha256_file(path))
    result["function_sha256"] = {}
    for name, expected in TORCH_CAPABILITY_FUNCTION_SHAS.items():
        function = getattr(torch.cuda, name)
        _require(Path(inspect.getsourcefile(function)).resolve(strict=True) == path and
                 function.__name__ == name, "Torch能力查询源码/函数身份变化: " + name)
        digest = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        _require(digest == expected, "Torch能力查询函数源码变化，停止升级: " + name)
        result["function_sha256"][name] = digest
    return result


@contextmanager
def _meta_guards(torch, *, capability_callers=(), evidence=None):
    """仅冻结caller可真实查询available；执行/分配/权重/未知query仍硬拒绝。"""
    evidence = {} if evidence is None else evidence
    original_initialized = torch.cuda.is_initialized
    _require(not original_initialized(), "PhaseA开始时Torch CUDA已经初始化")
    evidence.update(capability_decision=CAPABILITY_DECISION, cuda_capability_queries=[],
        cuda_nested_device_count_queries=[], cuda_driver_discovery_primitives=[],
        cuda_driver_discovery_may_initialize=True, cuda_execution_attempts=[], blocked_entry_attempts=[],
        cuda_tensor_allocation_observed=None, cuda_tensor_allocation_evidence="Python guard尚无实体分配；native/PID观察未证明",
        cuda_is_initialized_before=False, cuda_is_initialized_after=None,
        environment={name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "PYTORCH_NVML_BASED_CUDA_CHECK")},
        torch_cuda_version=torch.version.cuda, torch_hip_version=torch.version.hip,
        related_modules_before=[name for name in sys.modules if name.startswith(("transformers", "flash_attn", "triton", "modelopt"))])
    original_available = torch.cuda.is_available
    original_count = torch.cuda.device_count
    original_primitive = getattr(torch._C, "_cuda_getDeviceCount", None)
    query_source = Path(inspect.getsourcefile(original_available)).resolve(strict=True)
    source_hash = sha256_file(query_source)
    evidence["torch_cuda_query_source"] = {"path": str(query_source), "sha256": source_hash,
        "is_available_source": inspect.getsource(original_available),
        "device_count_source": inspect.getsource(original_count)}
    query_identity = {"path": str(query_source), "sha256": source_hash,
                      "function": original_available.__name__}
    count_source = Path(inspect.getsourcefile(original_count)).resolve(strict=True)
    count_identity = {"path": str(count_source), "sha256": sha256_file(count_source),
                      "function": original_count.__name__}
    query_state = threading.local()
    query_state.depth = 0
    restorations = []
    original_import, original_open, original_io_open, original_os_open = (
        builtins.__import__, builtins.open, io.open, os.open)
    def make_blocked(entry_name):
        # Dynamo按callable identity建规则表，各API必须有独立对象；不共享blocked。
        def blocked(*args, **kwargs):
            attempt = {"entry": entry_name, "caller": _frame_identity(sys._getframe(1)),
                       "phase": evidence.get("phase", "imports")}
            evidence["blocked_entry_attempts"].append(attempt)
            if "cuda" in entry_name:
                evidence["cuda_execution_attempts"].append(attempt)
            raise GateError("PhaseA 禁止 CUDA API 或实体权重加载入口: " + entry_name)
        blocked.__name__ = "blocked_" + entry_name.replace(".", "_")
        blocked.__qualname__ = "_meta_guards." + blocked.__name__
        return blocked
    def capability_query():
        caller = _frame_identity(sys._getframe(1))
        record = {"sequence": len(evidence["cuda_capability_queries"]) + 1,
                  "phase": evidence.get("phase", "imports"), "caller": caller}
        evidence["cuda_capability_queries"].append(record)
        _require(_matches_frame(caller, capability_callers), "未批准CUDA能力查询caller: " + repr(caller))
        _require(getattr(query_state, "depth", 0) == 0, "能力查询出现未批准递归")
        query_state.depth = 1
        try:
            result = original_available()
            _require(type(result) is bool, "原is_available未返回bool")
            record["result"] = result
            _require(not original_initialized(), "真实能力查询触发了Torch CUDA初始化")
            return result
        finally:
            query_state.depth = 0
    def nested_device_count():
        caller = _frame_identity(sys._getframe(1))
        _require(getattr(query_state, "depth", 0) == 1 and _matches_frame(caller, [query_identity]),
                 "device_count仅允许原is_available内部直接NVML分支")
        record = {"caller": caller, "phase": evidence.get("phase", "imports")}
        evidence["cuda_nested_device_count_queries"].append(record)
        result = original_count()
        record["result"] = result
        return result
    def discovery_primitive():
        caller = _frame_identity(sys._getframe(1))
        _require(getattr(query_state, "depth", 0) == 1 and
                 _matches_frame(caller, [query_identity, count_identity]),
                 "_C._cuda_getDeviceCount仅允许批准的原query内部")
        record = {"caller": caller, "phase": evidence.get("phase", "imports"),
                  "driver_discovery_may_initialize": True}
        evidence["cuda_driver_discovery_primitives"].append(record)
        result = original_primitive()
        record["result"] = result
        return result
    def guarded_import(name, *args, **kwargs):
        _require(name.split(".")[0] not in {"modelopt", "triton"}, "PhaseA 禁止import: " + name)
        return original_import(name, *args, **kwargs)
    def check_file(file):
        if isinstance(file, (str, bytes, os.PathLike)):
            suffix = Path(os.fsdecode(file)).suffix
            _require(suffix not in {".safetensors", ".bin", ".pt", ".pth"},
                     "PhaseA 禁止权重payload文件访问: " + os.fsdecode(file))
    def guarded_open(file, *args, **kwargs):
        check_file(file)
        return original_open(file, *args, **kwargs)
    def guarded_io_open(file, *args, **kwargs):
        check_file(file)
        return original_io_open(file, *args, **kwargs)
    def guarded_os_open(file, *args, **kwargs):
        check_file(file)
        return original_os_open(file, *args, **kwargs)
    try:
        with ExitStack() as stack:
            def guard(owner, name, replacement):
                restorations.append((owner, name, getattr(owner, name)))
                stack.enter_context(patch.object(owner, name, replacement))
            guard(builtins, "__import__", guarded_import)
            guard(builtins, "open", guarded_open)
            guard(io, "open", guarded_io_open)
            guard(os, "open", guarded_os_open)
            guard(torch, "load", make_blocked("torch.load"))
            guard(torch.cuda, "is_available", capability_query)
            guard(torch.cuda, "device_count", nested_device_count)
            if original_primitive is not None:
                guard(torch._C, "_cuda_getDeviceCount", discovery_primitive)
            for name in ("_cuda_init", "_cuda_setDevice", "_cuda_synchronize", "_cuda_getDeviceProperties",
                         "_cuda_getDevice", "_cuda_getCurrentStream", "_cuda_getDefaultStream", "_cuda_setStream",
                         "_cuda_emptyCache", "_cuda_resetPeakMemoryStats", "_cuda_beginAllocateToPool"):
                if hasattr(torch._C, name):
                    guard(torch._C, name, make_blocked("torch._C." + name))
            for name in ("current_device", "get_device_properties", "get_device_capability", "init",
                         "_lazy_init", "synchronize", "set_device", "memory_allocated",
                         "max_memory_allocated", "empty_cache"):
                if hasattr(torch.cuda, name):
                    guard(torch.cuda, name, make_blocked("torch.cuda." + name))
            guard(torch.Tensor, "cuda", make_blocked("torch.Tensor.cuda"))
            guard(torch.nn.Module, "cuda", make_blocked("torch.nn.Module.cuda"))
            yield evidence
    finally:
        query_state.depth = 0
        evidence["query_context_depth_after"] = query_state.depth
        evidence["guard_restore_table"] = [{"owner": getattr(owner, "__name__", type(owner).__name__),
            "entry": name, "exact_identity_restored": getattr(owner, name) is original}
            for owner, name, original in restorations]
        evidence["guard_restored_all"] = all(row["exact_identity_restored"] for row in evidence["guard_restore_table"])
        evidence["cuda_is_initialized_after"] = original_initialized()
        _require(evidence["guard_restored_all"], "guard原callable identity未恢复")
        _require(not evidence["cuda_is_initialized_after"], "PhaseA结束时Torch CUDA已初始化")


@contextmanager
def _allocation_guard(torch, evidence, *, default_device="cpu"):
    """执行前用meta输出推演拒绝CPU单次>1MiB与CUDA；未知CPU输出算子硬停。"""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten, tree_map
    limit = 1024 * 1024
    class AllocationGuard(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            baseline = evidence.get("rss_peak_constructor_baseline_bytes")
            if baseline is not None:
                _require(_rss_peak_bytes() - baseline <= 256 * 1024 * 1024,
                         "constructor运行中RSS增量超过256MiB")
            kwargs = {} if kwargs is None else kwargs
            flat, _ = tree_flatten((args, kwargs))
            inputs = [value for value in flat if isinstance(value, torch.Tensor)]
            schema = func._schema.arguments
            devices = [kwargs.get(argument.name, args[index] if index < len(args) else None)
                       for index, argument in enumerate(schema) if argument.name == "device"]
            if any(value.device.type == "cuda" for value in inputs) or any(
                    value is not None and torch.device(value).type == "cuda" for value in devices):
                evidence["cuda_execution_attempts"].append({"entry": str(func), "phase": evidence.get("phase")})
                raise GateError("PhaseA allocation/dispatch禁止CUDA目标: " + str(func))
            cpu = any(value.device.type == "cpu" for value in inputs) or any(
                value is not None and torch.device(value).type == "cpu" for value in devices)
            if cpu:
                for value in inputs:
                    _require(value.device.type != "cpu" or value.numel() * value.element_size() <= limit,
                             "constructor CPU tensor输入超过1MiB: " + str(func))
                def make_meta(value):
                    if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                        return torch.empty_strided(value.size(), value.stride(), dtype=value.dtype, device="meta")
                    return value
                meta_args = list(tree_map(make_meta, args))
                meta_kwargs = tree_map(make_meta, kwargs)
                for index, argument in enumerate(schema):
                    if argument.name == "device":
                        if index < len(meta_args): meta_args[index] = torch.device("meta")
                        else: meta_kwargs["device"] = torch.device("meta")
                try:
                    projected = func(*meta_args, **meta_kwargs)
                except Exception as error:
                    raise GateError("CPU操作无法分配前meta推演，停止升级: " + str(func)) from error
                values, _ = tree_flatten(projected)
                for value in values:
                    if isinstance(value, torch.Tensor):
                        _require(value.numel() * value.element_size() <= limit,
                                 "constructor CPU单次实体分配超过1MiB: " + str(func))
                evidence.setdefault("cpu_allocation_projections", []).append({"operation": str(func),
                    "max_tensor_bytes": max((v.numel() * v.element_size() for v in values
                                             if isinstance(v, torch.Tensor)), default=0)})
            return func(*args, **kwargs)
    # torch.tensor/from_numpy/as_tensor可先创建CPU storage再进入lift_fresh，需额外前置guard。
    factory_originals = {}
    def check_factory_restoration():
        table = {name: getattr(torch, name) is original for name, original in factory_originals.items()}
        evidence.setdefault("allocation_factory_restores", []).append({"phase": evidence.get("phase"), "table": table})
        _require(all(table.values()), "allocation factory identity恢复失败")
    with ExitStack() as stack:
        stack.callback(check_factory_restoration)
        for name in ("tensor", "as_tensor", "from_numpy"):
            original = getattr(torch, name)
            factory_originals[name] = original
            def factory(data, *args, _original=original, _name=name, **kwargs):
                # patch factory对象可能影响torch.device构造器缓存；显式保持本范围实际默认设备。
                device = "cpu" if _name == "from_numpy" else kwargs.get("device") or default_device
                if _name != "from_numpy":
                    kwargs["device"] = device
                _require(torch.device(device).type != "cuda", "PhaseA CUDA factory禁止: torch." + _name)
                # 列表/ndarray转meta也可能先经CPU ingress，不能借meta目标放行大输入。
                if torch.device(device).type == "cpu" or not (isinstance(data, torch.Tensor) and data.is_meta):
                    if isinstance(data, torch.Tensor): count = data.numel()
                    elif hasattr(data, "size") and type(getattr(data, "size")) is int: count = data.size
                    else:
                        count = len(tree_flatten(data)[0])
                    dtype = kwargs.get("dtype")
                    item_bytes = torch.empty((), device="meta", dtype=dtype).element_size() if dtype else 8
                    _require(count * item_bytes <= limit, "CPU factory单次实体输入超过1MiB: torch." + _name)
                return _original(data, *args, **kwargs)
            stack.enter_context(patch.object(torch, name, factory))
        stack.enter_context(AllocationGuard())
        yield


def _replace_meta_projections(model, plan):
    import torch
    class PackedMetaPlaceholder(torch.nn.Module):
        def __init__(self, spec):
            super().__init__()
            self.out_features, self.in_features = spec["shape"]
            self.register_buffer("bias", None)
            for name, tensor in spec["tensors"].items():
                self.register_buffer(name, torch.empty(tensor["shape"], device="meta",
                    dtype=torch.uint8 if tensor["dtype"] == "U8" else torch.bfloat16))
        def forward(self, *args, **kwargs):
            raise GateError("PhaseA meta占位禁止forward；不得当作AWQ数值验证")
    for name, spec in plan.packed.items():
        owner, attr = name.rsplit(".", 1)
        dense = model.get_submodule(owner)._modules[attr]
        _require(isinstance(dense, torch.nn.Linear) and dense.bias is None and dense.weight.is_meta,
                 "待替换projection不符: " + name)
        model.get_submodule(owner)._modules[attr] = PackedMetaPlaceholder(spec)
    for layer in model.vlm.model.language_model.layers:
        layer.mlp.intermediate_size = 5376


def _restore_small_state(model):
    """只重建官方小类；从不从meta拷出，也不构造实体MLP。"""
    import torch
    import hydra.utils as hyu
    from alpamayo1_5.models.action_in_proj import FourierEncoderV2
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionRotaryEmbedding, Qwen3VLTextRotaryEmbedding)
    with default_bf16(), torch.device("cpu"):
        old = model.vlm.model.visual.rotary_pos_emb
        dim = old.inv_freq.numel() * 2
        model.vlm.model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(dim)
        model.vlm.model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(
            model.vlm.config.text_config, device="cpu")
        model.expert.rotary_emb = Qwen3VLTextRotaryEmbedding(model.expert.config, device="cpu")
        config = model.config.action_in_proj_cfg
        for index in range(len(model.action_in_proj.sinus)):
            model.action_in_proj.sinus[index] = FourierEncoderV2(
                dim=config["num_fourier_feats"], max_freq=config["max_freq"])
        model.action_in_proj.timestep_fourier_encoder = FourierEncoderV2(
            dim=config["num_fourier_feats"], max_freq=config["max_freq"])
        model.action_space = hyu.instantiate(model.config.action_space_cfg)
        model.traj_tokenizer.action_space = hyu.instantiate(
            model.config.traj_tokenizer_cfg["action_space_cfg"])


def _expected_small_buffers():
    result = {"vlm.model.visual.rotary_pos_emb.inv_freq": ([18], "torch.float32"),
              "vlm.model.language_model.rotary_emb.inv_freq": ([64], "torch.float32"),
              "expert.rotary_emb.inv_freq": ([64], "torch.float32")}
    for name in ("sinus.0", "sinus.1", "timestep_fourier_encoder"):
        result[f"action_in_proj.{name}.freqs"] = ([1, 10], "torch.bfloat16")
    for name in ("accel_mean", "accel_std", "curvature_mean", "curvature_std"):
        result["action_space." + name] = ([], "torch.bfloat16")
    return result


def _check_small_buffers(rows, device):
    expected = _expected_small_buffers()
    _require(set(rows) == set(expected), "未知/缺少非persistent buffer: " + repr(set(rows) ^ set(expected)))
    _require(sum(row["bytes"] for row in rows.values()) == 652, "非persistent不是652B")
    for name, (shape, dtype) in expected.items():
        _require(rows[name]["shape"] == shape and rows[name]["dtype"] == dtype and
                 rows[name]["device"] == device, "小buffer shape/dtype/device不符: " + name)


def _check_ordinary(rows, device):
    expected = {"vlm.model.language_model.rotary_emb.original_inv_freq": ([64], "torch.float32"),
                "expert.rotary_emb.original_inv_freq": ([64], "torch.float32")}
    for name in ("accel_mean", "accel_std", "curvature_mean", "curvature_std"):
        expected[f"traj_tokenizer.action_space._buffers.{name}"] = ([], "torch.bfloat16")
    _require(set(rows) == set(expected), "未知/缺少普通tensor属性: " + repr(set(rows) ^ set(expected)))
    for name, (shape, dtype) in expected.items():
        _require(rows[name]["shape"] == shape and rows[name]["dtype"] == dtype and
                 rows[name]["device"] == device, "普通tensor shape/dtype/device不符: " + name)


def _runtime_module_evidence():
    evidence = {}
    for name, digest in SOURCE_HASHES.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve(strict=True)
        evidence[name] = verify_hash(path, digest)
    for name in ("torch", "transformers", "hydra", "safetensors", "accelerate"):
        try:
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve(strict=True)
            evidence[name] = {**fingerprint(path, digest=sha256_file(path)),
                              "version": getattr(module, "__version__", None)}
        except ModuleNotFoundError as error:
            if error.name != name or name != "accelerate":
                raise
            evidence[name] = {"available": False, "required": False}
    return evidence


def _rss_peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _native_import_evidence(evidence):
    """只读已导入flash-attn初始化/extension入口；hash不是无GPU活动的证明。"""
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if not name.startswith(("flash_attn", "flash_attn_2_cuda", "flash_attn_3_cuda")):
            continue
        path = getattr(module, "__file__", None)
        if path is None or not Path(path).is_file():
            modules[name] = {"path": path, "source_readable": False}
            continue
        record = fingerprint(path, digest=sha256_file(path))
        if Path(path).suffix == ".py":
            tree = ast.parse(Path(path).read_text())
            record["module_level_statements"] = []
            for statement in tree.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                calls = [ast.unparse(node.func) for node in ast.walk(statement) if isinstance(node, ast.Call)]
                imports = [ast.unparse(node) for node in ast.walk(statement)
                           if isinstance(node, (ast.Import, ast.ImportFrom))]
                if calls or imports:
                    record["module_level_statements"].append({"line": statement.lineno,
                                                             "calls": calls, "imports": imports})
            record["source_readable"] = True
        else:
            record.update(source_readable=False, native_initializers_and_launches_verified=False)
        modules[name] = record
    evidence["flash_attn_imported_modules"] = modules
    try:
        distribution = importlib.metadata.distribution("flash_attn")
        evidence["flash_attn_distribution"] = {"version": distribution.version,
            "installation_path": str(Path(distribution.locate_file("")).resolve())}
    except importlib.metadata.PackageNotFoundError:
        evidence["flash_attn_distribution"] = {"installed": False}
    evidence["native_gpu_activity_verified"] = None
    evidence["native_import_and_pid_observation_not_proven"] = [
        "native extension初始化/隐式GPU分配/launch不可仅凭Python source与文件hash证明",
        "探针PID的独立GPU进程/内存观察由主任务采集；短暂峰值可能漏采",
        "Torch is_initialized=False仅是Python状态；Driver discovery可能cuInit"]


def _tokenizer_contract(model, processor, base_module):
    from transformers import AutoProcessor
    tokenizer = model.tokenizer
    reference = AutoProcessor.from_pretrained(model.config.vlm_name_or_path,
        min_pixels=model.config.min_pixels, max_pixels=model.config.max_pixels,
        local_files_only=True).tokenizer
    discrete = [f"<i{index}>" for index in range(model.config.traj_vocab_size)]
    reference.add_tokens(discrete)
    reference.add_tokens(list(base_module.SPECIAL_TOKENS.values()), special_tokens=True)
    serial = tokenizer.backend_tokenizer.to_str()
    _require(serial == reference.backend_tokenizer.to_str() and len(tokenizer) == len(reference),
             "完整tokenizer序列化/长度不符")
    token_ids = tokenizer.convert_tokens_to_ids(discrete)
    _require(token_ids == reference.convert_tokens_to_ids(discrete), "全部4000 IDs不符")
    _require(tokenizer.traj_token_start_idx == model.config.traj_token_start_idx == 151669 and
             tokenizer.traj_token_ids == model.config.traj_token_ids, "trajectory边界ID不符")
    special = {name: tokenizer.convert_tokens_to_ids(value)
               for name, value in base_module.SPECIAL_TOKENS.items()}
    _require(special == model.special_token_ids, "全部special IDs不符")
    _require(tokenizer.padding_side == reference.padding_side, "padding_side不符")
    return {"length": len(tokenizer), "serialized_sha256": hashlib.sha256(serial.encode()).hexdigest(),
            "discrete_token_ids": token_ids, "special_token_ids": special,
            "traj_token_ids": tokenizer.traj_token_ids, "padding_side": tokenizer.padding_side,
            "config_padding_side": model.config.padding_side,
            "processor_tokenizer_same_object": processor.tokenizer is tokenizer,
            "processor_chat_template": processor.chat_template,
            "processor_image_config": processor.image_processor.to_dict(),
            "processor_video_config": processor.video_processor.to_dict()}


def _build_meta_skeleton(plan, phase_evidence):
    """真实原构造；meta通过前不允许实体阶段，任何前提失败原样抛栈。"""
    validate_plan(plan)
    # 原源码预检先于导入，避免意外导入本机不同base_model。
    source_root = Path(plan.paths["source_root"])
    for name, digest in SOURCE_HASHES.items():
        if name.startswith("alpamayo1_5"):
            verify_hash(within(source_root / (name.replace(".", "/") + ".py"), source_root), digest)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    helper = _load_frozen_helper(plan.paths["helper"])
    sys.path.insert(0, str(source_root))
    import torch
    original_dtype = str(torch.get_default_dtype())
    phase_evidence["default_dtype_before"] = original_dtype
    phase_evidence["torch_capability_source_preimport"] = _validate_torch_capability_source(torch)
    callers, hf_sources = _frozen_hf_capability_callers()
    phase_evidence["hf_capability_sources_preimport"] = hf_sources
    with helper._block_network(), _meta_guards(torch, capability_callers=callers,
                                              evidence=phase_evidence), ExitStack() as stack:
        from transformers import AutoProcessor, Qwen3VLConfig, PreTrainedModel
        from transformers.modeling_utils import no_init_weights
        from alpamayo1_5.config import Alpamayo1_5Config
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        from alpamayo1_5 import helper as official_helper
        from alpamayo1_5.models import base_model
        runtime = _runtime_module_evidence()
        allowed = {plan.paths["cosmos"], plan.paths["qwen"]}
        calls = []
        for cls in (AutoProcessor, Qwen3VLConfig):
            original = cls.from_pretrained
            def local_only(path, *args, _original=original, **kwargs):
                actual = str(Path(path).resolve(strict=True))
                _require(actual in allowed, "from_pretrained非允许配置路径: " + actual)
                calls.append({"class": _original.__self__.__name__, "path": actual})
                kwargs["local_files_only"] = True
                return _original(actual, *args, **kwargs)
            stack.enter_context(patch.object(cls, "from_pretrained", local_only))
        def forbidden_weights(*args, **kwargs):
            raise GateError("PhaseA 禁止PreTrainedModel.from_pretrained完整权重")
        stack.enter_context(patch.object(PreTrainedModel, "from_pretrained", forbidden_weights))
        local = dict(plan.model_config)
        local["vlm_name_or_path"] = plan.paths["cosmos"]
        config = Alpamayo1_5Config(**local)
        processor = AutoProcessor.from_pretrained(plan.paths["qwen"],
            min_pixels=official_helper.MIN_PIXELS, max_pixels=official_helper.MAX_PIXELS,
            local_files_only=True)
        baseline = _rss_peak_bytes()
        phase_evidence["rss_peak_constructor_baseline_bytes"] = baseline
        # 不修改原flash_attention_2配置；仅冻结HF调用点真实查询，未知query/实体CUDA仍停止。
        phase_evidence["phase"] = "constructor"
        with _allocation_guard(torch, phase_evidence, default_device="meta"), default_bf16(), torch.device("meta"), no_init_weights():
            model = Alpamayo1_5(config)
        phase_evidence["phase"] = "meta_audit"
        _require(str(torch.get_default_dtype()) == original_dtype, "构造default dtype未恢复")
        original_rows = audit_state(model, plan.original)
        original_order = list(model.state_dict())
        buffers_before = nonpersistent_buffers(model)
        ordinary_before = unregistered_tensors(model)
        _check_small_buffers(buffers_before, "meta")
        _check_ordinary(ordinary_before, "meta")
        generation_before = model.vlm.generation_config.to_dict()
        _require(model.vlm.config.text_config.tie_word_embeddings is False and
                 model.vlm.config.text_config.intermediate_size == 12288 and
                 model.original_vocab_size == 151936, "VLM config/original_vocab/alias前提不符")
        _replace_meta_projections(model, plan)
        target_rows = audit_state(model, plan.target)
        phase_evidence["phase"] = "small_state_restore"
        with _allocation_guard(torch, phase_evidence):
            _restore_small_state(model)
        phase_evidence["phase"] = "tokenizer_and_final_audit"
        buffers_after = nonpersistent_buffers(model)
        ordinary_after = unregistered_tensors(model)
        _check_small_buffers(buffers_after, "cpu")
        _check_ordinary(ordinary_after, "cpu")
        rope_aliases = {}
        for name in ("vlm.model.language_model.rotary_emb", "expert.rotary_emb"):
            rotary = model.get_submodule(name)
            rope_aliases[name] = {"original_inv_freq_is_inv_freq":
                                 rotary.original_inv_freq is rotary.inv_freq,
                                 "attention_scaling": rotary.attention_scaling,
                                 "max_seq_len_cached": rotary.max_seq_len_cached,
                                 "original_max_seq_len": rotary.original_max_seq_len,
                                 "mrope_section": rotary.mrope_section}
            _require(rope_aliases[name]["original_inv_freq_is_inv_freq"], "CPU恢复RoPE对象关联不符")
        processor.tokenizer = model.tokenizer
        tokenizer = _tokenizer_contract(model, processor, base_model)
        _require(model.vlm.generation_config.to_dict() == generation_before, "generation config发生变化")
        backend = {"outer": model.config.attn_implementation,
            "vlm": model.vlm.config._attn_implementation,
            "text": model.vlm.config.text_config._attn_implementation,
            "vision": model.vlm.config.vision_config._attn_implementation,
            "expert": model.expert.config._attn_implementation,
            "vlm_attention_class": type(model.vlm.model.language_model.layers[0].self_attn).__name__,
            "vision_attention_class": type(model.vlm.model.visual.blocks[0].attn).__name__,
            "expert_attention_class": type(model.expert.layers[0].self_attn).__name__}
        _require(all(backend[name] == "flash_attention_2" for name in ("outer", "vlm", "text", "vision"))
                 and backend["expert"] == "sdpa", "原flash2/Expert SDPA后端配置变化")
        phase_evidence["backend_contract"] = backend
        model.eval().requires_grad_(False)
        peak = _rss_peak_bytes()
        _require(peak - baseline <= 256 * 1024 * 1024, "meta阶段RSS增量超过256MiB")
        _require(str(torch.get_default_dtype()) == original_dtype, "恢复小状态后default dtype污染")
        small_values = {}
        for name, value in model.named_buffers():
            if name in buffers_after:
                _require(torch.isfinite(value).all().item(), "小buffer非finite: " + name)
                small_values[name] = {"raw_sha256": hashlib.sha256(
                    value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                    "values": value.float().tolist()}
        audit = {"schema": "alp15-meta-audit-v1", "decision": DECISION, "status": "meta_passed",
                 "plan_sha256": plan.plan_sha256, "runtime_sources": runtime,
                 "original_state": original_rows, "target_state": target_rows,
                 "original_ordered_keys": original_order,
                 "protected_ordered_keys": [k for k in original_order if k in plan.protected],
                 "parent_guard_ordered_keys": [k for k in original_order if not any(
                     k.startswith(name + ".") for name in plan.packed if ".mlp." in name)],
                 "protected_expected_sha256": PROTECTED_SHA, "parent_expected_sha256": PARENT_GUARD_SHA,
                 "protected_payload_hash_verified": False, "parent_payload_hash_verified": False,
                 "nonpersistent_before": buffers_before, "nonpersistent_after": buffers_after,
                 "ordinary_before": ordinary_before, "ordinary_after": ordinary_after,
                 "small_buffer_values": small_values, "generation_config": generation_before,
                 "rope_python_state_and_cpu_aliases": rope_aliases,
                 "model_config": model.config.to_dict(), "tokenizer_contract": tokenizer,
                 "from_pretrained_calls": calls, "default_dtype_before": original_dtype,
                 "default_dtype_after": str(torch.get_default_dtype()),
                 "rss_peak_baseline_bytes": baseline, "rss_peak_bytes": peak,
                 "rss_peak_increment_bytes": peak - baseline, "rss_budget_bytes": 256 * 1024 * 1024,
                 "capability_and_guard_evidence": phase_evidence, "original_payload_read_bytes": 0,
                 "entity_load_authorized": False,
                 "not_proven": ["旧实体loader配置/小状态对拍", "实体源与target逐tensor哈希", "系统pagecache预算",
                                "严格轨迹对拍", "Orin冷启动/16GB/max300ms/teacher损失"]}
        return model, processor, audit


def build_meta_skeleton(plan):
    """失败也携带已实际导入路径/版本/default dtype/RSS，不丢失停止现场。"""
    phase_evidence = {}
    try:
        model, processor, audit = _build_meta_skeleton(plan, phase_evidence)
        phase_evidence["default_dtype_after"] = str(sys.modules["torch"].get_default_dtype())
        _native_import_evidence(phase_evidence)
        # native extension/PID独立观察未完成不能把Python/meta检查算作完整通过。
        if phase_evidence["native_gpu_activity_verified"] is not True:
            phase_evidence["cpu_meta_checks_passed"] = True
            # 保留全部真实CPU/meta检查产物，但明确候选状态，不作为通过MetaAudit。
            audit["status"] = "cpu_meta_checks_passed_native_evidence_pending"
            audit["audit_sha256"] = canonical_sha(audit)
            error = GateError("CPU meta检查完成，但native import/PID GPU活动独立审核未证明，停止升级")
            error.cpu_meta_candidate_audit = audit
            raise error
        audit["audit_sha256"] = canonical_sha(audit)
        return model, processor, audit
    except Exception as error:
        modules = {}
        for name in (*SOURCE_HASHES, "torch", "transformers", "hydra", "accelerate", "safetensors"):
            module = sys.modules.get(name)
            path = getattr(module, "__file__", None)
            if path is not None and Path(path).is_file():
                modules[name] = {"path": str(Path(path).resolve()), "sha256": sha256_file(path),
                                 "version": getattr(module, "__version__", None)}
        torch = sys.modules.get("torch")
        if torch is not None:
            phase_evidence["default_dtype_after"] = str(torch.get_default_dtype())
            phase_evidence["cuda_is_initialized_at_failure"] = torch.cuda.is_initialized()
        _native_import_evidence(phase_evidence)
        error.phase_evidence = {**phase_evidence, "runtime_sources_imported_before_failure": modules,
            "default_dtype_at_failure": str(torch.get_default_dtype()) if torch else None,
            "rss_peak_bytes_at_failure": _rss_peak_bytes(),
            "forbidden_modules_present": [name for name in sys.modules
                if name.split(".")[0] in {"modelopt", "triton"}],
            "original_payload_read_bytes": 0, "entity_loaded": False}
        if hasattr(error, "cpu_meta_candidate_audit"):
            candidate = error.cpu_meta_candidate_audit
            candidate.pop("audit_sha256", None)
            candidate["audit_sha256"] = canonical_sha(candidate)
        raise


def validate_meta_audit(plan, audit):
    value = dict(audit)
    expected = value.pop("audit_sha256", None)
    _require(expected == canonical_sha(value), "MetaAudit内容hash不符")
    _require(audit.get("status") == "meta_passed" and audit.get("plan_sha256") == plan.plan_sha256
             and audit.get("decision") == DECISION, "未通过同来源真实MetaAudit")


def load_packed_only_model(plan, *, device, chunk_bytes=MAX_CHUNK_BYTES, audit_dir):
    """保留最终API；PhaseA真实验收、实体执行批准前硬拒绝，不伪返回模型。"""
    validate_plan(plan)
    audit = read_json(Path(audit_dir) / "MetaAudit.json")
    validate_meta_audit(plan, audit)
    row_chunks([1], "BF16", chunk_bytes)
    raise GateError("PhaseB 实体load未实施/未批准：须决策者验收真实PhaseA后新增执行授权")


def protected_hashes_chunked(model, plan, chunk_bytes=MAX_CHUNK_BYTES):
    raise GateError("PhaseB target readback未实施：禁止把header/source哈希冒作907实体复制验真")
