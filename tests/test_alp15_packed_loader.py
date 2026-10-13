"""PhaseA合成回归；不代表真实1159键官方模型meta构造成功。"""
from __future__ import annotations
import hashlib
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import alp15_packed_loader as loader


def fixture(path, tensors, metadata=None):
    header, offset = {}, 0
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, dtype, shape in tensors:
        size = math.prod(shape) * loader.DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.write(bytes(offset))
    return raw


class HeaderTests(unittest.TestCase):
    def test_header_reads_exactly_header_never_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.safetensors"
            raw = fixture(path, [("p", "BF16", [2, 3])])
            original = Path.open
            reads = []
            class ReadGuard:
                def __init__(self, stream): self.stream = stream
                def __enter__(self): return self
                def __exit__(self, *args): self.stream.close()
                def read(self, count):
                    reads.append(count)
                    self.stream.tell()
                    if self.stream.tell() + count > len(raw) + 8:
                        raise AssertionError("payload accessed")
                    return self.stream.read(count)
            def guarded(p, *args, **kwargs):
                return ReadGuard(original(p, *args, **kwargs))
            with patch.object(Path, "open", guarded):
                value = loader.read_safetensors_header(path)
            self.assertEqual(reads, [8, len(raw)])
            self.assertEqual(value["header_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(value["tensors"]["p"]["shape"], [2, 3])
            self.assertEqual(value["payload_read_bytes"], 0)

    def test_duplicate_key_rejected(self):
        with self.assertRaisesRegex(loader.GateError, "重复"):
            json.loads('{"x":1,"x":2}', object_pairs_hook=loader._unique_object)

    def test_real_overlay_weight_key_contract(self):
        # 主任务L20 mlp_29真实header键/shape；仅metadata，不分配大tensor/payload。
        header = {"tensors": {
            "gate_proj.weight": {"dtype": "BF16", "shape": [5376, 4096]},
            "up_proj.weight": {"dtype": "BF16", "shape": [5376, 4096]},
            "down_proj.weight": {"dtype": "BF16", "shape": [4096, 5376]}}}
        loader._validate_overlay_header(header)
        for invalid in (
            {key.removesuffix(".weight"): value for key, value in header["tensors"].items()},
            {key: value for key, value in header["tensors"].items() if key != "down_proj.weight"},
            {**header["tensors"], "protected.weight": {"dtype": "BF16", "shape": [1]}},
            {**header["tensors"], "down_proj.weight": {"dtype": "BF16", "shape": [5376, 4096]}},
            {**header["tensors"], "up_proj.weight": {"dtype": "F16", "shape": [5376, 4096]}},
        ):
            with self.assertRaises(loader.GateError):
                loader._validate_overlay_header({"tensors": invalid})

    def test_truncated_invalid_dtype_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.safetensors"
            path.write_bytes(struct.pack("<Q", 100))
            with self.assertRaises(loader.GateError): loader.read_safetensors_header(path)
            raw = b'{"p":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}'
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(8))
            with self.assertRaisesRegex(loader.GateError, "dtype"):
                loader.read_safetensors_header(path)
            raw = b'{"p":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}'
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(4))
            with self.assertRaisesRegex(loader.GateError, "重叠"):
                loader.read_safetensors_header(path)

    def test_path_escape_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            outside = Path(directory) / "outside"
            outside.write_text("x")
            (root / "link").symlink_to(outside)
            with self.assertRaises(loader.GateError): loader.within(root / "link", root)

    def test_chunk_limit_scalar_and_single_row_rejection(self):
        chunks = loader.row_chunks([155697, 4096], "BF16")
        self.assertEqual(chunks[0], (0, 4096, 33554432))
        self.assertEqual(chunks[-1][1], 155697)
        self.assertTrue(all(size <= 33554432 for _, _, size in chunks))
        self.assertEqual(loader.row_chunks([], "BF16"), [(0, 1, 2)])
        for shape, limit in (([1, 16777217], 33554432), ([1], 33554433)):
            with self.assertRaises(loader.GateError): loader.row_chunks(shape, "BF16", limit)

    def test_explicit_namespace_and_counts(self):
        names = loader.quantized_modules()
        self.assertEqual(len(set(names)), 252)
        self.assertTrue(all(name.startswith("vlm.model.language_model.layers.") for name in names))
        self.assertEqual(sum(".mlp." in name for name in names), 108)
        self.assertEqual(sum(".self_attn." in name for name in names), 144)

    def test_plan_binding_and_changed_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_text("one")
            plan = loader._freeze_plan(schema="test", decision=loader.DECISION, paths={},
                assets={"file": loader.fingerprint(path, digest=loader.sha256_file(path))},
                original={}, protected={}, packed={}, target={}, model_config={})
            loader.validate_plan(plan)
            plan.target["added"] = {}
            with self.assertRaisesRegex(loader.GateError, "内容变化"):
                loader.validate_plan(plan)
            del plan.target["added"]
            path.write_text("two")
            with self.assertRaises(loader.GateError): loader.validate_plan(plan)

    def test_meta_audit_binding_and_forged_digest_rejected(self):
        plan = loader._freeze_plan(schema="test", decision=loader.DECISION, paths={},
            assets={}, original={}, protected={}, packed={}, target={}, model_config={})
        audit = {"status": "meta_passed", "plan_sha256": plan.plan_sha256, "decision": loader.DECISION}
        audit["audit_sha256"] = loader.canonical_sha(audit)
        loader.validate_meta_audit(plan, audit)
        audit["plan_sha256"] = "wrong"
        with self.assertRaises(loader.GateError): loader.validate_meta_audit(plan, audit)

    def test_source_digests_are_distinct(self):
        self.assertNotEqual(loader.PROTECTED_SHA, loader.PARENT_GUARD_SHA)

    def test_no_entity_load_even_with_synthetic_passed_audit(self):
        plan = loader._freeze_plan(schema="test", decision=loader.DECISION, paths={},
            assets={}, original={}, protected={}, packed={}, target={}, model_config={})
        with tempfile.TemporaryDirectory() as directory:
            audit = {"status": "meta_passed", "plan_sha256": plan.plan_sha256, "decision": loader.DECISION}
            audit["audit_sha256"] = loader.canonical_sha(audit)
            (Path(directory) / "MetaAudit.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(loader.GateError, "未实施/未批准"):
                loader.load_packed_only_model(plan, device="cuda:0", audit_dir=directory)
        with self.assertRaises(loader.GateError): loader.protected_hashes_chunked(None, plan)


class SyntheticTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_dtype_context_restores_after_failure(self):
        torch = self.torch
        before = torch.get_default_dtype()
        with self.assertRaises(ValueError):
            with loader.default_bf16():
                self.assertEqual(torch.get_default_dtype(), torch.bfloat16)
                raise ValueError("synthetic failure")
        self.assertEqual(before, torch.get_default_dtype())

    def test_shape_dtype_kind_missing_and_object_alias(self):
        torch = self.torch
        model = torch.nn.Module()
        model.register_parameter("weight", torch.nn.Parameter(torch.empty([2, 3], device="meta", dtype=torch.bfloat16)))
        expected = {"weight": {"shape": [2, 3], "dtype": "BF16", "kind": "parameter"}}
        self.assertEqual(loader.audit_state(model, expected)["weight"]["device"], "meta")
        for changed in ({"weight": {"shape": [3, 2], "dtype": "BF16", "kind": "parameter"}},
                        {"weight": {"shape": [2, 3], "dtype": "U8", "kind": "parameter"}},
                        {"weight": {"shape": [2, 3], "dtype": "BF16", "kind": "buffer"}}, {}):
            with self.assertRaises(loader.GateError): loader.audit_state(model, changed)
        model.register_parameter("tied", model.weight)
        expected["tied"] = expected["weight"]
        with self.assertRaisesRegex(loader.GateError, "alias"): loader.audit_state(model, expected)

    def test_unregistered_python_action_space_and_nonpersistent(self):
        torch = self.torch
        model = torch.nn.Module()
        model.register_buffer("small", torch.ones(1), persistent=False)
        PythonObject = type("SyntheticTrajectoryTokenizer", (), {"__module__": "alpamayo1_5.synthetic"})
        model.traj_tokenizer = PythonObject()
        model.traj_tokenizer.action_space = torch.nn.Module()
        model.traj_tokenizer.action_space.register_buffer("freq", torch.ones(2, device="meta"), persistent=False)
        self.assertEqual(set(loader.nonpersistent_buffers(model)), {"small"})
        rows = loader.unregistered_tensors(model)
        self.assertEqual(rows["traj_tokenizer.action_space._buffers.freq"]["device"], "meta")

    def test_guards_reject_cuda_payload_modelopt_triton_and_restore(self):
        torch = self.torch
        original = torch.cuda.is_available
        with loader._meta_guards(torch):
            for operation in (lambda: torch.cuda.is_available(), lambda: __import__("triton"),
                              lambda: __import__("modelopt"), lambda: torch.load("synthetic.pt"),
                              lambda: Path("synthetic.safetensors").open("rb")):
                with self.assertRaises(loader.GateError): operation()
        self.assertIs(torch.cuda.is_available, original)

    def test_guard_rule_map_callable_identities_are_distinct(self):
        torch = self.torch
        # Dynamo rule map按callable对象键，不允许不同API共用一个函数对象。
        with loader._meta_guards(torch):
            entries = [(torch.load, "SkipFunctionVariable"),
                       (torch.cuda.is_available, "TorchInGraphFunctionVariable"),
                       (torch.cuda.device_count, "DeviceCountRule"),
                       (torch.Tensor.cuda, "TensorCudaRule"),
                       (torch.nn.Module.cuda, "ModuleCudaRule")]
            rule_map = {}
            for entry, rule in entries:
                if entry in rule_map:
                    self.assertEqual(rule_map[entry], rule, "Duplicate torch object with different rules")
                rule_map[entry] = rule
            self.assertEqual(len(rule_map), len(entries))

    def test_approved_capability_query_delegates_real_bool_once(self):
        torch = self.torch
        caller = {"path": str(Path(__file__).resolve()), "sha256": loader.sha256_file(__file__),
                  "function": "test_approved_capability_query_delegates_real_bool_once"}
        for result in (False, True):
            calls = []
            def is_available():
                calls.append("original query")
                return result
            evidence = {}
            with patch.object(torch.cuda, "is_available", is_available):
                with loader._meta_guards(torch, capability_callers=[caller], evidence=evidence):
                    self.assertIs(torch.cuda.is_available(), result)
            self.assertEqual(calls, ["original query"])
            self.assertEqual(evidence["cuda_capability_queries"][0]["result"], result)
            self.assertTrue(evidence["guard_restored_all"])

    def test_capability_unknown_caller_never_delegates(self):
        torch = self.torch
        calls = []
        def is_available():
            calls.append(True)
            return True
        with patch.object(torch.cuda, "is_available", is_available):
            with loader._meta_guards(torch):
                with self.assertRaisesRegex(loader.GateError, "未批准"):
                    torch.cuda.is_available()
        self.assertEqual(calls, [])

    def test_capability_exception_clears_state_and_restores(self):
        torch = self.torch
        evidence = {}
        caller = {"path": str(Path(__file__).resolve()), "sha256": loader.sha256_file(__file__),
                  "function": "test_capability_exception_clears_state_and_restores"}
        def is_available():
            raise ValueError("original-query-failure")
        with patch.object(torch.cuda, "is_available", is_available):
            with self.assertRaisesRegex(ValueError, "original-query-failure"):
                with loader._meta_guards(torch, capability_callers=[caller], evidence=evidence):
                    torch.cuda.is_available()
            self.assertIs(torch.cuda.is_available, is_available)
        self.assertEqual(evidence["query_context_depth_after"], 0)
        self.assertTrue(evidence["guard_restored_all"])
        self.assertNotIn("result", evidence["cuda_capability_queries"][0])

    def test_nested_device_count_only_from_original_query(self):
        torch = self.torch
        evidence = {}
        caller = {"path": str(Path(__file__).resolve()), "sha256": loader.sha256_file(__file__),
                  "function": "test_nested_device_count_only_from_original_query"}
        calls = []
        def device_count():
            calls.append("count")
            return 2
        def is_available():
            return torch.cuda.device_count() > 0
        with patch.object(torch.cuda, "is_available", is_available), patch.object(torch.cuda, "device_count", device_count):
            with loader._meta_guards(torch, capability_callers=[caller], evidence=evidence):
                with self.assertRaises(loader.GateError): torch.cuda.device_count()
                self.assertIs(torch.cuda.is_available(), True)
        self.assertEqual(calls, ["count"])
        self.assertEqual(len(evidence["cuda_nested_device_count_queries"]), 1)

    def test_discovery_primitive_only_from_original_query(self):
        torch = self.torch
        evidence = {}
        caller = {"path": str(Path(__file__).resolve()), "sha256": loader.sha256_file(__file__),
                  "function": "test_discovery_primitive_only_from_original_query"}
        calls = []
        def primitive():
            calls.append("primitive")
            return 2
        def is_available():
            return torch._C._cuda_getDeviceCount() > 0
        with patch.object(torch.cuda, "is_available", is_available), patch.object(
                torch._C, "_cuda_getDeviceCount", primitive, create=True):
            with loader._meta_guards(torch, capability_callers=[caller], evidence=evidence):
                with self.assertRaises(loader.GateError): torch._C._cuda_getDeviceCount()
                self.assertIs(torch.cuda.is_available(), True)
        self.assertEqual(calls, ["primitive"])
        self.assertEqual(len(evidence["cuda_driver_discovery_primitives"]), 1)

    def test_initial_cuda_initialized_is_rejected(self):
        torch = self.torch
        with patch.object(torch.cuda, "is_initialized", return_value=True):
            with self.assertRaisesRegex(loader.GateError, "开始时"):
                with loader._meta_guards(torch): pass

    def test_query_initialization_is_rejected_without_fake_result(self):
        torch = self.torch
        caller = {"path": str(Path(__file__).resolve()), "sha256": loader.sha256_file(__file__),
                  "function": "test_query_initialization_is_rejected_without_fake_result"}
        initialized = [False]
        def is_available():
            initialized[0] = True
            return True
        evidence = {}
        with patch.object(torch.cuda, "is_available", is_available), patch.object(
                torch.cuda, "is_initialized", lambda: initialized[0]):
            with self.assertRaisesRegex(loader.GateError, "CUDA.*初始化"):
                with loader._meta_guards(torch, capability_callers=[caller], evidence=evidence):
                    torch.cuda.is_available()
        self.assertEqual(evidence["cuda_capability_queries"][0]["result"], True)
        self.assertTrue(evidence["cuda_is_initialized_after"])
        self.assertTrue(evidence["guard_restored_all"])

    def test_cpu_allocation_limit_and_meta_large_allowed(self):
        torch = self.torch
        evidence = {"cuda_execution_attempts": [], "phase": "synthetic_constructor"}
        with loader._allocation_guard(torch, evidence):
            with self.assertRaisesRegex(loader.GateError, "超过1MiB"):
                torch.empty([1024 * 1024 + 1], dtype=torch.uint8, device="cpu")
            meta = torch.empty([1024 * 1024 + 1], dtype=torch.uint8, device="meta")
            self.assertTrue(meta.is_meta)
            small = torch.ones(4, device="cpu")
            self.assertEqual(small.shape, (4,))
            with self.assertRaisesRegex(loader.GateError, "超过1MiB"):
                torch.tensor([0] * (1024 * 1024 // 8 + 1), device="cpu")

    def test_guarded_factory_preserves_meta_default_device(self):
        torch = self.torch
        with loader._allocation_guard(torch, {"cuda_execution_attempts": []}, default_device="meta"), torch.device("meta"):
            self.assertTrue(torch.tensor(1.0).is_meta)
            self.assertTrue(torch.as_tensor([1.0, 2.0]).is_meta)
            self.assertTrue(torch.empty([2, 3]).is_meta)

    def test_cuda_factory_and_to_refused_before_allocation(self):
        torch = self.torch
        evidence = {}
        with loader._meta_guards(torch, evidence=evidence), loader._allocation_guard(torch, evidence):
            with self.assertRaises(loader.GateError): torch.empty(2, device="cuda")
            with self.assertRaises(loader.GateError): torch.ones(2).to("cuda")
        self.assertFalse(evidence["cuda_is_initialized_after"])
        self.assertTrue(evidence["guard_restored_all"])

    def test_direct_native_cuda_init_is_also_blocked(self):
        torch = self.torch
        calls = []
        def native_init(): calls.append("must not execute")
        with patch.object(torch._C, "_cuda_init", native_init, create=True):
            with loader._meta_guards(torch):
                with self.assertRaises(loader.GateError): torch._C._cuda_init()
            self.assertIs(torch._C._cuda_init, native_init)
        self.assertEqual(calls, [])

    def test_every_guard_entry_blocks_and_restores_after_exception(self):
        torch = self.torch
        owners_and_names = [(torch, "load"), (torch.Tensor, "cuda"), (torch.nn.Module, "cuda")]
        owners_and_names += [(torch.cuda, name) for name in (
            "is_available", "device_count", "current_device", "get_device_properties",
            "get_device_capability", "init", "_lazy_init", "synchronize", "set_device",
            "memory_allocated", "max_memory_allocated", "empty_cache") if hasattr(torch.cuda, name)]
        originals = [getattr(owner, name) for owner, name in owners_and_names]
        with self.assertRaisesRegex(ValueError, "restore"):
            with loader._meta_guards(torch):
                replacements = [getattr(owner, name) for owner, name in owners_and_names]
                self.assertEqual(len(set(replacements)), len(replacements))
                for function in replacements:
                    with self.assertRaises(loader.GateError):
                        function()
                raise ValueError("synthetic restore")
        for (owner, name), original in zip(owners_and_names, originals):
            self.assertIs(getattr(owner, name), original)

    def test_252_placeholders_remain_meta_and_forward_fails(self):
        torch = self.torch
        model = torch.nn.Module()
        model.vlm = torch.nn.Module()
        model.vlm.model = torch.nn.Module()
        model.vlm.model.language_model = torch.nn.Module()
        layers = torch.nn.ModuleList()
        for _ in range(36):
            layer = torch.nn.Module()
            for section, projections in (("mlp", ("gate_proj", "up_proj", "down_proj")),
                                         ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj"))):
                owner = torch.nn.Module()
                for projection in projections:
                    setattr(owner, projection, torch.nn.Linear(128, 2, bias=False,
                        device="meta", dtype=torch.bfloat16))
                setattr(layer, section, owner)
            layers.append(layer)
        model.vlm.model.language_model.layers = layers
        packed, target = {}, {}
        for name in loader.quantized_modules():
            tensors = {"qweight": {"shape": [2, 64], "dtype": "U8"},
                       "amax": {"shape": [2, 1], "dtype": "BF16"},
                       "pre_scale": {"shape": [128], "dtype": "BF16"}}
            packed[name] = {"shape": [2, 128], "tensors": tensors}
            for key, value in tensors.items():
                target[name + "." + key] = {**value, "kind": "buffer"}
        plan = loader._freeze_plan(schema="synthetic", decision=loader.DECISION,
            paths={}, assets={}, original={}, protected={}, packed=packed, target=target, model_config={})
        loader._replace_meta_projections(model, plan)
        self.assertEqual(len(loader.audit_state(model, target)), 756)
        self.assertEqual(len(list(model.parameters())), 0)
        with self.assertRaisesRegex(loader.GateError, "占位"):
            layers[0].mlp.gate_proj(torch.empty([1, 128], device="meta"))
        self.assertEqual(layers[0].mlp.intermediate_size, 5376)


if __name__ == "__main__":
    unittest.main()
