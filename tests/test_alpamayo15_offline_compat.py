"""离线兼容入口的无网络回归测试；不加载模型或真实 clip。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "test_alpamayo15_offline_compat.py"
)
SPEC = importlib.util.spec_from_file_location(
    "test_alpamayo15_offline_compat_script", SCRIPT
)
assert SPEC and SPEC.loader
offline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(offline)


def fixture_file(path: Path, content: bytes) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return len(content), hashlib.sha256(content).hexdigest()


class OfflineCompatTests(unittest.TestCase):
    def _fixture(self, root: Path):
        model = root / "model"
        dataset = root / "dataset"
        cosmos = root / "cosmos"
        qwen = root / "qwen"
        manifest_dir = root / "manifests"

        model_files = {
            "config.json": fixture_file(
                model / "config.json",
                json.dumps({"vlm_name_or_path": offline.MODEL_VLM_NAME}).encode(),
            ),
            "model.safetensors.index.json": fixture_file(
                model / "model.safetensors.index.json",
                json.dumps(
                    {"weight_map": {"weight": "model-00001-of-00001.safetensors"}}
                ).encode(),
            ),
            "model-00001-of-00001.safetensors": fixture_file(
                model / "model-00001-of-00001.safetensors", b"fake-model-shard"
            ),
        }

        features = {
            "LABELS.EGOMOTION": (
                "labels/egomotion/egomotion.chunk_3119.zip",
                *fixture_file(root / "payload" / "egomotion", b"fake-egomotion"),
            )
        }
        metadata = {
            "features.csv": fixture_file(
                root / "payload" / "features", b"fake-features"
            )
        }
        cache = dataset / offline.DATASET_CACHE_NAME
        snapshot = cache / "snapshots" / offline.DATASET_REVISION
        blobs = cache / "blobs"
        records = {
            relative: (size, digest) for relative, size, digest in features.values()
        }
        records.update(metadata)
        for relative, (_, digest) in records.items():
            source = (
                root
                / "payload"
                / ("egomotion" if relative.endswith(".zip") else "features")
            )
            blob = blobs / digest
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(source.read_bytes())
            link = snapshot / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(blob, link.parent))

        prefix = f"{offline.DATASET_CACHE_NAME}/snapshots/{offline.DATASET_REVISION}/"
        manifest = {
            "dataset_repo": offline.DATASET_REPO,
            "dataset_revision": offline.DATASET_REVISION,
            "clip_id": offline.CLIP_ID,
            "t0_us": offline.T0_US,
            "chunk_id": offline.CHUNK_ID,
            "physical_ai_av_version": "0.2.0",
            "physical_ai_av_source_commit": offline.PHYSICAL_AI_AV_COMMIT,
            "features": [
                {
                    "path": prefix + relative,
                    "feature": feature,
                    "chunk": offline.CHUNK_ID,
                    "size": size,
                    "sha256": digest,
                }
                for feature, (relative, size, digest) in features.items()
            ],
            "metadata": [
                {"path": prefix + relative, "size": size, "sha256": digest}
                for relative, (size, digest) in metadata.items()
            ],
        }
        (dataset / "dataset-manifest.json").write_text(json.dumps(manifest))
        for config_dir, repo, revision, source_name in (
            (
                cosmos,
                offline.MODEL_VLM_NAME,
                offline.COSMOS_REVISION,
                "cosmos-reason2-source.json",
            ),
            (
                qwen,
                "Qwen/Qwen3-VL-2B-Instruct",
                offline.QWEN_REVISION,
                "qwen3-vl-source.json",
            ),
        ):
            source_files = []
            for filename in offline.CONFIG_FILES:
                size, digest = fixture_file(config_dir / filename, b"{}")
                source_files.append({"path": filename, "size": size, "sha256": digest})
            fixture_file(
                manifest_dir / source_name,
                json.dumps(
                    {
                        "repo": repo,
                        "revision": revision,
                        "files": source_files,
                    }
                ).encode(),
            )
        return (
            model,
            dataset,
            cosmos,
            qwen,
            manifest_dir,
            model_files,
            features,
            metadata,
        )

    def test_preflight_uses_frozen_manifest_and_stays_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            model, dataset, cosmos, qwen, manifests, model_files, features, metadata = (
                fixture
            )
            args = [
                "--model-dir",
                str(model),
                "--dataset-dir",
                str(dataset),
                "--cosmos-config-dir",
                str(cosmos),
                "--qwen-config-dir",
                str(qwen),
                "--config-manifest-dir",
                str(manifests),
                "--preflight-only",
            ]
            with (
                mock.patch.object(offline, "MODEL_FILES", model_files),
                mock.patch.object(offline, "FEATURES", features),
                mock.patch.object(offline, "METADATA", metadata),
                mock.patch.object(
                    offline.importlib.metadata, "version", return_value="0.2.0"
                ),
                mock.patch.object(
                    offline, "_load_offline_dataset_interface"
                ) as load_interface,
                mock.patch.object(offline, "_run_inference") as inference,
                mock.patch.dict(
                    os.environ, {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
                ),
            ):
                self.assertEqual(offline.main(args), 0)
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
                load_interface.assert_called_once_with(dataset.resolve())
                inference.assert_not_called()

                blob = next((dataset / offline.DATASET_CACHE_NAME / "blobs").iterdir())
                blob.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "大小不匹配|SHA-256"):
                    offline.main(args)

    def test_config_source_requires_frozen_revision_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, _, cosmos, _, manifests, _, _, _ = self._fixture(Path(temporary))
            source = manifests / "cosmos-reason2-source.json"
            payload = json.loads(source.read_text())
            payload["revision"] = "other"
            source.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "revision"):
                offline._verify_config_source(
                    cosmos,
                    "Cosmos",
                    offline.MODEL_VLM_NAME,
                    offline.COSMOS_REVISION,
                    source.name,
                    manifests,
                )
            payload["revision"] = offline.COSMOS_REVISION
            source.write_text(json.dumps(payload))
            (cosmos / "tokenizer.json").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "大小不匹配|SHA-256"):
                offline._verify_config_source(
                    cosmos,
                    "Cosmos",
                    offline.MODEL_VLM_NAME,
                    offline.COSMOS_REVISION,
                    source.name,
                    manifests,
                )

    def test_config_remapping_is_memory_only(self) -> None:
        original = {"vlm_name_or_path": offline.MODEL_VLM_NAME, "vocab_size": 155697}
        mapped = offline._local_model_config(original, Path("/local/cosmos"))
        self.assertEqual(original["vlm_name_or_path"], offline.MODEL_VLM_NAME)
        self.assertEqual(mapped["vlm_name_or_path"], "/local/cosmos")
        self.assertEqual(mapped["vocab_size"], original["vocab_size"])

    def test_model_assembly_uses_local_config_and_local_qwen(self) -> None:
        calls = {}
        local_dtype = object()
        local_tokenizer = object()

        class Config:
            def __init__(self, **kwargs):
                calls["config"] = kwargs

        class Model:
            tokenizer = local_tokenizer

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls["model"] = (path, kwargs)
                return cls()

            def to(self, device):
                calls["device"] = device
                return self

        class Processor:
            tokenizer = None

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls["processor"] = (path, kwargs)
                return cls()

        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = local_dtype
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoProcessor = Processor
        fake_helper = types.ModuleType("alpamayo1_5.helper")
        fake_helper.MIN_PIXELS = 163840
        fake_helper.MAX_PIXELS = 196608
        fake_package = types.ModuleType("alpamayo1_5")
        fake_package.__path__ = []
        fake_package.helper = fake_helper
        fake_config = types.ModuleType("alpamayo1_5.config")
        fake_config.Alpamayo1_5Config = Config
        fake_models = types.ModuleType("alpamayo1_5.models")
        fake_models.__path__ = []
        fake_model_module = types.ModuleType("alpamayo1_5.models.alpamayo1_5")
        fake_model_module.Alpamayo1_5 = Model
        modules = {
            "torch": fake_torch,
            "transformers": fake_transformers,
            "alpamayo1_5": fake_package,
            "alpamayo1_5.helper": fake_helper,
            "alpamayo1_5.config": fake_config,
            "alpamayo1_5.models": fake_models,
            "alpamayo1_5.models.alpamayo1_5": fake_model_module,
        }
        original = {"vlm_name_or_path": offline.MODEL_VLM_NAME, "vocab_size": 155697}
        with mock.patch.dict(sys.modules, modules):
            model, processor = offline._load_local_model_and_processor(
                Path("/local/model"),
                Path("/local/cosmos"),
                Path("/local/qwen"),
                original,
            )
        self.assertIsInstance(model, Model)
        self.assertIs(processor.tokenizer, local_tokenizer)
        self.assertEqual(original["vlm_name_or_path"], offline.MODEL_VLM_NAME)
        self.assertEqual(calls["config"]["vlm_name_or_path"], "/local/cosmos")
        self.assertEqual(calls["model"][0], "/local/model")
        self.assertIsInstance(calls["model"][1]["config"], Config)
        self.assertTrue(calls["model"][1]["local_files_only"])
        self.assertIs(calls["model"][1]["dtype"], local_dtype)
        self.assertEqual(calls["processor"][0], "/local/qwen")
        self.assertTrue(calls["processor"][1]["local_files_only"])
        self.assertEqual(calls["device"], "cuda")

    def test_network_guard_blocks_ip_and_allows_unix_socket(self) -> None:
        with offline._block_network():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                with self.assertRaisesRegex(RuntimeError, "阻止外部"):
                    sock.connect(("127.0.0.1", 9))
            with self.assertRaisesRegex(RuntimeError, "阻止 socket.create_connection"):
                socket.create_connection(("127.0.0.1", 9))
            with tempfile.TemporaryDirectory() as temporary:
                address = str(Path(temporary) / "local.sock")
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                    server.bind(address)
                    server.listen()
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                        client.connect(address)
                        peer, _ = server.accept()
                        peer.close()

    def test_dataset_interface_uses_only_top_level_api_and_one_cached_probe(
        self,
    ) -> None:
        class HfApi:
            def file_exists(self, **kwargs):
                raise AssertionError("真实 file_exists 不应被调用")

        class Features:
            LABELS = SimpleNamespace(EGOMOTION="egomotion")

            @staticmethod
            def get_chunk_feature_filename(chunk, feature):
                assert chunk == offline.CHUNK_ID and feature == "egomotion"
                return "labels/egomotion/egomotion.chunk_3119.zip"

        class Interface:
            features = Features()
            extra_probe = False

            def __init__(self, revision, cache_dir, token):
                self.revision = revision
                self.cache_dir = cache_dir
                self.token = token
                self.metadata_exists = HfApi().file_exists(
                    repo_id=offline.DATASET_REPO,
                    repo_type="dataset",
                    revision=offline.DATASET_REVISION,
                    filename="metadata/feature_presence.parquet",
                )
                if self.extra_probe:
                    HfApi().file_exists(
                        repo_id=offline.DATASET_REPO,
                        repo_type="dataset",
                        revision=offline.DATASET_REVISION,
                        filename="unexpected.parquet",
                    )

            def get_clip_chunk(self, clip_id):
                assert clip_id == offline.CLIP_ID
                return offline.CHUNK_ID

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = HfApi
        fake_av = types.ModuleType("physical_ai_av")
        fake_av.PhysicalAIAVDatasetInterface = Interface
        with (
            mock.patch.dict(
                sys.modules, {"huggingface_hub": fake_hub, "physical_ai_av": fake_av}
            ),
            mock.patch.object(
                offline,
                "FEATURES",
                {"LABELS.EGOMOTION": offline.FEATURES["LABELS.EGOMOTION"]},
            ),
        ):
            interface = offline._load_offline_dataset_interface(Path("/cached/dataset"))
            self.assertTrue(interface.metadata_exists)
            self.assertEqual(interface.cache_dir, "/cached/dataset")
            self.assertFalse(interface.token)
            Interface.extra_probe = True
            with self.assertRaisesRegex(RuntimeError, "未批准"):
                offline._load_offline_dataset_interface(Path("/cached/dataset"))
            with self.assertRaises(AssertionError):
                HfApi().file_exists(filename="unexpected")


if __name__ == "__main__":
    unittest.main()
