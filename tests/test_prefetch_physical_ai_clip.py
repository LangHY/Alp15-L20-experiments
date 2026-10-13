"""单 clip 预取的无网络回归测试。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prefetch_physical_ai_clip.py"
SPEC = importlib.util.spec_from_file_location("prefetch_physical_ai_clip", SCRIPT)
assert SPEC and SPEC.loader
prefetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefetch)


class PrefetchTests(unittest.TestCase):
    def test_downloads_five_frozen_paths_one_by_one_and_reuses_cache(self) -> None:
        expected_paths = (
            "labels/egomotion/egomotion.chunk_3119.zip",
            "camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_3119.zip",
            "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_3119.zip",
            "camera/camera_cross_right_120fov/camera_cross_right_120fov.chunk_3119.zip",
            "camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_3119.zip",
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_dir = root / "persistent-cache"
            output_dir = root / "delivery"

            def add_cached_file(relative: str) -> Path:
                payload = f"fixture:{relative}".encode()
                blob_name = hashlib.sha256(payload).hexdigest()
                blob = cache_dir / prefetch.CACHE_REPO_DIR / "blobs" / blob_name
                snapshot = cache_dir / prefetch.SNAPSHOT_PREFIX / relative
                blob.parent.mkdir(parents=True, exist_ok=True)
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                blob.write_bytes(payload)
                snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))
                return snapshot

            cached_feature = add_cached_file(expected_paths[0])
            for relative in prefetch.METADATA_PATHS:
                add_cached_file(relative)

            class Features:
                LABELS = SimpleNamespace(EGOMOTION="LABELS.EGOMOTION")
                CAMERA = SimpleNamespace(**{
                    name: f"CAMERA.{name}" for name in prefetch.CAMERA_ATTRIBUTES
                })

                @staticmethod
                def get_chunk_feature_filename(chunk_id: int, feature: str) -> str:
                    return prefetch.FEATURE_TEMPLATES[feature].format(chunk_id=chunk_id)

            instances = []

            class FakeInterface:
                features = Features()

                def __init__(self, revision: str, cache_dir: str) -> None:
                    self.revision = revision
                    self.cache_dir = Path(cache_dir)
                    self.downloaded: list[str] = []
                    self.reused: list[str] = []
                    instances.append(self)

                def get_clip_chunk(self, clip_id: str) -> int:
                    self.clip_id = clip_id
                    return 3119

                def download_file(self, filename: str) -> str:
                    self.downloaded.append(filename)
                    snapshot = self.cache_dir / prefetch.SNAPSHOT_PREFIX / filename
                    if snapshot.is_symlink():
                        self.reused.append(filename)
                    else:
                        add_cached_file(filename)
                    return str(snapshot)

                def download_clip_features(self, clip_id: str, features: object) -> None:
                    raise AssertionError("不应调用旧的批量下载 API")

            argv = [str(SCRIPT), "--cache-dir", str(cache_dir),
                    "--output-dir", str(output_dir), "--execute"]
            with mock.patch.object(prefetch, "require_exact_api", return_value=(None, FakeInterface)), \
                    mock.patch.object(sys, "argv", argv):
                self.assertEqual(prefetch.main(), 0)

            interface = instances[0]
            self.assertEqual(interface.revision, prefetch.DATASET_REV)
            self.assertEqual(interface.cache_dir.resolve(), cache_dir.resolve())
            self.assertEqual(interface.clip_id, prefetch.CLIP_ID)
            self.assertEqual(tuple(interface.downloaded), expected_paths)
            self.assertEqual(interface.reused, [expected_paths[0]])
            self.assertTrue(cached_feature.is_symlink())
            manifest = json.loads((output_dir / "dataset-manifest.json").read_text())
            self.assertEqual([record["path"].split("/snapshots/", 1)[1].split("/", 1)[1]
                              for record in manifest["features"]], list(expected_paths))
            self.assertEqual(len(manifest["metadata"]), 4)


if __name__ == "__main__":
    unittest.main()
