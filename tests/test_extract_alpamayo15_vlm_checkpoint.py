"""VLM checkpoint 提取器的小型 safetensors 回归测试。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "extract_alpamayo15_vlm_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "extract_alpamayo15_vlm_checkpoint", SCRIPT
)
assert SPEC and SPEC.loader
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_RUNTIME, "需要 torch 与 safetensors")
class ExtractVlmCheckpointTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict]:
        model_dir = root / "model"
        model_dir.mkdir()
        shard1 = "model-00001-of-00002.safetensors"
        shard2 = "model-00002-of-00002.safetensors"
        save_file(
            {
                "vlm.model.language_model.embed_tokens.weight": torch.arange(
                    24, dtype=torch.float32
                ).reshape(6, 4),
                "vlm.model.visual.patch_embed.proj.weight": torch.arange(
                    8, dtype=torch.float32
                ).reshape(2, 1, 1, 2, 2),
                "expert.layers.0.mlp.up_proj.weight": torch.ones(3, 3),
            },
            model_dir / shard1,
        )
        save_file(
            {
                "vlm.lm_head.weight": torch.arange(24, dtype=torch.float32).reshape(
                    6, 4
                ),
                "vlm.model.language_model.layers.0.mlp.up_proj.weight": torch.ones(
                    5, 4
                ),
            },
            model_dir / shard2,
        )
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                "vlm.model.language_model.embed_tokens.weight": shard1,
                "vlm.model.visual.patch_embed.proj.weight": shard1,
                "expert.layers.0.mlp.up_proj.weight": shard1,
                "vlm.lm_head.weight": shard2,
                "vlm.model.language_model.layers.0.mlp.up_proj.weight": shard2,
            },
        }
        return model_dir, index

    def test_extracts_only_vlm_and_strips_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir, index = self._fixture(root)
            source_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in model_dir.glob("*.safetensors")
            }
            selected = extractor._select_vlm_tensors(index)
            self.assertEqual(len(selected), 4)
            specs = extractor._inspect_tensors(model_dir, selected)
            groups = extractor._plan_output_shards(specs, max_shard_bytes=128)
            output_dir = root / "output"
            output_dir.mkdir()
            weight_map = extractor._write_weight_shards(model_dir, output_dir, groups)
            extractor._verify_output_shards(model_dir, output_dir, groups, weight_map)

            self.assertEqual(set(weight_map), {item[1] for item in selected})
            self.assertNotIn("expert.layers.0.mlp.up_proj.weight", weight_map)
            self.assertFalse(any(key.startswith("vlm.") for key in weight_map))
            for filename in set(weight_map.values()):
                with safe_open(
                    output_dir / filename, framework="pt", device="cpu"
                ) as handle:
                    self.assertFalse(
                        any(key.startswith("vlm.") for key in handle.keys())
                    )
            self.assertEqual(
                source_hashes,
                {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in model_dir.glob("*.safetensors")
                },
            )

    def test_build_config_uses_extracted_embedding_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            processor = Path(temporary)
            (processor / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3VLForConditionalGeneration"],
                        "model_type": "qwen3_vl",
                        "text_config": {"vocab_size": 4},
                    }
                )
            )
            config = extractor._build_vlm_config(
                processor,
                {"model_dtype": "bfloat16"},
                vocab_size=155697,
            )
            self.assertEqual(config["vocab_size"], 155697)
            self.assertEqual(config["text_config"]["vocab_size"], 155697)
            self.assertEqual(config["dtype"], "bfloat16")

    def test_rejects_missing_required_vlm_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "关键权重"):
            extractor._select_vlm_tensors(
                {"weight_map": {"vlm.lm_head.weight": "model.safetensors"}}
            )


if __name__ == "__main__":
    unittest.main()
