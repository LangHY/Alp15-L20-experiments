# Third-party source snapshots

This archive preserves original source bytes and headers. It does not grant a new
blanket license over upstream projects, datasets, checkpoints, or model outputs.
The author has not selected a blanket license for the original experiment scripts.

| Upstream | Frozen source | License copy |
| --- | --- | --- |
| NVIDIA Alpamayo 1.5 | `NVlabs/alpamayo1.5@36aeb4c5938cbc2eb2aed33b22434773da4ab639` | `third_party/licenses/alpamayo/LICENSE` (Apache-2.0) |
| FlashDrive | `z-lab/flashdrive@26265174f0413db77c8a16d99993dbeecacf458f` | `third_party/licenses/flashdrive/LICENSE` (MIT) |
| LightVLA | `LiAutoAD/LightVLA@a4680fda5ffe73029190ac97328aa34b0e87a45a` | `third_party/licenses/lightvla/LICENSE` (MIT) |
| vLLM | `vllm-project/vllm@v0.16.0` | `third_party/licenses/vllm/LICENSE` (Apache-2.0) |

Upstream pinned license URLs:

- https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/LICENSE
- https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/LICENSE
- https://github.com/LiAutoAD/LightVLA/blob/a4680fda5ffe73029190ac97328aa34b0e87a45a/LICENSE
- https://github.com/vllm-project/vllm/blob/v0.16.0/LICENSE

`archive/SOURCE_INDEX.json` records every local source origin and its SHA-256.
`archive/framework_sources/` preserves the original research source/commit maps.
The framework snapshots are incomplete research references, not supported runtime
packages; they are not added to the experiment environment's import path.
In particular, `flashdrive_init.py` was saved from upstream `flashdrive/__init__.py`.

Dataset and checkpoint licenses remain separate. No weights, raw videos, USDZ,
tensor caches, prediction outputs, or benchmark result payloads are included.
Small filename/checksum/clip selection manifests are included only because the
download scripts require their exact hashes; access to the actual data still uses
the official gated repository and the user's own authorization.
