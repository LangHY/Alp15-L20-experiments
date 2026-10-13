"""阶段3固定层12/24确认入口；复用冻结阶段2算法，不导出权重。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import alp15_whitened_pilot as engine


CONFIRMATION_LAYERS = (12, 24)
ENGINE_SHA256 = "fa2e18bf86a768e724ec08536db1a0e6a9ed88125fe199fcebf04317ef86b007"
BASE_SHA256 = "c2975113b04a0a5e295ab14f0ba09db4152568fdb751ec957b45c87b3693ad69"


def main():
    # 只预读 output，其余参数由冻结引擎原有 CLI 严格解析。
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    if "--help" in sys.argv[1:]:
        engine.main()
        return
    args, _ = parser.parse_known_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"确认实验 output 必须尚不存在: {output}")
    harness_path = Path(__file__).resolve()
    engine_path = Path(engine.__file__).resolve()
    base_path = Path(engine.base.__file__).resolve()
    harness_sha = engine.base.sha256_file(harness_path)
    if engine.base.sha256_file(engine_path) != ENGINE_SHA256:
        raise ValueError("阶段2引擎哈希与冻结版本不符")
    if engine.base.sha256_file(base_path) != BASE_SHA256:
        raise ValueError("阶段1基础模块哈希与冻结版本不符")
    if CONFIRMATION_LAYERS != (12, 24):
        raise ValueError("确认入口只能执行预注册层12/24")
    # 只替换内存中的固定层配置；rank、split、seed 和正则保持引擎原值。
    engine.LAYERS = CONFIRMATION_LAYERS
    try:
        engine.main()
    finally:
        # 引擎自行创建新 output 后再保存源码，不抢先建立目录破坏其门禁。
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            code_dir = output / "code"
            code_dir.mkdir(exist_ok=True)
            code_hashes = {}
            for source in (harness_path, engine_path, base_path):
                destination = code_dir / source.name
                shutil.copy2(source, destination)
                code_hashes[str(destination.relative_to(output))] = engine.base.sha256_file(destination)
            manifest = engine.base._json(manifest_path)
            manifest.update({"phase": 3, "phase_id": "ALP15-WHITENED-CONFIRM-L12-L24",
                "entrypoint": str(harness_path), "entrypoint_code_sha256": harness_sha,
                "engine_code_sha256": ENGINE_SHA256, "base_code_sha256": BASE_SHA256,
                "confirmation_layers": list(CONFIRMATION_LAYERS), "copied_code_sha256": code_hashes})
            manifest.setdefault("artifact_sha256", {}).update(code_hashes)
            engine.base._write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
