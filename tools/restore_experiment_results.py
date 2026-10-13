"""还原公开脱敏结果；支持校验，拒绝覆盖不同内容。"""
import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    bundle = root / "results/packed_evidence"
    index = json.loads(gzip.decompress(base64.b64decode((bundle / "index.b64").read_text(), validate=False)))
    packed = base64.b64decode("".join((bundle / p).read_text().strip() for p in index["parts"]), validate=True)
    if hashlib.sha256(packed).hexdigest() != index["packed_sha256"]:
        raise RuntimeError("结果压缩包 SHA-256 校验失败")
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r:gz") as tar:
        for item in index["entries"]:
            path = Path(item["path"])
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("非法还原路径")
            target = root / "results/restored" / item["origin"] / path
            data = tar.extractfile(item["published_sha256"]).read()
            if hashlib.sha256(data).hexdigest() != item["published_sha256"]:
                raise RuntimeError(f"结果文件校验失败：{path}")
            if target.exists():
                if target.read_bytes() != data:
                    raise RuntimeError(f"拒绝覆盖不同内容：{path}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
    print(f"已还原并校验 {len(index['entries'])} 份来源结果；公开副本有脱敏，原始 SHA 记录在索引。")


if __name__ == "__main__":
    main()
