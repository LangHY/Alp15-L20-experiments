"""还原压缩分片的原始索引与数据清单，并校验 SHA-256；不下载任何数据。"""
import base64
import gzip
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "archive/packed_manifests/index.json").read_text())
    for item in manifest:
        target = root / item["path"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(f"拒绝覆盖已有且校验不匹配的文件：{item['path']}")
            print(f"已校验：{item['path']}")
            continue
        encoded = "".join((root / part).read_text().strip() for part in item["parts"])
        data = gzip.decompress(base64.b64decode(encoded, validate=True))
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError(f"分片校验失败：{item['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        print(f"已还原并校验：{item['path']}")


if __name__ == "__main__":
    main()
