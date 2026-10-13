#!/usr/bin/env bash
# 离线包只读验收：不联网、不安装、不写入、不加载或运行模型。
umask 077
set -Eeuo pipefail

readonly SOURCE_COMMIT="36aeb4c5938cbc2eb2aed33b22434773da4ab639"
readonly ENTRY_SHA256="dc69646feed09f92defa00a19ae6f2fc2a37a10a70ec35678f87946fbe6fe8e8"
readonly MODEL_REPO="nvidia/Alpamayo-1.5-10B"
readonly MODEL_REV="7aba8293c09993f2e125c6819df05d7fa3e873ea"
readonly MODELSCOPE_MODEL_REPO="nv-community/Alpamayo-1.5-10B"
readonly MODELSCOPE_MODEL_REV="d26524f2d3bd005149d7f23e2af7c3ea10123df3"
readonly MODEL_CONFIG_SHA256="824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e"
readonly MODEL_CONFIG_SIZE="3058"
readonly MODEL_INDEX_SHA256="b899e51816e15a96edc7cd57a01cd36c5e39fe1bcbf1537c4ac914f0cad43a0d"
readonly MODEL_INDEX_SIZE="104778"
readonly COSMOS_REPO="nvidia/Cosmos-Reason2-8B"
readonly COSMOS_REV="a9fae2cf89dc64db96b12860417f0eb403013bb9"
readonly QWEN_REPO="Qwen/Qwen3-VL-2B-Instruct"
readonly QWEN_REV="89644892e4d85e24eaac8bacfd4f463576704203"
readonly DATASET_REPO="nvidia/PhysicalAI-Autonomous-Vehicles"
readonly DATASET_REV="33f9bf447ed3bcb7d545ce13f4226f824214fafb"
readonly CLIP_ID="030c760c-ae38-49aa-9ad8-f5650a545d26"
readonly T0_US="5100000"

BUNDLE_ROOT="$PWD/alpamayo15-offline-bundle"
REQUIRE_RUNTIME=0
while (($#)); do
  case "$1" in
    --bundle-root) (($# >= 2)) || { echo "错误：--bundle-root 需要路径" >&2; exit 2; }; BUNDLE_ROOT="$2"; shift 2 ;;
    --require-runtime) REQUIRE_RUNTIME=1; shift ;;
    --help|-h) echo "用法：bash verify_alpamayo15_offline_bundle.sh [--bundle-root PATH] [--require-runtime]"; exit 0 ;;
    *) echo "错误：未知参数：$1" >&2; exit 2 ;;
  esac
done

failures=0
ok() { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*" >&2; failures=$((failures + 1)); }
sha256_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum -- "$1" | awk '{print $1}'; else shasum -a 256 -- "$1" | awk '{print $1}'; fi; }
check_file() { [[ -f "$1" ]] && ok "$2" || bad "$2：MISSING $1"; }
check_manifest_value() {
  local file="$1" expected_repo="$2" expected_rev="$3" label="$4"
  if [[ ! -f "$file" ]]; then bad "$label 来源 manifest 缺失：$file"; return; fi
  python3 -B - "$file" "$expected_repo" "$expected_rev" "$label" <<'PY' || bad "$label 来源 manifest 字段不匹配"
import json, sys
path, repo, revision, label = sys.argv[1:]
with open(path, encoding='utf-8') as handle:
    payload = json.load(handle)
if payload.get('repo') != repo or payload.get('revision') != revision or not isinstance(payload.get('files'), list):
    raise SystemExit(1)
print(f'PASS  {label} 来源 manifest：repo/revision/files')
PY
}

model_transport=""
check_model_manifest() {
  local file="$1"
  [[ -f "$file" ]] || { bad "模型 snapshot manifest 缺失：$file"; return; }
  model_transport="$(python3 -B - "$file" "$MODEL_REV" "$MODELSCOPE_MODEL_REPO" "$MODELSCOPE_MODEL_REV" <<'PY'
import json, sys
path, authority_revision, ms_repo, ms_revision = sys.argv[1:]
with open(path, encoding='utf-8') as handle:
    payload = json.load(handle)
if payload.get('authority_source') != 'huggingface:nvidia/Alpamayo-1.5-10B' or payload.get('authority_revision') != authority_revision or not isinstance(payload.get('files'), list):
    raise SystemExit(1)
transport = payload.get('transport_source')
revision = payload.get('transport_revision')
if transport == 'huggingface:nvidia/Alpamayo-1.5-10B' and revision == authority_revision:
    print('hf')
elif transport == f'modelscope:{ms_repo}' and revision == ms_revision:
    allowed = {
        'LICENSE', 'README.md', 'config.json', 'model.safetensors.index.json',
        'model-00001-of-00005.safetensors', 'model-00002-of-00005.safetensors',
        'model-00003-of-00005.safetensors', 'model-00004-of-00005.safetensors',
        'model-00005-of-00005.safetensors',
    }
    paths = {item.get('path') for item in payload['files'] if isinstance(item, dict)}
    if paths != allowed:
        raise SystemExit(1)
    print('modelscope')
else:
    raise SystemExit(1)
PY
  )" || { bad "模型 snapshot manifest authority/transport 字段不匹配"; return; }
  ok "模型 snapshot manifest authority/transport：$model_transport"
}

[[ -d "$BUNDLE_ROOT" ]] || { echo "错误：bundle 不存在：$BUNDLE_ROOT" >&2; exit 2; }
command -v git >/dev/null 2>&1 || { echo "错误：缺少 git" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "错误：缺少 python3" >&2; exit 2; }
command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 || { echo "错误：缺少 SHA256 工具" >&2; exit 2; }

echo "== Alpamayo 1.5 离线包只读校验 =="
echo "bundle=$BUNDLE_ROOT"
CODE_DIR="$BUNDLE_ROOT/source/alpamayo1.5"
MODEL_DIR="$BUNDLE_ROOT/models/Alpamayo-1.5-10B"
COSMOS_DIR="$BUNDLE_ROOT/dependencies/configs/Cosmos-Reason2-8B"
QWEN_DIR="$BUNDLE_ROOT/dependencies/configs/Qwen3-VL-2B-Instruct"
DATA_DIR="$BUNDLE_ROOT/datasets/physical-ai-av"
MANIFEST_DIR="$BUNDLE_ROOT/manifests"

if [[ -d "$CODE_DIR/.git" ]]; then
  [[ "$(git -C "$CODE_DIR" rev-parse HEAD 2>/dev/null || true)" == "$SOURCE_COMMIT" ]] && ok "官方源码 HEAD" || bad "官方源码 HEAD 不匹配"
  [[ -z "$(git -C "$CODE_DIR" status --porcelain --ignored=no 2>/dev/null || true)" ]] && ok "官方源码工作区 clean" || bad "官方源码工作区有未提交变更"
else
  bad "官方源码不是 Git 工作树：$CODE_DIR"
fi
ENTRY="$CODE_DIR/src/alpamayo1_5/test_inference.py"
if [[ -f "$ENTRY" && "$(sha256_file "$ENTRY")" == "$ENTRY_SHA256" ]]; then ok "官方入口 test_inference.py SHA256"; else bad "官方入口 test_inference.py 缺失或 SHA256 不匹配"; fi

check_model_manifest "$MANIFEST_DIR/model-source.json"
declare -a MODEL_FILES=(
  "model-00001-of-00005.safetensors:4928204944:537259bb56815f9dfdcb028d5606e84dff5ab0c8e0e559782d63d91b6671d15c"
  "model-00002-of-00005.safetensors:4915963032:18841e5049b7836c16c034123ae6d40b2ff0414f9f77008bc7ca72490302a47b"
  "model-00003-of-00005.safetensors:4983071160:e8953b27fe827a60604a006b07ab1216fe766375f9926d1c3bd5ecee84ecef78"
  "model-00004-of-00005.safetensors:4980341192:604f0c0f19986f363a23a1e0b5736eedadd231ca285beab28d5db968a1d37602"
  "model-00005-of-00005.safetensors:2349614196:9d889c09634e5a21b4c957957ede2cdab4418c268ee1d870c1f942d04242adf6"
)
for item in "${MODEL_FILES[@]}"; do
  name="${item%%:*}"; rest="${item#*:}"; expected_size="${rest%%:*}"; expected="${rest##*:}"; path="$MODEL_DIR/$name"
  if [[ ! -f "$path" ]]; then bad "模型分片缺失：$name"; continue; fi
  actual="$(sha256_file "$path")"; size="$(wc -c < "$path" | tr -d ' ')"
  [[ "$size" == "$expected_size" && "$actual" == "$expected" ]] && ok "模型分片 $name size=$size sha256=$actual" || bad "模型分片 $name size=$size/$expected_size 或 sha256 不匹配"
done
for item in "config.json:$MODEL_CONFIG_SIZE:$MODEL_CONFIG_SHA256" "model.safetensors.index.json:$MODEL_INDEX_SIZE:$MODEL_INDEX_SHA256"; do
  name="${item%%:*}"; rest="${item#*:}"; expected_size="${rest%%:*}"; expected="${rest##*:}"; path="$MODEL_DIR/$name"
  if [[ ! -f "$path" ]]; then bad "模型核心文件缺失：$name"; continue; fi
  actual="$(sha256_file "$path")"; size="$(wc -c < "$path" | tr -d ' ')"
  [[ "$size" == "$expected_size" && "$actual" == "$expected" ]] && ok "模型核心文件 $name size=$size sha256=$actual" || bad "模型核心文件 $name size=$size/$expected_size 或 sha256 不匹配"
done
if [[ -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
python3 -B - "$MODEL_DIR/model.safetensors.index.json" <<'PY' || bad "模型 index weight_map 未恰好引用冻结五片"
import json, sys
expected = {f'model-{number:05d}-of-00005.safetensors' for number in range(1, 6)}
with open(sys.argv[1], encoding='utf-8') as handle:
    weight_map = json.load(handle).get('weight_map')
if not isinstance(weight_map, dict) or not weight_map or set(weight_map.values()) != expected:
    raise SystemExit(1)
PY
fi
EXTRA_SAFETENSORS="$(find "$MODEL_DIR" -type f -name '*.safetensors' ! -name 'model-00001-of-00005.safetensors' ! -name 'model-00002-of-00005.safetensors' ! -name 'model-00003-of-00005.safetensors' ! -name 'model-00004-of-00005.safetensors' ! -name 'model-00005-of-00005.safetensors' -print 2>/dev/null || true)"
[[ -z "$EXTRA_SAFETENSORS" ]] && ok "未发现额外 safetensors" || bad "发现额外 safetensors：$EXTRA_SAFETENSORS"
if [[ "$model_transport" == "modelscope" ]]; then
  declare -a MODELSCOPE_CORE_FILES=(
    "LICENSE:2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df:0"
    "README.md:ab3a10f3abd36624b55d3f511ca6ac957a2a18a922ca6a27c4b0844b5f925240:0"
    "config.json:824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e:3058"
    "model.safetensors.index.json:b899e51816e15a96edc7cd57a01cd36c5e39fe1bcbf1537c4ac914f0cad43a0d:104778"
  )
  for item in "${MODELSCOPE_CORE_FILES[@]}"; do
    name="${item%%:*}"; rest="${item#*:}"; expected="${rest%%:*}"; expected_size="${rest##*:}"; path="$MODEL_DIR/$name"
    if [[ ! -f "$path" ]]; then bad "ModelScope 核心文件缺失：$name"; continue; fi
    actual="$(sha256_file "$path")"; size="$(wc -c < "$path" | tr -d ' ')"
    if [[ "$actual" == "$expected" && ( "$expected_size" == "0" || "$size" == "$expected_size" ) ]]; then ok "ModelScope 核心文件 $name size=$size sha256=$actual"; else bad "ModelScope 核心文件 $name 校验不匹配"; fi
  done
  for rejected in configuration.json .gitattributes; do [[ ! -e "$MODEL_DIR/$rejected" ]] && ok "ModelScope 未接收禁止文件 $rejected" || bad "ModelScope 接收到禁止文件 $rejected"; done
fi

declare -a CONFIG_FILES=(
  "config.json" "generation_config.json" "preprocessor_config.json" "video_preprocessor_config.json"
  "tokenizer.json" "tokenizer_config.json" "chat_template.json" "merges.txt" "vocab.json"
)
CONFIG_TARGETS=("Cosmos:$COSMOS_DIR" "Qwen:$QWEN_DIR")
for directory_label in "${CONFIG_TARGETS[@]}"; do
  label="${directory_label%%:*}"; directory="${directory_label#*:}"
  for name in "${CONFIG_FILES[@]}"; do check_file "$directory/$name" "$label 允许配置 $name"; done
done
check_manifest_value "$MANIFEST_DIR/cosmos-reason2-source.json" "$COSMOS_REPO" "$COSMOS_REV" "Cosmos snapshot"
check_manifest_value "$MANIFEST_DIR/qwen3-vl-source.json" "$QWEN_REPO" "$QWEN_REV" "Qwen snapshot"

DATASET_MANIFEST="$DATA_DIR/dataset-manifest.json"
if [[ -f "$DATASET_MANIFEST" ]]; then
  python3 -B - "$DATASET_MANIFEST" "$DATASET_REPO" "$DATASET_REV" "$CLIP_ID" "$T0_US" <<'PY' || bad "dataset manifest 字段、精确 feature/metadata 集合或文件校验失败"
import hashlib, json, os, re, sys
from pathlib import Path
path, repo, revision, clip_id, t0_us = sys.argv[1:]
root = Path(path).parent.resolve()
repo_dir = 'datasets--nvidia--PhysicalAI-Autonomous-Vehicles'
snapshot_prefix = f'{repo_dir}/snapshots/{revision}'
snapshot_root = root / snapshot_prefix
blobs_root = root / repo_dir / 'blobs'
with open(path, encoding='utf-8') as handle:
    payload = json.load(handle)
required = {
    'dataset_repo': repo,
    'dataset_revision': revision,
    'physical_ai_av_version': '0.2.0',
    'physical_ai_av_source_commit': '59d578b53ef186ffc3f1de1d1b2598717a982d64',
    'clip_id': clip_id,
    't0_us': int(t0_us),
}
if any(payload.get(key) != value for key, value in required.items()):
    raise SystemExit(1)
chunk_id = payload.get('chunk_id')
if isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or chunk_id < 0:
    raise SystemExit(1)
expected_features = {
    'LABELS.EGOMOTION': f'labels/egomotion/egomotion.chunk_{chunk_id:04d}.zip',
    'CAMERA.CAMERA_CROSS_LEFT_120FOV': f'camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_{chunk_id:04d}.zip',
    'CAMERA.CAMERA_FRONT_WIDE_120FOV': f'camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{chunk_id:04d}.zip',
    'CAMERA.CAMERA_CROSS_RIGHT_120FOV': f'camera/camera_cross_right_120fov/camera_cross_right_120fov.chunk_{chunk_id:04d}.zip',
    'CAMERA.CAMERA_FRONT_TELE_30FOV': f'camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_{chunk_id:04d}.zip',
}
features = payload.get('features')
if not isinstance(features, list) or len(features) != 5 or {item.get('feature') for item in features if isinstance(item, dict)} != set(expected_features):
    raise SystemExit(1)
metadata_paths = {
    f'{snapshot_prefix}/features.csv',
    f'{snapshot_prefix}/clip_index.parquet',
    f'{snapshot_prefix}/metadata/feature_presence.parquet',
    f'{snapshot_prefix}/metadata/data_collection.parquet',
}
metadata = payload.get('metadata')
if not isinstance(metadata, list) or len(metadata) != 4 or {item.get('path') for item in metadata if isinstance(item, dict)} != metadata_paths:
    raise SystemExit(1)
expected_paths = {f'{snapshot_prefix}/{name}' for name in expected_features.values()} | metadata_paths
seen_paths, seen_blobs = set(), set()
def check_record(item, require_feature):
    required_keys = {'path', 'size', 'sha256'} | ({'feature', 'chunk'} if require_feature else set())
    if not isinstance(item, dict) or set(item) != required_keys:
        raise SystemExit(1)
    if require_feature and item['chunk'] != chunk_id:
        raise SystemExit(1)
    relative = item['path']
    if not isinstance(relative, str) or relative not in expected_paths or relative in seen_paths:
        raise SystemExit(1)
    if require_feature and relative != f'{snapshot_prefix}/{expected_features[item["feature"]]}':
        raise SystemExit(1)
    seen_paths.add(relative)
    candidate = root / relative
    if not candidate.is_symlink():
        raise SystemExit(1)
    link_target = os.readlink(candidate)
    if os.path.isabs(link_target):
        raise SystemExit(1)
    blob = candidate.resolve()
    if (blob.parent != blobs_root or blob.is_symlink()
            or not re.fullmatch(r'[0-9a-f]{40}(?:[0-9a-f]{24})?', blob.name)
            or link_target != os.path.relpath(blob, candidate.parent)
            or not blob.is_file()):
        raise SystemExit(1)
    seen_blobs.add(blob.name)
    if (isinstance(item['size'], bool) or not isinstance(item['size'], int)
            or item['size'] < 0 or not isinstance(item['sha256'], str)
            or not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])):
        raise SystemExit(1)
    actual_size = blob.stat().st_size
    digest = hashlib.sha256()
    git_digest = hashlib.sha1() if len(blob.name) == 40 else None
    if git_digest is not None:
        git_digest.update(f'blob {actual_size}\0'.encode('ascii'))
    bytes_read = 0
    with open(candidate, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
            if git_digest is not None:
                git_digest.update(block)
            bytes_read += len(block)
    actual_identity = git_digest.hexdigest() if git_digest is not None else digest.hexdigest()
    if (bytes_read != actual_size or candidate.stat().st_size != actual_size
            or actual_size != item['size'] or digest.hexdigest() != item['sha256']
            or actual_identity != blob.name):
        raise SystemExit(1)
for item in features:
    check_record(item, True)
for item in metadata:
    check_record(item, False)
if seen_paths != expected_paths:
    raise SystemExit(1)
actual_snapshot = set()
for current, _, files in os.walk(snapshot_root):
    for name in files:
        actual_snapshot.add((Path(current) / name).relative_to(root).as_posix())
if actual_snapshot != expected_paths or {entry.name for entry in blobs_root.iterdir()} != seen_blobs:
    raise SystemExit(1)
print('PASS  dataset manifest：固定 revision/clip/chunk/九项 snapshot 链接、HF blob 内容摘要、size/sha256')
PY
else
  bad "dataset manifest 缺失：$DATASET_MANIFEST"
fi

check_file "$CODE_DIR/uv.lock" "官方源码 uv.lock"
if (( REQUIRE_RUNTIME )); then
  if [[ -d "$BUNDLE_ROOT/dependencies/linux-amd64/uv-cache" ]] &&
     [[ -n "$(find "$BUNDLE_ROOT/dependencies/linux-amd64/uv-cache" -type f -print -quit)" ]]; then
    ok "Linux/amd64 uv cache 非空"
  else
    bad "Linux/amd64 uv cache 缺失或为空"
  fi
  RUNTIME_EVIDENCE="$MANIFEST_DIR/linux-amd64-offline-validation.json"
  if [[ -f "$RUNTIME_EVIDENCE" ]]; then
    python3 -B - "$RUNTIME_EVIDENCE" "$BUNDLE_ROOT" <<'PY' || bad "Linux/amd64 离线同步验证记录字段无效"
import json, os, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    record = json.load(handle)
if (record.get('os') != 'Linux' or record.get('arch') not in {'x86_64', 'amd64'}
        or record.get('command') != 'uv sync --frozen --offline'
        or record.get('exit_code') != 0
        or record.get('log_path') != 'logs/linux-amd64-offline-sync.log'
        or not os.path.isfile(os.path.join(sys.argv[2], record['log_path']))):
    raise SystemExit(1)
print('PASS  干净 Linux/amd64 环境离线同步验证记录字段')
PY
  else
    bad "干净 Linux/amd64 环境离线同步验证记录缺失"
  fi
fi
FULL_MANIFEST="$MANIFEST_DIR/full-file-manifest.sha256"
if [[ -f "$FULL_MANIFEST" ]]; then
  python3 -B - "$BUNDLE_ROOT" "$FULL_MANIFEST" <<'PY' || bad "全文件 manifest 缺失、格式错误或 SHA256 不匹配"
import hashlib, os, sys
root, manifest = map(os.path.abspath, sys.argv[1:])
seen = set()
with open(manifest, encoding='utf-8') as handle:
    for line in handle:
        digest, separator, rel = line.rstrip('\n').partition('  ')
        path = os.path.abspath(os.path.join(root, rel))
        if len(digest) != 64 or not separator or not path.startswith(root + os.sep) or not os.path.isfile(path):
            raise SystemExit(1)
        actual = hashlib.sha256()
        with open(path, 'rb') as binary:
            for block in iter(lambda: binary.read(1024 * 1024), b''):
                actual.update(block)
        if actual.hexdigest() != digest:
            raise SystemExit(1)
        seen.add(rel)
for current, _, files in os.walk(root):
    for name in files:
        path = os.path.join(current, name)
        rel = os.path.relpath(path, root)
        if rel != 'manifests/full-file-manifest.sha256' and rel not in seen:
            raise SystemExit(1)
print('PASS  全文件 manifest SHA256')
PY
else
  bad "全文件 manifest 缺失：$FULL_MANIFEST"
fi

SENSITIVE_NAMES="$(find "$BUNDLE_ROOT" -type f \( -name '.env' -o -name '.netrc' -o -name 'id_rsa' -o -name 'id_ed25519' -o -name 'token' -o -name 'tokens' -o -name 'stored_tokens' -o -name '.git-credentials' -o -name '.pypirc' -o -name '*_history' -o -name '*.pem' -o -name '*.key' -o -name '*.p12' -o -path '*/huggingface/token' -o -path '*/huggingface/stored_tokens' \) -print)"
if [[ -n "$SENSITIVE_NAMES" ]]; then bad "发现疑似敏感文件名：$SENSITIVE_NAMES"; else ok "敏感文件名扫描"; fi

if (( failures )); then
  printf '校验失败：%d 项。未运行模型。\n' "$failures" >&2
  exit 1
fi
if (( REQUIRE_RUNTIME )); then
  echo "资产与依赖证据文件校验通过；仍须在 L20 实机断网运行推理，不能据此宣称模型已跑通。"
else
  echo "资产文件校验通过；未核验 Linux/amd64 离线依赖与 L20 推理，不能据此宣称完整可运行。"
fi
