#!/usr/bin/env bash
# 生成 Alpamayo 1.5 离线传输包。默认仅打印计划；必须显式传入 --execute 才会联网或写入资产。
umask 077
set -Eeuo pipefail

readonly SOURCE_REPO="https://github.com/NVlabs/alpamayo1.5.git"
readonly SOURCE_COMMIT="36aeb4c5938cbc2eb2aed33b22434773da4ab639"
readonly MODEL_REPO="nvidia/Alpamayo-1.5-10B"
readonly MODEL_REV="7aba8293c09993f2e125c6819df05d7fa3e873ea"
readonly MODELSCOPE_MODEL_REPO="nv-community/Alpamayo-1.5-10B"
readonly MODELSCOPE_MODEL_REV="d26524f2d3bd005149d7f23e2af7c3ea10123df3"
readonly MODEL_CONFIG_SHA256="824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e"
readonly MODEL_CONFIG_SIZE="3058"
readonly MODEL_INDEX_SHA256="b899e51816e15a96edc7cd57a01cd36c5e39fe1bcbf1537c4ac914f0cad43a0d"
readonly MODEL_INDEX_SIZE="104778"
readonly DATASET_REPO="nvidia/PhysicalAI-Autonomous-Vehicles"
readonly DATASET_REV="33f9bf447ed3bcb7d545ce13f4226f824214fafb"
readonly COSMOS_REPO="nvidia/Cosmos-Reason2-8B"
readonly COSMOS_REV="a9fae2cf89dc64db96b12860417f0eb403013bb9"
readonly QWEN_REPO="Qwen/Qwen3-VL-2B-Instruct"
readonly QWEN_REV="89644892e4d85e24eaac8bacfd4f463576704203"

readonly CLIP_ID="030c760c-ae38-49aa-9ad8-f5650a545d26"
readonly T0_US="5100000"

BUNDLE_ROOT="$PWD/alpamayo15-offline-bundle"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SOURCE="hf"
DATASET_SOURCE="hf"
CLIP_CACHE_DIR=""
EXECUTE=0
WITH_CLIP=0
PREPARE_LINUX_AMD64_DEPS=0
TEMP_DIRS=()
NEW_TEMP_DIR=""

cleanup_temp_dirs() {
  local directory
  for directory in "${TEMP_DIRS[@]:-}"; do
    [[ -n "$directory" && -d "$directory" ]] && rm -rf -- "$directory"
  done
  return 0
}
trap cleanup_temp_dirs EXIT

make_temp_dir() {
  local label="$1"
  NEW_TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/alp15-${label}.XXXXXX")" || fail "无法创建 bundle 外临时目录"
  TEMP_DIRS+=("$NEW_TEMP_DIR")
}

usage() {
  cat <<'EOF'
用法：
  bash prepare_alpamayo15_offline_bundle.sh [--bundle-root PATH] [--model-source hf|modelscope]
      [--dataset-source hf|modelscope] [--clip-cache-dir PATH]
      [--with-clip] [--prepare-linux-amd64-deps] [--execute]

默认 dry-run：只进行本机架构、磁盘和工具预检并输出计划；不会联网、下载、安装、写入
bundle 或创建虚拟环境。只有 --execute 才执行下载与写入。

--model-source hf    使用已冻结的 Hugging Face 模型仓库（默认）。认证仅使用本机已有的
                     Hugging Face 登录状态或环境，不接受 token 命令行参数，也不会输出 token。
--model-source modelscope
                     使用 Sol R1 核验的 ModelScope 传输源；仅接受固定 repo/revision 和
                     LICENSE、README、config、index、五个 checkpoint 分片。
--dataset-source hf  使用 physical_ai_av 官方 API 获取冻结数据集（默认）。
--dataset-source modelscope
                     当前拒绝执行：已知 nv-community/PhysicalAI-Autonomous-Vehicles 只含
                     calibration，缺少 metadata、egomotion 和四路 camera chunk，不能替代。
--with-clip          在模型资产后调用官方 physical_ai_av==0.2.0 API 预取冻结的一个 clip。
--clip-cache-dir PATH
                     指定持久 HF 下载缓存以复用已有文件；默认在 bundle 同级的
                     <bundle-name>-clip-hf-cache，失败时保留，不纳入离线包。
--prepare-linux-amd64-deps
                     仅允许 Linux/x86_64 主机准备锁定 uv cache/wheels；构建 venv 位于
                     bundle 外临时目录且结束即清理，不是可搬运交付物。
EOF
}

fail() {
  printf '错误：%s\n' "$*" >&2
  exit 2
}

note() { printf '%s\n' "$*"; }

quote_cmd() {
  local part
  printf '  '
  for part in "$@"; do printf '%q ' "$part"; done
  printf '\n'
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "缺少工具：$1"
}

while (($#)); do
  case "$1" in
    --bundle-root)
      (($# >= 2)) || fail "--bundle-root 需要路径"
      BUNDLE_ROOT="$2"
      shift 2
      ;;
    --model-source)
      (($# >= 2)) || fail "--model-source 需要来源名称"
      MODEL_SOURCE="$2"
      shift 2
      ;;
    --dataset-source)
      (($# >= 2)) || fail "--dataset-source 需要来源名称"
      DATASET_SOURCE="$2"
      shift 2
      ;;
    --clip-cache-dir)
      (($# >= 2)) || fail "--clip-cache-dir 需要路径"
      CLIP_CACHE_DIR="$2"
      shift 2
      ;;
    --with-clip) WITH_CLIP=1; shift ;;
    --prepare-linux-amd64-deps) PREPARE_LINUX_AMD64_DEPS=1; shift ;;
    --execute) EXECUTE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *token*|*TOKEN*) fail "不接受 token 参数；请使用本机已有认证状态" ;;
    *) fail "未知参数：$1（使用 --help 查看用法）" ;;
  esac
done

case "$MODEL_SOURCE" in hf|modelscope) ;; *) fail "--model-source 仅支持 hf 或 modelscope" ;; esac
case "$DATASET_SOURCE" in hf|modelscope) ;; *) fail "--dataset-source 仅支持 hf 或 modelscope" ;; esac

HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"
BUNDLE_PARENT="$(dirname "$BUNDLE_ROOT")"

note "== Alpamayo 1.5 离线包预检 =="
note "模式：$([[ "$EXECUTE" -eq 1 ]] && printf 'EXECUTE（将联网/写入）' || printf 'DRY-RUN（不联网/不写入）')"
note "主机：${HOST_OS}/${HOST_ARCH}"
note "目标包：$BUNDLE_ROOT"
note "冻结源码：$SOURCE_REPO @ $SOURCE_COMMIT"
note "模型传输来源：${MODEL_SOURCE}；权威模型：${MODEL_REPO} @ ${MODEL_REV}"
note "冻结数据：${DATASET_REPO} @ ${DATASET_REV}；clip=${CLIP_ID}；t0_us=${T0_US}"

[[ -d "$BUNDLE_PARENT" ]] || fail "目标父目录不存在：$BUNDLE_PARENT"
if (( WITH_CLIP )); then
  require_command python3
  if [[ -z "$CLIP_CACHE_DIR" ]]; then
    CLIP_CACHE_DIR="$(cd "$BUNDLE_PARENT" && pwd -P)/$(basename "$BUNDLE_ROOT")-clip-hf-cache"
  fi
  python3 -B - "$BUNDLE_ROOT" "$CLIP_CACHE_DIR" <<'PY' || fail "clip cache 必须位于 bundle 外，且不能包含 bundle"
import pathlib, sys
bundle, cache = (pathlib.Path(value).resolve() for value in sys.argv[1:])
if bundle == cache or bundle in cache.parents or cache in bundle.parents:
    raise SystemExit(1)
PY
  note "clip 持久缓存：$CLIP_CACHE_DIR"
fi
if command -v df >/dev/null 2>&1; then
  df -Pk "$BUNDLE_PARENT"
else
  note "警告：未找到 df，无法报告磁盘可用空间"
fi

for tool in git python3; do
  if command -v "$tool" >/dev/null 2>&1; then
    note "工具：$tool -> $(command -v "$tool")"
  else
    note "工具缺失：$tool"
  fi
done
if command -v docker >/dev/null 2>&1; then
  if docker image inspect ubuntu:22.04 --format '{{.Os}}/{{.Architecture}}' >/dev/null 2>&1; then
    note "Docker Ubuntu 22.04 镜像架构：$(docker image inspect ubuntu:22.04 --format '{{.Os}}/{{.Architecture}}')"
  else
    note "Docker Ubuntu 22.04 镜像架构：未核验（镜像不存在或 Docker daemon 未运行）"
  fi
else
  note "Docker：未找到；Ubuntu 22.04 镜像架构未核验"
fi

if [[ "$DATASET_SOURCE" == "modelscope" ]]; then
  fail "dataset ModelScope 当前不可用：nv-community/PhysicalAI-Autonomous-Vehicles 仅含 calibration，缺少 metadata、egomotion 和四路 camera chunk；未执行任何下载"
fi

if (( PREPARE_LINUX_AMD64_DEPS )) && [[ "$HOST_OS" != "Linux" || ( "$HOST_ARCH" != "x86_64" && "$HOST_ARCH" != "amd64" ) ]]; then
  fail "--prepare-linux-amd64-deps 仅能在 Linux/x86_64 执行；当前 ${HOST_OS}/${HOST_ARCH} 不得产出可交付 venv"
fi

if (( EXECUTE == 0 )); then
  note ""
  note "计划（未执行）："
  quote_cmd git clone "$SOURCE_REPO" "$BUNDLE_ROOT/source/alpamayo1.5"
  quote_cmd git -C "$BUNDLE_ROOT/source/alpamayo1.5" checkout --detach "$SOURCE_COMMIT"
  if [[ "$MODEL_SOURCE" == "hf" ]]; then
    quote_cmd hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir '<bundle外临时 staging>' "[仅 config、index、五个模型分片；验证后复制入 bundle]"
  else
    quote_cmd python3 -c 'from modelscope import snapshot_download; snapshot_download(...)' "[bundle外 staging；repo=${MODELSCOPE_MODEL_REPO} revision=${MODELSCOPE_MODEL_REV}；验证后复制入 bundle]"
  fi
  quote_cmd hf download "$COSMOS_REPO" --revision "$COSMOS_REV" --local-dir '<bundle外临时 staging>' "[仅九个 processor/config/tokenizer 文件；校验后复制入 bundle]"
  quote_cmd hf download "$QWEN_REPO" --revision "$QWEN_REV" --local-dir '<bundle外临时 staging>' "[仅九个 processor/config/tokenizer 文件；校验后复制入 bundle]"
  if (( WITH_CLIP )); then
    quote_cmd python3 "$SCRIPT_DIR/prefetch_physical_ai_clip.py" --cache-dir "$CLIP_CACHE_DIR" --output-dir "$BUNDLE_ROOT/datasets/physical-ai-av" --execute
  fi
  if (( PREPARE_LINUX_AMD64_DEPS )); then
    quote_cmd env "UV_CACHE_DIR=$BUNDLE_ROOT/dependencies/linux-amd64/uv-cache" 'UV_PROJECT_ENVIRONMENT=<bundle外临时 venv>' uv sync --frozen
    quote_cmd env "UV_CACHE_DIR=$BUNDLE_ROOT/dependencies/linux-amd64/uv-cache" 'UV_PROJECT_ENVIRONMENT=<另一个干净的bundle外临时venv>' uv sync --frozen --offline
  fi
  note "dry-run 完成：未创建 bundle，未访问网络。"
  exit 0
fi

require_command git
require_command python3
require_command hf  # Cosmos/Qwen 的冻结非权重配置仍从 Hugging Face 获取。
if [[ "$MODEL_SOURCE" == "modelscope" ]]; then
  python3 -B -c 'from modelscope import snapshot_download' >/dev/null 2>&1 || fail "缺少可用的 ModelScope SDK snapshot_download"
fi
[[ ! -e "$BUNDLE_ROOT" ]] || fail "拒绝覆盖已有 bundle：$BUNDLE_ROOT（请使用新的空路径）"

mkdir -p "$BUNDLE_ROOT"/{source,models,datasets,dependencies/configs,dependencies/linux-amd64,manifests,logs}

record_source() {
  local name="$1" repo="$2" revision="$3" directory="$4"
  python3 -B - "$BUNDLE_ROOT/manifests/${name}-source.json" "$repo" "$revision" "$directory" <<'PY'
import hashlib, json, os, sys
output, repo, revision, directory = sys.argv[1:]
records = []
for root, dirs, files in os.walk(directory):
    dirs[:] = sorted(d for d in dirs if d != '.cache')
    for filename in sorted(files):
        path = os.path.join(root, filename)
        rel = os.path.relpath(path, directory)
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        records.append({'path': rel, 'size': os.path.getsize(path), 'sha256': digest.hexdigest()})
with open(output, 'x', encoding='utf-8') as handle:
    json.dump({'repo': repo, 'revision': revision, 'files': records}, handle, indent=2, sort_keys=True)
    handle.write('\n')
PY
}

record_model_source() {
  local transport_source="$1" transport_revision="$2"
  python3 -B - "$BUNDLE_ROOT/manifests/model-source.json" "$transport_source" "$transport_revision" "$BUNDLE_ROOT/models/Alpamayo-1.5-10B" <<'PY'
import hashlib, json, os, sys
output, transport_source, transport_revision, directory = sys.argv[1:]
records = []
for root, dirs, files in os.walk(directory):
    dirs[:] = sorted(d for d in dirs if d != '.cache')
    for filename in sorted(files):
        path = os.path.join(root, filename)
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        records.append({'path': os.path.relpath(path, directory), 'size': os.path.getsize(path), 'sha256': digest.hexdigest()})
payload = {
    'authority_source': 'huggingface:nvidia/Alpamayo-1.5-10B',
    'authority_revision': '7aba8293c09993f2e125c6819df05d7fa3e873ea',
    'transport_source': transport_source,
    'transport_revision': transport_revision,
    'files': records,
}
with open(output, 'x', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write('\n')
PY
}

readonly -a CONFIG_ALLOW_PATTERNS=(
  "config.json" "generation_config.json" "preprocessor_config.json"
  "video_preprocessor_config.json" "tokenizer.json" "tokenizer_config.json"
  "chat_template.json" "merges.txt" "vocab.json"
)
readonly -a MODEL_ALLOW_PATTERNS=(
  "model-00001-of-00005.safetensors" "model-00002-of-00005.safetensors"
  "model-00003-of-00005.safetensors" "model-00004-of-00005.safetensors"
  "model-00005-of-00005.safetensors" "model.safetensors.index.json" "config.json"
)

verify_model_payload() {
  local directory="$1" source_kind="$2"
  python3 -B - "$directory" "$source_kind" <<'PY'
import hashlib, json, os, sys

directory, source_kind = sys.argv[1:]
shards = {
    'model-00001-of-00005.safetensors': (4928204944, '537259bb56815f9dfdcb028d5606e84dff5ab0c8e0e559782d63d91b6671d15c'),
    'model-00002-of-00005.safetensors': (4915963032, '18841e5049b7836c16c034123ae6d40b2ff0414f9f77008bc7ca72490302a47b'),
    'model-00003-of-00005.safetensors': (4983071160, 'e8953b27fe827a60604a006b07ab1216fe766375f9926d1c3bd5ecee84ecef78'),
    'model-00004-of-00005.safetensors': (4980341192, '604f0c0f19986f363a23a1e0b5736eedadd231ca285beab28d5db968a1d37602'),
    'model-00005-of-00005.safetensors': (2349614196, '9d889c09634e5a21b4c957957ede2cdab4418c268ee1d870c1f942d04242adf6'),
}
fixed = {
    'config.json': (3058, '824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e'),
    'model.safetensors.index.json': (104778, 'b899e51816e15a96edc7cd57a01cd36c5e39fe1bcbf1537c4ac914f0cad43a0d'),
}
if source_kind == 'modelscope':
    fixed.update({
        'LICENSE': (None, '2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df'),
        'README.md': (None, 'ab3a10f3abd36624b55d3f511ca6ac957a2a18a922ca6a27c4b0844b5f925240'),
    })
allowed = set(shards) | set(fixed)
def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()
for name, (size, expected) in {**shards, **fixed}.items():
    path = os.path.join(directory, name)
    exists = os.path.isfile(path)
    actual_size = os.path.getsize(path) if exists else None
    actual_sha256 = digest(path) if exists else None
    if not exists or (size is not None and actual_size != size) or actual_sha256 != expected:
        raise SystemExit(
            f'模型校验失败：{name}; exists={exists}; '
            f'actual_size={actual_size}; actual_sha256={actual_sha256}; '
            f'expected_size={size}; expected_sha256={expected}'
        )
actual_files = set()
for root, dirs, files in os.walk(directory):
    dirs[:] = [name for name in dirs if name != '.cache']
    for name in files:
        actual_files.add(os.path.relpath(os.path.join(root, name), directory))
if actual_files != allowed:
    raise SystemExit(f'模型 payload 白名单不匹配：缺失={sorted(allowed - actual_files)}；额外={sorted(actual_files - allowed)}')
with open(os.path.join(directory, 'model.safetensors.index.json'), encoding='utf-8') as handle:
    index = json.load(handle)
weight_map = index.get('weight_map')
if not isinstance(weight_map, dict) or not weight_map or set(weight_map.values()) != set(shards):
    raise SystemExit('模型 index weight_map 未恰好引用冻结的五个分片')
PY
}

copy_validated_model_payload() {
  local stage_dir="$1" target_dir="$2" source_kind="$3"
  [[ ! -e "$target_dir" ]] || fail "拒绝覆盖已有最终模型目录：$target_dir"
  mkdir -p "$target_dir"
  python3 -B - "$stage_dir" "$target_dir" "$source_kind" <<'PY'
import shutil, sys
stage, target, source_kind = sys.argv[1:]
names = [
    'config.json', 'model.safetensors.index.json',
    'model-00001-of-00005.safetensors', 'model-00002-of-00005.safetensors',
    'model-00003-of-00005.safetensors', 'model-00004-of-00005.safetensors',
    'model-00005-of-00005.safetensors',
]
if source_kind == 'modelscope':
    names = ['LICENSE', 'README.md', *names]
for name in names:
    shutil.copy2(f'{stage}/{name}', f'{target}/{name}')
PY
  verify_model_payload "$target_dir" "$source_kind"
}

note "克隆冻结官方源码（detached checkout）..."
git clone "$SOURCE_REPO" "$BUNDLE_ROOT/source/alpamayo1.5"
git -C "$BUNDLE_ROOT/source/alpamayo1.5" checkout --detach "$SOURCE_COMMIT"
[[ "$(git -C "$BUNDLE_ROOT/source/alpamayo1.5" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || fail "源码 HEAD 不等于冻结提交"

note "下载 Alpamayo checkpoint（不会传递或记录 token）..."
make_temp_dir "model-stage"
MODEL_STAGE_DIR="$NEW_TEMP_DIR"
MODEL_TARGET_DIR="$BUNDLE_ROOT/models/Alpamayo-1.5-10B"
if [[ "$MODEL_SOURCE" == "hf" ]]; then
  hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$MODEL_STAGE_DIR" --include "${MODEL_ALLOW_PATTERNS[@]}"
  verify_model_payload "$MODEL_STAGE_DIR" "hf"
  copy_validated_model_payload "$MODEL_STAGE_DIR" "$MODEL_TARGET_DIR" "hf"
  record_model_source "huggingface:$MODEL_REPO" "$MODEL_REV"
else
  python3 -B - "$MODEL_STAGE_DIR" "$MODELSCOPE_MODEL_REPO" "$MODELSCOPE_MODEL_REV" <<'PY'
import sys
try:
    from modelscope import snapshot_download
except ImportError as error:
    raise SystemExit(f'缺少 ModelScope SDK：{error}')
local_dir, repo, revision = sys.argv[1:]
allow_patterns = [
    'LICENSE', 'README.md', 'config.json', 'model.safetensors.index.json',
    'model-00001-of-00005.safetensors', 'model-00002-of-00005.safetensors',
    'model-00003-of-00005.safetensors', 'model-00004-of-00005.safetensors',
    'model-00005-of-00005.safetensors',
]
snapshot_download(repo, revision=revision, local_dir=local_dir, allow_patterns=allow_patterns)
unexpected = []
for root, dirs, files in __import__('os').walk(local_dir):
    dirs[:] = [name for name in dirs if name != '.cache']
    for name in files:
        relative = __import__('os').path.relpath(__import__('os').path.join(root, name), local_dir)
        if relative not in allow_patterns:
            unexpected.append(relative)
if unexpected:
    raise SystemExit(f'ModelScope 返回了白名单外文件：{sorted(unexpected)}')
PY
  for rejected in configuration.json .gitattributes; do
    [[ ! -e "$MODEL_STAGE_DIR/$rejected" ]] || fail "ModelScope staging 下载到禁止文件：$rejected"
  done
  verify_model_payload "$MODEL_STAGE_DIR" "modelscope"
  copy_validated_model_payload "$MODEL_STAGE_DIR" "$MODEL_TARGET_DIR" "modelscope"
  record_model_source "modelscope:$MODELSCOPE_MODEL_REPO" "$MODELSCOPE_MODEL_REV"
fi

note "下载 Cosmos/Qwen 的白名单配置；不会下载任何 Cosmos/Qwen 权重..."
copy_config_payload() {
  local stage_dir="$1" target_dir="$2"
  [[ ! -e "$target_dir" ]] || fail "拒绝覆盖已有配置目录：$target_dir"
  python3 -B - "$stage_dir" "$target_dir" "${CONFIG_ALLOW_PATTERNS[@]}" <<'PY'
import os, shutil, sys
stage, target, *names = sys.argv[1:]
for name in names:
    if not os.path.isfile(os.path.join(stage, name)):
        raise SystemExit(f'冻结配置文件缺失：{name}')
os.makedirs(target, exist_ok=False)
for name in names:
    shutil.copy2(os.path.join(stage, name), os.path.join(target, name))
PY
}
make_temp_dir "cosmos-stage"
COSMOS_STAGE_DIR="$NEW_TEMP_DIR"
hf download "$COSMOS_REPO" --revision "$COSMOS_REV" --local-dir "$COSMOS_STAGE_DIR" --include "${CONFIG_ALLOW_PATTERNS[@]}"
copy_config_payload "$COSMOS_STAGE_DIR" "$BUNDLE_ROOT/dependencies/configs/Cosmos-Reason2-8B"
record_source "cosmos-reason2" "$COSMOS_REPO" "$COSMOS_REV" "$BUNDLE_ROOT/dependencies/configs/Cosmos-Reason2-8B"
make_temp_dir "qwen-stage"
QWEN_STAGE_DIR="$NEW_TEMP_DIR"
hf download "$QWEN_REPO" --revision "$QWEN_REV" --local-dir "$QWEN_STAGE_DIR" --include "${CONFIG_ALLOW_PATTERNS[@]}"
copy_config_payload "$QWEN_STAGE_DIR" "$BUNDLE_ROOT/dependencies/configs/Qwen3-VL-2B-Instruct"
record_source "qwen3-vl" "$QWEN_REPO" "$QWEN_REV" "$BUNDLE_ROOT/dependencies/configs/Qwen3-VL-2B-Instruct"

if (( WITH_CLIP )); then
  python3 -B "$SCRIPT_DIR/prefetch_physical_ai_clip.py" --cache-dir "$CLIP_CACHE_DIR" --output-dir "$BUNDLE_ROOT/datasets/physical-ai-av" --execute
fi

if (( PREPARE_LINUX_AMD64_DEPS )); then
  require_command uv
  [[ -f "$BUNDLE_ROOT/source/alpamayo1.5/uv.lock" ]] || fail "官方源码缺少 uv.lock，停止依赖准备"
  make_temp_dir "build-venv"
  BUILD_VENV_DIR="$NEW_TEMP_DIR"
  (
    cd "$BUNDLE_ROOT/source/alpamayo1.5"
    UV_CACHE_DIR="$BUNDLE_ROOT/dependencies/linux-amd64/uv-cache" \
      UV_PROJECT_ENVIRONMENT="$BUILD_VENV_DIR" \
      uv sync --frozen
  )
  make_temp_dir "offline-check-venv"
  OFFLINE_VENV_DIR="$NEW_TEMP_DIR"
  OFFLINE_LOG="$BUNDLE_ROOT/logs/linux-amd64-offline-sync.log"
  (
    cd "$BUNDLE_ROOT/source/alpamayo1.5"
    UV_CACHE_DIR="$BUNDLE_ROOT/dependencies/linux-amd64/uv-cache" \
      UV_PROJECT_ENVIRONMENT="$OFFLINE_VENV_DIR" \
      uv sync --frozen --offline
  ) >"$OFFLINE_LOG" 2>&1 || fail "干净 Linux/amd64 环境离线同步失败；见 $OFFLINE_LOG"
  python3 -B - "$BUNDLE_ROOT/manifests/linux-amd64-offline-validation.json" <<'PY'
import json, platform, sys
record = {
    'os': platform.system(),
    'arch': platform.machine(),
    'command': 'uv sync --frozen --offline',
    'exit_code': 0,
    'log_path': 'logs/linux-amd64-offline-sync.log',
}
with open(sys.argv[1], 'x', encoding='utf-8') as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write('\n')
PY
fi

python3 -B - "$BUNDLE_ROOT" <<'PY'
import hashlib, os, sys
root = os.path.abspath(sys.argv[1])
output = os.path.join(root, 'manifests', 'full-file-manifest.sha256')
rows = []
for current, dirs, files in os.walk(root):
    dirs[:] = sorted(dirs)
    for name in sorted(files):
        path = os.path.join(current, name)
        rel = os.path.relpath(path, root)
        if rel == 'manifests/full-file-manifest.sha256':
            continue
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        rows.append(f'{digest.hexdigest()}  {rel}')
with open(output, 'x', encoding='utf-8') as handle:
    handle.write('\n'.join(sorted(rows)) + '\n')
PY

note "离线包已生成；下一步仅可运行只读校验："
quote_cmd bash "$SCRIPT_DIR/verify_alpamayo15_offline_bundle.sh" --bundle-root "$BUNDLE_ROOT"
