#!/usr/bin/env bash
# Alpamayo 1.5 L20 候选机只读盘点：仅写入本次审计日志目录，不更改服务器配置或软件。
umask 077
set -Eeuo pipefail

LOG_DIR="${1:-"$PWD/alp15_l20_audit_$(date +%Y%m%dT%H%M%S%z)"}"
LOG_FILE="$LOG_DIR/readiness.log"
if [[ -e "$LOG_FILE" ]]; then
  printf '拒绝执行：目标日志已存在，绝不覆盖：%s\n' "$LOG_FILE" >&2
  exit 2
fi
mkdir -p -- "$LOG_DIR"
if ! (set -C; : > "$LOG_FILE"); then
  printf '拒绝执行：目标日志已存在或无法以不覆盖方式创建：%s\n' "$LOG_FILE" >&2
  exit 2
fi

log() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

section() {
  log ""
  log "## $1"
}

run_if_available() {
  local label="$1"
  shift
  section "$label"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" >>"$LOG_FILE" 2>&1 || log "[命令退出码 $?：已记录，继续盘点]"
  else
    log "[未找到命令：$1]"
  fi
}

describe_dir() {
  local label="$1"
  local dir="$2"
  section "$label"
  if [[ -z "$dir" ]]; then
    log "[未设置]"
  elif [[ ! -e "$dir" ]]; then
    log "[路径不存在] $dir"
  elif [[ ! -d "$dir" ]]; then
    log "[不是目录] $dir"
  else
    log "[目录] $dir"
  fi
}

redact_remote_url() {
  # 不将 URL 中可能存在的认证信息写入日志。
  sed -E \
    -e 's#(https?://)[^/@[:space:]]+@#\1<redacted>@#g' \
    -e 's#(ssh://)[^/@[:space:]]+@#\1<redacted>@#g' \
    -e 's#[?#].*$#<query-or-fragment-redacted>#'
}

audit_code_dir() {
  local dir="$1"
  describe_dir "CODE_DIR" "$dir"
  [[ -n "$dir" && -d "$dir" ]] || return 0

  section "CODE_DIR Git 信息（只读）"
  if command -v git >/dev/null 2>&1 && git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "remote 名称："
    git -C "$dir" remote >>"$LOG_FILE" 2>&1 || true
    while IFS= read -r remote_name; do
      [[ -n "$remote_name" ]] || continue
      printf '%s\n' "remote $remote_name：" >>"$LOG_FILE"
      git -C "$dir" remote get-url --all "$remote_name" 2>/dev/null | redact_remote_url >>"$LOG_FILE" || true
    done < <(git -C "$dir" remote)
    log "HEAD："
    git -C "$dir" rev-parse HEAD >>"$LOG_FILE" 2>&1 || true
    log "工作区状态："
    git -C "$dir" status --short --branch >>"$LOG_FILE" 2>&1 || true
  else
    log "[未检测到可用 Git 工作树]"
  fi
}

audit_model_dir() {
  local dir="$1"
  describe_dir "MODEL_DIR" "$dir"
  [[ -n "$dir" && -d "$dir" ]] || return 0

  section "MODEL_DIR 文件清单与总大小（只读）"
  find "$dir" -type f -printf '%s\t%p\n' | LC_ALL=C sort >>"$LOG_FILE" 2>&1 || true
  du -sh -- "$dir" >>"$LOG_FILE" 2>&1 || true

  section "官方五个 safetensors 的 SHA256（仅对 MODEL_DIR 中明确文件，只读）"
  local -a filenames=(
    "model-00001-of-00005.safetensors"
    "model-00002-of-00005.safetensors"
    "model-00003-of-00005.safetensors"
    "model-00004-of-00005.safetensors"
    "model-00005-of-00005.safetensors"
  )
  local -a expected_sha256=(
    "537259bb56815f9dfdcb028d5606e84dff5ab0c8e0e559782d63d91b6671d15c"
    "18841e5049b7836c16c034123ae6d40b2ff0414f9f77008bc7ca72490302a47b"
    "e8953b27fe827a60604a006b07ab1216fe766375f9926d1c3bd5ecee84ecef78"
    "604f0c0f19986f363a23a1e0b5736eedadd231ca285beab28d5db968a1d37602"
    "9d889c09634e5a21b4c957957ede2cdab4418c268ee1d870c1f942d04242adf6"
  )
  local index file actual

  if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    log "[未找到 sha256sum 或 shasum，未计算校验值]"
    return 0
  fi

  for index in "${!filenames[@]}"; do
    file="$dir/${filenames[$index]}"
    if [[ ! -f "$file" ]]; then
      log "MISSING  ${filenames[$index]}  expected=${expected_sha256[$index]}"
      continue
    fi
    if command -v sha256sum >/dev/null 2>&1; then
      actual="$(sha256sum -- "$file" | awk '{print $1}')"
    else
      actual="$(shasum -a 256 -- "$file" | awk '{print $1}')"
    fi
    if [[ "$actual" == "${expected_sha256[$index]}" ]]; then
      log "MATCH    ${filenames[$index]}  $actual"
    else
      log "MISMATCH ${filenames[$index]}  actual=$actual expected=${expected_sha256[$index]}"
    fi
  done
}

audit_data_dir() {
  local dir="$1"
  describe_dir "DATA_DIR（最多两层清单，只读）" "$dir"
  [[ -n "$dir" && -d "$dir" ]] || return 0

  find "$dir" -maxdepth 2 -printf '%y\t%s\t%p\n' | LC_ALL=C sort >>"$LOG_FILE" 2>&1 || true
}

audit_torch_if_present() {
  section "PyTorch CUDA / BF16 能力（仅在已安装 torch 时查询）"
  if ! command -v python3 >/dev/null 2>&1; then
    log "[未找到 python3]"
  elif ! python3 -B -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("torch") else 1)' >/dev/null 2>&1; then
    log "[未安装 torch，未执行 PyTorch 查询]"
  else
    local python_rc=0
    python3 -B - <<'PY' >>"$LOG_FILE" 2>&1 || python_rc=$?
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            print(f"device[{index}].name={torch.cuda.get_device_name(index)}")
            print(
                f"device[{index}].bf16_supported="
                f"{torch.cuda.is_bf16_supported(including_emulation=False)}"
            )
PY
    if (( python_rc != 0 )); then
      log "[PyTorch CUDA / BF16 查询退出码 $python_rc：已记录，继续盘点]"
    fi
  fi
}

log "Alpamayo 1.5 L20 候选机只读盘点"
log "日志目录：$LOG_DIR"
log "提示：本脚本不记录 GPU UUID、主机 IP、SSH 配置或凭据。"

section "系统"
date -Is >>"$LOG_FILE" 2>&1 || true
if [[ -r /etc/os-release ]]; then
  sed -n '1,20p' /etc/os-release >>"$LOG_FILE" 2>&1 || true
else
  log "[无法读取 /etc/os-release]"
fi
uname -m >>"$LOG_FILE" 2>&1 || true

run_if_available "NVIDIA GPU 名称、显存、空闲显存、驱动与 Compute Capability" \
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader
run_if_available "NVIDIA GPU 拓扑" nvidia-smi topo -m
run_if_available "CUDA 编译器" nvcc --version
run_if_available "Python" python3 --version
run_if_available "uv" uv --version
run_if_available "Git" git --version
run_if_available "磁盘空间" df -h
run_if_available "主机内存" free -h

audit_torch_if_present
audit_code_dir "${CODE_DIR:-}"
audit_model_dir "${MODEL_DIR:-}"
audit_data_dir "${DATA_DIR:-}"

log ""
log "盘点完成。请仅回传日志文件；发送前确认其中不含 token、密码或私钥。"
log "日志文件：$LOG_FILE"
