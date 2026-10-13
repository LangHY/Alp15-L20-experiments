#!/usr/bin/env bash
# 新v3参考与真实激活捕获；旧v2和失败记录原样保留，不自动扩评。
set -euo pipefail
TASK_ROOT=/home/tuqiang/lang/Lighten_VLA
TASK_CODE="$TASK_ROOT/experiments/jetson_300ms_20261009_hybrid_m1_code_v3"
TASK_OUTPUT="$TASK_ROOT/experiments/jetson_300ms_20261009_hybrid_reference2_v3"
TASK_PY="$TASK_ROOT/alpamayo1.5/.venv/bin/python"
test ! -e "$TASK_OUTPUT"
exec 9>"$TASK_ROOT/experiments/jetson_300ms_20261009_hybrid_gpu.lock"
flock -n 9
export PYTHONPATH="$TASK_CODE/scripts:$TASK_ROOT/scripts:$TASK_ROOT/compression"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
printf 'START reference2_v3 %s launcher_pid=%s\n' "$(date -u +%FT%TZ)" "$$"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
set +e
"$TASK_PY" -u "$TASK_CODE/scripts/run_alp15_awq_hybrid_eval.py" \
  --root "$TASK_ROOT" \
  --awq-artifact "$TASK_ROOT/experiments/awq_full_20261008_best8_full20_v4" \
  --output "$TASK_OUTPUT" --candidate-decision D-ALP15-HYBRID-M1-20261008 \
  --awq-forward reference --clips 2 --steps 10 --capture-decode-activations
TASK_EXIT=$?
set -e
printf 'TERMINAL reference2_v3 %s exit=%s\n' "$(date -u +%FT%TZ)" "$TASK_EXIT"
exit "$TASK_EXIT"
