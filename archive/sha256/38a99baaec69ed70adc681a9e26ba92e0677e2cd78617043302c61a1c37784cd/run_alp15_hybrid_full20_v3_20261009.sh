#!/usr/bin/env bash
# 同代码20段开发集配对；不融合KV/少步，不覆盖旧证据。
set -euo pipefail
TASK_ROOT=/home/tuqiang/lang/Lighten_VLA
TASK_CODE="$TASK_ROOT/experiments/jetson_300ms_20261009_hybrid_m1_code_v3"
TASK_REFERENCE="$TASK_ROOT/experiments/jetson_300ms_20261009_hybrid_reference20_v3"
TASK_HYBRID="$TASK_ROOT/experiments/jetson_300ms_20261009_hybrid_full20_v3"
TASK_SMOKE="$TASK_ROOT/experiments/jetson_300ms_20261009_hybrid_smoke2_v3"
TASK_PROBE="$TASK_ROOT/experiments/jetson_300ms_20261009_hybrid252_v3"
TASK_PY="$TASK_ROOT/alpamayo1.5/.venv/bin/python"
test ! -e "$TASK_REFERENCE"
test ! -e "$TASK_HYBRID"
exec 9>"$TASK_ROOT/experiments/jetson_300ms_20261009_hybrid_gpu.lock"
flock -n 9
export PYTHONPATH="$TASK_CODE/scripts:$TASK_ROOT/scripts:$TASK_ROOT/compression"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
printf 'START full20_v3 %s launcher_pid=%s\n' "$(date -u +%FT%TZ)" "$$"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
set +e
"$TASK_PY" -u "$TASK_CODE/scripts/run_alp15_awq_hybrid_eval.py" \
  --root "$TASK_ROOT" \
  --awq-artifact "$TASK_ROOT/experiments/awq_full_20261008_best8_full20_v4" \
  --output "$TASK_REFERENCE" --candidate-decision D-ALP15-HYBRID-M1-20261008 \
  --awq-forward reference --clips 20 --steps 10 --approved-smoke "$TASK_SMOKE"
TASK_EXIT=$?
set -e
printf 'TERMINAL reference20_v3 %s exit=%s\n' "$(date -u +%FT%TZ)" "$TASK_EXIT"
if [ "$TASK_EXIT" -ne 0 ]; then exit "$TASK_EXIT"; fi
set +e
"$TASK_PY" -u "$TASK_CODE/scripts/run_alp15_awq_hybrid_eval.py" \
  --root "$TASK_ROOT" \
  --awq-artifact "$TASK_ROOT/experiments/awq_full_20261008_best8_full20_v4" \
  --output "$TASK_HYBRID" --candidate-decision D-ALP15-HYBRID-M1-20261008 \
  --awq-forward hybrid_m1 --clips 20 --steps 10 --approved-smoke "$TASK_SMOKE" \
  --matrix-probe "$TASK_PROBE" --paired-reference "$TASK_REFERENCE"
TASK_EXIT=$?
set -e
printf 'TERMINAL hybrid20_v3 %s exit=%s\n' "$(date -u +%FT%TZ)" "$TASK_EXIT"
exit "$TASK_EXIT"
