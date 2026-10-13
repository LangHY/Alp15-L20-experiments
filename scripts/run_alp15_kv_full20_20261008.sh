#!/usr/bin/env bash
# 隔离快照内执行：新口径smoke成功后，才由Python验真扩至完整20clip。
set -euo pipefail
ROOT=/home/tuqiang/lang/Lighten_VLA
CODE="$ROOT/experiments/jetson_300ms_20261008_kv_code_v3"
SMOKE="$ROOT/experiments/jetson_300ms_20261008_kv_candidate_smoke_v4"
FULL="$ROOT/experiments/jetson_300ms_20261008_kv_candidate_full20_v1"
cd "$ROOT"
export PYTHONPATH="$CODE/scripts:$ROOT/scripts:$ROOT/compression"
PYTHON="$ROOT/alpamayo1.5/.venv/bin/python"
for TARGET in "$SMOKE" "$FULL"; do
  if [[ -e "$TARGET" ]]; then
    echo "STOP: output exists: $TARGET" >&2
    exit 1
  fi
done
echo "START smoke $(date -u +%FT%TZ)"
"$PYTHON" -u "$CODE/scripts/alp15_kv_steps_probe.py" \
  --root "$ROOT" --output "$SMOKE" --clips 2 --steps 10 6 4 2 \
  --cache-policy new_candidate --candidate-decision D-JETSON-KV-CANDIDATE-20261008
echo "START full20 $(date -u +%FT%TZ)"
"$PYTHON" -u "$CODE/scripts/alp15_kv_steps_probe.py" \
  --root "$ROOT" --output "$FULL" --clips 20 --steps 10 6 4 2 \
  --cache-policy new_candidate --candidate-decision D-JETSON-KV-CANDIDATE-20261008 \
  --approved-smoke "$SMOKE"
echo "COMPLETE full20 $(date -u +%FT%TZ)"
