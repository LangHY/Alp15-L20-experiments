#!/usr/bin/env bash
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
out="${1:?Unique recovery path required}"
audit="${2:?Unique window audit path required}"
python="$root/alpamayo1.5/.venv/bin/python"
parent="$root/experiments/mlp_controlled8_recovery_20261002_132357_v2"
export PYTHONPATH="$root/compression:$root/scripts:$root/logs/alp15_audit_20261001_165853/official_source/src"
cd "$root"
finish() {
  rc=$?
  trap - EXIT
  for directory in "$audit" "$out"; do
    if [ -d "$directory" ]; then
      "$python" compression/export_curves.py --experiment "$directory" --language zh > "$directory/curves_export.log" 2>&1 || true
    fi
  done
  if [ -d "$out" ]; then
    "$python" compression/plot_controlled_recovery.py --experiment "$out" > "$out/task_curves_export.log" 2>&1 || true
  fi
  printf '%s\n' "$rc" > "$out.pipeline.exitcode"
  exit "$rc"
}
trap finish EXIT
"$python" -u compression/audit_training_windows.py --root "$root" --split "$parent/split.json" --output "$audit"
"$python" -u compression/run_interface_recovery.py --root "$root" \
  --teacher-experiment "$root/experiments/mlp_prune_distill_20261001_171143" \
  --parent-experiment "$parent" --output "$out" --steps 240 --eval-every 80 \
  --rank 16 --lr 5e-5 --rollout-prefix --logit-weight 0.5 --expert-weight 1.0 \
  --training-window-audit "$audit" \
  --controlled-validation-dir "$root/experiments/controlled8_expanded20_20261002_125000"
