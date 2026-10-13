#!/usr/bin/env bash
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
out="${1:?Unique experiment path required}"
python="$root/alpamayo1.5/.venv/bin/python"
export PYTHONPATH="$root/compression:$root/scripts:$root/logs/alp15_audit_20261001_165853/official_source/src"
cd "$root"
finish() {
  rc=$?
  trap - EXIT
  if [ -d "$out" ]; then
    "$python" compression/export_curves.py --experiment "$out" --language zh > "$out/curves_export.log" 2>&1 || true
    "$python" compression/plot_controlled_recovery.py --experiment "$out" > "$out/task_curves_export.log" 2>&1 || true
  fi
  printf '%s\n' "$rc" > "$out.pipeline.exitcode"
  exit "$rc"
}
trap finish EXIT
"$python" -u compression/run_interface_recovery.py --root "$root" \
  --teacher-experiment "$root/experiments/mlp_prune_distill_20261001_171143" \
  --parent-experiment "$root/experiments/mlp_stage8_joint_20261001_182631" \
  --output "$out" --steps 120 --eval-every 40 --rank 16 --lr 1e-4 \
  --rollout-prefix --logit-weight 0.5 --expert-weight 1.0 \
  --controlled-validation-dir "$root/experiments/controlled8_expanded20_20261002_125000"
