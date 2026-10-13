#!/usr/bin/env bash
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
out="${1:?Unique experiment path required}"
parent="$root/experiments/mlp_newdata8_recovery_20261002_165238"
python="$root/alpamayo1.5/.venv/bin/python"
export PYTHONPATH="$root/compression:$root/scripts:$root/logs/alp15_audit_20261001_165853/official_source/src"
cd "$root"
finish() {
    rc=$?
    trap - EXIT
    if [ -d "$out" ]; then
        cp compression/run_precision8_recovery.sh "$out/code/as_started_pipeline.sh"
        "$python" compression/export_curves.py --experiment "$out" --language zh > "$out/curves_export.log" 2>&1 || true
        "$python" compression/plot_controlled_recovery.py --experiment "$out" > "$out/task_curves_export.log" 2>&1 || true
    fi
    printf '%s\n' "$rc" > "$out.pipeline.exitcode"
    exit "$rc"
}
trap finish EXIT
"$python" -u compression/run_interface_recovery.py --root "$root" \
    --teacher-experiment "$root/experiments/mlp_prune_distill_20261001_171143" \
    --parent-experiment "$parent" --output "$out" --steps 320 --eval-every 20 \
    --rank 32 --lr 1e-5 --rollout-prefix --logit-weight 0.5 --expert-weight 1.0 --shuffle-training \
    --expert-probe-schedule balanced --save-adapters-every 1 \
    --new-training-audit "$root/experiments/newdata_chunk0000_audit_20261002_165238" \
    --teacher-interface-cache "$parent" \
    --controlled-validation-dir "$root/experiments/controlled8_expanded20_20261002_125000"
