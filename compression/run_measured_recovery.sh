#!/usr/bin/env bash
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
out="${1:?Unique experiment path required}"
parent="${2:?Verified recovered parent required}"
allocation="${3:?Completed measured allocation required}"
python="$root/alpamayo1.5/.venv/bin/python"
export PYTHONPATH="$root/compression:$root/scripts:$root/logs/alp15_audit_20261001_165853/official_source/src"
cd "$root"
finish() {
    rc=$?
    trap - EXIT
    if [ -d "$out" ]; then
        cp compression/run_measured_recovery.sh "$out/code/as_started_pipeline.sh"
        "$python" compression/export_curves.py --experiment "$out" --language zh > "$out/curves_export.log" 2>&1 || true
        "$python" compression/plot_controlled_recovery.py --experiment "$out" > "$out/task_curves_export.log" 2>&1 || true
    fi
    printf '%s\n' "$rc" > "$out.pipeline.exitcode"
    exit "$rc"
}
trap finish EXIT
"$python" -u compression/run_interface_recovery.py --root "$root" \
    --teacher-experiment "$root/experiments/mlp_prune_distill_20261001_171143" \
    --parent-experiment "$parent" --output "$out" --steps 800 --eval-every 100 \
    --rank 32 --lr 5e-5 --rollout-prefix --logit-weight 0.5 --expert-weight 1.0 --shuffle-training \
    --new-training-audit "$root/experiments/newdata_chunk0000_audit_20261002_165238" \
    --teacher-interface-cache "$parent" --pruning-allocation "$allocation" \
    --controlled-validation-dir "$root/experiments/controlled8_expanded20_20261002_125000"
