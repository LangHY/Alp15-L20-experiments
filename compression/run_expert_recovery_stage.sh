#!/usr/bin/env bash
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
audit="$root/logs/alp15_audit_20261001_165853"
previous="${1:?Latest completed experiment to audit}"
previous_pid="${2:?Latest experiment pipeline PID}"
parent="${3:?Best validation recovery parent}"
out="${4:?New Expert KD experiment directory}"
python="$root/alpamayo1.5/.venv/bin/python"
export PYTHONPATH="$root/compression:$root/scripts:$audit/official_source/src"
cd "$root"
while kill -0 "$previous_pid" 2>/dev/null; do sleep 10; done
test -f "$previous/stage_acceptance.json"
"$python" -u compression/evaluate_interface_recovery.py --root "$root" --experiment "$previous"
"$python" compression/export_curves.py --experiment "$previous" --language zh
finish() {
    rc=$?
    if [ -d "$out" ]; then
        "$python" compression/export_curves.py --experiment "$out" --language zh > "$out/curves_export.log" 2>&1 || true
    fi
    printf '%s\n' "$rc" > "$out.pipeline.exitcode"
    exit "$rc"
}
trap finish EXIT
"$python" -u compression/run_interface_recovery.py --root "$root" \
    --teacher-experiment "$root/experiments/mlp_prune_distill_20261001_171143" \
    --parent-experiment "$parent" --output "$out" --steps 240 --eval-every 24 \
    --rank 32 --lr 2e-4 --rollout-prefix --logit-weight 0.5 --expert-weight 5.0 &
training_pid=$!
wait "$training_pid"
bash compression/run_interface_checks.sh "$out" "$training_pid"
