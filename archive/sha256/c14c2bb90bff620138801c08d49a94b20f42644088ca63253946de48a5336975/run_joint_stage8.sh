#!/usr/bin/env bash
# Separate experiment; preserve every existing result and teacher reference.
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
audit="$root/logs/alp15_audit_20261001_165853"
teacher="$root/experiments/mlp_prune_distill_20261001_171143"
previous="$root/experiments/mlp_stage8_20261001_175032"
out="${1:?Pass a new experiment directory}"
export PYTHONPATH="$root/compression:$root/scripts:$audit/official_source/src"
python="$root/alpamayo1.5/.venv/bin/python"
cd "$root"
finish_experiment() {
    rc=$?
    if [ -d "$out" ]; then
        if "$python" compression/export_curves.py --experiment "$out"; then
            printf '0\n' > "$out/curves.exitcode"
        else
            printf '%s\n' "$?" > "$out/curves.exitcode"
        fi
    fi
    exit "$rc"
}
trap finish_experiment EXIT
"$python" - "$previous" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / 'expanded_checks'
if json.loads((p / 'result.json').read_text()).get('status') != 'completed':
    raise SystemExit('Complete the previous expanded evaluation before the next experiment')
PY
"$python" compression/audit_generation.py --log "$previous/stage8_checks.log" \
    --expanded-output "$previous/expanded_checks"
"$python" -u compression/run_width_sweep.py --root "$root" --teacher-experiment "$teacher" \
    --output "$out" --widths 5376 --joint-mlp --steps 300 --lr 3e-5 --eval-every 20
mkdir -p "$out/code"
cp compression/run_width_sweep.py compression/joint_mlp_distill.py \
    compression/evaluate_stage.py compression/evaluate_expanded.py \
    compression/run_joint_stage8.sh "$out/code/"
"$python" -u compression/evaluate_stage.py --root "$root" --teacher-experiment "$teacher" \
    --stage-experiment "$out" --output "$out/independent_checks" --repeat-reload
"$python" -u compression/evaluate_expanded.py --root "$root" --teacher-experiment "$teacher" \
    --stage-experiment "$out" --output "$out/expanded_checks" \
    --dense-reference "$previous/expanded_checks"
# The next decision uses expanded validation, never the expanded holdout metric.
