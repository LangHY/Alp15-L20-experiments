#!/usr/bin/env bash
# Require a completed calibration-only profile before any student is pruned.
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
audit="$root/logs/alp15_audit_20261001_165853"
teacher="$root/experiments/mlp_prune_distill_20261001_171143"
profile="${1:?Completed importance profile directory}"
budget="${2:?Reference mean width: 5376 or 3584}"
out="${3:?New student experiment directory}"
references="$root/experiments/mlp_stage8_20261001_175032/expanded_checks"
export PYTHONPATH="$root/compression:$root/scripts:$audit/official_source/src"
python="$root/alpamayo1.5/.venv/bin/python"
cd "$root"
finish_experiment() {
    rc=$?
    if [ -d "$out" ]; then
        "$python" compression/export_curves.py --experiment "$out" --language zh > "$out/curves_export.log" 2>&1 &&
            printf '0\n' > "$out/curves.exitcode" || printf '%s\n' "$?" > "$out/curves.exitcode"
    fi
    printf '%s\n' "$rc" > "$out.exitcode"
    exit "$rc"
}
trap finish_experiment EXIT
"$python" - "$profile" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
if json.loads((p/'importance.json').read_text())['status'] != 'completed':
    raise SystemExit('All layer importance measurements must finish before pruning')
PY
"$python" -u compression/run_width_sweep.py --root "$root" --teacher-experiment "$teacher" \
    --output "$out" --allocation-json "$profile/allocation_$budget.json" \
    --joint-mlp --steps 300 --lr 3e-5 --eval-every 20
cp compression/run_layerwise.sh compression/evaluate_stage.py compression/evaluate_expanded.py "$out/code/"
"$python" -u compression/evaluate_stage.py --root "$root" --teacher-experiment "$teacher" \
    --stage-experiment "$out" --output "$out/independent_checks" --repeat-reload
"$python" -u compression/evaluate_expanded.py --root "$root" --teacher-experiment "$teacher" \
    --stage-experiment "$out" --output "$out/expanded_checks" --dense-reference "$references"
"$python" - "$out" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
x=json.loads((p/'expanded_checks/result.json').read_text())
gate=x['next_6B_stage_gate']
payload={'accepted':gate['pass'], 'uses_holdout':False, 'criterion':gate['criterion'],
         'params':x['params'], 'expanded_validation_mean_ADE_m':x['expanded_validation']['student_minADE_m']['mean'],
         'expanded_holdout_mean_ADE_m':x['expanded_holdout']['student_minADE_m']['mean'],
         'missing_markers':x['missing_trajectory_start_counts']}
(p/'stage_acceptance.json').write_text(json.dumps(payload,indent=2)+'\n')
if not gate['pass']:
    raise SystemExit(2)
PY
