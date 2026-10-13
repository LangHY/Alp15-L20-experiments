#!/usr/bin/env bash
# Attach independent verification to an already started recovery process.
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
audit="$root/logs/alp15_audit_20261001_165853"
out="${1:?Recovery experiment directory}"
pid="${2:?Recovery training process PID}"
python="$root/alpamayo1.5/.venv/bin/python"
teacher="$root/experiments/mlp_prune_distill_20261001_171143"
export PYTHONPATH="$root/compression:$root/scripts:$audit/official_source/src"
cd "$root"
finish() {
    rc=$?
    "$python" compression/export_curves.py --experiment "$out" --language zh > "$out/curves_export.log" 2>&1 &&
        printf '0\n' > "$out/curves.exitcode" || printf '%s\n' "$?" > "$out/curves.exitcode"
    printf '%s\n' "$rc" > "$out.checks.exitcode"
    exit "$rc"
}
trap finish EXIT
while kill -0 "$pid" 2>/dev/null; do sleep 10; done
test -f "$out/selection.json"
mkdir -p "$out/code_checks"
cp compression/run_interface_checks.sh compression/export_curves.py "$out/code_checks/"
"$python" - "$out" <<'PY'
import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1])
files=[p/'split.json',p/'provenance.json',*sorted((p/'teacher_cache').glob('*.pt'))]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
(p/'teacher_interface_artifacts.json').write_text(json.dumps([{'file':str(x),'sha256':sha(x)} for x in files],indent=2)+'\n')
if json.loads((p/'selection.json').read_text())['selected'] is None:
 raise SystemExit('Recovery failed initial trajectory validation; retained as failed experiment')
PY
"$python" -u compression/evaluate_stage.py --root "$root" --teacher-experiment "$teacher" \
    --stage-experiment "$out" --output "$out/independent_checks" --repeat-reload
"$python" -u compression/evaluate_expanded.py --root "$root" --teacher-experiment "$teacher" \
    --stage-experiment "$out" --output "$out/expanded_checks" \
    --dense-reference "$root/experiments/mlp_stage8_20261001_175032/expanded_checks"
"$python" - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads((p/'expanded_checks/result.json').read_text()); gate=x['next_6B_stage_gate']
record={'accepted':gate['pass'],'uses_holdout':False,'params':x['params'],'criterion':gate['criterion'],
 'expanded_validation_mean_ADE_m':x['expanded_validation']['student_minADE_m']['mean'],
 'expanded_holdout_mean_ADE_m':x['expanded_holdout']['student_minADE_m']['mean'],
 'missing_markers':x['missing_trajectory_start_counts']}
(p/'stage_acceptance.json').write_text(json.dumps(record,indent=2)+'\n')
if not gate['pass']:raise SystemExit(2)
PY
