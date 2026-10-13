#!/usr/bin/env bash
# Run on the server, serially; do not restart the completed training job.
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
audit="$root/logs/alp15_audit_20261001_165853"
teacher="$root/experiments/mlp_prune_distill_20261001_171143"
stage="$root/experiments/mlp_stage8_20261001_175032"
export PYTHONPATH="$root/compression:$root/scripts:$audit/official_source/src"
python="$root/alpamayo1.5/.venv/bin/python"
cd "$root"
"$python" - "$stage" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
selection = json.loads((p / 'selection.json').read_text())
if not selection.get('selected') or selection['selected'].get('validation_pass') is not True:
    raise SystemExit('8B did not pass initial validation; retain the prior 10B stage')
print('8B_SELECTED', selection['selected']['params'], flush=True)
PY
if [ ! -e "$stage/independent_checks/evaluation.json" ]; then
    "$python" -u compression/evaluate_stage.py --root "$root" \
        --teacher-experiment "$teacher" --stage-experiment "$stage" \
        --output "$stage/independent_checks" --repeat-reload
fi
if [ ! -e "$stage/independent_checks/reload_verification.json" ]; then
    reload_out="$stage/independent_checks/reload_retry_$(date +%Y%m%d_%H%M%S)"
    "$python" -u compression/evaluate_stage.py --root "$root" \
        --teacher-experiment "$teacher" --stage-experiment "$stage" \
        --output "$reload_out" \
        --reference-evaluation "$stage/independent_checks/evaluation.json"
    "$python" - "$stage" "$reload_out" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[2]) / 'evaluation.json'
result = json.loads(p.read_text())
if result.get('status') != 'fresh_process_reload_passed':
    raise SystemExit('Reload retry did not pass')
target = Path(sys.argv[1]) / 'independent_checks' / 'reload_verification.json'
target.write_text(json.dumps({'status': result['status'], 'evaluation': str(p),
                             'checks': result['reload_verification']}, indent=2) + '\n')
PY
fi
"$python" - "$stage" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / 'independent_checks' / 'reload_verification.json'
if json.loads(p.read_text()).get('status') != 'fresh_process_reload_passed':
    raise SystemExit('Fresh-process reload must pass before expanded evaluation')
PY
if [ -d "$stage/expanded_checks" ]; then
    resume=(--resume)
else
    resume=()
fi
"$python" -u compression/evaluate_expanded.py --root "$root" \
    --teacher-experiment "$teacher" --stage-experiment "$stage" \
    --output "$stage/expanded_checks" "${resume[@]}"
# Inspect the expanded validation gate before starting the next 6B experiment.
