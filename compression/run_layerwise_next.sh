#!/usr/bin/env bash
# Advance the smaller allocation only after the first candidate's full checks.
set -euo pipefail
parent="${1:?8B experiment directory}"
profile="${2:?Completed importance profile}"
out="${3:?New 7.228B experiment directory}"
while [ ! -f "$parent.exitcode" ]; do sleep 5; done
if [ "$(cat "$parent.exitcode")" != 0 ]; then
    printf 'Parent experiment failed; smaller allocation not started\n'
    exit 2
fi
/home/tuqiang/lang/Lighten_VLA/alpamayo1.5/.venv/bin/python - "$parent" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
x=json.loads((p/'expanded_checks/result.json').read_text())
r=json.loads((p/'independent_checks/reload_verification.json').read_text())
if x['status'] != 'completed' or not x['next_6B_stage_gate']['pass']:
    raise SystemExit('Expanded validation gate failed; no smaller allocation')
if r['status'] != 'fresh_process_reload_passed':
    raise SystemExit('Independent reload failed; no smaller allocation')
PY
bash /home/tuqiang/lang/Lighten_VLA/compression/run_layerwise.sh "$profile" 3584 "$out"
