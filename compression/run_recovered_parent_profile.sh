#!/usr/bin/env bash
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
parent="${1:?Recovered parent required}"
out="${2:?Unique profile output required}"
python="$root/alpamayo1.5/.venv/bin/python"
export PYTHONPATH="$root/compression:$root/scripts:$root/logs/alp15_audit_20261001_165853/official_source/src"
cd "$root"
finish() {
    rc=$?
    trap - EXIT
    if [ -d "$out" ]; then
        cp compression/run_recovered_parent_profile.sh "$out/code/as_started_pipeline.sh"
        "$python" compression/export_curves.py --experiment "$out" --language zh > "$out/curves_export.log" 2>&1 || true
    fi
    printf '%s\n' "$rc" > "$out.pipeline.exitcode"
    exit "$rc"
}
trap finish EXIT
"$python" - "$parent" <<'PY'
import json,time,sys
from pathlib import Path
parent=Path(sys.argv[1]);status=Path(str(parent)+'.postprocess.json')
while True:
    result=json.loads(status.read_text())
    if result['status']=='failed':raise RuntimeError('Parent postprocessing failed; do not profile an unverified parent')
    if result['status']=='completed':break
    time.sleep(10)
assert result['reload_status']=='completed'
print('PARENT_RELOAD_VERIFIED',json.dumps(result),flush=True)
PY
"$python" -u compression/profile_recovered8_pruning.py --root "$root" --parent "$parent" --output "$out" --measurement-only
