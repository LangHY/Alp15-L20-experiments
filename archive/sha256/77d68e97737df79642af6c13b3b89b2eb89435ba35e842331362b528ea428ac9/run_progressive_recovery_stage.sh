#!/usr/bin/env bash
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
audit="$root/logs/alp15_audit_20261001_165853"
previous="${1:?Expert recovery experiment}"
previous_pid="${2:?Expert pipeline PID}"
parent="${3:?Prepared conservative overlay}"
out="${4:?New conservative recovery directory}"
python="$root/alpamayo1.5/.venv/bin/python"
export PYTHONPATH="$root/compression:$root/scripts:$audit/official_source/src"
cd "$root"
while kill -0 "$previous_pid" 2>/dev/null; do sleep 10; done
test -f "$previous/result.json"
"$python" -u compression/evaluate_interface_recovery.py --root "$root" --experiment "$previous"
"$python" compression/export_curves.py --experiment "$previous" --language zh
if "$python" - "$previous" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'stage_acceptance.json'
raise SystemExit(0 if p.exists() and json.loads(p.read_text())['accepted'] else 1)
PY
then
    echo 'Expert recovery accepted: conservative fallback is not needed'
    exit 0
fi
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
    --parent-experiment "$parent" --output "$out" --steps 120 --eval-every 12 \
    --rank 16 --lr 1e-4 --rollout-prefix --evaluate-parent --local-prefit-steps 100 &
training_pid=$!
wait "$training_pid"
bash compression/run_interface_checks.sh "$out" "$training_pid"
