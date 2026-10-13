#!/usr/bin/env bash
# Independent 6B candidate from the dense teacher, gated by the accepted 8B run.
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
audit="$root/logs/alp15_audit_20261001_165853"
teacher="$root/experiments/mlp_prune_distill_20261001_171143"
accepted="${1:?Pass the completed 8B joint experiment}"
out="${2:?Pass a new 6B experiment directory}"
references="$root/experiments/mlp_stage8_20261001_175032/expanded_checks"
export PYTHONPATH="$root/compression:$root/scripts:$audit/official_source/src"
python="$root/alpamayo1.5/.venv/bin/python"
cd "$root"
"$python" - "$accepted" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
result = json.loads((p / 'expanded_checks/result.json').read_text())
if result.get('status') != 'completed' or not result['next_6B_stage_gate']['pass']:
    raise SystemExit('8B expanded validation must pass before the 6B candidate')
reload = json.loads((p / 'independent_checks/reload_verification.json').read_text())
if reload['status'] != 'fresh_process_reload_passed':
    raise SystemExit('8B reload has not passed')
PY
"$python" -u compression/run_width_sweep.py --root "$root" --teacher-experiment "$teacher" \
    --output "$out" --widths 1408 --joint-mlp --steps 300 --lr 3e-5 --eval-every 20
mkdir -p "$out/code"
cp compression/run_width_sweep.py compression/joint_mlp_distill.py \
    compression/evaluate_stage.py compression/evaluate_expanded.py compression/run_stage6.sh "$out/code/"
"$python" -u compression/evaluate_stage.py --root "$root" --teacher-experiment "$teacher" \
    --stage-experiment "$out" --output "$out/independent_checks" --repeat-reload
"$python" -u compression/evaluate_expanded.py --root "$root" --teacher-experiment "$teacher" \
    --stage-experiment "$out" --output "$out/expanded_checks" --dense-reference "$references"


