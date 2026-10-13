#!/usr/bin/env bash
# Reporting-only 20+69 expansion for an exported but unselected candidate.
set -euo pipefail
root=/home/tuqiang/lang/Lighten_VLA
stage="${1:?Existing failed or unselected experiment directory}"
label="${2:?Completed candidate width label}"
out="${3:?New diagnostic expanded output}"
curves="${4:?New diagnostic chart output}"
export PYTHONPATH="$root/compression:$root/scripts:$root/logs/alp15_audit_20261001_165853/official_source/src"
python="$root/alpamayo1.5/.venv/bin/python"
cd "$root"
finish() {
    rc=$?
    if [ -d "$out" ]; then
        "$python" compression/export_curves.py --experiment "$stage" --output "$curves" --language zh \
            > "$out/curves_export.log" 2>&1 && printf '0\n' > "$out/curves.exitcode" || printf '%s\n' "$?" > "$out/curves.exitcode"
    fi
    printf '%s\n' "$rc" > "$out.exitcode"
    exit "$rc"
}
trap finish EXIT
test ! -e "$out"
test ! -e "$out.code"
mkdir -p "$out.code"
cp compression/evaluate_expanded.py compression/layerwise_allocation.py \
    compression/run_mlp_experiment.py compression/run_diagnostic_expanded.sh \
    compression/export_curves.py "$out.code/"
sha256sum "$out.code/"* > "$out.code/source_hashes.txt"
"$python" -u compression/evaluate_expanded.py --root "$root" \
    --teacher-experiment "$root/experiments/mlp_prune_distill_20261001_171143" \
    --stage-experiment "$stage" --diagnostic-candidate "$label" --output "$out" \
    --dense-reference "$root/experiments/mlp_stage8_20261001_175032/expanded_checks"
