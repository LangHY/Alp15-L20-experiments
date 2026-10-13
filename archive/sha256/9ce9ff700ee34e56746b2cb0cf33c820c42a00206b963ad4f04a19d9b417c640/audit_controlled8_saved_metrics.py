"""Reproduce read-only velocity proxy and conditional 15% diagnostic screen."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment', type=Path, required=True)
args = parser.parse_args()
p = args.experiment
x = json.loads((p / 'result.json').read_text())
assert x['status'] == 'completed'
records = []
for clip in sorted({r['clip_id'] for r in x['records']}):
    values = []
    for seed in x['trajectory_seeds']:
        cells = {(r['model'], r['prefix']): r for r in x['records'] if r['clip_id'] == clip and r['seed'] == seed}
        a, b = cells[('dense', 'teacher')], cells[('student8', 'teacher')]
        assert a['initial_noise_sha256'] == b['initial_noise_sha256']
        av, bv = np.asarray(a['first_velocity'], dtype=np.float64), np.asarray(b['first_velocity'], dtype=np.float64)
        values.append(float(np.mean((bv-av)**2) / max(float(np.mean(av**2)), 1e-12)))
    records.append({'clip_id': clip, 'mean_first_velocity_nmse': statistics.mean(values)})
audit = {'semantics': 'Same teacher prefix and same initial noise at t0; NMSE of first Expert velocity only, not the integrated trajectory', 'records': records}
old = json.loads((p/'first_velocity_audit.json').read_text())
assert {r['clip_id'] for r in old['records']} == {r['clip_id'] for r in records}
for r in records:
    before = next(a for a in old['records'] if a['clip_id'] == r['clip_id'])
    assert abs(before['mean_first_velocity_nmse'] - r['mean_first_velocity_nmse']) < 1e-6
(p/'first_velocity_audit_reproduced.json').write_text(json.dumps(audit, indent=2)+'\n')
summary = json.loads((p/'summary.json').read_text())['roles']['validation']
cells = {(r['model'],r['prefix']): r['mean_ADE_m'] for r in summary['conditions']}
teacher, student = cells[('dense','teacher')], cells[('student8','student8')]
screen = {'semantics':'Conditional reuse of historical 15% tolerance on controlled diagnostic; not official deployment acceptance', 'teacher_ADE_m':teacher, 'student_ADE_m':student, 'relative_increase_percent':100*(student/teacher-1), 'tolerance_fraction':0.15, 'limit_ADE_m':teacher*1.15, 'passed':student<=teacher*1.15, 'paired_clip_bootstrap_ci95':summary['paired_clip_bootstrap_ci95'], 'limitations':['20 independent clips with 3 repeated seeds, not 60 scenes','Standardized full-prefix KV diagnostic','Historical acceptance files unchanged','No training or weight modification']}
(p/'controlled_screen.json').write_text(json.dumps(screen, indent=2)+'\n')
print('velocity mean NMSE', statistics.mean(r['mean_first_velocity_nmse'] for r in records))
print('conditional screen passed', screen['passed'])
