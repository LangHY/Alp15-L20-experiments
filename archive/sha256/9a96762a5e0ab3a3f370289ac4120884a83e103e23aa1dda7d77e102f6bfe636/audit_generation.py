"""Attach trajectory-marker warnings to clips in an existing expanded run log."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--expanded-output', type=Path, required=True)
    args = parser.parse_args()
    progress = json.loads((args.expanded_output / 'progress.json').read_text())
    clips = set(progress['split']['expanded_validation'] + progress['split']['expanded_holdout'])
    pending, rows = [], []
    observed = {'dense': set(), 'student': set()}
    source = args.log.read_bytes()
    for line in source.decode().splitlines():
        if line.startswith('EXPANDED FAILED'):
            raise RuntimeError('Failed attempts require explicit per-attempt marker tracing; refusing legacy audit')
        if 'No <traj_future_start> token found' in line:
            pending.append(line)
        if line.startswith('STAGE HOLDOUT'):
            pending = []
        match = re.match(r'EXPANDED phase=(\w+) role=(\w+) clip=([\w-]+)', line)
        if match:
            phase, role, clip = match.groups()
            if phase not in observed or clip not in clips:
                raise RuntimeError('Unexpected expanded clip or phase in generation log')
            if clip in observed[phase]:
                raise RuntimeError('Repeated attempts require explicit per-attempt tracing; refusing legacy audit')
            observed[phase].add(clip)
            if pending:
                rows.append({'phase': phase, 'role': role, 'clip_id': clip, 'warnings': pending.copy()})
            pending = []
    if pending or any(ids != clips for ids in observed.values()):
        raise RuntimeError('Generation log is incomplete; cannot infer absence of marker warnings')
    (args.expanded_output / 'generation_audit_source.log').write_bytes(source)
    (args.expanded_output / 'generation_audit.json').write_text(json.dumps(rows, indent=2) + '\n')
    (args.expanded_output / 'generation_audit_provenance.json').write_text(json.dumps({
        'source_log_sha256': hashlib.sha256(source).hexdigest(),
        'observed_clips_per_phase': {k: len(v) for k, v in observed.items()},
        'missing_marker_counts': {k: sum(r['phase'] == k for r in rows) for k in observed},
        'note': 'Log audit supplements the original metrics; no trajectories or ADE values changed'}, indent=2) + '\n')
    print('GENERATION_AUDIT', {k: sum(r['phase'] == k for r in rows) for k in observed})


if __name__ == '__main__':
    main()


