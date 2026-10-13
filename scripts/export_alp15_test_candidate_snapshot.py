"""从已冻结的服务器CSV导出完整test候选快照；仅CPU文本，不下载。"""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path('/home/tuqiang/lang/Lighten_VLA')
SRC = ROOT / 'experiments/precision_holdout_prep_20261010_resume_v3'
OUT = ROOT / 'experiments/precision_full_pool_download_20261011_v1'
IDENTITIES = {
    'eligible-candidates.csv': 'cf620cf9b278bacd149378591d356e30a8efa3679e2696850a5bc743c717d8e7',
    'candidate-file-plan.csv': 'b76b687b43f3d9923c1a213e2b34ad3a3e64936d2c6c90a31470f7cf83a1b0ac',
    'usage-ledger.json': '0697cec28cd36ca841fe75ddeb95f1037269d69871bc2bda59ae831947e870d1',
}
raw = {}
for name, sha in IDENTITIES.items():
    p = SRC / name
    assert p.is_file() and not p.is_symlink()
    raw[name] = p.read_bytes()
    assert hashlib.sha256(raw[name]).hexdigest() == sha
rows = [r for r in csv.DictReader(io.StringIO(raw['eligible-candidates.csv'].decode())) if r['split'] == 'test']
chunks = sorted({int(r['chunk']) for r in rows})
assert len(rows) == len({r['clip_id'] for r in rows}) == 61301
assert len(chunks) == 633
ledger = json.loads(raw['usage-ledger.json'])
assert not set(chunks) & set(ledger['used_chunk_ids'])
files = [r for r in csv.DictReader(io.StringIO(raw['candidate-file-plan.csv'].decode())) if int(r['chunk']) in chunks]
assert len(files) == len({r['relative_path'] for r in files}) == 3165
features = {'egomotion', 'camera_cross_left_120fov', 'camera_cross_right_120fov', 'camera_front_tele_30fov', 'camera_front_wide_120fov'}
for chunk in chunks:
    assert {r['feature'] for r in files if int(r['chunk']) == chunk} == features
OUT.mkdir(exist_ok=False)
with (OUT / 'test-candidates.csv').open('x', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['clip_id', 'chunk', 't0_us'])
    writer.writeheader()
    writer.writerows({key: r[key] for key in writer.fieldnames} for r in rows)
snapshot = {'dataset': 'nvidia/PhysicalAI-Autonomous-Vehicles',
    'revision': '33f9bf447ed3bcb7d545ce13f4226f824214fafb',
    'scope': 'complete_metadata_test_candidate_pool_not_certified_independent',
    'coverage_complete': False, 'drive_disjoint_verified': False, 'clip_count': 61301,
    'chunks': chunks, 'excluded_chunks': ledger['used_chunk_ids'],
    'source_csv_sha256': IDENTITIES['eligible-candidates.csv'],
    'source_plan_sha256': IDENTITIES['candidate-file-plan.csv'],
    'test_candidates_sha256': hashlib.sha256((OUT / 'test-candidates.csv').read_bytes()).hexdigest(),
    'files': [{'path': r['relative_path'], 'chunk': int(r['chunk']), 'feature': r['feature']} for r in files]}
with (OUT / 'candidate-snapshot.json').open('x') as handle:
    json.dump(snapshot, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
for name, sha in IDENTITIES.items():
    assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == sha
print(json.dumps({'output': str(OUT), 'clip_count': len(rows), 'chunk_count': len(chunks),
    'file_count': len(files), 'artifacts': {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    for name in ('candidate-snapshot.json', 'test-candidates.csv')}}))
