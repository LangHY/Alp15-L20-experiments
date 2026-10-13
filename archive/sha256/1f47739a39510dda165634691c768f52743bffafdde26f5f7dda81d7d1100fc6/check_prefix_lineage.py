"""一次性 expanded20 prefix 来源查询；不读取 PT。"""
import csv
import hashlib
import json
import os
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

root = Path('/home/tuqiang/lang/Lighten_VLA')
out = root / 'experiments/controlled8_expanded20_20261002_125000'
census = root / 'experiments/precision_holdout_prep_20261010_resume_v3'
hashes = {}


def read(path):
    assert path.suffix in {'.json', '.csv', '.py'}
    assert stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    data = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return data.decode('utf-8', errors='strict')


started = datetime.now(timezone.utc).isoformat()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1'
result = json.loads(read(out / 'result.json'))
split = json.loads(read(out / 'split.json'))
source_path = out / 'code/controlled_8b_diagnostic.py'
source = read(source_path)
assert hashes[str(source_path)] == result['code_sha256'][source_path.name]
assert hashes[str(source_path)] == '92e858ca556deab9edb26b8f98231d332bdae1376c4907b52853baba27623313'
assert hashes[str(out / 'result.json')] == '67b328a1ecac9de414b47680ef6476c96f6cb30dab0c736a500a902970cebaa0'
assert hashes[str(out / 'split.json')] == 'e25cd24fa473a21cf14d9d46306bdd5b051ba1776d47bccd30be333ccaccd20c'
argv = result['argv']
assert argv[0].endswith('controlled_8b_diagnostic.py')
reference = Path(argv[argv.index('--expanded-validation-dir') + 1])
if not reference.is_absolute():
    reference = root / reference
assert reference.is_relative_to(root / 'experiments')
original = json.loads(read(reference / 'split.json'))
assert hashes[str(reference / 'split.json')] == hashes[str(out / 'split.json')]
assert result['inputs'][str(reference / 'split.json')] == hashes[str(reference / 'split.json')]
assert '--resume' not in argv
clips = split['expanded_validation']
assert len(clips) == len(set(clips)) == 20 and original['expanded_validation'] == clips
assert {row['clip_id'] for row in result['records']} == set(clips)
ledger = json.loads(read(census / 'usage-ledger.json'))
assert hashes[str(census / 'usage-ledger.json')] == '0697cec28cd36ca841fe75ddeb95f1037269d69871bc2bda59ae831947e870d1'
assert set(clips) <= set(ledger['used_clip_ids'])
assert result['dataset_revision'] == ledger['dataset_revision']
mapped = []
for role in ('teacher_prefix_cache', 'student_prefix_cache'):
    folder = out / role
    assert folder.is_dir() and not folder.is_symlink()
    expected = {f'validation_{index:02d}.pt' for index in range(20)}
    assert {p.name for p in folder.iterdir() if p.suffix == '.pt'} == expected
    for index, clip in enumerate(clips):
        path = folder / f'validation_{index:02d}.pt'
        info = path.lstat()
        assert stat.S_ISREG(info.st_mode) and not path.is_symlink()
        mapped.append({'source': str(path.relative_to(root)), 'clip_id': clip,
                       'role': role, 'index': index, 'bytes': info.st_size,
                       'mtime_ns': info.st_mtime_ns, 'payload_verified': False})
import io
rows = list(csv.DictReader(io.StringIO(read(census / 'unresolved-assets.csv'))))
assert hashes[str(census / 'unresolved-assets.csv')] == 'f398e3f054620d37544217e766fbafba2d6d052f084df75e22df9f41259a09c7'
sources = {entry['source'] for entry in mapped}
for reason in ('anonymous_cache_without_explicit_local_mapping', 'binary_payload_not_verified'):
    selected = [row['source'] for row in rows if row['reason'] == reason and row['source'] in sources]
    assert Counter(selected) == Counter({path: 1 for path in sources})
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in hashes.items())
print(json.dumps({'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'query_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'input_text_sha256': hashes, 'producer_source': source,
    'mappings': mapped, 'unique_files': 40, 'already_used_clips': 20,
    'input_texts_unchanged': True, 'new_excluded_chunks': [],
    'status': 'candidate_attribution_pending_source_semantic_review',
    'payload_verified': False, 'coverage_complete': False, 'exit_code': 0},
    ensure_ascii=False, indent=2))
