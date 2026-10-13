"""一次性只读核查：只读 UTF-8 文本；PT 仅 lstat，不读 payload。"""
import ast
import csv
import hashlib
import io
import json
import os
import platform
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/tuqiang/lang/Lighten_VLA')
CENSUS = ROOT / 'experiments/precision_holdout_prep_20261010_resume_v3'
EXPECTED = {
    'usage-ledger.json': '0697cec28cd36ca841fe75ddeb95f1037269d69871bc2bda59ae831947e870d1',
    'scan-manifest.jsonl': '2177978ecf1ab7b3efe8150fb6fcfa7aabe1d60b32b4354584c24aa28b06014b',
    'scan-final-inventory.json': 'fc608fb693fb047e6aa45675b2af1e5ea1ca0d0dc3140b115bc90680e41b5052',
    'unresolved-assets.csv': 'f398e3f054620d37544217e766fbafba2d6d052f084df75e22df9f41259a09c7',
}
GROUPS = [
    ('jetson_300ms_20261008_hybrid_reference2_v2',
     'jetson_300ms_20261008_hybrid_m1_code_v2', 'alp15-m1-activations-v2',
     'a2ccec42a954be1d8325c89afbc32337193808265121448ec770ec92a792e887'),
    ('jetson_300ms_20261009_hybrid_reference2_v3',
     'jetson_300ms_20261009_hybrid_m1_code_v3', 'alp15-m1-activations-v3',
     'fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd'),
]
before = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_text(path):
    assert path.suffix in {'.json', '.jsonl', '.csv', '.py'}
    assert stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    raw = path.read_bytes()
    before[str(path)] = digest(raw)
    return raw.decode('utf-8', errors='strict')


started = datetime.now(timezone.utc).isoformat()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('HF_HUB_OFFLINE') == '1'
assert os.environ.get('TRANSFORMERS_OFFLINE') == '1'
texts = {name: read_text(CENSUS / name) for name in EXPECTED}
assert all(before[str(CENSUS / name)] == sha for name, sha in EXPECTED.items())
ledger = json.loads(texts['usage-ledger.json'])
rows = list(csv.DictReader(io.StringIO(texts['unresolved-assets.csv'])))
original_counts = Counter(row['reason'] for row in rows)
assert original_counts == Counter({
    'external_path_reference': 63177, 'binary_payload_not_verified': 4104,
    'anonymous_cache_without_explicit_local_mapping': 1892,
    'foreign_or_unknown_revision': 247, 'revision_conflict': 171,
    'symlink_not_followed': 20})
groups, attributed = [], set()
for experiment, snapshot, schema, source_sha in GROUPS:
    out = ROOT / 'experiments' / experiment
    manifest_path = out / 'decode_activations/manifest.json'
    result_path, receipt_path = out / 'result.json', out / 'reference_final_receipt.json'
    manifest = json.loads(read_text(manifest_path))
    result = json.loads(read_text(result_path))
    receipt = json.loads(read_text(receipt_path))
    source_path = ROOT / 'experiments' / snapshot / 'scripts/run_alp15_awq_hybrid_eval.py'
    source = read_text(source_path)
    assert before[str(source_path)] == source_sha == result['code_sha256'][str(source_path)]
    assert receipt['reference_result_sha256'] == before[str(result_path)]
    assert receipt['activation_manifest_sha256'] == before[str(manifest_path)]
    assert receipt['reference_result'] == str(result_path)
    assert receipt['activation_manifest'] == str(manifest_path)
    assert manifest['reference_result'] == str(result_path)
    assert manifest['final_receipt'] == str(receipt_path)
    assert manifest['schema'] == schema
    assert manifest['clip_id'] == result['clips'][-1]
    assert manifest['clip_id'] in ledger['used_clip_ids']
    assert result['dataset_revision'] == ledger['dataset_revision']
    assert result['decode_activation_capture']['manifest'] == str(manifest_path)
    assert result['decode_activation_capture']['sha256'] == before[str(manifest_path)]
    for key in ('code_sha256', 'runtime_source_sha256', 'input_sha256', 'actual_import_sha256'):
        assert manifest[key] == result[key]
    assert '--capture-decode-activations' in result['argv']
    assert result['awq_forward'] == 'reference' and len(result['clips']) == 2
    entries = manifest['activations']
    names = [entry['file'] for entry in entries]
    assert len(entries) == len(set(names)) == 252
    assert all(Path(name).name == name and Path(name).suffix == '.pt' for name in names)
    actual = {path.name for path in manifest_path.parent.iterdir() if path.suffix == '.pt'}
    assert actual == set(names)
    file_stats = []
    for entry in entries:
        path = manifest_path.parent / entry['file']
        info = path.lstat()
        assert stat.S_ISREG(info.st_mode) and not path.is_symlink()
        assert len(entry['sha256']) == 64 and all(c in '0123456789abcdef' for c in entry['sha256'])
        rel = str(path.relative_to(ROOT))
        attributed.add(rel)
        file_stats.append({'path': rel, 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns})
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    snippets = {name: ast.get_source_segment(source, functions[name]) for name in
                ('capture_decode_activations', 'write_final_capture_manifest')}
    lines = source.splitlines()
    call_context = []
    for i, line in enumerate(lines):
        if 'pending_capture = capture_decode_activations(' in line:
            call_context.append({'line': i + 1, 'text': '\n'.join(lines[max(0, i-3):i+2])})
    assert len(call_context) == 1
    groups.append({'experiment': experiment, 'clip_id': manifest['clip_id'],
                   'files': 252, 'producer_sha256': source_sha, 'producer': str(source_path.relative_to(ROOT)),
                   'result_sha256': before[str(result_path)],
                   'manifest_sha256': before[str(manifest_path)],
                   'receipt_sha256': before[str(receipt_path)],
                   'file_stat_list_sha256': digest(json.dumps(file_stats, sort_keys=True).encode()),
                   'file_stats': file_stats,
                   'producer_functions': snippets, 'capture_call_context': call_context,
                   'status': 'candidate_attribution_pending_source_semantic_review',
                   'payload_verified': False})
counts = Counter()
for row in rows:
    bounded = row['source'] in attributed and row['reason'] in {
        'anonymous_cache_without_explicit_local_mapping', 'binary_payload_not_verified'}
    counts[(row['reason'], 'candidate_attributed_used' if bounded else 'pending_evidence')] += 1
for reason, number in original_counts.items():
    assert sum(n for (r, disposition), n in counts.items() if r == reason) == number
for reason in ('anonymous_cache_without_explicit_local_mapping', 'binary_payload_not_verified'):
    assert counts[(reason, 'candidate_attributed_used')] == 504
    selected = [row['source'] for row in rows if row['reason'] == reason and row['source'] in attributed]
    assert Counter(selected) == Counter({path: 1 for path in attributed})
assert all(digest(Path(path).read_bytes()) == sha for path, sha in before.items())
print(json.dumps({'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'python': platform.python_version(), 'platform': platform.platform(),
    'query_sha256': digest(Path(__file__).read_bytes()), 'input_text_sha256': before,
    'groups': groups, 'record_counts': [{'reason': r, 'disposition': d, 'count': n}
                                       for (r,d),n in sorted(counts.items())],
    'unique_activation_files': len(attributed), 'conservation_passed': True,
    'input_texts_unchanged': True, 'new_excluded_chunks': [],
    'coverage_complete': False, 'drive_disjoint_verified': False,
    'payload_verified': False, 'exit_code': 0}, ensure_ascii=False, indent=2))
