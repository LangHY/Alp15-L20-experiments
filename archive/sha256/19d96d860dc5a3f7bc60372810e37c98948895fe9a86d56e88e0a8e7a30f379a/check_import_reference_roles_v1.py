"""仅核查一个冻结结果中 import 指纹引用的角色，不打开被引用文件。"""
import ast
import csv
import hashlib
import io
import json
import os
import re
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/tuqiang/lang/Lighten_VLA')
CENSUS = ROOT / 'experiments/precision_holdout_prep_20261010_resume_v3'
RESULT = ROOT / 'experiments/jetson_300ms_20261009_hybrid_full20_v3/result.json'
PRODUCER = ROOT / 'experiments/jetson_300ms_20261009_hybrid_m1_code_v3/scripts/run_alp15_awq_hybrid_eval.py'
EXPECTED = {
    CENSUS / 'unresolved-assets.csv': 'f398e3f054620d37544217e766fbafba2d6d052f084df75e22df9f41259a09c7',
    CENSUS / 'scan-manifest.jsonl': '2177978ecf1ab7b3efe8150fb6fcfa7aabe1d60b32b4354584c24aa28b06014b',
    RESULT: 'a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6',
    PRODUCER: 'fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd',
}


def read(path):
    assert stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED[path]
    return raw.decode('utf-8', errors='strict')


assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1'
started = datetime.now(timezone.utc).isoformat()
texts = {p: read(p) for p in EXPECTED}
result = json.loads(texts[RESULT])
assert result['code_sha256'][str(PRODUCER)] == EXPECTED[PRODUCER]
scan = [json.loads(line) for line in texts[CENSUS / 'scan-manifest.jsonl'].splitlines()]
result_rel = str(RESULT.relative_to(ROOT))
assert [r['sha256'] for r in scan if r['path'] == result_rel] == [EXPECTED[RESULT]]
rows = list(csv.DictReader(io.StringIO(texts[CENSUS / 'unresolved-assets.csv'])))
external = [r for r in rows if r['reason'] == 'external_path_reference']
assert len(external) == 63177
selected = [r for r in external if r['source'] == result_rel]
assert len(selected) == 6280
source = texts[PRODUCER]
tree = ast.parse(source)
functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
function = functions['imported_source_receipt']
source_lines = source.splitlines()
function_text = '\n'.join(source_lines[function.lineno - 1:function.end_lineno])
assert 'sys.modules.items()' in function_text and "getattr(module, '__file__', None)" in function_text
assert 'receipt[str(canonical)] = sha256_file(canonical)' in function_text
assert "result['actual_import_sha256_start'] = source_receipt()" in source
assert "result['actual_import_sha256'] = source_receipt()" in source
role_by_line, active = {}, None
for number, line in enumerate(texts[RESULT].splitlines(), 1):
    if line in ('  "actual_import_sha256_start": {', '  "actual_import_sha256": {'):
        assert active is None
        active = json.loads('{' + line.strip()[:-1] + 'null}')
        active = next(iter(active))
        continue
    if active is not None and line in ('  },', '  }'):
        active = None
        continue
    if active is None:
        continue
    assert line.startswith('    "')
    pair = json.loads('{' + line.strip().rstrip(',') + '}')
    assert len(pair) == 1
    path, digest = next(iter(pair.items()))
    assert result[active][path] == digest and re.fullmatch('[0-9a-f]{64}', digest)
    role_by_line['line:' + str(number)] = (active, path, digest)
assert active is None
assert Counter(v[0] for v in role_by_line.values()) == Counter({
    'actual_import_sha256_start': 2839, 'actual_import_sha256': 3432})
dispositions = []
for row in selected:
    role = role_by_line.get(row['location'])
    if role is None:
        continue
    key, path, digest = role
    # 保留 census 的引用字符串语法；必须精确匹配该行唯一 key 的 hash。
    line = texts[RESULT].splitlines()[int(row['location'].split(':')[1]) - 1]
    matches = re.findall(r'(?<![A-Za-z0-9:])/(?:[^\s\"\'<>;,]+)', line)
    assert len(matches) == 1 and matches[0].rstrip(').]}') == path
    assert hashlib.sha256(path.encode()).hexdigest() == row['reference_sha256']
    dispositions.append({**row, 'json_field': key,
        'declared_import_digest': digest, 'disposition': 'proven_non_usage',
        'scope': 'this_reference_is_a_code_fingerprint_not_a_data_input',
        'referenced_file_read': False, 'payload_verified': False})
assert len(dispositions) == 6271
assert len({(r['location'], r['reference_sha256']) for r in dispositions}) == 6271
for path in EXPECTED:
    read(path)
print(json.dumps({'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'query_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'input_sha256': {str(p.relative_to(ROOT)): sha for p, sha in EXPECTED.items()},
    'producer_source': source, 'function_lines': [function.lineno, function.end_lineno],
    'dispositions': dispositions, 'external_total': 63177,
    'selected_source_total': 6280, 'selected_source_pending': 9,
    'other_external_pending': 56897, 'role_classified': 6271,
    'overall_external_pending': 56906, 'input_texts_unchanged': True,
    'coverage_complete': False, 'drive_disjoint_verified': False,
    'limitations': ['does_not_prove_imported_code_never_reads_data',
        'does_not_rehash_import_targets_or_validate_payload',
        'does_not_classify_other_references_in_this_or_other_results'],
    'exit_code': 0}, ensure_ascii=False, indent=2))
