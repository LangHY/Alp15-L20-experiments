"""补充七组实际 argv/root/output 绑定，只读取已声明文本。"""
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/tuqiang/lang/Lighten_VLA')
QUERY = ROOT / 'experiments/precision_usage_gap_20261010_v1/newdata-teacher-lineage-query-v2.json'
EXPECTED_QUERY_SHA = 'e062044eab401ab989c6183e505e69d00f663c0996e534ca7d9a8cdc730fbab9'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1'
assert stat.S_ISREG(QUERY.lstat().st_mode) and not QUERY.is_symlink()
started = datetime.now(timezone.utc).isoformat()
raw = QUERY.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED_QUERY_SHA
previous = json.loads(raw.decode('utf-8'))
bindings = []
for group in previous['groups']:
    out = ROOT / 'experiments' / group['experiment']
    path = out / 'provenance.json'
    assert stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == previous['input_text_sha256'][str(path)]
    provenance = json.loads(raw.decode('utf-8'))
    argv = provenance['argv']
    assert argv.count('--root') == argv.count('--output') == 1
    root_arg = Path(argv[argv.index('--root') + 1])
    output_arg = Path(argv[argv.index('--output') + 1])
    assert root_arg.is_absolute() and root_arg == ROOT
    assert output_arg.is_absolute() and output_arg == out
    # 历史 argv[0] 是项目内相对脚本路径；不声称现在的同名源码仍是历史版本。
    script_arg = Path(argv[0])
    assert script_arg == Path('compression/run_interface_recovery.py')
    source_path = out / 'code/run_interface_recovery.py'
    assert stat.S_ISREG(source_path.lstat().st_mode) and not source_path.is_symlink()
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert source_sha == group['producer_sha256'] == provenance['code_sha256'][source_path.name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    bindings.append({'experiment': group['experiment'], 'provenance_sha256': sha,
        'argv0': str(script_arg), 'root_argument': str(root_arg),
        'output_argument': str(output_arg), 'source_snapshot': str(source_path.relative_to(ROOT)),
        'source_snapshot_sha256': source_sha, 'payload_verified': False})
assert len(bindings) == 7
assert hashlib.sha256(QUERY.read_bytes()).hexdigest() == EXPECTED_QUERY_SHA
print(json.dumps({'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'query_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'parent_query_sha256': EXPECTED_QUERY_SHA, 'bindings': bindings,
    'input_texts_unchanged': True, 'coverage_complete': False, 'exit_code': 0},
    ensure_ascii=False, indent=2))
