"""仅导出本地实验文本源码；拒绝凭据和覆盖，保留原字节及历史来源索引。"""
import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

SUFFIXES = {'.py', '.sh', '.ps1'}
SECRET_PATTERNS = [
    rb'\bhf_[A-Za-z0-9]{20,}\b',
    rb'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}\b',
    rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b',
    rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    rb'https?://[^\s/:]+:[^\s/@]+@',
    rb'sshpass\s+-p\s+[\x27\x22][^\x27\x22]+[\x27\x22]',
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def scan(data, label):
    for pattern in SECRET_PATTERNS:
        hit = re.search(pattern, data)
        if hit:
            line = data[:hit.start()].count(b'\n') + 1
            raise ValueError(f'Sensitive pattern: {label}:{line}; content withheld')


def put(root, relative, data):
    path = root / relative
    if '..' in PurePosixPath(relative).parts or Path(relative).is_absolute():
        raise ValueError('Unsafe destination')
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError('Symlink destination forbidden')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError('Existing destination differs: ' + relative)
    else:
        with path.open('xb') as handle:
            handle.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1]
    source = args.source_root.resolve()
    if target == source or target.is_relative_to(source):
        raise ValueError('Use an independent export repository')
    unique, active, origin_count = {}, {}, 0

    def record(data, origin, active_path=None):
        nonlocal origin_count
        scan(data, origin['path'])
        digest = sha(data)
        entry = unique.setdefault(digest, {'data': data, 'name': PurePosixPath(origin['path']).name,
                                           'origins': []})
        entry['origins'].append(origin)
        origin_count += 1
        if active_path:
            active[active_path] = data

    directories = ['scripts', 'tests', 'deliverables/alp15_windows_949', 'docs/baselines/evidence']
    for directory in directories:
        for path in sorted((source / directory).rglob('*')):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            if path.is_symlink() or any(p.startswith('.') or p == '__pycache__' for p in path.relative_to(source).parts):
                continue
            relative = path.relative_to(source).as_posix()
            chosen = relative if not relative.startswith('docs/') else None
            record(path.read_bytes(), {'source': 'local_workspace', 'path': relative}, chosen)
    if (source / 'main.py').exists():
        record((source / 'main.py').read_bytes(), {'source': 'local_workspace', 'path': 'main.py'})
    with ZipFile(args.handoff) as bundle:
        for info in sorted(bundle.infolist(), key=lambda item: item.filename):
            name = info.filename
            if '..' in PurePosixPath(name).parts or name.startswith('/'):
                raise ValueError('Unsafe handoff member')
            if PurePosixPath(name).suffix not in SUFFIXES or not name.startswith('Lighten_VLA/'):
                continue
            relative = name.removeprefix('Lighten_VLA/')
            if not relative.startswith(('compression/', 'dashboard/', 'handover/')):
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Handoff symlink forbidden')
            active_path = None
            if relative.startswith('compression/') and len(PurePosixPath(relative).parts) == 2:
                active_path = relative
            elif relative.startswith(('dashboard/', 'handover/')):
                active_path = relative
            record(bundle.read(info), {'source': 'handoff_20261004', 'path': relative}, active_path)
        # 研究快照的来源、commit及原始文件哈希，不包含实验结果或模型。
        prefix = 'Lighten_VLA/compression/research/framework_budget_20261002/'
        for name in ['source_versions.json', 'download_sha256.json', 'compatibility_sources.json', 'flashdrive_pyproject.toml']:
            data = bundle.read(prefix + name)
            scan(data, name)
            put(target, 'archive/framework_sources/' + name, data)
    # 这两个快照的原字节哈希是后续AWQ门禁要求；ZIP版本另存archive，不归一行尾。
    for name in ['run_mlp_experiment.py', 'mlp_pruning.py']:
        active['compression/' + name] = (source / 'docs/baselines/evidence/quant_route_20261008/full20/code' / name).read_bytes()
    for relative, data in sorted(active.items()):
        put(target, relative, data)
    catalog = []
    for digest, entry in sorted(unique.items()):
        relative = 'archive/sha256/' + digest + '/' + entry['name']
        put(target, relative, entry['data'])
        catalog.append({'sha256': digest, 'bytes': len(entry['data']), 'archive_path': relative,
                        'origins': entry['origins']})
    assets = [
        ('deliverables/alp15_windows_949', ['small-download-plan.json', 'test-candidates.csv', 'README.md']),
        ('docs/baselines/evidence/full_pool_download_20261011', ['candidate-snapshot.json', 'download-plan.json',
         'test-candidates.csv', 'features.csv', 'README.md']),
    ]
    for directory, names in assets:
        for name in names:
            relative = directory + '/' + name
            data = (source / relative).read_bytes()
            scan(data, relative)
            put(target, relative, data)
    # 自编基线/数据操作说明供早期setup脚本使用；不复制实验结果、日志或执行环境凭据。
    for path in sorted((source / 'docs').rglob('*.md')):
        relative = path.relative_to(source).as_posix()
        if path.parent == source / 'docs/baselines' and path.name in {
                'ALP15-PRECISION-HOLDOUT-PREP-DECISION-20261009.md',
                'ALP15-PRECISION-HOLDOUT-PREP-DECISION-V2-20261009.md',
                'ALP15-PRECISION-HOLDOUT-PREP-DECISION-V3-20261010.md'}:
            data = path.read_bytes()
            scan(data, relative)
            put(target, relative, data)
    manifest = {'schema': 'alp15-source-export-v1', 'source_records': origin_count,
                'unique_source_sha256': len(unique), 'active_source_files': len(active),
                'scope': 'Available local workspace plus 20261004 handoff; no live L20 collection',
                'excluded': ['weights', 'raw sensor data', 'runtime caches', 'logs', 'results',
                             'upstream full repositories', 'credentials'],
                'active_sha256': {name: sha(data) for name, data in sorted(active.items())},
                'sources': catalog}
    put(target, 'archive/SOURCE_INDEX.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps({'source_records': origin_count, 'unique_source_sha256': len(unique),
                      'active_source_files': len(active), 'credential_pattern_hits': 0}))


if __name__ == '__main__':
    main()
