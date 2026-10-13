"""冻结精度候选的数据资格准备；纯CPU离线，不导入模型或数据下载接口。"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

REVISION = '33f9bf447ed3bcb7d545ce13f4226f824214fafb'
FEATURES = ('egomotion', 'camera_cross_left_120fov', 'camera_cross_right_120fov',
            'camera_front_tele_30fov', 'camera_front_wide_120fov')
COLLECTION = ('country', 'month', 'hour_of_day', 'platform_class', 'radar_config')
METADATA_IDENTITIES = {
    'clip_index.parquet': (11132300, '9630a3de5ee057ded194deaa94d3b9b42af6f18c301785706e50141314a6935c'),
    'metadata/feature_presence.parquet': (11209157, 'a9946feb269124e3a7cbfe22d97669585735134f634c2ca97332e7abc4314790'),
    'metadata/data_collection.parquet': (11339400, '9624750cadabf8028264cc1404e75e81777654be9def06b00e35c9f834b173d5'),
}
DATASET_SHA = 'b35f5f1bbb42ba92c83c835f7525c33e8ee915eb5ef071ac3dd972d91b0d9560'
INIT_SHA = '40f7c62c601c74f9b44460745a137db8d4fafdff7d9719b402e962e6d97c22f0'
CSV_BLOB = '65de094a4037902a40bd42de7b17082029b78fe2'
DECISIONS = {
    'ALP15-PRECISION-HOLDOUT-PREP-DECISION-20261009.md': '3efbfea76b5b336d5199c93337863bff49c120f2b3bd92c9919d6f26e979297c',
    'ALP15-PRECISION-HOLDOUT-PREP-DECISION-V2-20261009.md': '32b534ea574677928fca9c6e9277498807f00839fe8de852f14f01d7a9e7a23b',
    'ALP15-PRECISION-HOLDOUT-PREP-DECISION-V3-20261010.md': 'e53284d0ebcba64787b4f6ca4f5d60bbef8e779f62f29d7d1845b1910c1731f8',
}
TEXT_EXTENSIONS = {'.json', '.jsonl', '.csv', '.txt', '.log', '.md', '.yaml', '.yml', '.toml', '.sh', '.py'}
CACHE_EXTENSIONS = {'.pt', '.pth', '.pkl', '.pickle', '.cache'}
UUID = re.compile(r'(?<![0-9a-fA-F])[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}(?![0-9a-fA-F])')
ROLES = ('training', 'calibration', 'monitor', 'development', 'smoke', 'holdout', 'simulation')


class PreparationError(ValueError):
    pass


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': sha256(path)}


def within(path, root):
    return path == root or root in path.parents


def require_identity(path, expected):
    actual = identity(path)
    if (actual['bytes'], actual['sha256']) != expected:
        raise PreparationError('固定来源身份不符: ' + Path(path).name)
    return actual


def require_stable_inputs(metadata_before, metadata_after, csv_before, csv_after):
    if metadata_before != metadata_after or csv_before.get('sha256') != csv_after.get('sha256'):
        raise PreparationError('元数据/CSV运行中变动')


def environment_identity():
    # 读取发行版文件和AST，不import执行physical_ai_av。
    distribution = importlib.metadata.distribution('physical-ai-av')
    dataset = Path(distribution.locate_file('physical_ai_av/dataset.py'))
    init = Path(distribution.locate_file('physical_ai_av/__init__.py'))
    public = None
    for node in ast.parse(init.read_bytes()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets):
            public = ast.literal_eval(node.value)
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'physical_ai_av': distribution.version, 'dataset_sha256': sha256(dataset),
              'init_sha256': sha256(init), 'public_api': public,
              'pandas': importlib.metadata.version('pandas'),
              'pyarrow': importlib.metadata.version('pyarrow'), 'official_api_called': False}
    if (distribution.version != '0.2.0' or result['dataset_sha256'] != DATASET_SHA
            or result['init_sha256'] != INIT_SHA or public != ['HfRepoInterface', 'PhysicalAIAVDatasetInterface']):
        raise PreparationError('固定包来源或公开入口不符')
    return result


def inventory(root, usage_roots, exclusions=()):
    """显式目录清单；symlink不跟随，重复root按绝对词法路径去重。"""
    root = Path(root).resolve()
    excluded = [Path(path).absolute() for path in exclusions]
    raw_roots = [Path(path).absolute() for path in usage_roots]
    if any(path.is_symlink() for path in raw_roots):
        raise PreparationError('usage-root不能是symlink')
    declared = sorted(set(path.resolve() for path in raw_roots), key=str)
    entries, errors, seen = [], [], set()
    for declared_root in declared:
        if not within(declared_root, root) or not within(declared_root.resolve(), root) or declared_root.is_symlink():
            raise PreparationError('usage-root必须是项目内真实目录')
        if not declared_root.is_dir():
            raise PreparationError('usage-root不存在或不是目录')
        def onerror(error):
            errors.append({'reason': 'inventory_error', 'exception_type': type(error).__name__})
        for base, directories, files in os.walk(declared_root, followlinks=False, onerror=onerror):
            paths = [Path(base)] + [Path(base) / name for name in sorted(directories + files)]
            directories[:] = [name for name in directories if not any(within(Path(base) / name, e) for e in excluded)]
            for path in paths:
                if path in seen:
                    continue
                seen.add(path)
                row = {'path': str(path.relative_to(root)), 'declared_root': str(declared_root.relative_to(root))}
                try:
                    stat = path.lstat()
                    row.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
                    if any(within(path, e) for e in excluded):
                        row.update(kind='excluded', reason='output_or_frozen_code_directory')
                    elif path.is_symlink():
                        # 目标只登记hash，避免任意链接文本泄漏秘密。
                        target = path.resolve()
                        row.update(kind='symlink', target_sha256=hashlib.sha256(os.readlink(path).encode()).hexdigest(),
                                   target_in_declared_scope=any(within(target, d) for d in declared), reason='not_followed')
                    elif path.is_dir():
                        row['kind'] = 'directory'
                    elif path.is_file():
                        row.update(kind='file', extension=path.suffix.lower())
                    else:
                        row.update(kind='special', reason='not_read')
                except OSError as error:
                    row.update(kind='unreadable', reason='stat_error', exception_type=type(error).__name__)
                    errors.append(row.copy())
                entries.append(row)
    return sorted(entries, key=lambda row: row['path']), errors, declared


def census_usage(root, usage_roots, metadata, exclusions=()):
    root = Path(root).resolve()
    initial, inventory_errors, declared = inventory(root, usage_roots, exclusions)
    clip_chunks = {clip: int(chunk) for clip, chunk in metadata['chunk'].items()}
    references, unknown, unresolved, mappings = defaultdict(list), [], [], {}
    receipts = []
    chunk_ids, file_counts = {0, 3119}, Counter()
    text_complete = True

    def reference(value, source, location, role='unknown'):
        for match in UUID.finditer(value):
            clip = match.group(0).lower()
            ref = {'source': source['path'], 'location': location, 'source_sha256': source.get('sha256'), 'role': role}
            if clip in clip_chunks:
                if ref not in references[clip]:
                    references[clip].append(ref)
                chunk_ids.add(clip_chunks[clip])
            else:
                unknown.append({'uuid': clip, **ref})

    def classify(value):
        names = [role for role in ROLES if role in value.lower()]
        return names[0] if names else 'unknown'

    def walk_json(value, source, location='$', role='unknown', revision=None, repository=None):
        if isinstance(value, dict):
            explicit_dataset_revision = value.get('dataset_revision')
            revision = value.get('revision', value.get('dataset_revision', revision))
            repository = value.get('repository', repository)
            if ((explicit_dataset_revision is not None and explicit_dataset_revision != REVISION)
                    or (repository == 'nvidia/PhysicalAI-Autonomous-Vehicles' and revision is not None and revision != REVISION)):
                unresolved.append({'source': source['path'], 'location': location, 'reason': 'revision_conflict'})
            elif 'revision' in value and value['revision'] != REVISION and repository != 'nvidia/PhysicalAI-Autonomous-Vehicles':
                unresolved.append({'source': source['path'], 'location': location, 'reason': 'foreign_or_unknown_revision'})
            receipt_path, receipt_bytes, receipt_sha = value.get('path'), value.get('bytes'), value.get('sha256')
            if (revision == REVISION and repository == 'nvidia/PhysicalAI-Autonomous-Vehicles'
                    and isinstance(receipt_path, str) and receipt_path.endswith('.zip')
                    and type(receipt_bytes) is int and receipt_bytes >= 0
                    and isinstance(receipt_sha, str) and re.fullmatch('[0-9a-f]{64}', receipt_sha)):
                target = Path(receipt_path) if Path(receipt_path).is_absolute() else (root / source['path']).parent / receipt_path
                resolved = target.resolve()
                if any(within(resolved, d) for d in declared) and within(resolved, root) and not target.is_symlink():
                    receipts.append({'local_path': str(resolved.relative_to(root)), 'reported_bytes': receipt_bytes,
                                     'reported_sha256': receipt_sha, 'source': source['path'],
                                     'source_sha256': source['sha256'], 'location': location,
                                     'payload_sha256_verified': False, 'official_quote_verified': False})
            # 仅同一对象中的显式clip_id+缓存文件字段建立映射，不根据数组顺序猜。
            clip = value.get('clip_id')
            if isinstance(clip, str) and UUID.fullmatch(clip):
                clip = clip.lower()
            if isinstance(clip, str) and clip in clip_chunks:
                for key in ('cache_path', 'cache_file', 'filename', 'path'):
                    cached = value.get(key)
                    if isinstance(cached, str) and Path(cached).suffix.lower() in CACHE_EXTENSIONS:
                        candidate = Path(cached) if Path(cached).is_absolute() else (root / source['path']).parent / cached
                        if any(within(candidate.resolve(), d) for d in declared):
                            mappings.setdefault(str(candidate.absolute()), set()).add(clip)
            for ordinal, (key, item) in enumerate(value.items()):
                pointer = location + '/object_key[' + str(ordinal) + ']'
                next_role = classify(key) if isinstance(key, str) else role
                if isinstance(key, str):
                    reference(key, source, pointer, next_role)
                walk_json(item, source, pointer + '/value', next_role if next_role != 'unknown' else role, revision, repository)
        elif isinstance(value, list):
            for ordinal, item in enumerate(value):
                walk_json(item, source, location + '/list[' + str(ordinal) + ']', role, revision, repository)
        elif isinstance(value, str):
            reference(value, source, location, role)

    for row in initial:
        if row['kind'] == 'symlink':
            unresolved.append({'source': row['path'], 'reason': 'symlink_not_followed'})
            reference(row['path'], row, 'filename')
            continue
        if row['kind'] != 'file':
            continue
        extension = row['extension']
        file_counts[extension.lstrip('.') or 'no_extension'] += 1
        reference(row['path'], row, 'filename')
        if extension not in TEXT_EXTENSIONS:
            row['read_status'] = 'binary_payload_not_read'
            if extension == '.zip':
                matches = re.findall(r'chunk_?(\d+)', Path(row['path']).name, flags=re.IGNORECASE)
                chunk_ids.update(int(v) for v in matches)
            if extension in CACHE_EXTENSIONS or extension in {'.zip', '.mp4', '.mkv', '.bin', '.safetensors', '.npy', '.npz'}:
                unresolved.append({'source': row['path'], 'reason': 'binary_payload_not_verified'})
            continue
        file_counts['text'] += 1
        path = root / row['path']
        try:
            row['sha256'] = sha256(path)
            # 逐行扫描包括大日志，不按size静默跳过。
            with path.open('r', encoding='utf-8', errors='strict') as stream:
                for number, line in enumerate(stream, 1):
                    reference(line, row, 'line:' + str(number), classify(row['path']))
                    for match in re.finditer(r'(?<![A-Za-z0-9:])/(?:[^\s\"\'<>;,]+)', line):
                        value = match.group(0).rstrip(').]}')
                        candidate = Path(value)
                        if not any(within(candidate.resolve(), d) for d in declared):
                            unresolved.append({'source': row['path'], 'location': 'line:' + str(number),
                                               'reason': 'external_path_reference',
                                               'reference_sha256': hashlib.sha256(value.encode()).hexdigest()})
                    for match in re.finditer(r'\b(dataset_revision|revision)[\"\'\s:=]+([A-Za-z0-9_.-]+)', line):
                        if match.group(2) != REVISION:
                            reason = 'revision_conflict' if match.group(1) == 'dataset_revision' else 'foreign_or_unknown_revision'
                            unresolved.append({'source': row['path'], 'location': 'line:' + str(number), 'reason': reason})
            if extension == '.json':
                with path.open('r', encoding='utf-8') as stream:
                    walk_json(json.load(stream), row)
            elif extension == '.jsonl':
                with path.open('r', encoding='utf-8') as stream:
                    for number, line in enumerate(stream, 1):
                        if line.strip():
                            walk_json(json.loads(line), row, '$line:' + str(number))
            row['read_status'] = 'text_scanned_and_json_parsed' if extension in {'.json', '.jsonl'} else 'text_scanned'
            if sha256(path) != row['sha256']:
                row['read_status'] = 'changed_during_read'
                unresolved.append({'source': row['path'], 'reason': 'changed_during_read'})
                text_complete = False
        except (OSError, UnicodeError, ValueError, RecursionError) as error:
            row.update(read_status='read_or_parse_failed', exception_type=type(error).__name__)
            unresolved.append({'source': row['path'], 'reason': 'read_or_parse_failed', 'exception_type': type(error).__name__})
            text_complete = False
    cache_unresolved = []
    for row in initial:
        if row.get('extension') not in CACHE_EXTENSIONS:
            continue
        path = root / row['path']
        filename_clips = [m.group(0).lower() for m in UUID.finditer(path.name) if m.group(0).lower() in clip_chunks]
        bound = mappings.get(str(path.absolute()), set())
        if not filename_clips and not bound:
            cache_unresolved.append({'source': row['path'], 'reason': 'anonymous_cache_without_explicit_local_mapping'})
        else:
            row['explicit_index_clip_ids'] = sorted(set(filename_clips) | bound)
    unresolved.extend(cache_unresolved)
    final, final_errors, _ = inventory(root, usage_roots, exclusions)
    def signature(rows):
        return [(r['path'], r['kind'], r.get('bytes'), r.get('mtime_ns'), r.get('target_sha256')) for r in rows]
    stable = signature(initial) == signature(final) and not any(r['reason'] == 'changed_during_read' for r in unresolved)
    # 冻结尾端清单与文本hash，避免相同长度或mtime的并发写入被忽略。
    for row in initial:
        if 'sha256' in row:
            try:
                row['final_sha256'] = sha256(root / row['path'])
                stable = stable and row['final_sha256'] == row['sha256']
            except OSError:
                stable = False
    if not stable:
        unresolved.append({'source': '', 'reason': 'scan_changed'})
    return {
        'dataset_revision': REVISION, 'coverage_complete': False, 'drive_disjoint_verified': False,
        'used_clip_ids': sorted(references), 'used_or_referenced_clip_ids': sorted(references),
        'used_chunk_ids': sorted(chunk_ids), 'excluded_chunk_ids': sorted(chunk_ids),
        'audit_evidence': dict(sorted(references.items())), 'unmapped_uuid_refs': unknown,
        'existing_receipt_evidence': receipts,
        'unresolved_assets': unresolved, 'unresolved_cache_indices': cache_unresolved,
        'unresolved_cache_count': len(cache_unresolved),
        'unresolved_external_path_count': sum(r['reason'] == 'external_path_reference' for r in unresolved),
        'declared_roots_inventory_complete': not inventory_errors and not final_errors,
        'all_declared_text_parsed': text_complete, 'scan_stable': stable,
        'text_parse_contract': 'UTF8_reference_scan_for_all_text_plus_JSON_grammar_for_json_and_jsonl',
        'file_counts': dict(sorted(file_counts.items())), 'scan_manifest': initial,
        'final_inventory': final, 'inventory_errors': inventory_errors + final_errors,
        'declared_roots': [str(d) for d in declared],
        'limitations': ['coverage_requires_independent_scope_review', 'binary_payloads_not_read',
                        'input_window_qc_pending', 'drive_session_ids_unavailable'],
    }


def counts(frame):
    result = {'clips': len(frame), 'chunks': int(frame['chunk'].nunique()),
              'missing_by_column': {column: int(frame[column].isna().sum()) for column in frame.columns}}
    for column in ('split', 'chunk') + COLLECTION:
        groups = {str(k): int(v) for k, v in frame.groupby(column, dropna=True).size().items()}
        if sum(groups.values()) + result['missing_by_column'][column] != len(frame):
            raise PreparationError('分组计数与缺失计数合计不符')
        result['by_' + column] = groups
    return result


def eligibility(metadata, excluded):
    valid = metadata[metadata['clip_is_valid'] & metadata[list(FEATURES)].all(axis=1)]
    baseline = valid[~valid['chunk'].isin({0, 3119})]
    remaining = valid[~valid['chunk'].isin(excluded)].sort_index().copy()
    remaining['t0_us'] = 5100000
    remaining['radar_config_is_missing'] = remaining['radar_config'].isna()
    remaining['state'] = 'candidate_only_pending_usage_coverage_and_input_window_qc'
    return remaining, {'all_metadata': counts(metadata), 'qualified': counts(valid),
                       'known_chunk_only_baseline': counts(baseline), 'after_usage_exclusion': counts(remaining),
                       'official_test_priority_pool': counts(remaining[remaining['split'] == 'test'])}


def require_production_counts(summary):
    baseline = summary['known_chunk_only_baseline']
    if (summary['all_metadata']['clips'] != 306152 or summary['qualified']['clips'] != 306152
            or baseline['clips'] != 305952 or baseline['chunks'] != 3144
            or baseline['by_split'] != {'train': 153425, 'val': 90928, 'test': 61599}
            or summary['all_metadata']['missing_by_column']['radar_config'] != 145391
            or summary['qualified']['missing_by_column']['radar_config'] != 145391):
        raise PreparationError('官方元数据基线计数不符，见metadata-counts.json')


def file_plan(chunks, allowed_chunks, templates, csv_identity, metadata_root, project_root=None, receipt_evidence=()):
    rows = []
    metadata_root = Path(metadata_root).resolve()
    for chunk in sorted(chunks):
        paths = set()
        for feature in FEATURES:
            relative = resolve_chunk_path(templates[feature], int(chunk), allowed_chunks)
            if relative in paths:
                raise PreparationError('五feature路径碰撞')
            paths.add(relative)
            path = metadata_root / relative
            if not within(path.resolve(), metadata_root):
                raise PreparationError('候选ZIP路径symlink逃逸')
            existing_receipts = []
            if project_root is not None:
                for receipt in receipt_evidence:
                    if (Path(project_root) / receipt['local_path']).resolve() == path.resolve():
                        existing_receipts.append(receipt)
            reported = {(r['reported_bytes'], r['reported_sha256']) for r in existing_receipts}
            receipt_state = 'no_bound_fixed_revision_receipt'
            local_sha = None
            if len(reported) > 1:
                receipt_state = 'conflicting_receipts_pending_review'
            elif len(reported) == 1:
                size, local_sha = next(iter(reported))
                receipt_state = 'bound_receipt_payload_sha_unverified'
                if not path.is_file() or size != path.stat().st_size:
                    receipt_state = 'bound_receipt_missing_or_size_mismatch'
            rows.append({'chunk': int(chunk), 'feature': feature, 'relative_path': relative,
                         'local_exists': path.is_file(), 'local_bytes': path.stat().st_size if path.is_file() else None,
                         'local_sha256': local_sha, 'local_sha_source': 'receipt_reported_unverified_payload' if local_sha else 'not_hashed_binary',
                         'receipt_status': receipt_state, 'receipt_sources': json.dumps([r['source'] for r in existing_receipts]), 'official_bytes': None,
                         'official_sha256': None, 'quote_status': 'pending_fixed_revision_query',
                         'resolution_method': 'verified_features_csv_chunk_path_format',
                         'features_csv_sha256': csv_identity['sha256'], 'original_template': templates[feature],
                         'official_api_called': False})
    return rows


def draft_selection(candidates, requested, seed):
    pool = candidates[candidates['split'] == 'test']
    rng = random.Random(seed)
    chunks = sorted(int(c) for c in pool['chunk'].unique())
    rng.shuffle(chunks)
    order = []
    for chunk in chunks:
        clips = sorted(pool[pool['chunk'] == chunk].index)
        rng.shuffle(clips)
        order.extend(clips)
    selected = order[:requested]
    return {'state': 'draft_pending_coverage_review_quote_and_input_qc', 'sampling_seed': seed,
            'python': platform.python_version(), 'algorithm': 'sorted_chunks_then_rng_shuffle_then_sorted_clip_rng_shuffle',
            'requested': requested, 'available': len(order), 'insufficient': len(order) < requested,
            'clip_ids': selected, 'full_candidate_order': order, 'chunk_order': chunks,
            'counts': counts(pool.loc[selected]), 'drive_disjoint_verified': False}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('root', 'metadata-root', 'output'):
        parser.add_argument('--' + flag, required=True, type=Path)
    parser.add_argument('--usage-root', required=True, action='append', type=Path)
    parser.add_argument('--draft-feasibility-clips', type=int, choices=(100,))
    parser.add_argument('--sampling-seed', type=int, choices=(20261009,), default=20261009)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        print('output已存在，拒绝覆盖', file=sys.stderr)
        return 2
    if not args.root.is_dir() or not within(output.resolve(), args.root.resolve()):
        print('output必须在存在的显式项目root内', file=sys.stderr)
        return 2
    # 存在目录的拒覆盖检查先于任何来源读取；失败产物仅写这个新目录。
    output.mkdir(parents=True, exist_ok=False)
    code_root = Path(__file__).resolve().parents[1]
    manifest = {'argv': list(sys.argv if argv is None else [str(Path(__file__))] + list(argv)),
                'started_utc': utc_now(), 'dataset_revision': REVISION, 'official_api_called': False,
                'coverage_complete': False, 'drive_disjoint_verified': False,
                'script_sha256': sha256(Path(__file__)), 'decisions': DECISIONS,
                'status': 'failed', 'diagnostics': [], 'decision_source_root': str(code_root),
                'root': str(args.root.absolute()), 'metadata_root': str(args.metadata_root.absolute()),
                'usage_roots': [str(p.absolute()) for p in args.usage_root], 'stage': 'source_preflight'}
    manifest['execution_controls'] = {'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE') == '1',
                                      'hf_datasets_offline': os.environ.get('HF_DATASETS_OFFLINE') == '1',
                                      'cuda_visible_devices_empty': os.environ.get('CUDA_VISIBLE_DEVICES') == ''}
    ledger = None
    exit_code = 1
    try:
        root = args.root.resolve(strict=True)
        metadata_root = args.metadata_root.resolve(strict=True)
        if not within(metadata_root, root) or not within(output.resolve(), root):
            raise PreparationError('metadata/output必须在显式项目root内')
        for name, expected in DECISIONS.items():
            if sha256(code_root / 'docs/baselines' / name) != expected:
                raise PreparationError('冻结决策SHA不符')
        manifest['environment'] = environment_identity()
        manifest['metadata_expected'] = METADATA_IDENTITIES
        manifest['features_csv_expected'] = {'bytes': 7076, 'git_blob': CSV_BLOB}
        manifest['metadata_before'] = {relative: identity(metadata_root / relative) for relative in METADATA_IDENTITIES}
        manifest['features_csv_before'] = identity(metadata_root / 'features.csv') if (metadata_root / 'features.csv').is_file() else {'missing': True}
        metadata, before = load_metadata(metadata_root)
        manifest['metadata_before'] = before
        templates, csv_before = read_templates(metadata_root / 'features.csv')
        manifest['features_csv_before'] = csv_before
        manifest['stage'] = 'usage_census'
        exclusions = [output.resolve(), code_root] if code_root != root else [output.resolve(), Path(__file__).resolve(), code_root / 'tests/test_prepare_alp15_precision_holdout.py']
        manifest['exclusions'] = [{'path': str(p), 'reason': 'output_or_frozen_preparation_code'} for p in exclusions]
        ledger = census_usage(root, args.usage_root, metadata, exclusions)
        scan = ledger.pop('scan_manifest')
        final = ledger.pop('final_inventory')
        with (output / 'scan-manifest.jsonl').open('w', encoding='utf-8') as stream:
            for row in scan:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        write_json(output / 'scan-final-inventory.json', final)
        write_json(output / 'usage-ledger.json', ledger)
        references = []
        for clip, sources in ledger['audit_evidence'].items():
            for source in sources:
                references.append({'clip_id': clip, 'chunk': int(metadata.at[clip, 'chunk']), **source})
        write_csv(output / 'referenced-clips.csv', references, ('clip_id', 'chunk', 'source', 'location', 'source_sha256', 'role'))
        write_csv(output / 'unresolved-assets.csv', ledger['unresolved_assets'], ('source', 'reason', 'location', 'exception_type', 'reference_sha256'))
        candidates, summary = eligibility(metadata, ledger['used_chunk_ids'])
        manifest['counts'] = summary
        write_json(output / 'metadata-counts.json', summary)
        require_production_counts(summary)
        manifest['stage'] = 'candidate_file_plan'
        candidates[list(('split', 'chunk') + COLLECTION + ('radar_config_is_missing', 't0_us', 'state'))].to_csv(output / 'eligible-candidates.csv', index_label='clip_id')
        plan = file_plan(candidates['chunk'].unique(), set(int(c) for c in metadata['chunk'].unique()), templates, csv_before,
                         metadata_root, root, ledger['existing_receipt_evidence'])
        plan_fields = ('chunk', 'feature', 'relative_path', 'local_exists', 'local_bytes', 'local_sha256', 'local_sha_source',
                       'receipt_status', 'receipt_sources', 'official_bytes', 'official_sha256', 'quote_status', 'resolution_method',
                       'features_csv_sha256', 'original_template', 'official_api_called')
        write_csv(output / 'candidate-file-plan.csv', plan, plan_fields)
        if args.draft_feasibility_clips:
            write_json(output / 'draft-feasibility-selection.json', draft_selection(candidates, 100, args.sampling_seed))
        manifest['metadata_after'] = {relative: identity(metadata_root / relative) for relative in METADATA_IDENTITIES}
        manifest['features_csv_after'] = identity(metadata_root / 'features.csv')
        require_stable_inputs(before, manifest['metadata_after'], csv_before, manifest['features_csv_after'])
        if not ledger['scan_stable'] or not ledger['declared_roots_inventory_complete']:
            raise PreparationError('扫描变动或目录读取失败，台账仅为待复核部分结果')
        manifest['status'] = 'completed_usage_census_pending_coverage_review'
        manifest['stage'] = 'completed'
        manifest['counts'] = summary
        manifest['draft_generated'] = bool(args.draft_feasibility_clips)
        exit_code = 0
    except Exception as error:
        # 不序列化任意异常文本，避免日志内容、键名或凭据进入失败报告。
        manifest['diagnostics'].append({'exception_type': type(error).__name__,
                                        'reason': str(error) if isinstance(error, PreparationError) else 'unexpected_failure_redacted'})
    finally:
        if 'metadata_before' in manifest:
            manifest['metadata_after'] = {}
            for relative in METADATA_IDENTITIES:
                try:
                    manifest['metadata_after'][relative] = identity(metadata_root / relative)
                except OSError as error:
                    manifest['metadata_after'][relative] = {'exception_type': type(error).__name__}
            try:
                manifest['features_csv_after'] = identity(metadata_root / 'features.csv')
            except OSError as error:
                manifest['features_csv_after'] = {'exception_type': type(error).__name__}
            before_csv = manifest.get('features_csv_before', {})
            if exit_code == 0:
                try:
                    require_stable_inputs(manifest['metadata_before'], manifest['metadata_after'], before_csv, manifest['features_csv_after'])
                except PreparationError:
                    manifest['status'] = 'failed'
                    manifest['diagnostics'].append({'reason': 'input_changed_at_final_verification'})
                    exit_code = 1
        manifest['finished_utc'] = utc_now()
        manifest['exit_code'] = exit_code
        report_details = ''
        if ledger is not None:
            report_details = '\n\n扫描计数：' + json.dumps(ledger['file_counts'], ensure_ascii=False)
            report_details += '\n\n扫描证据：' + json.dumps({key: ledger[key] for key in (
                'declared_roots_inventory_complete', 'all_declared_text_parsed', 'scan_stable',
                'unresolved_cache_count', 'unresolved_external_path_count')}, ensure_ascii=False)
            report_details += '\n\n保守引用clip数：' + str(len(ledger['used_clip_ids'])) + '；排除chunk数：' + str(len(ledger['used_chunk_ids']))
        if 'counts' in manifest:
            report_details += '\n\n资格/缺失计数：' + json.dumps({layer: {'clips': values['clips'], 'missing_by_column': values['missing_by_column']}
                for layer, values in manifest['counts'].items()}, ensure_ascii=False)
        (output / 'report.md').write_text(
            '# 未见数据资格准备\n\n状态：' + manifest['status'] +
            '\n\ncoverage_complete=False；drive_disjoint_verified=False。纯CSV模板解析，official_api_called=False。'
            '\n\n本输出不是最终冻结测试名单；窗口QC、范围验收、文件报价、采集前统计方案均待完成。'
            '\n\nradar_config 缺失保留为原始空字段与布尔标记；不影响摄像头/egomotion 输入资格，不证明场景覆盖充足。'
            + report_details + '\n\n诊断：' + json.dumps(manifest['diagnostics'], ensure_ascii=False) + '\n', encoding='utf-8')
        manifest['artifact_sha256'] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'run-manifest.json'}
        write_json(output / 'run-manifest.json', manifest)
    print(manifest['status'])
    return exit_code


def normalize_metadata(clip_index, presence, collection):
    import pandas as pd
    frames = []
    for original in (clip_index, presence, collection):
        frame = original.copy()
        if not frame.columns.is_unique:
            raise PreparationError('metadata列不唯一')
        if 'clip_id' in frame.columns:
            if frame.index.name == 'clip_id' and list(frame.index) != list(frame['clip_id']):
                raise PreparationError('clip_id列与索引冲突')
            frame = frame.set_index('clip_id')
        if frame.index.name != 'clip_id' or not frame.index.is_unique or frame.index.hasnans:
            raise PreparationError('clip_id索引缺失、重复或空值')
        if any(not isinstance(value, str) or not value for value in frame.index):
            raise PreparationError('clip_id必须是非空字符串')
        if any(UUID.fullmatch(value) is None or value != value.lower() for value in frame.index):
            raise PreparationError('metadata clip_id必须为小写canonical UUID')
        if not pd.api.types.is_string_dtype(frame.index.dtype):
            raise PreparationError('clip_id必须为字符串dtype')
        frames.append(frame)
    index, presence, collection = frames
    if set(index.index) != set(presence.index) or set(index.index) != set(collection.index):
        raise PreparationError('三个metadata键集合不一致')
    required = ((index, ('chunk', 'clip_is_valid', 'split')),
                (presence, FEATURES), (collection, COLLECTION))
    for frame, columns in required:
        if any(column not in frame for column in columns):
            raise PreparationError('metadata缺少必需列')
        non_nullable = [column for column in columns if not (frame is collection and column == 'radar_config')]
        if frame[non_nullable].isna().any().any():
            raise PreparationError('metadata必需列含空值')
    if (pd.api.types.is_bool_dtype(index['chunk'].dtype)
            or not pd.api.types.is_integer_dtype(index['chunk'].dtype)
            or (index['chunk'] < 0).any()):
        raise PreparationError('chunk必须是非负整数dtype，拒绝bool/float/string')
    for frame, columns in ((index, ('clip_is_valid',)), (presence, FEATURES)):
        for column in columns:
            if not pd.api.types.is_bool_dtype(frame[column].dtype):
                raise PreparationError('资格列必须为严格布尔dtype')
    if not pd.api.types.is_string_dtype(index['split'].dtype) or any(not isinstance(v, str) or not v for v in index['split']):
        raise PreparationError('split必须保留非空真实字符串')
    result = index[['chunk', 'clip_is_valid', 'split']].join(presence[list(FEATURES)]).join(collection[list(COLLECTION)])
    result['chunk'] = [int(v) for v in result['chunk']]
    return result


def load_metadata(root, identities=METADATA_IDENTITIES):
    import pandas as pd
    before, frames = {}, []
    for relative, expected in identities.items():
        path = root / relative
        if not within(path.resolve(), root.resolve()):
            raise PreparationError('metadata路径逃逸')
        before[relative] = require_identity(path, expected)
        frames.append(pd.read_parquet(path))
    return normalize_metadata(*frames), before


def read_templates(path, expected_bytes=7076, expected_blob=CSV_BLOB):
    raw = Path(path).read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()
    if len(raw) != expected_bytes or blob != expected_blob:
        raise PreparationError('features.csv固定bytes/Git blob不符')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8')), strict=True))
    if not rows or 'feature' not in rows[0] or 'chunk_path' not in rows[0]:
        raise PreparationError('features.csv缺少列')
    features = [row['feature'] for row in rows]
    if any(not isinstance(f, str) or not f for f in features) or len(features) != len(set(features)):
        raise PreparationError('features.csv重复feature')
    templates = {row['feature']: row['chunk_path'] for row in rows if row['feature'] in FEATURES}
    if set(templates) != set(FEATURES) or any(not isinstance(v, str) or not v for v in templates.values()):
        raise PreparationError('features.csv五feature不完整或chunk_path为空')
    return templates, {'bytes': len(raw), 'git_blob': blob, 'sha256': hashlib.sha256(raw).hexdigest()}


def resolve_chunk_path(template, chunk_id, allowed_chunks):
    """只解析经校验CSV的chunk_path，不调用内部Features接口。"""
    if type(chunk_id) is not int or chunk_id < 0 or chunk_id not in allowed_chunks:
        raise ValueError('chunk_id必须是metadata中的非负Python整数')
    if not isinstance(template, str) or not template:
        raise ValueError('chunk_path必须是非空字符串')
    fields = []
    for literal, field, spec, conversion in string.Formatter().parse(template):
        if literal and not re.fullmatch(r'[A-Za-z0-9_./-]+', literal):
            raise ValueError('模板字面量含不允许字符')
        if field is not None:
            if field != 'chunk_id' or conversion is not None:
                raise ValueError('模板仅允许chunk_id，无conversion/属性/索引')
            if spec not in ('', 'd'):
                match = re.fullmatch(r'0?([1-9][0-9]?)d', spec)
                if not match or not 1 <= int(match[1]) <= 12:
                    raise ValueError('不允许的整数format_spec')
            fields.append(field)
    if fields != ['chunk_id']:
        raise ValueError('模板必须恰有一次chunk_id')
    # 格式化前后均拒绝路径穿越，不使用normalize吞掉痕迹。
    def safe(value):
        if (not value or value.startswith('/') or not re.fullmatch(r'[A-Za-z0-9_./-]+', value)
                or any(part in ('', '.', '..') for part in value.split('/'))
                or not value.endswith('.zip') or PurePosixPath(value).is_absolute()):
            raise ValueError('必须是安全POSIX相对ZIP路径')
    safe(re.sub(r'\{chunk_id(?::[^{}]*)?\}', '0', template))
    result = template.format(chunk_id=chunk_id)
    safe(result)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
