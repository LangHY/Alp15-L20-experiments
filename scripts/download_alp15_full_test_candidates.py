"""完整候选池：默认仅展示计划；大文件下载必须由用户显式启动。"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath

REVISION = '33f9bf447ed3bcb7d545ce13f4226f824214fafb'
REPO = 'nvidia/PhysicalAI-Autonomous-Vehicles'
BASE = Path(__file__).resolve().parents[1] / 'docs/baselines/evidence/full_pool_download_20261011'
os.environ['HF_HUB_DISABLE_XET'] = '1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(snapshot, plan=None):
    if snapshot['dataset'] != REPO or snapshot['revision'] != REVISION:
        raise ValueError('数据仓或revision不符')
    if snapshot['clip_count'] != 61301 or len(set(snapshot['chunks'])) != 633:
        raise ValueError('不是已批准的完整候选快照')
    if set(snapshot['chunks']) & set(snapshot['excluded_chunks']):
        raise ValueError('与旧排除块重叠')
    paths = [r['path'] for r in snapshot['files']]
    if len(paths) != 3165 or len(set(paths)) != 3165:
        raise ValueError('文件不全或重复')
    for p in paths:
        parts = PurePosixPath(p).parts
        if not p.endswith('.zip') or p.startswith('/') or '..' in parts or '\\' in p:
            raise ValueError('非法相对路径')
    if plan is not None:
        if plan['revision'] != REVISION or plan['dataset'] != REPO:
            raise ValueError('报价身份不符')
        if {r['path'] for r in plan['files']} != set(paths) or len(plan['files']) != 3165:
            raise ValueError('报价文件集合不符')
        for r in plan['files']:
            if type(r['size']) is not int or r['size'] <= 0 or not re.fullmatch('[0-9a-f]{64}', r['sha256']):
                raise ValueError('缺少真实size/SHA256')
        if sum(r['size'] for r in plan['files']) != plan['total_bytes']:
            raise ValueError('总容量不符')


def prepare_output(output, plan, plan_sha):
    out = Path(output).expanduser()
    if not out.is_absolute() or '..' in out.parts:
        raise ValueError('输出目录必须是无..的独立绝对路径')
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError('输出目录祖先不允许symlink')
    out = out.resolve()
    if out in (Path('/'), Path.home().resolve()):
        raise ValueError('不能把根目录或用户主目录作为输出')
    cache = out / '.cache'
    if cache.exists() and not cache.is_dir():
        raise ValueError('缓存路径不是目录，保留不覆盖')
    # 只检查缓存的目录项类型，不打开缓存payload或跟随链接。
    if cache.is_symlink():
        raise ValueError('缓存目录不允许symlink')
    if cache.exists():
        for parent, folders, files in os.walk(cache, followlinks=False):
            if any((Path(parent) / name).is_symlink() for name in folders + files):
                raise ValueError('缓存内部存在symlink，保留不改动')
        hf_cache = cache / 'huggingface'
        if hf_cache.exists() and not hf_cache.is_dir():
            raise ValueError('HF缓存路径冲突，保留不覆盖')
    expected = {r['path']: r for r in plan['files']}
    for name in ('verified-files.jsonl', 'download-complete.jsonl'):
        receipt = out / name
        if receipt.is_symlink() or (receipt.exists() and not receipt.is_file()):
            raise ValueError('receipt路径冲突或symlink，保留不覆盖')
        if receipt.exists():
            for line in receipt.read_text().splitlines():
                row = json.loads(line)
                if row.get('plan_sha256') != plan_sha:
                    raise ValueError('已有receipt属于其他计划，保留不追加')
                if name == 'verified-files.jsonl':
                    if row.get('path') not in expected:
                        raise ValueError('已有receipt文件范围冲突')
                    record = expected[row['path']]
                    if row.get('size') != record['size'] or row.get('sha256') != record['sha256']:
                        raise ValueError('已有receipt身份冲突')
                elif row.get('bytes_verified') != plan['total_bytes'] or row.get('files') != 3165:
                    raise ValueError('已有completion容量或文件数冲突')
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=BASE / 'candidate-snapshot.json')
    parser.add_argument('--plan', type=Path, default=BASE / 'download-plan.json')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--quote', action='store_true', help='只查询小型官方size/SHA，不下载ZIP')
    actions.add_argument('--download', action='store_true', help='用户显式启动全部ZIP下载')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--confirm-bytes', type=int)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    validate(snapshot)
    if digest(args.snapshot.parent / 'test-candidates.csv') != snapshot['test_candidates_sha256']:
        raise ValueError('test clip名单SHA不符')
    snapshot_sha = digest(args.snapshot)
    if args.quote:
        if args.plan.exists():
            raise FileExistsError('报价已存在；保留原件，改用新的 --plan 路径')
        from huggingface_hub import HfApi
        api = HfApi(endpoint='https://huggingface.co')
        if api.dataset_info(REPO, revision=REVISION).sha != REVISION:
            raise ValueError('官方revision不符')
        paths = sorted(r['path'] for r in snapshot['files'])
        records = []
        for offset in range(0, len(paths), 100):
            batch = paths[offset:offset + 100]
            infos = api.get_paths_info(REPO, batch, revision=REVISION, repo_type='dataset')
            if {x.path for x in infos} != set(batch):
                raise ValueError('官方缺少所列文件')
            for info in infos:
                if not info.lfs:
                    raise ValueError('ZIP缺少LFS SHA256')
                records.append({'path': info.path, 'size': info.size, 'sha256': info.lfs.sha256})
            print('Quoted', len(records), '/', len(paths), flush=True)
        plan = {'dataset': REPO, 'revision': REVISION, 'snapshot_sha256': snapshot_sha,
            'authority_source': 'https://huggingface.co', 'transport_source': 'https://huggingface.co',
            'scope': snapshot['scope'], 'coverage_complete': False, 'drive_disjoint_verified': False,
            'clip_count': 61301, 'chunk_count': 633, 'files': sorted(records, key=lambda r: r['path']),
            'total_bytes': sum(r['size'] for r in records)}
        validate(snapshot, plan)
        with args.plan.open('x') as handle:
            json.dump(plan, handle, indent=2)
            handle.write('\n')
    elif not args.plan.exists():
        print('633 chunks / 3165 ZIP / 61301 candidate clips; no quote yet. No network/download.')
        return
    plan = json.loads(args.plan.read_text())
    validate(snapshot, plan)
    if plan['snapshot_sha256'] != snapshot_sha:
        raise ValueError('报价绑定的候选快照SHA不符')
    total = plan['total_bytes']
    print(json.dumps({'bytes': total, 'TB': total / 10**12, 'TiB': total / 2**40,
        'plan_sha256': digest(args.plan), 'scope': plan['scope']}, indent=2), flush=True)
    if not args.download:
        print('PLAN ONLY: no ZIP download; not a certified independent test set.')
        return
    if args.plan_sha256 != digest(args.plan) or args.confirm_bytes != total or args.output is None:
        raise ValueError('下载需明确 --output、--plan-sha256、--confirm-bytes')
    if plan['transport_source'] != 'https://huggingface.co':
        raise ValueError('尚无验收通过的第三方传输源；不得擅自发送token')
    plan_sha = digest(args.plan)
    out = prepare_output(args.output, plan, plan_sha)
    existing_bytes = 0
    for r in plan['files']:
        target = out / r['path']
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError('输出路径不允许symlink')
        if target.exists():
            if not target.is_file() or target.stat().st_size != r['size'] or digest(target) != r['sha256']:
                raise ValueError('已有文件冲突，保留不覆盖：' + r['path'])
            existing_bytes += r['size']
    if shutil.disk_usage(out).free < (total - existing_bytes) * 1.05:
        raise ValueError('磁盘余量不足（剩余ZIP容量另留5%）；不启动下载')
    from huggingface_hub import hf_hub_download
    for index, r in enumerate(plan['files'], 1):
        target = out / r['path']
        if not target.exists():
            path = hf_hub_download(REPO, r['path'], repo_type='dataset', revision=REVISION,
                local_dir=out, endpoint='https://huggingface.co')
            target = Path(path)
        if target.stat().st_size != r['size'] or digest(target) != r['sha256']:
            raise ValueError('下载校验失败，文件与缓存保留：' + r['path'])
        with (out / 'verified-files.jsonl').open('a') as handle:
            handle.write(json.dumps({'path': r['path'], 'size': r['size'], 'sha256': r['sha256'],
                'plan_sha256': plan_sha}) + '\n')
        print('Verified', index, '/', len(plan['files']), flush=True)
    with (out / 'download-complete.jsonl').open('a') as handle:
        handle.write(json.dumps({'plan_sha256': digest(args.plan), 'files': 3165,
            'bytes_verified': total, 'independence_verified': False, 'input_qc_verified': False}) + '\n')


if __name__ == '__main__':
    main()
