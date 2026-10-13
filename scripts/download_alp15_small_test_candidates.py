"""固定949个候选clip；默认只打印计划，不下载。"""
import argparse
import csv
import io
import json
import random
import shutil
import hashlib

from download_alp15_full_test_candidates import BASE, REPO, REVISION, digest, prepare_output, validate

TOTAL = 54382445102
FULL_SHA = '2a2352b5e8d690763c7d66c40d3baed2154095987b64bd33c8206614ac93e8f0'


def selection():
    if digest(BASE / 'download-plan.json') != FULL_SHA:
        raise ValueError('原始报价SHA不符，停止')
    full = json.loads((BASE / 'download-plan.json').read_text())
    snapshot = json.loads((BASE / 'candidate-snapshot.json').read_text())
    if digest(BASE / 'candidate-snapshot.json') != full['snapshot_sha256']:
        raise ValueError('原始快照SHA不符')
    if digest(BASE / 'test-candidates.csv') != snapshot['test_candidates_sha256']:
        raise ValueError('clip名单SHA不符')
    validate(snapshot, full)
    chunks = sorted(random.Random(20261011).sample(sorted(snapshot['chunks']), 10))
    paths = {r['path'] for r in snapshot['files'] if r['chunk'] in chunks}
    files = [r for r in full['files'] if r['path'] in paths]
    with (BASE / 'test-candidates.csv').open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if int(r['chunk']) in chunks]
    if len(files) != 50 or len(rows) != 949 or sum(r['size'] for r in files) != TOTAL:
        raise ValueError('子集身份或容量不符')
    plan = dict(full, files=files, total_bytes=TOTAL, clip_count=949, chunk_count=10,
                chunks=chunks, seed=20261011, scope='small_candidate_pool_not_certified_independent')
    plan_text = json.dumps(plan, sort_keys=True, indent=2) + '\n'
    plan_sha = hashlib.sha256(plan_text.encode()).hexdigest()
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=['clip_id', 'chunk', 't0_us'])
    writer.writeheader()
    writer.writerows(rows)
    return plan, plan_text, plan_sha, buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--confirm-bytes', type=int)
    args = parser.parse_args()
    plan, plan_text, plan_sha, clip_text = selection()
    print(json.dumps({k: plan[k] for k in ('chunks', 'clip_count', 'total_bytes', 'scope')}, indent=2))
    if not args.download:
        print('PLAN ONLY: no network/download.')
        return
    if args.confirm_bytes != TOTAL or not args.output:
        raise ValueError('必须明确 --output 和 --confirm-bytes 54382445102')
    out = prepare_output(args.output, plan, plan_sha)
    remaining = TOTAL
    for row in plan['files']:
        target = out / row['path']
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError('文件路径存在symlink')
        if target.exists():
            if not target.is_file() or target.stat().st_size != row['size'] or digest(target) != row['sha256']:
                raise ValueError('已有文件冲突，保留不覆盖：' + row['path'])
            remaining -= row['size']
    if shutil.disk_usage(out).free < remaining * 1.05:
        raise ValueError('磁盘余量不足；首次至少约57.11GB，解压另算')
    for name, value in (('small-download-plan.json', plan_text), ('test-candidates.csv', clip_text)):
        path = out / name
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != value.encode())):
            raise ValueError('子集记录冲突，保留不覆盖：' + name)
    for name, value in (('small-download-plan.json', plan_text), ('test-candidates.csv', clip_text)):
        path = out / name
        if not path.exists():
            with path.open('x', newline='') as handle:
                handle.write(value)
    from huggingface_hub import hf_hub_download
    for index, row in enumerate(plan['files'], 1):
        target = out / row['path']
        if not target.exists():
            target = hf_hub_download(REPO, row['path'], repo_type='dataset', revision=REVISION,
                                     local_dir=out, endpoint='https://huggingface.co')
        from pathlib import Path
        target = Path(target)
        if target.stat().st_size != row['size'] or digest(target) != row['sha256']:
            raise ValueError('下载校验失败，文件和缓存保留：' + row['path'])
        with (out / 'verified-files.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(row, plan_sha256=plan_sha)) + '\n')
        print('Verified', index, '/ 50', flush=True)
    print('50 ZIP verified. Only evaluate IDs in test-candidates.csv; independence/QC not verified.')


if __name__ == '__main__':
    main()
