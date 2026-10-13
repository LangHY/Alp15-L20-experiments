"""本任务专用 SSH JSONL 接收器；仅写固定数据根，不打印第三方异常。"""
import base64
import fcntl
import getpass
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/tuqiang/lang/datasets/alp15-small-test-949-20261013')
PLAN_SHA = '7a2356ea86921d535a96ef84e49f1018f8209dd31f7011423c6574372703471f'
CSV_SHA = '9fbf80b944f8b810557838da52e7318d9cc7d3855746bc7f39b183dd539e76af'


def reject_links(path):
    for item in (path, *path.parents):
        try:
            mode = item.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError('UNSAFE_PATH')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verified(path, row):
    reject_links(path)
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != row['size'] or digest(path) != row['sha256']:
        raise ValueError('EXISTING_FINAL_CONFLICT')
    return True


def write_new(path, content):
    reject_links(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError('TASK_METADATA_CONFLICT')
    else:
        with path.open('xb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())


class Receiver:
    def __init__(self, request):
        if (getpass.getuser() != 'tuqiang' or platform.system() != 'Linux'
                or socket.gethostname() != request['hostname']):
            raise ValueError('REMOTE_IDENTITY_MISMATCH')
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20, check=True).stdout.splitlines()
        if not gpu or any(name.strip() != 'NVIDIA L20' for name in gpu):
            raise ValueError('REMOTE_GPU_MISMATCH')
        plan_data, csv_data = [base64.b64decode(request[key], validate=True) for key in ('plan', 'csv')]
        if hashlib.sha256(plan_data).hexdigest() != PLAN_SHA or hashlib.sha256(csv_data).hexdigest() != CSV_SHA:
            raise ValueError('FROZEN_METADATA_MISMATCH')
        self.plan = json.loads(plan_data)
        self.rows = {row['path']: row for row in self.plan['files']}
        self.identity = {'plan_sha256': PLAN_SHA, 'candidates_sha256': CSV_SHA,
                         'revision': self.plan['revision'], 'root': str(ROOT),
                         'hostname': request['hostname']}
        reject_links(ROOT)
        for ancestor in (ROOT, *ROOT.parents):
            if ancestor.exists() and not ancestor.is_dir():
                raise ValueError('ROOT_DIRECTORY_CONFLICT')
        # 检查整个专用目录；拒绝 staging、partial、receipt 的隐藏链接。
        if ROOT.exists():
            for parent, folders, files in os.walk(ROOT, followlinks=False):
                for name in folders + files:
                    reject_links(Path(parent) / name)
        ROOT.mkdir(parents=True, exist_ok=True)
        lock = ROOT / '.pipeline.lock'
        reject_links(lock)
        self.lock = lock.open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        binding = json.dumps(self.identity, sort_keys=True).encode()
        write_new(ROOT / '.pipeline-identity.json', binding)
        write_new(ROOT / 'small-download-plan.json', plan_data)
        write_new(ROOT / 'test-candidates.csv', csv_data)
        receipt = ROOT / 'remote-verified.jsonl'
        if receipt.exists():
            for line in receipt.read_text().splitlines():
                value = json.loads(line)
                if any(value.get(key) != item for key, item in self.identity.items()):
                    raise ValueError('RECEIPT_IDENTITY_CONFLICT')
                row = self.rows.get(value.get('path'))
                if row is None or any(value.get(key) != row[key] for key in ('size', 'sha256')):
                    raise ValueError('RECEIPT_FILE_CONFLICT')
        self.done = {name for name, row in self.rows.items() if verified(ROOT / name, row)}
        self.space()

    def space(self):
        needed = sum(row['size'] for name, row in self.rows.items() if name not in self.done)
        free = shutil.disk_usage(ROOT).free
        if free < math.ceil(needed * 1.05):
            raise ValueError('REMOTE_SPACE_INSUFFICIENT')
        return free

    def handle(self, request):
        action = request['action']
        if action == 'complete':
            if not all(verified(ROOT / name, row) for name, row in self.rows.items()):
                raise ValueError('REMOTE_INCOMPLETE')
            result = dict(self.identity, files=len(self.rows), bytes_verified=self.plan['total_bytes'])
            write_new(ROOT / 'remote-complete.json', json.dumps(result, sort_keys=True).encode())
            return result
        name = request['path']
        row = self.rows[name]  # 仅允许冻结清单中的相对路径。
        final = ROOT / name
        if verified(final, row):
            self.done.add(name)
            return {'verified': True}
        self.space()
        stage = ROOT / '.incoming' / (name + '.part')
        reject_links(stage)
        reject_links(stage.parent / '.rsync-partial')
        if stage.exists() and not stage.is_file():
            raise ValueError('STAGE_CONFLICT')
        if action == 'prepare':
            partial = stage.parent / '.rsync-partial'
            if partial.exists():
                for parent, folders, files in os.walk(partial, followlinks=False):
                    for entry in folders + files:
                        reject_links(Path(parent) / entry)
            stage.parent.mkdir(parents=True, exist_ok=True)
            return {'verified': False, 'stage': str(stage)}
        if action != 'commit' or not verified(stage, row):
            raise ValueError('STAGE_NOT_VERIFIED')
        final.parent.mkdir(parents=True, exist_ok=True)
        reject_links(final)
        # link 排他发布；保留 staging 硬链接，不删除之前的文件，不另占一份空间。
        os.link(stage, final)
        verified(final, row)
        self.done.add(name)
        value = dict(self.identity, **row, at=datetime.now(timezone.utc).isoformat())
        with (ROOT / 'remote-verified.jsonl').open('a') as handle:
            handle.write(json.dumps(value, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        return {'verified': True}


def serve():
    try:
        receiver = Receiver(json.loads(sys.stdin.readline()))
        print(json.dumps({'ready': True, 'verified': sorted(receiver.done),
                          'free_bytes': receiver.space(), 'identity_verified': True}), flush=True)
        for line in sys.stdin:
            print(json.dumps(receiver.handle(json.loads(line))), flush=True)
    except Exception as exc:
        # 仅显示已知固定错误码或类型；绝不输出 SSH 环境或认证内容。
        code = str(exc) if type(exc) is ValueError and str(exc).isupper() else type(exc).__name__
        print(json.dumps({'error': code}), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(serve())
