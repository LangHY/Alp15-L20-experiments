"""固定949候选资产：默认离线计划；显式运行才下载并流水线同步L20。"""
import argparse
import base64
import fcntl
import importlib.metadata
import importlib.util
import json
import os
import queue
import re
import selectors
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).absolute().parents[1]
PACKAGE = BASE / 'deliverables/alp15_windows_949'
spec = importlib.util.spec_from_file_location('alp15_frozen_downloader', PACKAGE / 'download_alp15_windows.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
REMOTE = '/home/tuqiang/lang/datasets/alp15-small-test-949-20261013'
DEFAULT_LOCAL = '/Users/lang/Downloads/alp15-small-test-949-20261013'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'


class PipelineError(ValueError):
    pass


def retry(operation, transient, log, stop):
    """首次尝试后最多6次重试；停止事件解除退避等待。"""
    for attempt in range(7):
        if stop.is_set():
            raise PipelineError('STOP_REQUESTED')
        try:
            return operation()
        except Exception as exc:
            log('attempt_failed', attempt=attempt + 1, error_type=type(exc).__name__,
                exit_code=getattr(exc, 'returncode', None))
            if attempt == 6 or not transient(exc):
                raise
            if stop.wait(min(2 ** (attempt + 1), 60)):
                raise PipelineError('STOP_REQUESTED') from None


def hf_transient(exc):
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status is not None:
        return status in (408, 429, 500, 502, 503, 504)
    if type(exc).__name__ == 'LocalEntryNotFoundError' and exc.__cause__ is not None:
        return hf_transient(exc.__cause__)
    return type(exc).__name__ in {'ConnectTimeout', 'ReadTimeout', 'WriteTimeout', 'PoolTimeout',
        'ConnectError', 'ReadError', 'WriteError', 'RemoteProtocolError', 'TimeoutError', 'ConnectionError'}


def hf_preflight(log, stop):
    if importlib.metadata.version('huggingface_hub') != core.HUB_VERSION:
        raise PipelineError('REQUIRE_HUGGINGFACE_HUB_1.31.0')
    from huggingface_hub import HfApi, get_token, get_hf_file_metadata, hf_hub_url, hf_hub_download
    token = get_token()
    if not token:
        raise PipelineError('HF_LOGIN_REQUIRED')
    retry(lambda: HfApi(endpoint=core.ENDPOINT).whoami(token=token), hf_transient, log, stop)
    # HEAD实际下载目标，验证认证、冻结commit、LFS SHA及每个文件的真实大小。
    for row in core.load_plan()['files']:
        url = hf_hub_url(core.REPO, row['path'], repo_type='dataset',
                         revision=core.REVISION, endpoint=core.ENDPOINT)
        metadata = retry(lambda: get_hf_file_metadata(url, token=token, timeout=30),
                         hf_transient, log, stop)
        if (metadata.commit_hash != core.REVISION or metadata.size != row['size']
                or metadata.etag != row['sha256']):
            raise PipelineError('HF_METADATA_MISMATCH')
    log('hf_preflight_verified', files=50, hub_version=core.HUB_VERSION)
    return lambda row, out: hf_hub_download(core.REPO, row['path'], repo_type='dataset',
        revision=core.REVISION, local_dir=str(out), endpoint=core.ENDPOINT, token=token)


def ssh_args(args):
    result = ['ssh', '-p', str(args.port), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
              '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2']
    if args.control_path:
        result += ['-o', 'ControlPath=' + args.control_path]
    return result


def rsync_args(args, local, stage):
    return ['rsync', '-a', '--partial', '--partial-dir=.rsync-partial', '--timeout=60',
            '-e', shlex.join(ssh_args(args)), str(local), args.host + ':' + str(stage)]


def rsync_transient(exc):
    if isinstance(exc, subprocess.TimeoutExpired):
        return True
    if not isinstance(exc, subprocess.CalledProcessError):
        return False
    error = (exc.stderr or '').lower()
    if any(word in error for word in ('permission denied', 'host key', 'authentication',
                                      'remote host identification', 'no such file', 'no space left')):
        return False
    return exc.returncode in (10, 12, 30, 35, 255)


def transfer(command, stop):
    # rsync --timeout 约束停滞；不限制健康大文件的总传输时间。
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        while True:
            try:
                _, error = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if stop.is_set():
                    raise PipelineError('STOP_REQUESTED')
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, stderr=error)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


class Remote:
    def __init__(self, args, log, stop=None):
        self.args, self.log, self.stop = args, log, stop or threading.Event()
        self.needs_reconnect = False
        self.connect()

    def connect(self):
        args, log = self.args, self.log
        code = base64.b64encode((BASE / 'scripts/alp15_remote_receiver.py').read_bytes()).decode()
        command = 'python3 -u -c ' + shlex.quote("import base64;exec(base64.b64decode(%r))" % code)
        self.process = subprocess.Popen(ssh_args(args) + [args.host, command], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        request = dict(hostname=args.expected_hostname,
            plan=base64.b64encode((PACKAGE / 'small-download-plan.json').read_bytes()).decode(),
            csv=base64.b64encode((PACKAGE / 'test-candidates.csv').read_bytes()).decode())
        try:
            self.initial = self.call(request, timeout=1800)
            if not self.initial.get('ready'):
                raise PipelineError('REMOTE_NOT_READY')
            log('remote_preflight_verified', **self.initial)
        except BaseException:
            self.close()
            raise

    def call(self, request, timeout=120):
        try:
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            raise ConnectionError('SSH_SESSION_LOST') from None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + timeout
            while not selector.select(timeout=0.2):
                if self.stop.is_set():
                    raise PipelineError('STOP_REQUESTED')
                if time.monotonic() >= deadline:
                    raise ConnectionError('REMOTE_RESPONSE_TIMEOUT')
        line = self.process.stdout.readline()
        if not line:
            self.process.wait(timeout=5)
            error = self.process.stderr.read().lower()
            if any(word in error for word in ('permission denied', 'host key', 'authentication',
                                              'remote host identification')):
                raise PipelineError('SSH_AUTH_OR_HOSTKEY_FAILED')
            raise ConnectionError('SSH_SESSION_LOST')
        result = json.loads(line)
        if 'error' in result:
            raise PipelineError('REMOTE_' + result['error'])
        return result

    def reconnect(self):
        self.close()
        self.connect()
        self.needs_reconnect = False

    def close(self):
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def pipeline(rows, download, upload, stop):
    """一个下载线程（调用线程）、一个上传线程，队列容量2。"""
    pending, failures = queue.Queue(maxsize=2), []
    def worker():
        try:
            while not stop.is_set():
                try:
                    item = pending.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    return
                upload(item)
        except BaseException as exc:
            failures.append(exc)
            stop.set()
    def put(item):
        while not stop.is_set():
            try:
                pending.put(item, timeout=0.2)
                return
            except queue.Full:
                pass
        raise PipelineError('UPLOAD_STOPPED')
    thread = threading.Thread(target=worker, name='alp15-upload')
    thread.start()
    try:
        for row in rows:
            if stop.is_set():
                raise PipelineError('STOP_REQUESTED')
            download(row)
            put(row)
        put(None)
    except BaseException as exc:
        failures.append(exc)
        stop.set()
    finally:
        thread.join()
    if failures:
        raise failures[0]


def run(args, plan):
    if args.confirm_bytes != core.TOTAL:
        raise PipelineError('REQUIRE_CONFIRM_BYTES_54382445102')
    if sys.version_info < (3, 10) or sys.maxsize <= 2 ** 32:
        raise PipelineError('REQUIRE_64BIT_PYTHON_3.10_OR_NEWER')
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*', args.host)
            or not 1 <= args.port <= 65535 or not args.expected_hostname):
        raise PipelineError('INVALID_SSH_CONFIGURATION')
    out = core.validate_output(args.local_dir)
    core.inspect_cache(out)
    core.inspect_records(out, plan)
    core.inspect_metadata(out)
    out.mkdir(parents=True, exist_ok=True)
    lockpath = out / '.pipeline.lock'
    core.regular_file(lockpath)
    stop, remote = threading.Event(), None
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verified = core.inspect_files(out, plan)
        remaining = sum(row['size'] for row in plan['files'] if row['path'] not in verified)
        core.check_space(out, remaining)
        logfile = out / ('pipeline-' + uuid.uuid4().hex + '.jsonl')
        loglock = threading.Lock()
        def log(event, **details):
            record = dict(core.identity(), at=datetime.now(timezone.utc).isoformat(), event=event, **details)
            with loglock:
                core.append_record(logfile, record)
                print(json.dumps(record, sort_keys=True), flush=True)
        def interrupted(signum, frame):
            stop.set()
        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            log('environment', python=platform_python(), mode='download_only' if args.download_only else 'pipeline')
            downloader = hf_preflight(log, stop)
            if not args.download_only:
                remote = retry(lambda: Remote(args, log, stop), hf_transient, log, stop)
            for name in ('small-download-plan.json', 'test-candidates.csv'):
                target = out / name
                if not target.exists():
                    with target.open('xb') as handle:
                        handle.write((PACKAGE / name).read_bytes())
            def download(row):
                nonlocal remaining
                target = out / row['path']
                core.regular_file(target)
                core.inspect_cache(out)
                core.check_space(out, remaining)
                if row['path'] not in verified:
                    if target.exists():
                        raise PipelineError('TARGET_APPEARED_DURING_RUN')
                    actual = retry(lambda: downloader(row, out), hf_transient,
                                   lambda event, **info: log(event, path=row['path'], phase='download', **info), stop)
                    if Path(actual).absolute() != target.absolute():
                        raise PipelineError('UNEXPECTED_DOWNLOAD_PATH')
                    remaining -= row['size']
                core.regular_file(target)
                if target.stat().st_size != row['size'] or core.digest(target) != row['sha256']:
                    raise PipelineError('LOCAL_SHA_MISMATCH')
                verified.add(row['path'])
                core.append_record(out / 'verified-files.jsonl', dict(core.identity(), **row))
                log('local_verified', **row)
            def upload(row):
                def attempt():
                    try:
                        if remote.needs_reconnect or remote.process.poll() is not None:
                            remote.reconnect()
                        result = remote.call(dict(action='prepare', path=row['path']))
                        if not result.get('verified'):
                            transfer(rsync_args(args, out / row['path'], result['stage']), stop)
                            result = remote.call(dict(action='commit', path=row['path']))
                        if not result.get('verified'):
                            raise PipelineError('REMOTE_NOT_VERIFIED')
                    except ConnectionError:
                        remote.needs_reconnect = True
                        raise
                retry(attempt, lambda exc: hf_transient(exc) or rsync_transient(exc),
                      lambda event, **info: log(event, path=row['path'], phase='upload', **info), stop)
                log('remote_verified', **row)
            if args.download_only:
                for row in plan['files']:
                    if stop.is_set():
                        raise PipelineError('STOP_REQUESTED')
                    download(row)
            else:
                pipeline(plan['files'], download, upload, stop)
            if stop.is_set() or len(core.inspect_files(out, plan)) != 50:
                raise PipelineError('LOCAL_INCOMPLETE_OR_STOPPED')
            if remote:
                def complete():
                    try:
                        if remote.needs_reconnect or remote.process.poll() is not None:
                            remote.reconnect()
                        return remote.call({'action': 'complete'}, timeout=1800)
                    except ConnectionError:
                        remote.needs_reconnect = True
                        raise
                retry(complete, hf_transient, log, stop)
            core.append_record(out / 'download-complete.jsonl', dict(core.identity(), files=50,
                bytes_verified=core.TOTAL, clip_count=949, independence_verified=False, input_qc_verified=False))
            log('complete', files=50, bytes_verified=core.TOTAL, remote_verified=remote is not None)
        except BaseException as exc:
            log('stopped', error_type=type(exc).__name__)
            raise
        finally:
            stop.set()
            if remote:
                remote.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def platform_python():
    return sys.version.split()[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--download-only', action='store_true', help='只下载Mac；稍后重跑完整流水线复用ZIP')
    parser.add_argument('--confirm-bytes', type=int)
    parser.add_argument('--local-dir', '--output', default=DEFAULT_LOCAL)
    parser.add_argument('--host', default='tuqiang@127.0.0.1')
    parser.add_argument('--port', type=int, default=2222)
    parser.add_argument('--expected-hostname', default='zwl-acloud')
    parser.add_argument('--control-path')
    args = parser.parse_args(argv)
    plan = core.load_plan()
    print(json.dumps(dict(files=50, clips=949, bytes=core.TOTAL, local=args.local_dir,
                         remote=REMOTE, download_only=args.download_only, revision=core.REVISION)), flush=True)
    if not args.run:
        print('PLAN ONLY: no network or output writes.')
        return 0
    run(args, plan)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        message = str(exc) if isinstance(exc, (PipelineError, core.DownloadError)) else type(exc).__name__
        print('STOP: ' + message + '; files/cache/partial preserved.', file=sys.stderr)
        sys.exit(1)
