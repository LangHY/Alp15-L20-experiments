"""离线、小文件与mock验证；不得连接HF/SSH或下载真实ZIP。"""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).absolute().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load('pipeline', 'download_alp15_mac_l20_pipeline.py')
receiver = load('receiver', 'alp15_remote_receiver.py')


class FastStop:
    def __init__(self):
        self.waits = []
    def is_set(self):
        return False
    def wait(self, seconds):
        self.waits.append(seconds)
        return False


class PipelineTests(unittest.TestCase):
    def test_frozen_plan(self):
        plan = pipeline.core.load_plan()
        self.assertEqual(len(plan['files']), 50)
        self.assertEqual(sum(row['size'] for row in plan['files']), pipeline.core.TOTAL)

    def test_default_has_no_network_or_write(self):
        with patch.object(pipeline, 'run') as run, patch('subprocess.Popen') as proc, \
                patch('pathlib.Path.mkdir') as mkdir, patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(pipeline.main([]), 0)
        run.assert_not_called()
        proc.assert_not_called()
        mkdir.assert_not_called()

    def test_missing_byte_confirmation_fails_before_write(self):
        with patch('pathlib.Path.mkdir') as mkdir, patch('sys.stdout', new=io.StringIO()):
            with self.assertRaises(pipeline.PipelineError):
                pipeline.main(['--run'])
        mkdir.assert_not_called()

    def test_retry_bounded_exponential(self):
        stop = FastStop()
        attempts = []
        def fail():
            attempts.append(1)
            raise ConnectionError('no secret output')
        with self.assertRaises(ConnectionError):
            pipeline.retry(fail, pipeline.hf_transient, lambda *a, **k: None, stop)
        self.assertEqual(len(attempts), 7)
        self.assertEqual(stop.waits, [2, 4, 8, 16, 32, 60])

    def test_hf_auth_fails_fast(self):
        stop = FastStop()
        error = RuntimeError('secret token')
        error.response = types.SimpleNamespace(status_code=403)
        with self.assertRaises(RuntimeError):
            pipeline.retry(lambda: (_ for _ in ()).throw(error), pipeline.hf_transient,
                           lambda *a, **k: None, stop)
        self.assertEqual(stop.waits, [])

    def test_hf_wrapped_connection_retry(self):
        wrapped = type('LocalEntryNotFoundError', (Exception,), {})('wrapped')
        wrapped.__cause__ = ConnectionError('network')
        self.assertTrue(pipeline.hf_transient(wrapped))

    def test_metadata_checks_all_files_and_frozen_revision(self):
        plan = pipeline.core.load_plan()
        rows = {row['path']: row for row in plan['files']}
        heads = []
        def metadata(url, **kwargs):
            heads.append(url)
            row = rows[url]
            return types.SimpleNamespace(commit_hash=pipeline.core.REVISION,
                                         size=row['size'], etag=row['sha256'])
        api = types.SimpleNamespace(whoami=lambda **kwargs: {})
        fake = types.SimpleNamespace(HfApi=lambda **kwargs: api, get_token=lambda: 'test-private',
            get_hf_file_metadata=metadata, hf_hub_url=lambda repo, name, **kwargs: name,
            hf_hub_download=lambda *a, **k: None)
        with patch.dict('sys.modules', huggingface_hub=fake), \
                patch('importlib.metadata.version', return_value='1.31.0'):
            pipeline.hf_preflight(lambda *a, **k: None, FastStop())
            self.assertEqual(len(heads), 50)
            fake.get_hf_file_metadata = lambda *a, **k: types.SimpleNamespace(
                commit_hash='wrong', size=1, etag='wrong')
            with self.assertRaises(pipeline.PipelineError):
                pipeline.hf_preflight(lambda *a, **k: None, FastStop())

    def test_queue_download_then_upload(self):
        events = []
        rows = list(range(7))
        pipeline.pipeline(rows, lambda row: events.append(('d', row)),
                          lambda row: events.append(('u', row)), threading.Event())
        for row in rows:
            self.assertLess(events.index(('d', row)), events.index(('u', row)))
        self.assertEqual(sum(kind == 'u' for kind, _ in events), 7)

    def test_upload_failure_releases_full_queue(self):
        result = []
        def run():
            try:
                pipeline.pipeline(range(20), lambda row: None,
                    lambda row: (_ for _ in ()).throw(ValueError('conflict')), threading.Event())
            except ValueError as exc:
                result.append(str(exc))
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, ['conflict'])

    def test_download_failure_releases_idle_worker(self):
        with self.assertRaises(ValueError):
            pipeline.pipeline([1], lambda row: (_ for _ in ()).throw(ValueError('bad SHA')),
                              lambda row: self.fail('not verified'), threading.Event())

    def test_rsync_resume_options_and_auth_classification(self):
        args = argparse.Namespace(port=2222, host='tuqiang@127.0.0.1', control_path='/tmp/ssh-control')
        command = pipeline.rsync_args(args, Path('/tmp/asset.zip'), receiver.ROOT / 'file.part')
        self.assertIn('--partial-dir=.rsync-partial', command)
        self.assertIn('--timeout=60', command)
        self.assertNotIn('--append-verify', command)
        self.assertIn('StrictHostKeyChecking=yes', command[command.index('-e') + 1])
        self.assertFalse(pipeline.rsync_transient(subprocess.CalledProcessError(255, [], stderr='Permission denied')))
        self.assertFalse(pipeline.rsync_transient(subprocess.CalledProcessError(255, [], stderr='Host key verification failed')))
        self.assertTrue(pipeline.rsync_transient(subprocess.CalledProcessError(12, [], stderr='connection reset')))

    def test_remote_reconnect_keeps_same_arguments(self):
        args = object()
        with patch.object(pipeline.Remote, 'connect') as connect:
            remote = pipeline.Remote(args, lambda *a, **k: None)
            remote.needs_reconnect = True
            with patch.object(remote, 'close') as close:
                remote.reconnect()
        self.assertIs(remote.args, args)
        self.assertEqual(connect.call_count, 2)
        close.assert_called_once()
        self.assertFalse(remote.needs_reconnect)

    def test_two_failed_reconnections_then_recovery(self):
        with patch.object(pipeline.Remote, 'connect'):
            remote = pipeline.Remote(object(), lambda *a, **k: None)
        remote.needs_reconnect = True
        stop = FastStop()
        states = []
        def connect():
            states.append(remote.needs_reconnect)
            if len(states) < 3:
                raise ConnectionError('simulated SSH interruption')
        with patch.object(remote, 'close'), patch.object(remote, 'connect', side_effect=connect):
            pipeline.retry(remote.reconnect, pipeline.hf_transient, lambda *a, **k: None, stop)
        self.assertEqual(states, [True, True, True])
        self.assertEqual(stop.waits, [2, 4])
        self.assertFalse(remote.needs_reconnect)

    def test_ssh_auth_not_retried(self):
        stop = FastStop()
        with self.assertRaises(pipeline.PipelineError):
            pipeline.retry(lambda: (_ for _ in ()).throw(pipeline.PipelineError('SSH_AUTH_OR_HOSTKEY_FAILED')),
                           pipeline.hf_transient, lambda *a, **k: None, stop)
        self.assertEqual(stop.waits, [])


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()  # macOS /var 别名；测试目录用真实路径。
        self.rootpatch = patch.object(receiver, 'ROOT', self.root)
        self.rootpatch.start()
        content = b'synthetic zip asset'
        self.row = {'path': 'camera/test.zip', 'size': len(content),
                    'sha256': hashlib.sha256(content).hexdigest()}
        self.content = content
        self.server = object.__new__(receiver.Receiver)
        self.server.rows = {self.row['path']: self.row}
        self.server.done = set()
        self.server.plan = {'total_bytes': len(content)}
        self.server.identity = {'plan_sha256': 'synthetic'}
    def tearDown(self):
        self.rootpatch.stop()
        self.tmp.cleanup()

    def test_remote_conflict_preserves_final(self):
        final = self.root / self.row['path']
        final.parent.mkdir()
        final.write_bytes(b'previous data')
        with self.assertRaisesRegex(ValueError, 'EXISTING_FINAL_CONFLICT'):
            self.server.handle({'action': 'prepare', 'path': self.row['path']})
        self.assertEqual(final.read_bytes(), b'previous data')

    def test_publication_is_exclusive_and_preserves_stage(self):
        result = self.server.handle({'action': 'prepare', 'path': self.row['path']})
        stage = Path(result['stage'])
        stage.write_bytes(self.content)
        self.assertTrue(self.server.handle({'action': 'commit', 'path': self.row['path']})['verified'])
        final = self.root / self.row['path']
        self.assertEqual(stage.stat().st_ino, final.stat().st_ino)
        self.assertTrue(stage.exists())
        self.assertEqual(self.server.handle({'action': 'complete'})['files'], 1)

    def test_bad_stage_does_not_publish(self):
        result = self.server.handle({'action': 'prepare', 'path': self.row['path']})
        Path(result['stage']).write_bytes(b'bad sha')
        with self.assertRaises(ValueError):
            self.server.handle({'action': 'commit', 'path': self.row['path']})
        self.assertFalse((self.root / self.row['path']).exists())

    def test_symlink_and_space_fail_before_transfer(self):
        staging = self.root / '.incoming'
        staging.symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, 'UNSAFE_PATH'):
            self.server.handle({'action': 'prepare', 'path': self.row['path']})
        with patch('shutil.disk_usage', return_value=types.SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ValueError, 'REMOTE_SPACE_INSUFFICIENT'):
                self.server.space()

    def test_receiver_identity_and_exclusive_task_lock(self):
        request = dict(hostname='zwl-acloud',
            plan=base64.b64encode((pipeline.PACKAGE / 'small-download-plan.json').read_bytes()).decode(),
            csv=base64.b64encode((pipeline.PACKAGE / 'test-candidates.csv').read_bytes()).decode())
        with patch('getpass.getuser', return_value='tuqiang'), \
                patch('platform.system', return_value='Linux'), \
                patch('socket.gethostname', return_value='wrong-host'):
            with self.assertRaisesRegex(ValueError, 'REMOTE_IDENTITY_MISMATCH'):
                receiver.Receiver(request)
        self.assertEqual(list(self.root.iterdir()), [])
        with patch('getpass.getuser', return_value='tuqiang'), \
                patch('platform.system', return_value='Linux'), \
                patch('socket.gethostname', return_value='zwl-acloud'), \
                patch('subprocess.run', return_value=types.SimpleNamespace(stdout='NVIDIA L20\n')), \
                patch('shutil.disk_usage', return_value=types.SimpleNamespace(free=10**15)):
            first = receiver.Receiver(request)
            try:
                with self.assertRaises(BlockingIOError):
                    receiver.Receiver(request)
            finally:
                first.lock.close()


if __name__ == '__main__':
    unittest.main()
