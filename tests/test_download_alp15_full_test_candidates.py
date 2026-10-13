"""用户下载工具的纯CPU合成边界测试；不触发真实网络/下载。"""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/download_alp15_full_test_candidates.py'
spec = importlib.util.spec_from_file_location('candidate_downloader', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    files = [{'path': f'camera/f{f}/f{f}.chunk_{c:04d}.zip', 'chunk': c, 'feature': f'f{f}'}
             for c in range(1, 634) for f in range(5)]
    return {'dataset': module.REPO, 'revision': module.REVISION, 'clip_count': 61301,
        'chunks': list(range(1, 634)), 'excluded_chunks': [0, 1000], 'files': files}


class CandidateDownloadTests(unittest.TestCase):
    def test_complete_snapshot_is_accepted(self):
        module.validate(fixture())

    def test_duplicate_or_missing_paths_are_rejected(self):
        for change in ('missing', 'duplicate'):
            value = fixture()
            if change == 'missing':
                value['files'].pop()
            else:
                value['files'][-1] = value['files'][0]
            with self.assertRaises(ValueError):
                module.validate(value)

    def test_escape_paths_are_rejected(self):
        for path in ('/tmp/input.zip', '../input.zip', 'a\\input.zip'):
            value = fixture()
            value['files'][0]['path'] = path
            with self.assertRaises(ValueError):
                module.validate(value)

    def test_used_chunk_is_rejected(self):
        value = fixture()
        value['excluded_chunks'].append(298)
        with self.assertRaises(ValueError):
            module.validate(value)

    def test_output_aliases_and_cache_symlinks_are_rejected_without_touching_target(self):
        plan = {'files': [], 'total_bytes': 1}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            with self.assertRaises(ValueError):
                module.prepare_output(root / 'new' / '..' / 'other', plan, 'a' * 64)
            self.assertFalse((root / 'new').exists())
            outside = root / 'outside'
            outside.mkdir()
            output = root / 'output'
            output.mkdir()
            (output / '.cache').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                module.prepare_output(output, plan, 'a' * 64)
            self.assertEqual(list(outside.iterdir()), [])

    def test_receipt_symlink_and_foreign_receipt_are_preserved(self):
        plan = {'files': [], 'total_bytes': 1}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            output = root / 'output'
            output.mkdir()
            target = root / 'user-log.txt'
            target.write_text('untouched\n')
            (output / 'verified-files.jsonl').symlink_to(target)
            with self.assertRaises(ValueError):
                module.prepare_output(output, plan, 'a' * 64)
            self.assertEqual(target.read_text(), 'untouched\n')
            second = root / 'second'
            second.mkdir()
            receipt = second / 'download-complete.jsonl'
            receipt.write_text('{"plan_sha256": "foreign"}\n')
            with self.assertRaises(ValueError):
                module.prepare_output(second, plan, 'a' * 64)
            self.assertEqual(receipt.read_text(), '{"plan_sha256": "foreign"}\n')

    def test_masked_sha_and_bad_total_are_rejected(self):
        value = fixture()
        plan = {'dataset': module.REPO, 'revision': module.REVISION,
            'files': [{'path': r['path'], 'size': 1, 'sha256': 'a' * 64} for r in value['files']],
            'total_bytes': 3165}
        module.validate(value, plan)
        for bad in ('total', 'sha'):
            altered = copy.deepcopy(plan)
            if bad == 'total':
                altered['total_bytes'] += 1
            else:
                altered['files'][0]['sha256'] = '*' * 64
            with self.assertRaises(ValueError):
                module.validate(value, altered)

    def test_default_cli_has_no_network_or_download(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            value = fixture()
            data = b'clip_id,chunk,t0_us\n'
            (root / 'test-candidates.csv').write_bytes(data)
            value['test_candidates_sha256'] = hashlib.sha256(data).hexdigest()
            snapshot = root / 'snapshot.json'
            snapshot.write_text(json.dumps(value))
            result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--snapshot', str(snapshot),
                '--plan', str(root / 'absent-plan.json')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('No network/download', result.stdout)
            self.assertFalse((root / 'absent-plan.json').exists())

    def test_download_requires_explicit_hash_and_bytes_before_creating_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            value = fixture()
            data = b'clip_id,chunk,t0_us\n'
            (root / 'test-candidates.csv').write_bytes(data)
            value['test_candidates_sha256'] = hashlib.sha256(data).hexdigest()
            value['scope'] = 'synthetic_fixture'
            snapshot = root / 'snapshot.json'
            snapshot.write_text(json.dumps(value))
            plan = {'dataset': module.REPO, 'revision': module.REVISION,
                'snapshot_sha256': module.digest(snapshot), 'scope': value['scope'],
                'files': [{'path': r['path'], 'size': 1, 'sha256': 'a' * 64} for r in value['files']],
                'total_bytes': 3165}
            quote = root / 'plan.json'
            quote.write_text(json.dumps(plan))
            output = root / 'no-download-created'
            result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--snapshot', str(snapshot),
                '--plan', str(quote), '--download', '--output', str(output)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--confirm-bytes', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
