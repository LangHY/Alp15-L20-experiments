import json
import tempfile
import unittest
from pathlib import Path
import torch
from runtime_memory_record import MemoryRecorder, storage_inventory


class MemoryRecordingTests(unittest.TestCase):
    def test_shared_storage_views_count_full_backing_once(self):
        module = torch.nn.Module()
        full = torch.arange(128, dtype=torch.float32)
        module.register_buffer('view', full[:8])
        result = storage_inventory(module, [full[16:24]])
        self.assertEqual(result['unique_storage_bytes_by_device']['cpu'], 512)
        self.assertEqual(result['unique_storages'], 1)
        self.assertEqual(result['logical_tensor_bytes_including_aliases'], 64)

    def test_failure_keeps_measurement_and_rethrows(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'phases.jsonl'
            recorder = MemoryRecorder(p)
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with recorder.phase('failed_load'):
                    raise RuntimeError('test failure')
            row = json.loads(p.read_text())
            self.assertEqual(row['status'], 'failed')
            self.assertIsNone(row['cuda_peak_allocated_bytes'])
            with self.assertRaises(FileExistsError):
                MemoryRecorder(p)


if __name__ == '__main__':
    unittest.main()
