"""不依赖模型的指标聚合与输出保护回归测试。"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/alp15_quant_route_pilot.py'
spec = importlib.util.spec_from_file_location('quant_route', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RouteTests(unittest.TestCase):
    def test_percentile_linear_known_values(self):
        self.assertAlmostEqual(module.percentile([0, 10], .95), 9.5)
        self.assertEqual(module.percentile([4], .95), 4)

    def test_clip_equal_aggregation_not_seed_pseudoreplication(self):
        rows = [{'clip_id': c, 'ADE_m': ade, 'FDE_m': ade*2,
                 'paired_ADE_difference_m': 0} for c, ade in [('a', 1), ('a', 3), ('b', 10)]]
        result = module.summary(rows)
        self.assertEqual(result['clips'], 2)
        self.assertEqual(result['trajectories'], 3)
        self.assertEqual(result['mean_ADE_m'], 6)
        self.assertEqual(result['mean_FDE_m'], 12)
        self.assertAlmostEqual(result['clip_mean_ADE_p95_m'], 9.6)

    def test_preflight_failure_retains_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'new'
            run = subprocess.run([sys.executable, str(SCRIPT), '--root', '/missing-alp15-root',
                                  '--output', str(out)], capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            result = json.loads((out/'result.json').read_text())
            self.assertEqual(result['status'], 'failed_preflight')
            self.assertIn('type', result['error'])

    def test_existing_empty_directory_stays_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'existing'
            out.mkdir()
            run = subprocess.run([sys.executable, str(SCRIPT), '--root', '/missing-alp15-root',
                                  '--output', str(out)], capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(list(out.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
