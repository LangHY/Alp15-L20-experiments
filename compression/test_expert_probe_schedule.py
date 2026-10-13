import random
import unittest
from expert_probe_schedule import expert_probe_index


class ProbeScheduleTests(unittest.TestCase):
    def test_each_epoch_balanced_and_each_clip_visits_all_times(self):
        visits = {i: [] for i in range(80)}
        for epoch in range(4):
            order = list(range(80))
            random.Random(42 + epoch).shuffle(order)
            probes = []
            for offset, example in enumerate(order):
                p = expert_probe_index(epoch * 80 + offset + 1, 80, example, 'balanced')
                probes.append(p)
                visits[example].append(p)
            self.assertEqual([probes.count(i) for i in range(4)], [20] * 4)
        self.assertTrue(all(sorted(p) == [0, 1, 2, 3] for p in visits.values()))

    def test_legacy_schedule_unchanged(self):
        self.assertEqual(expert_probe_index(80, 80, 19), 0)
        self.assertEqual(expert_probe_index(81, 80, 19), 1)


if __name__ == '__main__':
    unittest.main()
