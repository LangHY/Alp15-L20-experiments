import json
import tempfile
import unittest
from pathlib import Path
from server import load_experiment, read_trace


class RecordedMetricsTests(unittest.TestCase):
    def test_partial_validation_never_becomes_full_ade(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); path=root/'experiment';path.mkdir()
            (path/'provenance.json').write_text(json.dumps({'steps':320,'eval_every':20}))
            (path/'training_trace.jsonl').write_text(json.dumps({'phase':'training','step':1,'loss':.3})+'\n{"phase":')
            directory=path/'task_validation/step_000020';directory.mkdir(parents=True)
            (directory/'progress.json').write_text(json.dumps({'records':[{'clip_id':'one','ADE_m':.1,'seed':42}]}))
            data=load_experiment(path)
            self.assertEqual(data['current_step'],1)
            self.assertIsNone(data['best_ADE_m'])
            self.assertIsNone(data['task_validation'][0]['ADE_m'])
            self.assertEqual(data['task_validation'][0]['records'],1)
            self.assertTrue(data['warnings'])

    def test_complete_pairs_and_invalid_boundaries_stay_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);(path/'provenance.json').write_text('{}')
            records=[{'clip_id':str(c),'seed':seed,'ADE_m':1.2,'dense_ADE_m':1.0} for c in range(20) for seed in [42,43,44]]
            for step,failures,value in [(0,[],1.2),(20,[{'clip_id':'3'}],None)]:
                p=path/f'task_validation/step_{step:06d}';p.mkdir(parents=True)
                (p/'result.json').write_text(json.dumps({'records':records if not failures else records[:-3],
                                                        'failures':failures,'mean_ADE_m':value}))
            data=load_experiment(path)
            self.assertEqual(data['best_step'],0)
            self.assertEqual(len(data['clip_comparison']),20)
            self.assertEqual(data['clip_comparison'][0]['seeds'],3)
            self.assertEqual(data['task_validation'][1]['status'],'invalid')


if __name__=='__main__':
    unittest.main()
