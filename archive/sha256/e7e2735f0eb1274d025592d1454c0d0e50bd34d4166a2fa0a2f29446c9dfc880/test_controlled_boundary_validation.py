import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import controlled_recovery_validation as module


class BoundaryValidationTests(unittest.TestCase):
    def run_validation(self, failed=False, unexpected=False):
        with tempfile.TemporaryDirectory() as directory:
            validator=module.ControlledValidation.__new__(module.ControlledValidation)
            validator.output=Path(directory)
            validator.clips=[str(i) for i in range(20)]
            validator.examples={c:{'clip_id':c} for c in validator.clips}
            validator.seeds=[42,43,44]
            validator.rows=[{'ADE_m':1.0}]*60
            validator.dense={(c,s):{'ADE_m':1.0,'initial_noise_sha256':'noise'} for c in validator.clips for s in validator.seeds}
            validator.check_parent_reproduction=False
            def prefix(model, example):
                if example['clip_id']=='3':
                    if unexpected: raise RuntimeError('unexpected computation failure')
                    if failed:
                        error=module.TrajectoryBoundaryError.__new__(module.TrajectoryBoundaryError)
                        error.details={'clip_id':'3','status':'trajectory_start_missing'}
                        raise error
                return {'inputs':{'input_ids':SimpleNamespace(cuda=lambda:None)}},None
            model=SimpleNamespace(vlm=SimpleNamespace(model=lambda **kwargs:None))
            with patch.object(module,'generate_prefix',side_effect=prefix), \
                 patch.object(module,'sampled',return_value={'ADE_m':1.0,'initial_noise_sha256':'noise'}), \
                 patch.object(module,'digest',return_value='prefix'), \
                 patch.object(module,'mlp_tensor_digest',return_value='model'), \
                 patch.object(module,'exported_weights',return_value=contextlib.nullcontext()), \
                 patch.object(module.torch.random,'fork_rng',return_value=contextlib.nullcontext()), \
                 patch.object(module.torch.cuda,'current_device',return_value=0), \
                 patch.object(module.torch,'autocast',return_value=contextlib.nullcontext()), \
                 patch('alpamayo1_5.load_physical_aiavdataset.load_physical_aiavdataset',return_value={}):
                return validator.evaluate(model,None,[],[],0)

    def test_missing_boundary_is_recorded_and_checkpoint_disqualified(self):
        score,result=self.run_validation(failed=True)
        self.assertEqual(score,float('inf'))
        self.assertIsNone(result['mean_ADE_m'])
        self.assertEqual(result['missing_markers'],1)
        self.assertEqual(len(result['records']),57)

    def test_valid_checkpoint_uses_all_twenty_clips(self):
        score,result=self.run_validation()
        self.assertEqual(score,1.0)
        self.assertEqual(len(result['records']),60)

    def test_unexpected_failure_is_not_swallowed(self):
        with self.assertRaisesRegex(RuntimeError,'unexpected computation'):
            self.run_validation(unexpected=True)


if __name__=='__main__':
    unittest.main()
