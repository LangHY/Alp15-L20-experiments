import tempfile
import unittest
from pathlib import Path
import torch
from student_interface_distill import RecoveryLinear, exported_weights
from adapter_checkpoint import restore_adapter_checkpoint


class AdapterCheckpointTests(unittest.TestCase):
    def test_saved_adapter_reconstructs_identical_bf16_export(self):
        torch.manual_seed(42)
        adapter = RecoveryLinear(torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16), 2)
        with torch.no_grad():
            adapter.b.normal_()
        base = adapter.base.weight.detach().clone()
        with exported_weights([adapter], [base]):
            expected = adapter.base.weight.detach().clone()
        state = {'step':17,'rank':2,'scale':1.0,'adapters':[(adapter.a.detach().clone(),adapter.b.detach().clone())],
                 'parent_overlay_manifest_sha256':'parent','module_order':['0.gate_proj']}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.pt'
            torch.save(state,path)
            with torch.no_grad():
                adapter.a.zero_();adapter.b.zero_()
            with self.assertRaisesRegex(ValueError,'parent identity'):
                restore_adapter_checkpoint([adapter],path,'wrong',['0.gate_proj'])
            self.assertEqual(restore_adapter_checkpoint([adapter],path,'parent',['0.gate_proj']),17)
            with exported_weights([adapter],[base]):
                self.assertTrue(torch.equal(adapter.base.weight,expected))


if __name__ == '__main__':
    unittest.main()
