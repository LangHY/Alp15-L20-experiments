import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from profile_recovered8_pruning import measure


class NonuniformMeasurementTests(unittest.TestCase):
    def test_each_actual_source_width_reproduces_parent(self):
        torch.manual_seed(42)
        for width in (256,384):
            mlp=SimpleNamespace(gate_proj=torch.nn.Linear(4096,width,bias=False),up_proj=torch.nn.Linear(4096,width,bias=False),down_proj=torch.nn.Linear(width,4096,bias=False),act_fn=torch.nn.functional.silu)
            x=torch.randn(3,4096)
            y=mlp.down_proj(mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x)).detach()
            before=mlp.down_proj.weight.detach().clone()
            with patch.object(torch.Tensor,'cuda',lambda self:self):
                rows=measure(mlp,x,y,torch.rand(width),grid=list(range(128,width+1,128)))
            self.assertEqual(rows[-1]['width'],width)
            self.assertEqual(rows[-1]['calibration_nmse'],0.)
            self.assertGreater(rows[0]['calibration_nmse'],0.)
            torch.testing.assert_close(mlp.down_proj.weight,before)

    def test_grid_cannot_exceed_source_width(self):
        mlp=SimpleNamespace(gate_proj=SimpleNamespace(out_features=256))
        with patch.object(torch.Tensor,'cuda',lambda self:self):
            with self.assertRaises(AssertionError):measure(mlp,torch.zeros(1,4096),torch.zeros(1,4096),torch.ones(256),grid=[128,384])


if __name__=='__main__':unittest.main()
