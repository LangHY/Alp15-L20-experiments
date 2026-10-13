import copy,unittest
import torch
from torch import nn
from apply_measured_pruning import apply_plan

def fixture():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__();self.gate_proj=nn.Linear(16,256,bias=False);self.up_proj=nn.Linear(16,256,bias=False)
            self.down_proj=nn.Linear(256,16,bias=False);self.act_fn=nn.SiLU();self.intermediate_size=256
    model=nn.Module();model.vlm=nn.Module();model.vlm.model=nn.Module();model.vlm.model.language_model=nn.Module()
    layers=model.vlm.model.language_model.layers=nn.ModuleList([nn.Module(),nn.Module()])
    for layer in layers:layer.mlp=MLP()
    manifest={'layers':[{'layer':i,'original_width':512,'keep_width':256,'keep_indices':list(range(0,512,2))} for i in range(2)]}
    candidate={'validation_mean_minADE_m':1.29}
    plan={'layer_widths':[128,256],'parent_local_channel_ids':[list(range(128)),list(range(256))],
        'original_dense_channel_ids':[list(range(0,256,2)),list(range(0,512,2))],
        'ranking_uses_holdout':False,'ranking_uses_validation':False,'actual_total_params':18432,
        'keep_width':128,'profile_directory':'profile'}
    return model,manifest,candidate,plan

class Tests(unittest.TestCase):
    def test_physical_linked_shapes_and_dense_mapping(self):
        model,manifest,candidate,plan=fixture();before=copy.deepcopy(manifest)
        original=model.vlm.model.language_model.layers[0].mlp.down_proj.weight.detach().clone()
        result,selected=apply_plan(model,manifest,candidate,plan)
        mlp=model.vlm.model.language_model.layers[0].mlp
        self.assertEqual(mlp.gate_proj.weight.shape,(128,16));self.assertEqual(mlp.down_proj.weight.shape,(16,128))
        self.assertTrue(torch.equal(mlp.down_proj.weight,original[:,:128]))
        self.assertEqual(selected['params'],18432);self.assertEqual(manifest,before)
        self.assertEqual(result['layers'][0]['keep_indices'],list(range(0,256,2)))
    def test_bad_mapping_budget_or_leakage_rejected_before_mutation(self):
        for error in ['mapping','budget','leakage']:
            model,manifest,candidate,plan=fixture()
            if error=='mapping':plan['original_dense_channel_ids'][1][0]=1
            if error=='budget':plan['actual_total_params']+=1
            if error=='leakage':plan['ranking_uses_holdout']=True
            with self.assertRaises(ValueError):apply_plan(model,manifest,candidate,plan)
            self.assertTrue(all(layer.mlp.gate_proj.out_features==256 for layer in model.vlm.model.language_model.layers))

if __name__=='__main__':unittest.main()
