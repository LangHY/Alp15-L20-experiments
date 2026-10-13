"""Independent-process reload check on two fixed validation clips, no training."""
import argparse,json,torch
from pathlib import Path
from run_mlp_experiment import (_enable_offline_mode,_block_network,_verify_model,_verify_dataset,
    _load_local_model_and_processor,_load_offline_dataset_interface,load_overlay,protected_digest,write_json)
from controlled_8b_diagnostic import generate_prefix,sampled
from student_interface_distill import mlp_tensor_digest

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--experiment',type=Path,required=True)
args=parser.parse_args();root=args.root;p=args.experiment
x=json.loads((p/'controlled_recovery_result.json').read_text());selected=json.loads((p/'selected_task_validation.json').read_text());reference=json.loads((p/'task_validation_reference.json').read_text())
_enable_offline_mode();torch.set_num_threads(8)
with _block_network(),torch.no_grad():
    dataset=root/'alp15-test-clip-delivery-20260917';_verify_dataset(dataset)
    config=_verify_model(root/'models');configs=root/'alp15-configs-delivery-20260919'
    model,processor=_load_local_model_and_processor(root/'models',configs/'dependencies/configs/Cosmos-Reason2-8B',configs/'dependencies/configs/Qwen3-VL-2B-Instruct',config)
    model.eval().requires_grad_(False);load_overlay(model,Path(x['overlay_directory']))
    assert protected_digest(model)==x['protected_sha256_before']
    assert sum(t.numel() for t in model.parameters())==x['params']
    assert mlp_tensor_digest(model)==selected['mlp_tensor_sha256']
    interface=_load_offline_dataset_interface(dataset)
    examples={}
    for f in (Path(reference['directory'])/'teacher_prefix_cache').glob('*.pt'):
        a=torch.load(f,weights_only=True,map_location='cpu');examples[a['clip_id']]=a
    checks=[]
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    for clip in sorted(examples)[:2]:
        prefix,actual=generate_prefix(model,examples[clip]);del actual
        inputs={k:v.cuda() for k,v in prefix['inputs'].items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):output=model.vlm.model(**inputs,use_cache=True,return_dict=True)
        data=load_physical_aiavdataset(clip,t0_us=5_100_000,avdi=interface,maybe_stream=False)
        for seed in reference['trajectory_seeds']:
            got=sampled(model,output,inputs,data,seed)
            before=next(r for r in selected['records'] if r['clip_id']==clip and r['seed']==seed)
            assert got['initial_noise_sha256']==before['initial_noise_sha256'] and prefix['prefix_sha256']==before['prefix_sha256']
            a,b=torch.tensor(got['pred_xyz']),torch.tensor(before['pred_xyz'])
            nmse=((a-b).square().sum()/b.square().sum().clamp_min(1e-12)).item()
            assert nmse<=1e-3
            checks.append({'clip_id':clip,'seed':seed,'trajectory_nmse':nmse,'ADE_m':got['ADE_m'],'selected_ADE_m':before['ADE_m']})
        del output,inputs,data
    write_json(p/'controlled_reload_check.json',{'status':'completed','checks':checks,'scope':'First two lexicographically fixed validation clips, three paired seeds; not full holdout','protected_sha256':protected_digest(model),'params':x['params']})
    print('RELOAD_VERIFIED',len(checks),max(r['trajectory_nmse'] for r in checks))
