"""Read-only dense/8B teacher-forced state-shift audit; no training or trajectories."""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
import torch
from run_interface_recovery import (forward_targets, _enable_offline_mode, _block_network,
    _verify_model, _verify_dataset, _verify_config_dir, _verify_config_source,
    _load_local_model_and_processor, COSMOS_REVISION, QWEN_REVISION, MODEL_VLM_NAME,
    protected_digest, file_digest, load_overlay)
from student_interface_distill import relative_mse, interface_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False); (out/'code').mkdir(); (out/'state_cache').mkdir()
    source = root/'experiments/mlp_progressive76_20261002_003000'
    for name in ['audit_8b_recovery.py','run_interface_recovery.py','student_interface_distill.py','run_mlp_experiment.py','mlp_pruning.py']:
        shutil.copy2(Path(__file__).parent/name, out/'code'/name)
    shutil.copy2(source/'split.json', out/'split.json')
    result = {'status':'running','argv':sys.argv,'seed':42,'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'source_experiment':str(source),
        'scope':'Original four calibration and two validation clips, actual complete dense CoC prefix; no training, no trajectories, no holdout selection',
        'metrics':'NMSE; local output on dense input vs full-graph output on student input; selected positions from frozen cache',
        'missing_metrics':['Training losses, learning rate, gradient norm: no optimization in this diagnostic'],
        'inputs':{},'models':{},'code_sha256':{p.name:file_digest(p) for p in (out/'code').glob('*.py')}}
    _enable_offline_mode(); torch.set_num_threads(8); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    started=time.perf_counter()
    def save():
        result['elapsed_seconds']=time.perf_counter()-started
        result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with _block_network(), torch.no_grad():
        config=_verify_model(root/'models'); _verify_dataset(root/'alp15-test-clip-delivery-20260917')
        configs=root/'alp15-configs-delivery-20260919'; cosmos=configs/'dependencies/configs/Cosmos-Reason2-8B'; qwen=configs/'dependencies/configs/Qwen3-VL-2B-Instruct'
        for directory,label,repo,revision,name in [(cosmos,'Cosmos',MODEL_VLM_NAME,COSMOS_REVISION,'cosmos-reason2-source.json'),(qwen,'Qwen','Qwen/Qwen3-VL-2B-Instruct',QWEN_REVISION,'qwen3-vl-source.json')]:
            _verify_config_dir(directory,label); _verify_config_source(directory,label,repo,revision,name,configs/'manifests')
        model,_=_load_local_model_and_processor(root/'models',cosmos,qwen,config); model.eval().requires_grad_(False)
        result['protected_sha256_before']=protected_digest(model)
        paths=sorted((source/'teacher_cache').glob('*.pt'))
        assert len(paths)==6
        for label, overlay in [('dense',None),('accepted_uniform8',root/'experiments/mlp_stage8_joint_20261001_182631/width_5376/distilled'),('nonuniform8',root/'experiments/mlp_layerwise8_20261001_220500/width_layerwise_5376/distilled')]:
            if overlay:
                load_overlay(model,overlay); result['inputs'][str(overlay/'manifest.json')]=file_digest(overlay/'manifest.json')
            records=[]
            for path in paths:
                example=torch.load(path,weights_only=True,map_location='cpu'); positions=example['positions']; states={}; handles=[]
                def hook(index):
                    def capture(module,inputs,output):
                        states[index]={'x':inputs[0][:,positions.to(inputs[0].device)].detach().cpu(),'y':output[:,positions.to(output.device)].detach().cpu()}
                    return capture
                for index,layer in enumerate(model.vlm.model.language_model.layers):handles.append(layer.mlp.register_forward_hook(hook(index)))
                try:output=forward_targets(model,example)
                finally:
                    for handle in handles:handle.remove()
                loss,pieces=interface_loss(output,example['targets']); layer_rows=[]
                if label=='dense':torch.save(states,out/'state_cache'/path.name)
                else:
                    dense=torch.load(out/'state_cache'/path.name,weights_only=True,map_location='cpu')
                    for index,layer in enumerate(model.vlm.model.language_model.layers):
                        target=dense[index]; actual=states[index]
                        with torch.autocast('cuda',dtype=torch.bfloat16):local=layer.mlp(target['x'].cuda()).cpu()
                        layer_rows.append({'layer':index,'input_state_nmse':relative_mse(actual['x'],target['x']).item(),'local_output_nmse':relative_mse(local,target['y']).item(),'full_graph_output_nmse':relative_mse(actual['y'],target['y']).item(),'key_nmse':relative_mse(output['keys'][index],example['targets']['keys'][index]).item(),'value_nmse':relative_mse(output['values'][index],example['targets']['values'][index]).item()})
                records.append({'clip_id':example['clip_id'],'role':path.stem.split('_')[0],'interface_loss':loss.item(),**{k:v.item() for k,v in pieces.items()},'layers':layer_rows})
                result['inputs'][str(path)]=file_digest(path); print('AUDIT',label,path.stem,loss.item(),flush=True)
            result['models'][label]=records; save()
        result['protected_sha256_after']=protected_digest(model)
        assert result['protected_sha256_after']==result['protected_sha256_before']
    result['status']='completed'; save()
    rows=[{'model':m,'clip_id':r['clip_id'],'role':r['role'],**v} for m,records in result['models'].items() for r in records for v in r['layers']]
    with (out/'state_shift.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('AUDIT_COMPLETED',out,flush=True)


if __name__=='__main__':main()
