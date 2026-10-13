"""Check dense generated KV vs standardized full-prefill on two fixed clips."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from controlled_8b_diagnostic import generate_prefix,sampled,digest
from run_mlp_experiment import (_enable_offline_mode,_block_network,_verify_model,_verify_dataset,
    _load_local_model_and_processor,_load_offline_dataset_interface,protected_digest,file_digest)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.experiment.resolve()
    baseline=json.loads((out/'result.json').read_text());assert baseline['status']=='completed'
    _enable_offline_mode();torch.set_num_threads(8)
    configs=root/'alp15-configs-delivery-20260919';cosmos=configs/'dependencies/configs/Cosmos-Reason2-8B';qwen=configs/'dependencies/configs/Qwen3-VL-2B-Instruct'
    result={'status':'running','scope':'Original two validation clips; seed42; actual generated dense KV compared to saved controlled full-prefill trajectories','records':[]}
    def save():(out/'dense_operator_check.json').write_text(json.dumps(result,indent=2)+'\n')
    with _block_network(),torch.no_grad():
        config=_verify_model(root/'models');_verify_dataset(root/'alp15-test-clip-delivery-20260917');interface=_load_offline_dataset_interface(root/'alp15-test-clip-delivery-20260917')
        model,_=_load_local_model_and_processor(root/'models',cosmos,qwen,config);model.eval().requires_grad_(False)
        protected=protected_digest(model)
        assert protected==baseline['protected_sha256_before']
        result['protected_sha256']=protected
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        for path in sorted((root/'experiments/mlp_progressive76_20261002_003000/teacher_cache').glob('validation_*.pt')):
            example=torch.load(path,weights_only=True,map_location='cpu');generated,actual=generate_prefix(model,example)
            assert generated['prefix_sha256']==digest(example['inputs']['input_ids'])
            inputs={k:v.cuda() for k,v in generated['inputs'].items()};data=load_physical_aiavdataset(example['clip_id'],t0_us=5_100_000,avdi=interface,maybe_stream=False)
            # Preview legacy coupled noise without changing the generator state.
            state=torch.cuda.get_rng_state();noise=torch.randn(1,*model.action_space.get_action_space_dims(),device='cuda');coupled_sha=digest(noise);torch.cuda.set_rng_state(state)
            measured=sampled(model,actual,inputs,data,42)
            ref=next(r for r in baseline['records'] if r['model']=='dense' and r['prefix']=='teacher' and r['seed']==42 and r['clip_id']==example['clip_id'])
            assert measured['initial_noise_sha256']==ref['initial_noise_sha256']
            a=torch.tensor(measured['pred_xyz']);b=torch.tensor(ref['pred_xyz']);nmse=((a-b).square().sum()/a.square().sum().clamp_min(1e-12)).item()
            row={'clip_id':example['clip_id'],'prefix_sha256':generated['prefix_sha256'],'generation_steps':generated['generated_tokens']+1,'coupled_noise_sha256':coupled_sha,'controlled_noise_sha256':measured['initial_noise_sha256'],'same_noise_between_protocols':coupled_sha==measured['initial_noise_sha256'],'trajectory_nmse':nmse,'max_abs_xyz_difference_m':(a-b).abs().max().item(),'actual_generated_KV_ADE_m':measured['ADE_m'],'full_prefill_ADE_m':ref['ADE_m'],'tolerance_nmse':1e-3}
            result['records'].append(row);save();print('DENSE_OPERATOR',row,flush=True)
            assert nmse<=1e-3
        assert protected_digest(model)==protected
    result['status']='completed';save();shutil.copy2(__file__,out/'code'/Path(__file__).name)
    print('DENSE_OPERATOR_COMPLETED',flush=True)


if __name__=='__main__':main()
