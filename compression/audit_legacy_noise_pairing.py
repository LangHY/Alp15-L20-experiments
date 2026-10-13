"""Audit legacy coupled noise vs identical teacher noise, reusing legacy teacher refs."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from controlled_8b_diagnostic import generate_prefix,digest
from expert_interface_distill import expert_velocity
from run_mlp_experiment import (_enable_offline_mode,_block_network,_verify_model,_verify_dataset,
    _load_local_model_and_processor,_load_offline_dataset_interface,load_overlay,protected_digest,file_digest)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.experiment.resolve()
    baseline=json.loads((out/'result.json').read_text());operator=json.loads((out/'dense_operator_check.json').read_text())
    assert baseline['status']==operator['status']=='completed'
    parent=root/'experiments/mlp_stage8_joint_20261001_182631';candidate=json.loads((parent/'width_5376/result.json').read_text())
    teacher_path=root/'experiments/mlp_prune_distill_20261001_171143/teacher_calibration.json'
    teacher=json.loads(teacher_path.read_text())
    old_dense={r['clip_id']:r for r in teacher['validation']};old_student={r['clip_id']:r for r in candidate['validation']}
    assert set(old_dense)==set(old_student)=={r['clip_id'] for r in operator['records']}
    _enable_offline_mode();torch.set_num_threads(8)
    configs=root/'alp15-configs-delivery-20260919';cosmos=configs/'dependencies/configs/Cosmos-Reason2-8B';qwen=configs/'dependencies/configs/Qwen3-VL-2B-Instruct'
    result={'status':'running','scope':'Two original validation clips; original text seed42; reconstruct legacy noise draw states, verified against actual CUDA RNG; reuse original dense legacy trajectory refs only','source_sha256':{'teacher':file_digest(teacher_path),'old_student':file_digest(parent/'width_5376/result.json'),'dense_operator':file_digest(out/'dense_operator_check.json')},'records':[]}
    def save():(out/'legacy_pairing_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with _block_network(),torch.no_grad():
        config=_verify_model(root/'models');_verify_dataset(root/'alp15-test-clip-delivery-20260917');interface=_load_offline_dataset_interface(root/'alp15-test-clip-delivery-20260917')
        model,_=_load_local_model_and_processor(root/'models',cosmos,qwen,config);model.eval().requires_grad_(False)
        load_overlay(model,parent/'width_5376/distilled');protected=protected_digest(model)
        assert protected==baseline['protected_sha256_before']
        vocab=model.vlm.lm_head.weight.shape[0]
        def set_legacy_state(steps):
            torch.manual_seed(42);torch.cuda.manual_seed(42)
            probs=torch.ones(1,vocab,device='cuda',dtype=torch.float32)/vocab
            for _ in range(steps):torch.multinomial(probs,num_samples=1)
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        for path in sorted((root/'experiments/mlp_progressive76_20261002_003000/teacher_cache').glob('validation_*.pt')):
            example=torch.load(path,weights_only=True,map_location='cpu');clip=example['clip_id'];generated,actual=generate_prefix(model,example)
            auto_state=torch.cuda.get_rng_state();auto_hash=digest(torch.randn(1,*model.action_space.get_action_space_dims(),device='cuda'));torch.cuda.set_rng_state(auto_state)
            check=next(r for r in operator['records'] if r['clip_id']==clip)
            student_steps=generated['generated_tokens']+1;teacher_steps=check['generation_steps']
            inputs={k:v.cuda() for k,v in generated['inputs'].items()};data=load_physical_aiavdataset(clip,t0_us=5_100_000,avdi=interface,maybe_stream=False)
            assert generated['prefix_sha256']==next(r['prefix_sha256'] for r in baseline['records'] if r['clip_id']==clip and r['model']=='student8' and r['prefix']=='student8')
            row={'clip_id':clip,'teacher_sampling_steps':teacher_steps,'student_sampling_steps':student_steps,'old_dense_ADE_m':old_dense[clip]['minADE_m'],'old_student_ADE_m':old_student[clip]['minADE_m'],'protocols':{}}
            for name,steps,expected in [('legacy_student_noise',student_steps,auto_hash),('paired_teacher_noise',teacher_steps,check['coupled_noise_sha256'])]:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    set_legacy_state(steps);state=torch.cuda.get_rng_state();noise_hash=digest(torch.randn(1,*model.action_space.get_action_space_dims(),device='cuda'));torch.cuda.set_rng_state(state)
                    assert noise_hash==expected,'Legacy RNG reconstruction did not match measured generation'
                    def step_fn(x,t):return expert_velocity(model,actual,inputs,{'x':x,'t':t})
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        action=model.diffusion.sample(batch_size=1,step_fn=step_fn,device=torch.device('cuda'),return_all_steps=False)
                        xyz,_=model.action_space.action_to_traj(action,data['ego_history_xyz'][:,-1].cuda(),data['ego_history_rot'][:,-1].cuda())
                    ade=torch.linalg.vector_norm(xyz[0,:,:2].float().cpu()-data['ego_future_xyz'][0,0,:,:2].float().cpu(),dim=-1).mean().item()
                row['protocols'][name]={'ADE_m':ade,'initial_noise_sha256':noise_hash,'pred_xyz':xyz[0].float().cpu().tolist()}
            reference=torch.tensor(old_student[clip]['pred_xyz']);reproduction=torch.tensor(row['protocols']['legacy_student_noise']['pred_xyz'])
            row['legacy_student_reproduction_nmse']=((reference-reproduction).square().sum()/reference.square().sum().clamp_min(1e-12)).item()
            assert row['legacy_student_reproduction_nmse']<=1e-3
            result['records'].append(row);save();print('LEGACY_NOISE_PAIRING',row['clip_id'],row['old_dense_ADE_m'],{k:v['ADE_m'] for k,v in row['protocols'].items()},flush=True)
        assert protected_digest(model)==protected
    result['status']='completed';save();shutil.copy2(__file__,out/'code'/Path(__file__).name)
    print('LEGACY_NOISE_PAIRING_COMPLETED',flush=True)


if __name__=='__main__':main()
