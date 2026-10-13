"""Read-only crossed model/prefix diagnostic with independent trajectory RNG."""
import argparse
import copy
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
import torch
from run_mlp_experiment import (_enable_offline_mode,_block_network,_verify_model,_verify_dataset,
    _verify_config_dir,_verify_config_source,_load_local_model_and_processor,
    _load_offline_dataset_interface,COSMOS_REVISION,QWEN_REVISION,MODEL_VLM_NAME,
    DATASET_REVISION,protected_digest,file_digest,load_overlay)
from expert_interface_distill import expert_velocity


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class TrajectoryBoundaryError(RuntimeError):
    def __init__(self, clip_id, sequence, prompt_length, tokenizer):
        super().__init__('Student failed verified trajectory-start boundary')
        self.details = {'clip_id':clip_id, 'status':'trajectory_start_missing',
            'generated_token_ids':sequence[0,prompt_length:].tolist(),
            'generated_text':tokenizer.decode(sequence[0,prompt_length:],skip_special_tokens=False),
            'max_new_tokens':256}


def generate_prefix(model, example):
    from transformers import StoppingCriteriaList,LogitsProcessorList
    from alpamayo1_5.models.token_utils import StopAfterEOS
    from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
    prompt_length=int(example['positions'][example['logit_mask']][0])+1
    inputs={k:v.cuda() for k,v in example['inputs'].items()}
    inputs['input_ids']=inputs['input_ids'][:,:prompt_length]
    inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
    future=model.tokenizer.convert_tokens_to_ids('<|traj_future_start|>')
    gen=copy.deepcopy(model.vlm.generation_config)
    gen.top_p,gen.temperature,gen.top_k,gen.do_sample=0.98,0.6,None,True
    gen.num_return_sequences,gen.max_new_tokens=1,256
    gen.output_logits,gen.return_dict_in_generate=False,True
    gen.pad_token_id=model.tokenizer.pad_token_id
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=model.vlm.generate(**inputs,generation_config=gen,
            stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=future)]),
            logits_processor=LogitsProcessorList([ExpertLogitsProcessor(model.config.traj_token_start_idx,model.config.traj_vocab_size)]))
    sequence=output.sequences
    output.rope_deltas=model.vlm.model.rope_deltas.detach().clone()
    prefix=sequence[:,:-1]
    if prefix[0,-1].item()!=future:
        raise TrajectoryBoundaryError(example['clip_id'],sequence,prompt_length,model.tokenizer)
    inputs['input_ids']=prefix;inputs['attention_mask']=torch.ones_like(prefix)
    generated={'inputs':{k:v.cpu() for k,v in inputs.items()},'clip_id':example['clip_id'],
        'prompt_length':prompt_length,'prefix_sha256':digest(prefix),
        'generated_tokens':prefix.shape[1]-prompt_length,
        'generated_text':model.tokenizer.decode(prefix[0,prompt_length:],skip_special_tokens=False)}
    # Preserve actual autoregressive KV for checking the full-prefill diagnostic.
    return generated,output


def sampled(model, output, inputs, data, seed):
    captured={}
    def step_fn(x,t):
        value=expert_velocity(model,output,inputs,{'x':x,'t':t})
        if 'velocity' not in captured:captured['velocity']=value.detach().float().cpu()
        return value
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        state=torch.cuda.get_rng_state()
        noise=torch.randn(1,*model.action_space.get_action_space_dims(),device='cuda')
        noise_hash=digest(noise);torch.cuda.set_rng_state(state)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            action=model.diffusion.sample(batch_size=1,step_fn=step_fn,device=torch.device('cuda'),return_all_steps=False)
            xyz,rot=model.action_space.action_to_traj(action,data['ego_history_xyz'][:,-1].cuda(),data['ego_history_rot'][:,-1].cuda())
    gt=data['ego_future_xyz'][0,0,:,:2].float().cpu();pred=xyz[0,:,:2].float().cpu()
    ade=torch.linalg.vector_norm(pred-gt,dim=-1).mean().item()
    if not torch.isfinite(xyz).all():raise RuntimeError('Nonfinite diagnostic trajectory')
    return {'ADE_m':ade,'initial_noise_sha256':noise_hash,'pred_xyz':xyz[0].float().cpu().tolist(),
        'pred_xyz_sha256':digest(xyz),'first_velocity':captured['velocity'].tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--expanded-validation-dir',type=Path)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if args.resume and args.expanded_validation_dir:raise ValueError('Expanded diagnostic cannot resume original six-clip run')
    (out/'code').mkdir();(out/'student_prefix_cache').mkdir()
    source=root/'experiments/mlp_progressive76_20261002_003000'
    for name in ['controlled_8b_diagnostic.py','expert_interface_distill.py','run_mlp_experiment.py','mlp_pruning.py','run_interface_recovery.py','student_interface_distill.py']:
        shutil.copy2(Path(__file__).parent/name,out/'code'/name)
    shutil.copy2(source/'split.json',out/'split.json')
    shutil.copy2(source/'provenance.json',out/'teacher_cache_provenance.json')
    cache_prov=json.loads((source/'provenance.json').read_text())
    assert cache_prov['source_commit']=='36aeb4c5938cbc2eb2aed33b22434773da4ab639'
    assert cache_prov['checkpoint_revision']=='7aba8293c09993f2e125c6819df05d7fa3e873ea'
    assert cache_prov['dataset_revision']==DATASET_REVISION
    _enable_offline_mode();torch.set_num_threads(8)
    started=time.perf_counter();trace=(out/'events.jsonl').open('x')
    elapsed_offset=0.0
    result={'status':'running','argv':sys.argv,'text_seed':42,'trajectory_seeds':[42,43,44],
        'scope':'Four original calibration and two original validation clips; no training, no holdout; crossed models and fixed full prefixes',
        'teacher_prefix':'Existing verified dense CoC token cache; full-prefill KV standardized for crossed comparisons',
        'source_commit':'36aeb4c5938cbc2eb2aed33b22434773da4ab639','checkpoint_revision':'7aba8293c09993f2e125c6819df05d7fa3e873ea',
        'dataset_revision':DATASET_REVISION,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'code_sha256':{p.name:file_digest(p) for p in (out/'code').glob('*.py')},'inputs':{},'records':[],'autoregressive_checks':[],
        'missing_metrics':['Training loss, learning rate, gradient norm: no optimizer in this diagnostic']}
    if args.resume:
        previous=args.resume.resolve();prior=json.loads((previous/'result.json').read_text())
        assert prior['text_seed']==42 and prior['trajectory_seeds']==[42,43,44]
        assert prior['checkpoint_revision']==result['checkpoint_revision'] and prior['dataset_revision']==DATASET_REVISION
        reused=[r for r in prior['records'] if r['model']=='dense' and r['prefix']=='teacher'];assert len(reused)==18
        result['records']=reused;elapsed_offset=prior['elapsed_seconds']
        result['resumed_from']={'path':str(previous),'result_sha256':file_digest(previous/'result.json'),'reused_records':18,'discarded_partial_student_records':len(prior['records'])-18}
        shutil.copy2(previous/'result.json',out/'previous_failed_result.json');shutil.copytree(previous/'code',out/'previous_code')
        for line in (previous/'events.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['model']=='dense' and row['prefix']=='teacher':trace.write(line+'\n')
        trace.flush()
    def save():
        result['elapsed_seconds']=elapsed_offset+time.perf_counter()-started
        temporary=out/'result.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(out/'result.json')
    def event(row):
        row={**row,'elapsed_seconds':elapsed_offset+time.perf_counter()-started,'cuda_allocated_bytes':torch.cuda.memory_allocated(),'cuda_reserved_bytes':torch.cuda.memory_reserved()}
        trace.write(json.dumps(row)+'\n');trace.flush();print('CONTROLLED',row.get('model'),row.get('prefix'),row.get('clip_id'),row.get('seed'),row.get('ADE_m'),flush=True)
    configs=root/'alp15-configs-delivery-20260919';cosmos=configs/'dependencies/configs/Cosmos-Reason2-8B';qwen=configs/'dependencies/configs/Qwen3-VL-2B-Instruct'
    paths=sorted((source/'teacher_cache').glob('*.pt')) if not args.expanded_validation_dir else []
    if not args.expanded_validation_dir:assert len(paths)==6
    with _block_network(),torch.no_grad():
        config=_verify_model(root/'models');_verify_dataset(root/'alp15-test-clip-delivery-20260917');interface=_load_offline_dataset_interface(root/'alp15-test-clip-delivery-20260917')
        for directory,label,repo,revision,name in [(cosmos,'Cosmos',MODEL_VLM_NAME,COSMOS_REVISION,'cosmos-reason2-source.json'),(qwen,'Qwen','Qwen/Qwen3-VL-2B-Instruct',QWEN_REVISION,'qwen3-vl-source.json')]:
            _verify_config_dir(directory,label);_verify_config_source(directory,label,repo,revision,name,configs/'manifests')
        model,processor=_load_local_model_and_processor(root/'models',cosmos,qwen,config);model.eval().requires_grad_(False)
        result['protected_sha256_before']=protected_digest(model)
        overlay=root/'experiments/mlp_stage8_joint_20261001_182631/width_5376/distilled'
        result['inputs'][str(overlay/'manifest.json')]=file_digest(overlay/'manifest.json')
        if args.expanded_validation_dir:
            reference=args.expanded_validation_dir.resolve();split=json.loads((reference/'split.json').read_text());clips=split['expanded_validation'];assert len(clips)==len(set(clips))==20
            original=json.loads((source/'split.json').read_text());assert not set(clips)&set(original['calibration']+original['validation']+original['holdout'])
            result['scope']='Fixed twenty expanded validation clips; three independent trajectory seeds; four crossed conditions; no training and no holdout'
            result['teacher_prefix']='New dense CoC tokens only on fixed twenty validation clips; no reuse of old unpaired trajectory references'
            result['inputs'][str(reference/'split.json')]=file_digest(reference/'split.json')
            shutil.copy2(reference/'split.json',out/'split.json');(out/'teacher_prefix_cache').mkdir()
            from run_interface_recovery import prepare
            for index,clip in enumerate(clips):
                inputs,positions,logit_mask=prepare(model,processor,interface,{'clip_id':clip},rollout_prefix=True)
                path=out/'teacher_prefix_cache'/f'validation_{index:02d}.pt'
                torch.save({'clip_id':clip,'inputs':inputs,'positions':positions,'logit_mask':logit_mask},path);paths.append(path)
                print('CONTROLLED_TEACHER_PREFIX',index,clip,flush=True)
        for phase in ['dense_teacher_prefix','student_both_prefixes','dense_student_prefix']:
            if phase=='dense_teacher_prefix' and args.resume:
                print('CONTROLLED_REUSED_DENSE_TEACHER_PREFIX',18,flush=True)
                continue
            if phase=='student_both_prefixes':load_overlay(model,overlay)
            if phase=='dense_student_prefix':
                del model;gc.collect();torch.cuda.empty_cache()
                model,processor=_load_local_model_and_processor(root/'models',cosmos,qwen,config);model.eval().requires_grad_(False)
            label='student8' if phase=='student_both_prefixes' else 'dense'
            from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
            for path in paths:
                example=torch.load(path,weights_only=True,map_location='cpu');clip=example['clip_id'];role=path.stem.split('_')[0]
                result['inputs'][str(path)]=file_digest(path)
                data=load_physical_aiavdataset(clip,t0_us=5_100_000,avdi=interface,maybe_stream=False)
                prefixes=[('teacher',example['inputs'])]
                actual=None
                if phase=='student_both_prefixes':
                    generated,actual=generate_prefix(model,example);torch.save(generated,out/'student_prefix_cache'/path.name)
                    prefixes.append(('student8',generated['inputs']))
                elif phase=='dense_student_prefix':
                    generated=torch.load(out/'student_prefix_cache'/path.name,weights_only=True,map_location='cpu')
                    prefixes=[('student8',generated['inputs'])]
                for prefix,values in prefixes:
                    inputs={k:v.cuda() for k,v in values.items()}
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=model.vlm.model(**inputs,use_cache=True,return_dict=True)
                    for seed in result['trajectory_seeds']:
                        measured=sampled(model,output,inputs,data,seed)
                        record={'model':label,'prefix':prefix,'clip_id':clip,'role':role,'seed':seed,'prefix_sha256':digest(inputs['input_ids']),'prefix_length':inputs['input_ids'].shape[1],**measured}
                        result['records'].append(record);event({k:v for k,v in record.items() if k not in ['pred_xyz','first_velocity']});save()
                        if actual is not None and prefix=='student8' and seed==42:
                            # Actual generated cache contains future-start, extra token is not cached.
                            check_output=type('CachedOutput',(),{'past_key_values':actual.past_key_values,'rope_deltas':actual.rope_deltas})()
                            check=sampled(model,check_output,inputs,data,seed)
                            a=torch.tensor(check['pred_xyz']);b=torch.tensor(measured['pred_xyz'])
                            nmse=(a-b).square().sum()/a.square().sum().clamp_min(1e-12)
                            result['autoregressive_checks'].append({'clip_id':clip,'model':label,'seed':seed,'trajectory_nmse':nmse.item(),'max_abs_xyz_difference_m':(a-b).abs().max().item(),'autoregressive_ADE_m':check['ADE_m'],'full_prefill_ADE_m':measured['ADE_m']})
                            if nmse>1e-3:raise RuntimeError('Full-prefill trajectory differs from actual generated cache beyond diagnostic tolerance')
                    del output
                if actual is not None:del actual
                del data;gc.collect();torch.cuda.empty_cache()
        result['protected_sha256_after']=protected_digest(model);assert result['protected_sha256_after']==result['protected_sha256_before']
    assert len(result['records'])==len(paths)*4*len(result['trajectory_seeds'])
    for clip in {r['clip_id'] for r in result['records']}:
        for seed in result['trajectory_seeds']:
            rows=[r for r in result['records'] if r['clip_id']==clip and r['seed']==seed];assert len(rows)==4
            assert len({r['initial_noise_sha256'] for r in rows})==1
    result['status']='completed';save();trace.close();print('CONTROLLED_COMPLETED',out,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])/'result.json'
            if path.exists():
                result=json.loads(path.read_text());result['status']='failed';result['error']={'type':type(exc).__name__,'message':str(exc)}
                path.write_text(json.dumps(result,indent=2)+'\n')
        raise
