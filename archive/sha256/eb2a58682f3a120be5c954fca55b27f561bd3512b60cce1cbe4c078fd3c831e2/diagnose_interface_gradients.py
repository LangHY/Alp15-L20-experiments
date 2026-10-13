"""Read-only, fixed-protocol evaluation of saved adapters; no gradient updates."""
import argparse, json, shutil, sys, time, subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch
from adapter_checkpoint import restore_adapter_checkpoint
from controlled_recovery_validation import ControlledValidation
from student_interface_distill import install_adapters
from run_mlp_experiment import (_enable_offline_mode, _block_network, _verify_model,
    _verify_dataset, _verify_config_dir, _verify_config_source,
    _load_local_model_and_processor, _load_offline_dataset_interface,
    COSMOS_REVISION, QWEN_REVISION, MODEL_VLM_NAME,
    protected_digest, file_digest, load_overlay, write_json)

def main():
    parser=argparse.ArgumentParser()
    for arg in ('root','source','output'):parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args();root=args.root;source=args.source;out=args.output
    out.mkdir(parents=True,exist_ok=False)
    (out/'code').mkdir()
    for name in ('diagnose_interface_gradients.py','adapter_checkpoint.py','controlled_recovery_validation.py','student_interface_distill.py','controlled_8b_diagnostic.py','run_mlp_experiment.py'):
        shutil.copy2(root/'compression'/name,out/'code'/name)
    provenance=json.loads((source/'provenance.json').read_text())
    parent=Path(provenance['parent_experiment'])
    candidate=json.loads((parent/'result.json').read_text())['candidates'][0]
    overlay=Path(candidate['overlay_directory'])
    if file_digest(overlay/'manifest.json')!=provenance['parent_overlay_sha256']:raise RuntimeError('Parent manifest identity changed')
    manifest=json.loads((overlay/'manifest.json').read_text())
    planned=[]
    provenance.update(started_at_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        source_experiment=str(source),source_provenance_sha256=file_digest(source/'provenance.json'),
        purpose='Measure per-objective gradient magnitudes and cosine conflicts; no updates/checkpoints',
        checkpoint_schedule=planned,eval_every=1,steps=19,save_adapters_every=0,
        code_sha256={x.name:file_digest(x) for x in (out/'code').iterdir()},
        limitations=['Validation reused for checkpoint selection; no holdout or external generalization claim',
                      'Gradient conflicts are local proxies; they do not prove the cause of ADE degradation'])
    write_json(out/'provenance.json',provenance)
    shutil.copy2(source/'split.json',out/'split.json')
    shutil.copy2(source/'optimization_split.json',out/'optimization_split.json')
    _enable_offline_mode();torch.set_num_threads(8);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    configs=root/'alp15-configs-delivery-20260919'
    cosmos=configs/'dependencies/configs/Cosmos-Reason2-8B'
    qwen=configs/'dependencies/configs/Qwen3-VL-2B-Instruct'
    dataset=root/'alp15-test-clip-delivery-20260917'
    completed=[]
    write_json(out/'audit_progress.json',{'status':'initializing','planned_steps':planned})
    try:
        with _block_network():
            config=_verify_model(root/'models');_verify_dataset(dataset)
            for directory,label,repo,revision,name in ((cosmos,'Cosmos',MODEL_VLM_NAME,COSMOS_REVISION,'cosmos-reason2-source.json'),(qwen,'Qwen','Qwen/Qwen3-VL-2B-Instruct',QWEN_REVISION,'qwen3-vl-source.json')):
                _verify_config_dir(directory,label);_verify_config_source(directory,label,repo,revision,name,configs/'manifests')
            interface=_load_offline_dataset_interface(dataset)
            model,_=_load_local_model_and_processor(root/'models',cosmos,qwen,config)
            model.eval().requires_grad_(False)
            protected=protected_digest(model)
            if protected!=manifest['protected_sha256']:raise RuntimeError('Protected weights identity mismatch')
            load_overlay(model,overlay)
            if sum(x.numel() for x in model.parameters())!=candidate['params']:raise RuntimeError('Student parameter count changed')
            adapters,_=install_adapters(model,provenance['rank'],checkpoint_mlp=False)
            model.eval()
            from run_interface_recovery import forward_targets, text_logits
            from student_interface_distill import interface_loss, logits_kl, relative_mse
            import itertools, math
            donor=root/'experiments/mlp_newdata8_recovery_20261002_165238'
            optimization=json.loads((source/'optimization_split.json').read_text())
            controlled=json.loads((root/'experiments/controlled8_expanded20_20261002_125000/split.json').read_text())
            forbidden=set(controlled['expanded_validation']+controlled['expanded_holdout'])
            training=set(optimization['training']);monitor=set(optimization['monitor_validation'])
            if training&monitor or (training|monitor)&forbidden:raise RuntimeError('Data split overlap')
            samples=[]
            for role,n,allowed in [('calibration',12,training),('validation',8,monitor)]:
                paths=sorted((donor/'teacher_cache').glob(role+'_*.pt'))
                indices=[round(i*(len(paths)-1)/(n-1)) for i in range(n)]
                for i,index in enumerate(indices):
                    path=paths[index];example=torch.load(path,weights_only=True,map_location='cpu')
                    if example['clip_id'] not in allowed:raise RuntimeError('Teacher cache split mismatch')
                    samples.append((role,i%4,path,example['clip_id']))
            write_json(out/'diagnostic_split.json',{'samples':[{'role':r,'probe':i,'file':str(p),'sha256':file_digest(p),'clip_id':clip} for r,i,p,clip in samples],'holdout_used':False,'updates':0})
            trainable=[p for a in adapters for p in (a.a,a.b)]
            rows=[]
            for index,(role,probe,path,clip) in enumerate(samples):
                torch.cuda.reset_peak_memory_stats();started=time.monotonic()
                example=torch.load(path,weights_only=True,map_location='cpu')
                output=forward_targets(model,example,probe)
                _,pieces=interface_loss(output,example['targets'])
                logits=text_logits(model,output['hidden'][:,example['logit_mask'].cuda()])
                objectives={'hidden':pieces['hidden_nmse'],'kv':(pieces['key_nmse']+pieces['value_nmse'])/2,
                    'token':logits_kl(logits,example['teacher_logits']),
                    'expert':relative_mse(output['expert_velocities'][0],example['targets']['expert_velocities'][probe])}
                gradients={}
                for j,(name,loss) in enumerate(objectives.items()):
                    values=torch.autograd.grad(loss,trainable,retain_graph=j<len(objectives)-1,allow_unused=True)
                    gradients[name]=[g.detach().cpu() if g is not None else torch.zeros_like(p,device='cpu') for p,g in zip(trainable,values)]
                    del values
                def dot(a,b):return sum((x.double()*y.double()).sum().item() for x,y in zip(a,b))
                norms={k:math.sqrt(dot(g,g)) for k,g in gradients.items()}
                pairs={a+'__'+b:(dot(gradients[a],gradients[b])/max(norms[a]*norms[b],1e-30)) for a,b in itertools.combinations(gradients,2)}
                weights={'hidden':1,'kv':1,'token':.5,'expert':1}
                total=[sum(weights[k]*gradients[k][i] for k in gradients) for i in range(len(trainable))]
                align={k:dot(g,total)/max(norms[k]*math.sqrt(dot(total,total)),1e-30) for k,g in gradients.items()}
                row={'index':index,'role':role,'clip_id':clip,'expert_probe_index':probe,
                    'losses':{k:v.detach().item() for k,v in objectives.items()},'gradient_norms':norms,
                    'weighted_gradient_norms':{k:norms[k]*weights[k] for k in norms},
                    'pairwise_cosine':pairs,'cosine_to_combined_gradient':align,
                    'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'elapsed_seconds':time.monotonic()-started}
                if not all(math.isfinite(v) for v in norms.values()):raise RuntimeError('Nonfinite gradient')
                rows.append(row)
                with (out/'gradient_records.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
                write_json(out/'audit_progress.json',{'status':'running','completed_samples':len(rows),'planned_samples':len(samples)})
                print('GRADIENT_SAMPLE',json.dumps(row),flush=True)
                del example,output,logits,objectives,pieces,gradients,total
            if protected_digest(model)!=protected:raise RuntimeError('Protected weights changed')
            write_json(out/'gradient_summary.json',{'status':'completed','params':candidate['params'],'updates':0,'checkpoints_saved':0,
                'samples':len(rows),'roles':{role:{'pair_mean_cosine':{key:sum(r['pairwise_cosine'][key] for r in rows if r['role']==role)/sum(r['role']==role for r in rows),
                'conflict_fraction':sum(r['pairwise_cosine'][key]<0 for r in rows if r['role']==role)/sum(r['role']==role for r in rows)} for key in rows[0]['pairwise_cosine']} for role in ['calibration','validation']},
                'protected_sha256_after':protected,'ADE_measured':False})
            write_json(out/'audit_progress.json',{'status':'completed','completed_samples':len(rows),'planned_samples':len(samples)})
            import csv
            with (out/'gradient_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.writer(f);writer.writerow(['角色','片段','探针','目标','损失','梯度范数','加权梯度范数','与联合梯度余弦'])
                for r in rows:
                    for k in r['losses']:writer.writerow([r['role'],r['clip_id'],r['expert_probe_index'],k,r['losses'][k],r['gradient_norms'][k],r['weighted_gradient_norms'][k],r['cosine_to_combined_gradient'][k]])
    except BaseException as exc:
        write_json(out/'audit_progress.json',{'status':'failed','error':repr(exc),'planned_steps':planned,'completed':completed});raise
    finally:
        for script in ('export_curves.py','plot_controlled_recovery.py'):
            command=[sys.executable,str(root/'compression'/script),'--experiment',str(out)]
            if script=='export_curves.py':command+=['--language','zh']
            with (out/(script+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)

if __name__=='__main__':main()
