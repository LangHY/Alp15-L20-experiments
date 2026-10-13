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
    for name in ('reevaluate_saved_adapters.py','adapter_checkpoint.py','controlled_recovery_validation.py','student_interface_distill.py','controlled_8b_diagnostic.py','run_mlp_experiment.py'):
        shutil.copy2(root/'compression'/name,out/'code'/name)
    provenance=json.loads((source/'provenance.json').read_text())
    parent=Path(provenance['parent_experiment'])
    candidate=json.loads((parent/'result.json').read_text())['candidates'][0]
    overlay=Path(candidate['overlay_directory'])
    if file_digest(overlay/'manifest.json')!=provenance['parent_overlay_sha256']:raise RuntimeError('Parent manifest identity changed')
    manifest=json.loads((overlay/'manifest.json').read_text())
    planned=[0,*range(1,20)]
    checkpoints={step:source/'adapter_checkpoints'/f'step_{step:06d}.pt' for step in planned}
    # Exact filenames checked before GPU load.
    for path in checkpoints.values():
        if not path.is_file():raise FileNotFoundError(path)
    provenance.update(started_at_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        source_experiment=str(source),source_provenance_sha256=file_digest(source/'provenance.json'),
        purpose='Early-step checkpoint ADE audit; no training or pruning',
        checkpoint_schedule=planned,eval_every=1,steps=19,save_adapters_every=0,
        checkpoints={str(step):{'file':str(path),'sha256':file_digest(path)} for step,path in checkpoints.items()},
        code_sha256={x.name:file_digest(x) for x in (out/'code').iterdir()},
        limitations=['Validation reused for checkpoint selection; no holdout or external generalization claim',
                      'Only steps 0 through 19 audited; later unmeasured steps remain unknown'])
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
            model.eval().requires_grad_(False)
            base_weights=[m.base.weight.detach().cpu().clone() for m in adapters]
            order=[f'{i}.{name}' for i in range(len(model.vlm.model.language_model.layers)) for name in ('gate_proj','up_proj','down_proj')]
            reference=json.loads((source/'task_validation_reference.json').read_text())
            validator=ControlledValidation(reference['directory'],out,provenance,json.loads((source/'split.json').read_text()),parent_experiment=parent)
            for step,path in checkpoints.items():
                if file_digest(path)!=provenance['checkpoints'][str(step)]['sha256']:raise RuntimeError('Adapter checksum changed')
                restored=restore_adapter_checkpoint(adapters,path,provenance['parent_overlay_sha256'],order)
                if restored!=step:raise RuntimeError('Checkpoint step mismatch')
                started=time.monotonic();ade,result=validator.evaluate(model,interface,adapters,base_weights,step)
                if step==0:
                    expected=json.loads((source/'task_validation/step_000000/result.json').read_text())
                    if result['mlp_tensor_sha256']!=expected['mlp_tensor_sha256'] or abs(ade-expected['mean_ADE_m'])>1e-4:raise RuntimeError('Step zero reproduction failed')
                completed.append({'step':step,'ADE_m':result['mean_ADE_m'],'status':result['status'],'elapsed_seconds':time.monotonic()-started,'adapter_sha256':provenance['checkpoints'][str(step)]['sha256']})
                write_json(out/'audit_progress.json',{'status':'running','planned_steps':planned,'completed':completed})
            if protected_digest(model)!=protected:raise RuntimeError('Protected weights changed during evaluation')
            valid=[x for x in completed if x['ADE_m'] is not None]
            best=min(valid,key=lambda x:x['ADE_m'])
            selected=json.loads((out/f"task_validation/step_{best['step']:06d}/result.json").read_text())
            write_json(out/'selected_task_validation.json',selected)
            write_json(out/'audit_result.json',{'status':'completed','params':candidate['params'],'best_step':best['step'],'best_ADE_m':best['ADE_m'],'parent_ADE_m':completed[0]['ADE_m'],'completed':completed,'all_saved_steps_evaluated':False,'selection_uses_holdout':False,'protected_sha256_after':protected})
            write_json(out/'audit_progress.json',{'status':'completed','planned_steps':planned,'completed':completed})
    except BaseException as exc:
        write_json(out/'audit_progress.json',{'status':'failed','error':repr(exc),'planned_steps':planned,'completed':completed});raise
    finally:
        for script in ('export_curves.py','plot_controlled_recovery.py'):
            command=[sys.executable,str(root/'compression'/script),'--experiment',str(out)]
            if script=='export_curves.py':command+=['--language','zh']
            with (out/(script+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)

if __name__=='__main__':main()
