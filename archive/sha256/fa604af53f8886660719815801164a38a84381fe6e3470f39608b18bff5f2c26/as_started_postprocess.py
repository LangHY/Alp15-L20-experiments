"""Wait for a launched recovery pipeline, then independently reload and export records."""
import argparse,json,os,shutil,subprocess,time,traceback
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--launch',type=Path,required=True)
args=parser.parse_args()
launch=json.loads(args.launch.read_text());out=Path(launch['experiment'])
root=out.parent.parent;python=root/'alpamayo1.5/.venv/bin/python'
exitcode=Path(str(out)+'.pipeline.exitcode')
status_path=Path(str(out)+'.postprocess.json')
status={'status':'waiting_pipeline','launch':str(args.launch),'pid':os.getpid(),'holdout_evaluated':False}
def save():
    status['updated_at_epoch']=time.time();status_path.write_text(json.dumps(status,indent=2)+'\n')
save()
try:
    while not exitcode.exists():
        if not Path('/proc/'+str(launch['pid'])).exists():raise RuntimeError('Pipeline disappeared without exit code; records retained')
        time.sleep(10)
    status['pipeline_exitcode']=int(exitcode.read_text())
    if status['pipeline_exitcode']!=0:raise RuntimeError('Pipeline failed; automatic failure export already attempted')
    status['status']='independent_reload';save()
    env=dict(os.environ);env['PYTHONPATH']=':'.join(str(root/p) for p in ['compression','scripts','logs/alp15_audit_20261001_165853/official_source/src'])
    def run(name,*extra):
        with (out/(name+'.log')).open('wb') as log:
            subprocess.run([str(python),str(root/'compression'/name),*extra],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    run('verify_controlled_recovery.py','--root',str(root),'--experiment',str(out))
    # Keep training split and task validation identities distinct for exported comparisons.
    optimization=json.loads((out/'optimization_split.json').read_text())
    reference=json.loads((out/'task_validation_reference.json').read_text())
    width=out/'width_5376'
    prior=width/'split.json'
    if prior.exists() and not (width/'as_started_split.json').exists():shutil.copy2(prior,width/'as_started_split.json')
    prior.write_text(json.dumps({'calibration':optimization['training'],'validation':reference['clips'],'holdout':[],
        'monitor_validation':optimization['monitor_validation'],'semantics':'New training/monitor clips; fixed old controlled20 task validation; no holdout run'},indent=2)+'\n')
    shutil.copy2(launch['log'],out/'run.log')
    export_code=out/'export_code';export_code.mkdir(exist_ok=True)
    for name in ['export_curves.py','plot_controlled_recovery.py','summarize_controlled_recovery.py','verify_controlled_recovery.py']:
        shutil.copy2(root/'compression'/name,export_code/name)
    status['status']='exporting_curves';save()
    run('export_curves.py','--experiment',str(out),'--language','zh')
    run('plot_controlled_recovery.py','--experiment',str(out))
    run('summarize_controlled_recovery.py','--experiment',str(out))
    result=json.loads((out/'controlled_recovery_result.json').read_text())
    reload=json.loads((out/'controlled_reload_check.json').read_text())
    status.update({'status':'completed','selected_step':result['selected_step'],
        'parent_ADE_m':result['parent_mean_ADE_m'],'selected_ADE_m':result['selected_mean_ADE_m'],
        'teacher_ADE_m':result['dense_mean_ADE_m'],'params':result['params'],
        'reload_status':reload['status'],'max_reload_trajectory_nmse':max(r['trajectory_nmse'] for r in reload['checks']),
        'limitations':['Validation-selected checkpoint; no untouched holdout run','One chunk of new training clips; recording-drive independence unknown']})
    save();print(json.dumps(status),flush=True)
except Exception as exc:
    status.update({'status':'failed','error':{'type':type(exc).__name__,'message':str(exc)}});save();traceback.print_exc();raise
