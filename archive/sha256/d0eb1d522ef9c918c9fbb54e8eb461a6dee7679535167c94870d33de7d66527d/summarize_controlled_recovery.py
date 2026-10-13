"""Paired parent/recovered clip metrics; validation-selected checkpoint is not an untouched test."""
import argparse,json,statistics,csv,random
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',type=Path,required=True)
args=parser.parse_args();p=args.experiment
parent=json.loads((p/'task_validation/step_000000/result.json').read_text())
selected=json.loads((p/'selected_task_validation.json').read_text())
parent_map={(r['clip_id'],r['seed']):r for r in parent['records']};rows=[]
for r in selected['records']:
    before=parent_map[(r['clip_id'],r['seed'])]
    assert r['initial_noise_sha256']==before['initial_noise_sha256']
    rows.append({'clip_id':r['clip_id'],'seed':r['seed'],'parent_ADE_m':before['ADE_m'],
        'recovered_ADE_m':r['ADE_m'],'teacher_ADE_m':r['dense_ADE_m'],
        'recovered_minus_parent_m':r['ADE_m']-before['ADE_m'],
        'recovered_minus_teacher_m':r['ADE_m']-r['dense_ADE_m'],
        'same_text_prefix_as_parent':r['prefix_sha256']==before['prefix_sha256']})
clips=sorted({r['clip_id'] for r in rows});assert len(clips)==20 and len(rows)==60
optimization=p/'optimization_split.json'
training_scope='Only four original training clips (possibly multiple related time windows)'
if optimization.exists():
    training_split=json.loads(optimization.read_text())
    training_scope=f"{len(training_split['training'])} training clips from one new chunk; {len(training_split['monitor_validation'])} monitor clips receive no gradients"
keys=['parent_ADE_m','recovered_ADE_m','teacher_ADE_m','recovered_minus_parent_m','recovered_minus_teacher_m']
clip_rows=[{'clip_id':clip,**{k:statistics.mean(r[k] for r in rows if r['clip_id']==clip) for k in keys}} for clip in clips]
values=[r['recovered_minus_parent_m'] for r in clip_rows];rng=random.Random(20261002)
bootstrap=sorted(statistics.mean(rng.choices(values,k=20)) for _ in range(5000))
summary={'selected_step':selected['step'],'means':{k:statistics.mean(r[k] for r in clip_rows) for k in keys},
    'improved_clips':sum(v<0 for v in values),'worse_clips':sum(v>0 for v in values),
    'descriptive_paired_clip_bootstrap_ci95_m':[bootstrap[124],bootstrap[4874]],
    'bootstrap_seed':20261002,'resamples':5000,
    'limitations':['20 clips used for checkpoint selection; this CI does not correct selection bias','Three seeds averaged within clip, not 60 independent scenes',training_scope,'No holdout validation or external generalization claim'],
    'per_clip':clip_rows}
summary['relative_improvement_vs_parent_percent']=100*(1-summary['means']['recovered_ADE_m']/summary['means']['parent_ADE_m'])
(p/'paired_recovery_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name,data in [('paired_recovery_seeds.csv',rows),('paired_recovery_clips.csv',clip_rows)]:
    with (p/'curves'/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
print(json.dumps({k:v for k,v in summary.items() if k!='per_clip'},indent=2))
