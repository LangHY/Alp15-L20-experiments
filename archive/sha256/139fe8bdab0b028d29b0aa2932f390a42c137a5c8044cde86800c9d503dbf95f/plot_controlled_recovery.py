"""Export observed controlled task ADE checkpoints without mixing evaluation protocols."""
import argparse,csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',type=Path,required=True)
args=parser.parse_args();p=args.experiment;rows=[];sources=[]
allocation=p/'as_applied_allocation.json'
model_label='8B'
plot_title='保持 8B 结构：整网恢复受控对照'
if allocation.exists():
    plan=json.loads(allocation.read_text())
    model_label=f"{plan['actual_total_params']/1e9:.3f}B 学生"
    plot_title='按层剪枝后：整网恢复受控对照'
for file in sorted((p/'task_validation').glob('step_*/result.json')):
    x=json.loads(file.read_text());assert x['trajectory_seeds']==[42,43,44]
    rows.append({'step':x['step'],'student_ADE_m':x['mean_ADE_m'],'teacher_ADE_m':x['dense_mean_ADE_m'],'clips':20,'seeds':'42,43,44'})
    sources.append({'file':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
out=p/'curves';out.mkdir(exist_ok=True)
if not rows:
    (out/'controlled_task_manifest.json').write_text(json.dumps({'missing':'No completed controlled validation checkpoints','inputs':sources})+'\n')
else:
    csvpath=out/'controlled_task_ADE.csv'
    with csvpath.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc');plt.rcParams['svg.fonttype']='path'
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    ax.plot([r['step'] for r in rows],[r['student_ADE_m'] for r in rows],'o-',label=model_label+'：固定 20 段、三种子均值')
    ax.axhline(rows[0]['teacher_ADE_m'],color='black',linestyle='--',label='配对噪声的原教师')
    ax.axhline(rows[0]['teacher_ADE_m']*1.15,color='gray',linestyle=':',label='沿用 15% 的诊断容忍线')
    ax.set_xlabel('优化步数（仅显示实际验证点）',fontproperties=font);ax.set_ylabel('轨迹 ADE（米）',fontproperties=font)
    ax.set_title(plot_title,fontproperties=font);ax.legend(prop=font);ax.grid(alpha=.2)
    paths=[csvpath]
    for suffix in ['png','svg']:
        path=out/f'controlled_task_ADE_zh.{suffix}';fig.savefig(path,dpi=160);paths.append(path)
    plt.close(fig)
    (out/'controlled_task_manifest.json').write_text(json.dumps({'inputs':sources,'observed_steps':[r['step'] for r in rows],'semantics':'Fixed 20 validation clips; average three independently paired seeds; checkpoint selection uses validation; no holdout','outputs':[{'file':x.name,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in paths]},indent=2)+'\n')
print('Exported controlled task checkpoints',len(rows))
