"""Export observed crossed diagnostic metrics, no fitting or training curves."""
import argparse
import csv
import hashlib
import json
import shutil
import statistics as st
import random
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args();out=args.experiment.resolve();x=json.loads((out/'result.json').read_text())
    assert x['status']=='completed' and len(x['records'])==len({r['clip_id'] for r in x['records']})*4*len(x['trajectory_seeds'])
    rows=x['records'];curves=out/'curves';curves.mkdir(exist_ok=True)
    conditions=[('dense','teacher'),('student8','teacher'),('dense','student8'),('student8','student8')]
    summary={'status':x['status'],'scope':x['scope'],'noise_seeds':x['trajectory_seeds'],'text_seed':x['text_seed'],'roles':{},'autoregressive_checks':x['autoregressive_checks']}
    paired=[]
    roles=[role for role in ['calibration','validation'] if any(r['role']==role for r in rows)]
    for role in roles:
        count=len({r['clip_id'] for r in rows if r['role']==role})
        clips=sorted({r['clip_id'] for r in rows if r['role']==role});assert len(clips)==count
        groups=[]
        for model,prefix in conditions:
            subset=[r for r in rows if r['role']==role and r['model']==model and r['prefix']==prefix];assert len(subset)==count*3
            seed_means={str(seed):st.mean(r['ADE_m'] for r in subset if r['seed']==seed) for seed in x['trajectory_seeds']}
            groups.append({'model':model,'prefix':prefix,'mean_ADE_m':st.mean(r['ADE_m'] for r in subset),'seed_means':seed_means,'seed_mean_range_m':[min(seed_means.values()),max(seed_means.values())], 'clips':count,'trajectories':len(subset)})
        for clip in clips:
            for seed in x['trajectory_seeds']:
                cells={(r['model'],r['prefix']):r for r in rows if r['clip_id']==clip and r['seed']==seed}
                assert len(cells)==4;assert len({r['initial_noise_sha256'] for r in cells.values()})==1
                a,b,c,d=[cells[k]['ADE_m'] for k in conditions]
                paired.append({'role':role,'clip_id':clip,'seed':seed,'dense_teacher_ADE_m':a,'student_teacher_ADE_m':b,'dense_student_ADE_m':c,'student_student_ADE_m':d,'weight_effect_m':b-a,'prefix_effect_m':c-a,'interaction_m':d-b-c+a,'total_effect_m':d-a})
        rs=[r for r in paired if r['role']==role]
        effects={k:st.mean(r[k] for r in rs) for k in ['weight_effect_m','prefix_effect_m','interaction_m','total_effect_m']}
        assert abs(effects['weight_effect_m']+effects['prefix_effect_m']+effects['interaction_m']-effects['total_effect_m'])<1e-10
        clip_effects=[{'clip_id':clip,**{k:st.mean(r[k] for r in rs if r['clip_id']==clip) for k in effects}} for clip in clips]
        uncertainty={}
        if count>=10:
            for key in effects:
                values=[r[key] for r in clip_effects];rng=random.Random(20261002)
                means=sorted(st.mean(rng.choices(values,k=count)) for _ in range(5000))
                uncertainty[key]={'lower_m':means[124],'upper_m':means[4874],'resamples':5000,'seed':20261002,'method':'Paired clip bootstrap; first average three seeds within each clip; empirical percentile endpoints'}
        summary['roles'][role]={'conditions':groups,'factorial_ADE_differences':effects,'per_clip_seed_averaged_effects':clip_effects,'paired_clip_bootstrap_ci95':uncertainty,'note':'Descriptive ADE contrasts for fixed clips; interactions present; not percentages of root-cause attribution or an external benchmark'}
    for name,data in [('crossed_ADE.csv',[{k:r[k] for k in ['model','prefix','clip_id','role','seed','prefix_sha256','prefix_length','initial_noise_sha256','ADE_m','pred_xyz_sha256']} for r in rows]),('paired_effects.csv',paired)]:
        with (curves/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc');plt.rcParams['svg.fonttype']='path'
    labels=['教师模型\n教师前缀','8B 模型\n教师前缀','教师模型\n8B 前缀','8B 模型\n8B 前缀']
    fig,axes=plt.subplots(1,len(roles),figsize=(6.5*len(roles),5),constrained_layout=True)
    if len(roles)==1:axes=[axes]
    for ax,role in zip(axes,roles):
        count=summary['roles'][role]['conditions'][0]['clips']
        title=f'固定 {count} 段'+('校准' if role=='calibration' else '验证')+'：受控诊断'
        gs=summary['roles'][role]['conditions'];means=[g['mean_ADE_m'] for g in gs]
        low=[g['mean_ADE_m']-g['seed_mean_range_m'][0] for g in gs];high=[g['seed_mean_range_m'][1]-g['mean_ADE_m'] for g in gs]
        ax.bar(range(4),means,yerr=[low,high],capsize=5,color=['#3b7197','#c88554','#79a4b6','#a25850'])
        ax.set_xticks(range(4),labels,fontproperties=font);ax.set_title(title,fontproperties=font);ax.set_ylabel('平均轨迹 ADE（米，越低越好）',fontproperties=font)
        ax.grid(axis='y',alpha=.2)
        for i,m in enumerate(means):ax.text(i,m+high[i]+.06,f'{m:.3f}',ha='center')
        ax.set_ylim(0,max(m+h for m,h in zip(means,high))*1.18)
    fig.suptitle('相同轨迹噪声配对；误差线为 3 个种子均值的范围（非置信区间）',fontproperties=font)
    for suffix in ['png','svg']:fig.savefig(curves/f'controlled_prefix_comparison_zh.{suffix}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    if (out/'legacy_pairing_audit.json').exists():
        legacy=json.loads((out/'legacy_pairing_audit.json').read_text());assert legacy['status']=='completed'
        lr=legacy['records'];assert len(lr)==2
        legacy_means={'dense_legacy_ADE_m':st.mean(r['old_dense_ADE_m'] for r in lr),
            'student_unpaired_legacy_ADE_m':st.mean(r['protocols']['legacy_student_noise']['ADE_m'] for r in lr),
            'student_paired_teacher_noise_ADE_m':st.mean(r['protocols']['paired_teacher_noise']['ADE_m'] for r in lr)}
        summary['legacy_noise_pairing']=legacy_means
        summary['legacy_noise_pairing']['note']='Same two clips, same student weights and student prefixes; only student initial noise pairing changes; exact legacy student reproduction checked'
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        with (curves/'legacy_noise_pairing.csv').open('w',newline='',encoding='utf-8-sig') as f:
            data=[{'clip_id':r['clip_id'],'teacher_sampling_steps':r['teacher_sampling_steps'],'student_sampling_steps':r['student_sampling_steps'],'dense_legacy_ADE_m':r['old_dense_ADE_m'],'student_original_ADE_m':r['protocols']['legacy_student_noise']['ADE_m'],'student_paired_ADE_m':r['protocols']['paired_teacher_noise']['ADE_m'],'reproduction_nmse':r['legacy_student_reproduction_nmse']} for r in lr]
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for ax,key,title in [(axes[0],'student_unpaired_legacy_ADE_m','旧协议：共用 RNG，噪声未配对'),(axes[1],'student_paired_teacher_noise_ADE_m','仅修正 8B 噪声：与教师相同')]:
            values=[legacy_means['dense_legacy_ADE_m'],legacy_means[key]]
            ax.bar([0,1],values,color=['#3b7197','#c88554']);ax.set_xticks([0,1],['教师','同一 8B'],fontproperties=font)
            ax.set_title(title,fontproperties=font);ax.set_ylabel('原两段平均 ADE（米）',fontproperties=font);ax.set_ylim(0,5.8);ax.grid(axis='y',alpha=.2)
            for i,v in enumerate(values):ax.text(i,v+.08,f'{v:.3f}',ha='center')
        fig.suptitle('原结果精确复现；只改变轨迹噪声配对，模型优劣反转',fontproperties=font)
        for suffix in ['png','svg']:fig.savefig(curves/f'legacy_noise_ranking_zh.{suffix}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    events=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
    with (curves/'inference_resource_samples.csv').open('w',newline='') as f:
        keys=['model','prefix','clip_id','seed','elapsed_seconds','cuda_allocated_bytes','cuda_reserved_bytes'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in events)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    ax.scatter([r['elapsed_seconds'] for r in events],[r['cuda_allocated_bytes']/2**30 for r in events],s=12,label='PyTorch 已分配')
    ax.scatter([r['elapsed_seconds'] for r in events],[r['cuda_reserved_bytes']/2**30 for r in events],s=12,label='PyTorch 已预留')
    ax.set_xlabel('实验启动后的累计秒数（含加载）',fontproperties=font);ax.set_ylabel('当前进程显存 GiB',fontproperties=font);ax.set_title('每条轨迹完成时实际采集的资源点',fontproperties=font);ax.legend(prop=font);ax.grid(alpha=.2)
    for suffix in ['png','svg']:fig.savefig(curves/f'controlled_resources_zh.{suffix}',dpi=160,bbox_inches='tight')
    plt.close(fig);shutil.copy2(__file__,out/'code'/Path(__file__).name)
    manifest={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':hashlib.sha256((out/'result.json').read_bytes()).hexdigest(),'semantics':f'{len(rows)} actual diagnostic trajectories, three seeds, four crossed conditions; no interpolation, no training loss curves','outputs':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in curves.iterdir() if p.is_file() and p.name!='controlled_charts_manifest.json']}
    (curves/'controlled_charts_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
