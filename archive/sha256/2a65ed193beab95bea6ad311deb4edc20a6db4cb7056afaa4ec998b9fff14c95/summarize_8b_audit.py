"""Summarize measured 8B diagnostic data without training or model selection."""
import argparse
import csv
import hashlib
import json
import shutil
import statistics as st
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.experiment.resolve()
    x=json.loads((out/'audit.json').read_text());assert x['status']=='completed'
    summary={'models':{},'old8_local_training':{},'historical_trajectory':{},'compression_scope':{}}
    for model,records in x['models'].items():
        summary['models'][model]={}
        for role in ['calibration','validation']:
            rows=[r for r in records if r['role']==role]
            scalar={k:st.mean(r[k] for r in rows) for k in ['interface_loss','hidden_nmse','key_nmse','value_nmse']}
            scalar['clips']=len(rows)
            scalar['layers']=[{'layer':i,**{k:st.mean(r['layers'][i][k] for r in rows) for k in ['input_state_nmse','local_output_nmse','full_graph_output_nmse','key_nmse','value_nmse']}} for i in range(36)] if model!='dense' else []
            summary['models'][model][role]=scalar
    historical=out/'historical_inputs';historical.mkdir(exist_ok=True)
    for name in ['mlp_stage8_joint_20261001_182631','mlp_layerwise8_20261001_220500']:
        base=root/'experiments'/name
        for relative in ['provenance.json','expanded_checks/result.json']:
            dst=historical/name/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/relative,dst)
        data=json.loads((base/'expanded_checks/result.json').read_text())
        summary['historical_trajectory'][name]={r:{k:v for k,v in data[r].items() if k!='paired_records'} for r in ['expanded_validation','expanded_holdout']}
    local=root/'experiments/mlp_stage8_joint_20261001_182631/width_5376/distillation_progress.json'
    shutil.copy2(local,historical/'accepted8_local_distillation.json');layers=json.loads(local.read_text())
    summary['old8_local_training']={'layers':len(layers),'best_at_final_step':sum(r['best_step']==300 for r in layers),'best_before_final_step':sum(r['best_step']<300 for r in layers),'baseline_val_nmse_mean':st.mean(r['baseline_val_nmse'] for r in layers),'final_val_nmse_mean':st.mean(r['final_val_nmse'] for r in layers),'final_train_nmse_mean':st.mean(r['final_train_nmse'] for r in layers),'worst_layers':[{k:r[k] for k in ['layer','baseline_val_nmse','final_val_nmse','final_train_nmse','best_step']} for r in sorted(layers,key=lambda r:r['final_val_nmse'],reverse=True)[:9]]}
    old=8020878578;fixed=5642708210
    for name,total in [('old8',old),('progressive76',7624516850),('layerwise7',7228155122)]:
        summary['compression_scope'][name]={'total_params':total,'mlp_params':total-fixed,'total_reduction_from8':1-total/old,'mlp_reduction_from8':1-(total-fixed)/(old-fixed)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    curves=out/'curves';curves.mkdir(exist_ok=True)
    flat=[{'model':m,'role':r,**v} for m,rs in summary['models'].items() for r,z in rs.items() for v in z['layers']]
    with (curves/'layer_state_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc');plt.rcParams['svg.fonttype']='path'
    fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    rows=summary['models']['accepted_uniform8']['validation']['layers'];indices=[v['layer'] for v in rows]
    axes[0].plot(indices,[v['input_state_nmse'] for v in rows]);axes[0].set_title('旧 8B：进入每层 MLP 的状态偏差',fontproperties=font)
    for key,label in [('local_output_nmse','固定教师输入：局部误差'),('full_graph_output_nmse','实际学生输入：整网输出误差')]:
        axes[1].plot(indices,[v[key] for v in rows],label=label)
    axes[1].set_title('旧 8B：相同位置的局部与整网误差',fontproperties=font);axes[1].legend(prop=font)
    for model,label in [('accepted_uniform8','旧 8B 均匀版'),('nonuniform8','非均匀 8B 版')]:
        rs=summary['models'][model]['validation']['layers'];axes[2].plot(indices,[(v['key_nmse']+v['value_nmse'])/2 for v in rs],label=label)
    axes[2].set_title('两种 8B：KV 偏差',fontproperties=font);axes[2].legend(prop=font)
    for ax in axes:
        ax.set_xlabel('层编号（0–35）',fontproperties=font);ax.set_ylabel('NMSE（越低越好）',fontproperties=font);ax.grid(alpha=.2)
    fig.suptitle('固定原 2 段验证、完整教师 CoC 前缀：不代表轨迹 ADE 或泛化',fontproperties=font)
    for suffix in ['png','svg']:fig.savefig(curves/f'8b_state_shift_zh.{suffix}',dpi=160,bbox_inches='tight')
    plt.close(fig);shutil.copy2(__file__,out/'code'/Path(__file__).name)
    manifest={'inputs':[{'file':str(out/'audit.json'),'sha256':hashlib.sha256((out/'audit.json').read_bytes()).hexdigest()}],'semantics':'Equal clip-weight mean on original two validation clips, layer axis, no interpolation or trajectories','outputs':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in curves.iterdir() if p.is_file()]}
    (curves/'state_shift_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({m:{r:{k:v for k,v in z.items() if k!='layers'} for r,z in rs.items()} for m,rs in summary['models'].items()},indent=2))


if __name__=='__main__':main()
