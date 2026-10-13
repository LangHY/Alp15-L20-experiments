"""Read-only ground-truth XY diagnostics on two largest validation regressions."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from run_mlp_experiment import _enable_offline_mode,_block_network,_load_offline_dataset_interface,DATASET_REVISION


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.experiment.resolve()
    result=json.loads((out/'result.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert result['status']=='completed';role=summary['roles']['validation']
    worst=sorted(role['per_clip_seed_averaged_effects'],key=lambda r:r['total_effect_m'],reverse=True)[:2]
    _enable_offline_mode()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc');plt.rcParams['svg.fonttype']='path'
    records=[];fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
    with _block_network():
        interface=_load_offline_dataset_interface(root/'alp15-test-clip-delivery-20260917')
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        for row,item in enumerate(worst):
            clip=item['clip_id'];data=load_physical_aiavdataset(clip,t0_us=5_100_000,avdi=interface,maybe_stream=False)
            gt=data['ego_future_xyz'][0,0].float().cpu().numpy();gt_xy=gt[:,:2]
            record={'clip_id':clip,'dataset_revision':DATASET_REVISION,'t0_us':5100000,'ground_truth_xyz':gt.tolist(),'total_mean_ADE_difference_m':item['total_effect_m'],'models':{}}
            a,b=axes[row];a.plot(gt[:,0],gt[:,1],color='black',linewidth=2,label='真实轨迹')
            for model,prefix,color,label in [('dense','teacher','#3b7197','教师'),('student8','student8','#bd7549','8B')]:
                values=[r for r in result['records'] if r['clip_id']==clip and r['model']==model and r['prefix']==prefix];assert len(values)==3
                preds=np.array([r['pred_xyz'] for r in values]);delta=preds[:,:,:2]-gt_xy
                record['models'][model]={'mean_ADE_m':float(np.linalg.norm(delta,axis=-1).mean()),'mean_abs_X_error_m':float(np.abs(delta[:,:,0]).mean()),'mean_abs_Y_error_m':float(np.abs(delta[:,:,1]).mean()),'mean_final_point_X_error_m':float(delta[:,-1,0].mean()),'mean_final_point_Y_error_m':float(delta[:,-1,1].mean())}
                for i,(r,pred) in enumerate(zip(values,preds)):
                    a.plot(pred[:,0],pred[:,1],color=color,alpha=.65,label=label+'：三个噪声种子' if i==0 else None)
                b.plot(np.arange(1,65)/10,np.linalg.norm(delta,axis=-1).mean(axis=0),color=color,label=label+'：三种子平均')
            a.set_aspect('equal',adjustable='datalim');a.set_xlabel('局部坐标 X（米）',fontproperties=font);a.set_ylabel('局部坐标 Y（米）',fontproperties=font)
            a.set_title(clip[:8]+'：预测与真实轨迹',fontproperties=font);b.set_title('逐时间点的位置误差',fontproperties=font)
            b.set_xlabel('预测时间（秒，官方 10 Hz）',fontproperties=font);b.set_ylabel('XY 距离误差（米）',fontproperties=font)
            for ax in [a,b]:ax.legend(prop=font);ax.grid(alpha=.2)
            records.append(record)
    fig.suptitle('两个最大净退化片段：保持全部 20 段统计，未剔除失败片段',fontproperties=font)
    curves=out/'curves'
    for suffix in ['png','svg']:fig.savefig(curves/f'worst_two_trajectories_zh.{suffix}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    (out/'failure_case_metrics.json').write_text(json.dumps({'scope':'Post-hoc diagnostic only; not allocation or training data selection','records':records},indent=2)+'\n')
    shutil.copy2(__file__,out/'code'/Path(__file__).name)
    manifest={'input_result_sha256':hashlib.sha256((out/'result.json').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'outputs':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in curves.glob('worst_two*')]}
    (curves/'failure_cases_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FAILURE_CASE_METRICS',json.dumps([{k:v for k,v in r.items() if k!='ground_truth_xyz'} for r in records]))


if __name__=='__main__':main()
