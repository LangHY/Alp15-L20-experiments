"""Measure all current 8B MLPs on the 80 training clips; never use monitor/test clips."""
import argparse,csv,json,shutil,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from run_mlp_experiment import (_enable_offline_mode,_block_network,_verify_model,
    _load_local_model_and_processor,load_overlay,protected_digest,file_digest,write_json)
from mlp_pruning import compute_activation_scores,select_channels
from student_interface_distill import mlp_tensor_digest
from layerwise_allocation import allocate_widths

GRID=[768,1024,1280,1536,1792,2048,2560,3072,3584,4096,4608,5120,5376]

@torch.no_grad()
def measure(mlp,x,y,scores,batch=128):
    ranked=torch.argsort(scores,descending=True,stable=True).cuda();energy=0.;input_energy=0.
    error={w:0. for w in GRID};width=mlp.gate_proj.out_features
    assert width==5376 and x.shape==y.shape and x.shape[1]==4096
    for first in range(0,len(x),batch):
        a=x[first:first+batch].cuda();target=y[first:first+batch].cuda().float()
        features=mlp.act_fn(mlp.gate_proj(a))*mlp.up_proj(a)
        energy+=target.square().sum().item();input_energy+=a.float().square().sum().item()
        for w in GRID:
            ids=ranked[:w]
            pred=mlp.down_proj(features) if w==width else F.linear(features.index_select(1,ids),mlp.down_proj.weight.index_select(1,ids))
            error[w]+=(pred.float()-target).square().sum().item()
    assert energy>0 and input_energy>0
    return [{'width':w,'calibration_nmse':error[w]/energy,'input_relative_error':error[w]/input_energy,
        'tokens':len(x),'squared_error_sum':error[w],'target_squared_energy':energy} for w in GRID]

def plots(out,curves,importance,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc');plt.rcParams['svg.fonttype']='path'
    figures=[]
    fig,axes=plt.subplots(6,6,figsize=(18,16))
    for i,ax in enumerate(axes.flat):
        ax.plot([r['width'] for r in curves[i]],[r['calibration_nmse'] for r in curves[i]],'.-')
        ax.set_title(f'第 {i} 层',fontproperties=font);ax.grid(alpha=.2)
    fig.supxlabel('保留通道数',fontproperties=font);fig.supylabel('训练数据局部输出 NMSE（未恢复）',fontproperties=font)
    figures.append(('all_layer_width_sensitivity_zh',fig))
    fig,ax=plt.subplots(figsize=(14,5))
    ax.bar(range(36),[r['nmse_at_3072'] for r in importance],color=['#d95f02' if r['protected'] else '#3182bd' for r in importance])
    ax.set_xlabel('MLP 层编号',fontproperties=font);ax.set_ylabel('3072 通道时局部 NMSE',fontproperties=font)
    ax.set_title('80 段训练数据：全层局部敏感性；橙色层优先保护',fontproperties=font)
    figures.append(('all_layer_importance_zh',fig))
    fig,ax=plt.subplots(figsize=(14,5));ax.bar(range(36),plan['layer_widths']);ax.axhline(5376,color='gray',ls='--')
    ax.set_xlabel('MLP 层编号',fontproperties=font);ax.set_ylabel('保留通道数',fontproperties=font)
    ax.set_title(f"约 7B 非均匀宽度分配：实际 {plan['actual_total_params']/1e9:.6f}B",fontproperties=font)
    figures.append(('stage7_allocation_zh',fig));artifacts=[]
    for name,fig in figures:
        fig.tight_layout()
        for suffix in ['png','svg']:
            f=out/'curves'/f'{name}.{suffix}';fig.savefig(f,dpi=150);artifacts.append({'file':f.name,'sha256':file_digest(f)})
        plt.close(fig)
    return artifacts

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root,parent,out=args.root.resolve(),args.parent.resolve(),args.output.resolve()
    assert not out.exists();out.mkdir(parents=True);(out/'cache').mkdir();(out/'curves').mkdir();(out/'code').mkdir()
    for name in ['profile_recovered8_pruning.py','mlp_pruning.py','layerwise_allocation.py']:
        shutil.copy2(Path(__file__).parent/name,out/'code'/name)
    split=json.loads((parent/'optimization_split.json').read_text());selected=json.loads((parent/'selected_task_validation.json').read_text())
    result=json.loads((parent/'controlled_recovery_result.json').read_text());reload=json.loads((parent/'controlled_reload_check.json').read_text())
    parent_provenance=json.loads((parent/'provenance.json').read_text())
    assert reload['status']=='completed' and result['params']==8020878578
    train=set(split['training']);assert len(train)==80 and not train&set(split['monitor_validation'])
    examples=[]
    for path in sorted((parent/'teacher_cache').glob('calibration_*.pt')):
        e=torch.load(path,map_location='cpu',weights_only=True);assert e['clip_id'] in train
        examples.append((path,{'clip_id':e['clip_id'],'inputs':e['inputs'],'positions':e['positions']}))
    assert len(examples)==80 and {e['clip_id'] for path,e in examples}==train
    provenance={'status':'running','argv':sys.argv,'parent':str(parent),'parent_MLP_SHA256':selected['mlp_tensor_sha256'],
        **{k:parent_provenance[k] for k in ['source_commit','checkpoint_revision','dataset_revision']},
        'parent_ADE_m':result['selected_mean_ADE_m'],'training_clips':sorted(train),'monitor_used':False,'holdout_used':False,
        'grid':GRID,'seed':42,'source_sha256':file_digest(Path(__file__)),'parent_split_sha256':file_digest(parent/'optimization_split.json'),
        'parent_overlay_manifest_sha256':file_digest(Path(result['overlay_directory'])/'manifest.json'),
        'teacher_input_cache_sha256':{str(path):file_digest(path) for path,e in examples},
        'metric':'Unrecovered linked-channel MLP output NMSE on current parent inputs; local proxy, not causal trajectory importance',
        'scope':'All 80 training clips; identical teacher-generated full reasoning prefix, current student hidden inputs; no monitor/validation/holdout ranking',
        'torch':torch.__version__,'python':sys.version,'limitations':['One new data chunk','Local proxy does not establish task accuracy','No 6B acceptance implied']}
    write_json(out/'provenance.json',provenance);_enable_offline_mode();torch.set_num_threads(8);torch.manual_seed(42)
    start=time.perf_counter()
    try:
        with _block_network(),torch.no_grad():
            config=_verify_model(root/'models');configs=root/'alp15-configs-delivery-20260919'
            model,processor=_load_local_model_and_processor(root/'models',configs/'dependencies/configs/Cosmos-Reason2-8B',configs/'dependencies/configs/Qwen3-VL-2B-Instruct',config)
            model.eval().requires_grad_(False);load_overlay(model,Path(result['overlay_directory']))
            assert mlp_tensor_digest(model)==selected['mlp_tensor_sha256'];protected=protected_digest(model)
            layers=model.vlm.model.language_model.layers;assert len(layers)==36
            caches=[{'x':[],'y':[]} for _ in layers];state={};handles=[]
            for i,layer in enumerate(layers):
                def hook(module,inputs,output,i=i):
                    pos=state['positions'];caches[i]['x'].append(inputs[0][0].index_select(0,pos).detach().cpu())
                    caches[i]['y'].append(output[0].index_select(0,pos).detach().cpu())
                handles.append(layer.mlp.register_forward_hook(hook))
            try:
                for n,(path,e) in enumerate(examples):
                    state['positions']=e['positions'].cuda();inputs={k:v.cuda() for k,v in e['inputs'].items()}
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=model.vlm.model(**inputs,use_cache=False,return_dict=True)
                    del output,inputs
                    if (n+1)%8==0:print('PROFILE_CLIPS',n+1,flush=True)
            finally:
                for h in handles:h.remove()
            curves=[];scores_all=[]
            for i,layer in enumerate(layers):
                cache={k:torch.cat(v) for k,v in caches[i].items()};caches[i]=None
                assert torch.isfinite(cache['x']).all() and torch.isfinite(cache['y']).all()
                torch.save(cache,out/'cache'/f'train_{i:02d}.pt')
                scores=compute_activation_scores(layer.mlp,cache['x']);torch.save(scores,out/f'scores_{i:02d}.pt');scores_all.append(scores)
                rows=measure(layer.mlp,cache['x'],cache['y'],scores)
                assert rows[-1]['calibration_nmse']<=1e-3,('Parent reproduction failed',i,rows[-1])
                curves.append(rows);write_json(out/'measurement_progress.json',{'status':'profiling','completed_layers':len(curves),'curves':curves})
                print('SENSITIVITY_LAYER',i,'NMSE3072',next(r['calibration_nmse'] for r in rows if r['width']==3072),flush=True);del cache
            assert protected_digest(model)==protected and mlp_tensor_digest(model)==selected['mlp_tensor_sha256']
            original_maps=json.loads((Path(result['overlay_directory'])/'manifest.json').read_text())['layers']
            fixed=result['params']-36*5376*3*4096
        rank=sorted(range(36),key=lambda i:(-next(r['calibration_nmse'] for r in curves[i] if r['width']==3072),i))
        protected_layers=rank[:9];budget=((7000000000-fixed)//12288)//128*128
        plan=allocate_widths(curves,channel_budget=budget,minimum_width=1536,protected_layers=protected_layers,protected_minimum=5376)
        plan.update({'actual_total_params':fixed+plan['sum_widths']*12288,'fixed_non_mlp_params':fixed,'parent_experiment':str(parent),
            'parent_MLP_SHA256':selected['mlp_tensor_sha256'],'profile_directory':str(out),'ranking_uses_holdout':False,'ranking_uses_validation':False,
            'keep_width':3072,'source_widths':[5376]*36,'profile_provenance_sha256':file_digest(out/'provenance.json'),
            'parent_local_channel_ids':[select_channels(s,w).tolist() for s,w in zip(scores_all,plan['layer_widths'])]})
        plan['original_dense_channel_ids']=[[original_maps[i]['keep_indices'][j] for j in ids] for i,ids in enumerate(plan['parent_local_channel_ids'])]
        importance=[{'layer':i,'rank':rank.index(i)+1,'protected':i in protected_layers,'nmse_at_3072':next(r['calibration_nmse'] for r in c if r['width']==3072)} for i,c in enumerate(curves)]
        write_json(out/'allocation_7B.json',plan)
        for name,rows in [('layer_importance.csv',importance),('sensitivity_measurements.csv',[dict(layer=i,**r) for i,c in enumerate(curves) for r in c]),
            ('channel_importance.csv',[{'layer':i,'channel':j,'score':float(v)} for i,s in enumerate(scores_all) for j,v in enumerate(s)])]:
            f=out/'curves'/name
            with f.open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        charts=plots(out,curves,importance,plan)
        write_json(out/'curves/importance_manifest.json',{'metric':provenance['metric'],'outputs':charts,'missing_metrics':['No optimization loss or task ADE collected in importance audit']})
        write_json(out/'importance.json',{'status':'completed','layers':importance,'curves':curves,'allocation_file':'allocation_7B.json','elapsed_seconds':time.perf_counter()-start})
        print('PROFILE_COMPLETE',plan['actual_total_params'],plan['layer_widths'],flush=True)
    except Exception as exc:
        write_json(out/'importance.json',{'status':'failed','error':{'type':type(exc).__name__,'message':str(exc)}});raise

if __name__=='__main__':main()
