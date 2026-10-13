"""Header-only deployment storage scenarios, never a measured Jetson memory claim."""
import argparse,json,math,struct,hashlib,datetime
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    index=root/'models/model.safetensors.index.json';mapping=json.loads(index.read_text());tensors={};headers={}
    for name in sorted(set(mapping['weight_map'].values())):
        with (root/'models'/name).open('rb') as stream:
            length=struct.unpack('<Q',stream.read(8))[0];raw=stream.read(length)
        headers[name]=hashlib.sha256(raw).hexdigest()
        for key,row in json.loads(raw).items():
            if key!='__metadata__':tensors[key]={'shape':row['shape'],'dtype':row['dtype']}
    variants=[('original_dense',None)]
    for label,file in [('best8B','logs/latest_newdata8_launch.json'),('best7B','logs/latest_measured7_launch.json'),('current6p8B','logs/latest_measured6p8_launch.json')]:
        launch=json.loads((root/file).read_text());p=Path(launch['experiment']);result=p/'controlled_recovery_result.json'
        if result.exists():
            x=json.loads(result.read_text());overlay=Path(x['overlay_directory'])/'manifest.json';variants.append((label,{'params':x['params'],'widths':json.loads(overlay.read_text())['layer_widths'] if 'layer_widths' in json.loads(overlay.read_text()) else [json.loads(overlay.read_text())['keep_width']]*36,'ADE_m':x['selected_mean_ADE_m'],'source':str(result),'sha256':sha(result)}))
        else:
            allocation=json.loads(Path(launch['allocation']).read_text());variants.append((label,{'params':allocation['actual_total_params'],'widths':allocation['layer_widths'],'ADE_m':None,'source':launch['allocation'],'sha256':sha(Path(launch['allocation']))}))
    cases=[]
    for label,variant in variants:
        rows={k:dict(v,shape=list(v['shape'])) for k,v in tensors.items()}
        if variant:
            for i,w in enumerate(variant['widths']):
                for proj in ['gate_proj','up_proj','down_proj']:
                    key=f'vlm.model.language_model.layers.{i}.mlp.{proj}.weight'
                    rows[key]['shape']=[4096,w] if proj=='down_proj' else [w,4096]
        params=sum(math.prod(v['shape']) for v in rows.values())
        if variant:assert params==variant['params'],(label,params,variant['params'])
        for scheme in ['FP16_all','language_linear_W8_rest_FP16','language_MLP_W4_attention_W8_rest_FP16','language_linear_W4_rest_FP16']:
            storage=0;quantized=0;scales=0
            for name,v in rows.items():
                n=math.prod(v['shape']);bits=16
                eligible=name.startswith('vlm.model.language_model.layers.') and name.endswith('.weight') and len(v['shape'])==2
                if eligible and scheme!='FP16_all':bits=4 if scheme=='language_linear_W4_rest_FP16' or (scheme=='language_MLP_W4_attention_W8_rest_FP16' and '.mlp.' in name) else 8
                storage+=math.ceil(n*bits/8)
                if bits<16:
                    quantized+=n;scale_bytes=v['shape'][0]*math.ceil(v['shape'][1]/128)*2;storage+=scale_bytes;scales+=scale_bytes
            cases.append({'model':label,'params':params,'unquantized_validation_ADE_m':variant['ADE_m'] if variant else None,'scheme':scheme,'weight_storage_bytes':storage,'weight_storage_GiB':storage/2**30,'quantized_params':quantized,'group128_FP16_scale_bytes':scales})
    kv=[{'batch':1,'tokens':t,'dtype_bytes':2,'bytes':2*36*8*128*t*2,'GiB':2*36*8*128*t*2/2**30} for t in [2048,4096,8192]]
    result={'status':'storage_scenarios_complete','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_index_sha256':sha(index),'checkpoint_header_sha256':headers,'source_sha256':sha(Path(__file__)),'variants':{label:v for label,v in variants},'cases':cases,'language_KV_scenarios':kv,'kv_formula':'2(K,V) *36 layers *8 KV heads *128 head dimension *batch *tokens *dtype bytes','scenarios_only':True,'assumptions':['Quantized language transformer linear matrices only; Expert, vision, embeddings and lm_head stay FP16','Two-byte group128 scales; zero-point, layout padding, engine serialization and kernel metadata not included','No persistent duplicate FP16 matrix or CPU model assumed; loader must be audited','KV assumes one unshared standard language cache; extra copies, Expert working state, activations and vision tensors not included'],'missing_metrics':['Jetson MemTotal/system-services memory','Actual quantized ADE and boundary success rate','Quantization-kernel support for installed JetPack/TensorRT','Engine/runtime workspace and activation peaks','Loading and sustained whole-system peak memory','Jetson end-to-end latency'],'selection_policy':'Maximize validated accuracy subject to model plus whole-system peak memory fitting actual Jetson RAM with documented operating headroom; no fixed 6B requirement'}
    (out/'memory_scenarios.json').write_text(json.dumps(result,indent=2)+'\n')
    import shutil
    shutil.copy2(Path(__file__),out/'audit_deployment_memory.py')
    print('MEMORY_SCENARIOS',json.dumps(cases));print('KV_SCENARIOS',json.dumps(kv))

if __name__=='__main__':main()
