"""Offline data coverage audit on the new training/monitor partition, no model inference."""
import argparse,json,shutil,sys,time
from pathlib import Path
import torch
from run_mlp_experiment import _enable_offline_mode,_block_network,_load_offline_dataset_interface,DATASET_REVISION,file_digest,write_json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();dataset=args.dataset.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
partition=json.loads((dataset/'training_partition.json').read_text());receipt=json.loads((dataset/'delivery_receipt.json').read_text())
assert receipt['status']=='completed' and receipt['revision']==DATASET_REVISION
assert file_digest(dataset/'training_partition.json')==receipt['training_partition_sha256']
shutil.copy2(__file__,out/'as_started_source.py');shutil.copy2(dataset/'training_partition.json',out/'initial_partition.json')
start=time.perf_counter();result={'status':'running','dataset_revision':DATASET_REVISION,'dataset_directory':str(dataset),
    'receipt_sha256':file_digest(dataset/'delivery_receipt.json'),'argv':sys.argv,'torch':torch.__version__,'python':sys.version,
    'records':[],'training_records':[],'monitor_records':[],
    'policy':'Only predeclared new chunk0000 official train clips. Exclusions only for input/label coverage failures, no ADE or model output inspected.'}
def save():
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)
_enable_offline_mode();torch.set_num_threads(8)
try:
    with _block_network():
        interface=_load_offline_dataset_interface(dataset)
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        for role,key in [('training','training_records'),('monitor_validation','monitor_records')]:
            for clip in partition[role]:
                t0=partition['t0_us']
                try:
                    motion=interface.get_clip_feature(clip,interface.features.LABELS.EGOMOTION,maybe_stream=False)
                    lo,hi=[int(v) for v in motion.time_range]
                    assert lo<=t0-1500000 and t0+6400000<=hi
                    data=load_physical_aiavdataset(clip,t0_us=t0,avdi=interface,maybe_stream=False)
                    assert tuple(data['image_frames'].shape[:3])==(4,4,3)
                    for field in ['ego_history_xyz','ego_history_rot','ego_future_xyz','ego_future_rot']:
                        assert torch.isfinite(data[field]).all()
                    error=(data['absolute_timestamps']-torch.tensor([t0-300000,t0-200000,t0-100000,t0])).abs().max().item()
                    assert error<=50000
                    result[key].append({'clip_id':clip,'t0_us':t0})
                    result['records'].append({'clip_id':clip,'role':role,'status':'completed','max_camera_timestamp_error_us':error,'egomotion_bounds_us':[lo,hi]})
                    del data
                except Exception as exc:
                    result['records'].append({'clip_id':clip,'role':role,'status':'excluded','error':{'type':type(exc).__name__,'message':str(exc)}})
                save();print('NEW_DATA_CLIP',role,clip,result['records'][-1]['status'],flush=True)
    if len(result['training_records'])<40 or len(result['monitor_records'])<10:
        raise RuntimeError('Insufficient complete clips; refusing to start training')
    assert not {r['clip_id'] for r in result['training_records']}&{r['clip_id'] for r in result['monitor_records']}
    result['status']='completed';save()
except Exception as exc:
    result['status']='failed';result['error']={'type':type(exc).__name__,'message':str(exc)};save();raise
