"""Validate only original training clips at fixed extra timestamps, offline."""
import argparse,hashlib,json,sys,time,shutil
from pathlib import Path
import torch
from run_mlp_experiment import _enable_offline_mode,_block_network,_load_offline_dataset_interface,DATASET_REVISION,file_digest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True);parser.add_argument('--split',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--times-us',default='3100000,5100000,7100000,9100000,11100000')
args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
started=time.perf_counter();shutil.copy2(__file__,out/'as_started_source.py');shutil.copy2(args.split,out/'split.json')
split=json.loads(args.split.read_text());times=[int(v) for v in args.times_us.split(',')]
assert len(times)==len(set(times)) and all(v>1600000 for v in times)
assert not set(split['calibration'])&set(split['validation']+split['holdout'])
result={'status':'running','dataset_revision':DATASET_REVISION,'split_sha256':file_digest(args.split),
    'source_commit':'36aeb4c5938cbc2eb2aed33b22434773da4ab639','python':sys.version,'torch':torch.__version__,
    'dataset_manifest_sha256':file_digest(args.root/'alp15-test-clip-delivery-20260917/dataset-manifest.json'),
    'source_sha256':file_digest(Path(__file__)),'argv':sys.argv,'times_us':times,'records':[],
    'semantics':'Temporal augmentation on four existing training clips only; not new independent scenes; no validation or holdout accessed'}
def save():
    result['elapsed_seconds']=time.perf_counter()-started
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
_enable_offline_mode();torch.set_num_threads(8)
try:
    with _block_network():
        interface=_load_offline_dataset_interface(args.root/'alp15-test-clip-delivery-20260917')
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        for clip in split['calibration']:
            motion=interface.get_clip_feature(clip,interface.features.LABELS.EGOMOTION,maybe_stream=False)
            bounds=tuple(int(v) for v in motion.time_range)
            for t0 in times:
                assert bounds[0]<=t0-1500000 and t0+6400000<=bounds[1]
                data=load_physical_aiavdataset(clip,t0_us=t0,avdi=interface,maybe_stream=False)
                for key in ['ego_history_xyz','ego_history_rot','ego_future_xyz','ego_future_rot']:
                    assert torch.isfinite(data[key]).all()
                assert tuple(data['image_frames'].shape[:3])==(4,4,3)
                target=torch.tensor([t0-300000,t0-200000,t0-100000,t0])
                error=(data['absolute_timestamps']-target).abs()
                # 30Hz sensors: fail if any decoded frame is more than 50ms away.
                assert error.max().item()<=50000, ('Camera timestamps outside tolerance',clip,t0,error.tolist())
                xyz=data['ego_future_xyz'].contiguous()
                r={'clip_id':clip,'t0_us':t0,'egomotion_bounds_us':bounds,'image_shape':list(data['image_frames'].shape),
                    'camera_timestamps_us':data['absolute_timestamps'].tolist(),'max_camera_timestamp_error_us':error.max().item(),
                    'future_xyz_sha256':hashlib.sha256(xyz.view(torch.uint8).numpy().tobytes()).hexdigest()}
                result['records'].append(r);save();print('TRAIN_WINDOW_VALID',clip,t0,flush=True);del data
    result['status']='completed';save()
except Exception as exc:
    result['status']='failed';result['error']={'type':type(exc).__name__,'message':str(exc)};save();raise
