"""Verify manual chunk0000 delivery and build a separate immutable offline HF cache."""
import argparse,json,random,shutil,sys,tarfile,zipfile
from pathlib import Path,PurePosixPath
import pandas as pd
from run_mlp_experiment import DATASET_REVISION,file_digest,write_json

REPO='nvidia/PhysicalAI-Autonomous-Vehicles'
CACHE='datasets--nvidia--PhysicalAI-Autonomous-Vehicles'
METADATA=['clip_index.parquet','features.csv','metadata/data_collection.parquet','metadata/feature_presence.parquet']
CAMERAS=['camera_cross_left_120fov','camera_cross_right_120fov','camera_front_tele_30fov','camera_front_wide_120fov']
FILES=METADATA+['calibration/camera_intrinsics/camera_intrinsics.chunk_0000.parquet',
    'calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet','labels/egomotion/egomotion.chunk_0000.zip']+[
    f'camera/{name}/{name}.chunk_0000.zip' for name in CAMERAS]


def safe_members(archive):
    members=archive.getmembers();names=[m.name for m in members]
    if len(names)!=len(set(names)) or set(names)!=set(FILES+['download_manifest.json']):
        raise ValueError('Archive must contain exactly the fixed requested files and manifest')
    for member in members:
        path=PurePosixPath(member.name)
        if not member.isfile() or path.is_absolute() or '..' in path.parts or '\\' in member.name:
            raise ValueError('Unsafe archive entry')
    return members


def frozen_partition(clips,forbidden,seed=20261002):
    clips=sorted(clips)
    if len(clips)<40 or len(set(clips))!=len(clips) or set(clips)&set(forbidden):
        raise ValueError('Too few unique complete clips or leakage into old evaluation split')
    random.Random(seed).shuffle(clips)
    monitor_count=max(10,len(clips)//5)
    return {'training':sorted(clips[monitor_count:]),'monitor_validation':sorted(clips[:monitor_count]),
        'seed':seed,'policy':'New chunk0000 official train clips; random partition fixed before inference; new monitor clips never receive gradients'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--checksum',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    expected=args.checksum.read_text().split()[0]
    if len(expected)!=64 or file_digest(args.archive)!=expected:raise ValueError('Archive SHA256 mismatch')
    if out.exists():raise ValueError('Refusing to overwrite a data delivery')
    out.mkdir(parents=True);assets=out/'files';assets.mkdir()
    receipt={'status':'running','argv':sys.argv,'repository':REPO,'revision':DATASET_REVISION,
        'archive_sha256':expected,'source_sha256':file_digest(Path(__file__))}
    try:
        with tarfile.open(args.archive) as tar:
            tar.extractall(assets,members=safe_members(tar),filter='data')
        manifest=json.loads((assets/'download_manifest.json').read_text())
        if (manifest['repository'],manifest['revision'],manifest['chunk_id'])!=(REPO,DATASET_REVISION,0):
            raise ValueError('Wrong repository, revision or chunk')
        rows=manifest['files']
        if len(rows)!=len(FILES) or {r['path'] for r in rows}!=set(FILES):raise ValueError('Manifest paths differ from fixed request')
        for row in rows:
            file=assets/row['path']
            if file.stat().st_size!=row['bytes'] or file_digest(file)!=row['sha256']:
                raise ValueError(f'File digest mismatch: {row["path"]}')
        old=root/'alp15-test-clip-delivery-20260917'/CACHE/'snapshots'/DATASET_REVISION
        for name in METADATA:
            if file_digest(assets/name)!=file_digest(old/name):raise ValueError('Metadata differs from original frozen revision')
        snapshot=out/CACHE/'snapshots'/DATASET_REVISION;snapshot.mkdir(parents=True)
        for name in FILES:
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(assets/name)
        old_split=json.loads((root/'experiments/mlp_controlled8_recovery_20261002_132357_v2/split.json').read_text())
        expanded=json.loads((root/'experiments/controlled8_expanded20_20261002_125000/split.json').read_text())
        forbidden=set(sum((old_split[k] for k in ['calibration','validation','holdout']),[]))
        forbidden.update(expanded['expanded_validation']);forbidden.update(expanded['expanded_holdout'])
        forbidden.add('030c760c-ae38-49aa-9ad8-f5650a545d26')
        index=pd.read_parquet(assets/'clip_index.parquet')
        eligible=set(index.index[(index.chunk==0)&(index.clip_is_valid==True)&(index['split']=='train')])
        coverage={}
        for name in CAMERAS:
            with zipfile.ZipFile(assets/f'camera/{name}/{name}.chunk_0000.zip') as z:
                names=set(z.namelist())
            covered={clip for clip in eligible if f'{clip}.{name}.mp4' in names and f'{clip}.{name}.timestamps.parquet' in names}
            coverage[name]=len(covered);eligible &= covered
        with zipfile.ZipFile(assets/'labels/egomotion/egomotion.chunk_0000.zip') as z:motion_names=set(z.namelist())
        eligible={clip for clip in eligible if f'{clip}.egomotion.parquet' in motion_names}
        partition=frozen_partition(eligible,forbidden)
        partition.update({'dataset_revision':DATASET_REVISION,'chunk_id':0,'t0_us':5100000,
            'coverage':coverage,'forbidden_clip_count':len(forbidden),'ready_for_training':False,
            'limitations':['Full offline loading and timestamp coverage preflight still required','Distinct clips do not establish independence of recording drives']})
        write_json(out/'training_partition.json',partition)
        receipt.update({'status':'completed','files':rows,'training_partition_sha256':file_digest(out/'training_partition.json'),
            'training_clips':len(partition['training']),'monitor_validation_clips':len(partition['monitor_validation'])})
        write_json(out/'delivery_receipt.json',receipt)
        print('DELIVERY_VERIFIED',receipt['training_clips'],receipt['monitor_validation_clips'])
    except Exception as exc:
        receipt.update({'status':'failed','error':{'type':type(exc).__name__,'message':str(exc)}})
        write_json(out/'delivery_receipt.json',receipt);raise


if __name__=='__main__':main()
