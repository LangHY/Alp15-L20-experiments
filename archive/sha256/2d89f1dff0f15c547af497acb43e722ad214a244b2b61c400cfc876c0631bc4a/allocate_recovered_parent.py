"""Allocate the next small step from completed, training-only recovered-parent measurements."""
import argparse,json,hashlib
from pathlib import Path
import torch
from layerwise_allocation import allocate_widths
from mlp_pruning import select_channels
from run_mlp_experiment import file_digest,write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--target-params',type=int,default=6800000000)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=args.profile.resolve()
    if args.output.exists():raise RuntimeError('Refuse to overwrite a prior allocation')
    x=json.loads((p/'importance.json').read_text());v=json.loads((p/'provenance.json').read_text())
    assert x['status']=='completed' and len(x['curves'])==36 and v['measurement_only']
    assert v['monitor_used'] is False and v['holdout_used'] is False
    mapping=p/'parent_overlay_manifest.json';assert file_digest(mapping)==v['parent_overlay_manifest_sha256']
    manifest=json.loads(mapping.read_text());protected=sorted(r['layer'] for r in sorted(x['layers'],key=lambda r:r['rank'])[:9])
    floors=[x['source_widths'][i] if i in protected else 1024 for i in range(36)]
    curves=[[r for r in c if r['width']>=floors[i]] for i,c in enumerate(x['curves'])]
    fixed=x['fixed_non_mlp_params'];budget=((args.target_params-fixed)//12288)//128*128
    plan=allocate_widths(curves,channel_budget=budget,minimum_width=128)
    plan.update({'actual_total_params':fixed+plan['sum_widths']*12288,'fixed_non_mlp_params':fixed,'requested_total_params':args.target_params,
        'parent_experiment':v['parent'],'parent_ADE_m':v['parent_ADE_m'],'parent_MLP_SHA256':v['parent_MLP_SHA256'],
        'profile_directory':str(p),'profile_provenance_sha256':file_digest(p/'provenance.json'),
        'ranking_uses_holdout':False,'ranking_uses_validation':False,'keep_width':2560,
        'source_widths':x['source_widths'],'protected_layers':protected,'per_layer_minimum_widths':floors,
        'protection_policy':'Nine highest measured half-width local sensitivities retain all current channels; other layers >=1024',
        'minimum_width':1024,'allocation_source_sha256':file_digest(Path(__file__)),
        'limitations':['Local proxy, not trajectory importance or recovered ADE','Protection and next total budget are explicit engineering constraints, not a globally optimal task allocation']})
    scores=[torch.load(p/f'scores_{i:02d}.pt',map_location='cpu',weights_only=True) for i in range(36)]
    plan['score_sha256']={f'scores_{i:02d}.pt':file_digest(p/f'scores_{i:02d}.pt') for i in range(36)}
    plan['parent_local_channel_ids']=[select_channels(s,w).tolist() for s,w in zip(scores,plan['layer_widths'])]
    plan['original_dense_channel_ids']=[[manifest['layers'][i]['keep_indices'][j] for j in ids] for i,ids in enumerate(plan['parent_local_channel_ids'])]
    write_json(args.output,plan);print('ALLOCATION_COMPLETE',plan['actual_total_params'],plan['calibration_nmse_sum'],plan['layer_widths'],flush=True)

if __name__=='__main__':main()
