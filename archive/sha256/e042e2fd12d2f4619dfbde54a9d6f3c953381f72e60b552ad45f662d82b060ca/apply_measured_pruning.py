"""Apply a complete calibration-only plan and preserve original dense channel identities."""
import copy
import torch
from mlp_pruning import prune_mlp

def apply_plan(model,manifest,candidate,plan):
    layers=model.vlm.model.language_model.layers
    widths=plan['layer_widths'];local_ids=plan['parent_local_channel_ids'];dense_ids=plan['original_dense_channel_ids']
    if plan.get('ranking_uses_holdout') is not False or plan.get('ranking_uses_validation') is not False:
        raise ValueError('Allocation must use training calibration only')
    if not (len(layers)==len(widths)==len(local_ids)==len(dense_ids)==len(manifest['layers'])):
        raise ValueError('Plan must contain every layer')
    # Validate all shapes and mappings before changing any layer.
    for i,(layer,w,ids,mapped) in enumerate(zip(layers,widths,local_ids,dense_ids)):
        if type(w) is not int or w%128 or not 128<=w<=layer.mlp.gate_proj.out_features:
            raise ValueError('Invalid measured width')
        if len(ids)!=w or ids!=sorted(set(ids)) or min(ids)<0 or max(ids)>=layer.mlp.gate_proj.out_features:
            raise ValueError('Invalid parent channel identities')
        if mapped!=[manifest['layers'][i]['keep_indices'][j] for j in ids]:
            raise ValueError('Dense channel map does not match parent')
    new_manifest=copy.deepcopy(manifest);new_candidate=copy.deepcopy(candidate)
    hidden=layers[0].mlp.gate_proj.in_features
    expected=sum(p.numel() for p in model.parameters())-sum(layer.mlp.gate_proj.out_features*hidden*3 for layer in layers)+sum(widths)*hidden*3
    if expected!=plan['actual_total_params']:raise ValueError('Plan parameter budget mismatch')
    for i,(layer,ids,w,mapped) in enumerate(zip(layers,local_ids,widths,dense_ids)):
        prune_mlp(layer.mlp,torch.tensor(ids,dtype=torch.long))
        new_manifest['layers'][i].update({'keep_width':w,'keep_indices':mapped})
    actual=sum(p.numel() for p in model.parameters())
    if actual!=expected:raise RuntimeError('Actual pruned parameter count mismatch')
    new_manifest.update({'keep_width':plan['keep_width'],'layer_widths':widths,
        'method':'Measured current-parent linked-channel pruning; original dense channel map retained'})
    new_candidate.update({'keep_width':plan['keep_width'],'layer_widths':widths,'params':actual,
        'vlm_params':sum(p.numel() for p in model.vlm.parameters()),'bf16_weight_gib':actual*2/2**30,
        'parent_validation_mean_minADE_m':candidate['validation_mean_minADE_m'],
        'pruning_profile_directory':plan['profile_directory']})
    return new_manifest,new_candidate
