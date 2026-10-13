"""Differentiate frozen Expert velocity predictions through the VLM KV prefix.

The probes use fixed linear interpolation between noise and ground-truth actions;
this is a distillation probe distribution, not the original training pipeline.
It neither runs dense trajectory sampling nor trains any Expert parameter.
"""
import hashlib
import torch
from transformers.cache_utils import DynamicCache


def make_probes(model, interface, clip):
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    data = load_physical_aiavdataset(clip, t0_us=5_100_000, avdi=interface, maybe_stream=False)
    with torch.no_grad():
        action = model.action_space.traj_to_action(
            data['ego_history_xyz'][:, -1].cuda(), data['ego_history_rot'][:, -1].cuda(),
            data['ego_future_xyz'][:, -1].cuda(), data['ego_future_rot'][:, -1].cuda()).float().cpu()
    expected = (1, *model.action_space.get_action_space_dims())
    if tuple(action.shape) != expected or not torch.isfinite(action).all():
        raise RuntimeError('Invalid normalized action labels for Expert probes')
    seed = 12345 + int(hashlib.sha256(clip.encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(action.shape, generator=generator)
    return [{'x': (1 - t) * noise + t * action, 't': torch.full((1, 1, 1), t)}
            for t in (0.0, 0.25, 0.5, 0.75)]


def expert_velocity(model, output, inputs, probe):
    cache = DynamicCache(config=model.vlm.config.text_config)
    for index, (key, value) in enumerate(output.past_key_values):
        cache.update(key, value, index)
    sequence_length = cache.get_seq_length()
    future = model.tokenizer.convert_tokens_to_ids('<|traj_future_start|>')
    if inputs['input_ids'][0, -1].item() != future:
        raise RuntimeError('Expert KD requires the verified complete trajectory-start prefix')
    offset = model._find_eos_offset(inputs['input_ids'], future, inputs['input_ids'].device, warn=False)
    n_tokens = model.action_space.get_action_space_dims()[0]
    positions, mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=output.rope_deltas, kv_cache_seq_len=sequence_length,
        n_diffusion_tokens=n_tokens, b_star=1, device=inputs['input_ids'].device,
        prefix_mask=inputs.get('attention_mask'))
    embeds = model.action_in_proj(probe['x'].cuda(), probe['t'].cuda())
    if embeds.dim() == 2:
        embeds = embeds.view(1, n_tokens, -1)
    kwargs = {'is_causal': False} if model.config.expert_non_causal_attention else {}
    expert = model.expert(inputs_embeds=embeds, position_ids=positions,
        past_key_values=cache, attention_mask=mask, use_cache=True, **kwargs)
    return model.action_out_proj(expert.last_hidden_state[:, -n_tokens:]).view(
        1, *model.action_space.get_action_space_dims())
