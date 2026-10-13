"""LightVLA eval selection primitive, not a completed Alpamayo/streaming adapter.

Caller supplies exact visual/text positions from the tokenizer. Camera/frame
labels, image delimiters, history and trajectory tokens remain untouched.
This deliberately does not infer a contiguous image span or alter model cache.
"""
import math
import torch
import torch.nn.functional as F


def select_visual_positions(hidden_states, visual_positions, task_positions,
                            query_chunk=128, key_chunk=256):
    """Batch-one unique LightVLA winners with bounded score-matrix memory.

Returns sorted absolute positions. FP32 scoring defines this prototype's
numerical policy; compatibility with upstream BF16 selections needs measuring.
Selections are inference-only: train the adapted model separately before ADE
claims. Streaming cache capacity, deepstack indices and three-axis RoPE still
need the caller's aligned remapping.
"""
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError('Variable-length selection currently requires batch one')
    if query_chunk < 1 or key_chunk < 1:
        raise ValueError('Chunks must be positive')
    positions = []
    for name, value in [('visual', visual_positions), ('task', task_positions)]:
        if value.ndim != 1 or value.dtype != torch.long or value.numel() == 0:
            raise ValueError(f'{name} positions must be nonempty 1D int64')
        if value.device != hidden_states.device:
            raise ValueError('Position tensors must share the hidden-state device')
        if int(value.min()) < 0 or int(value.max()) >= hidden_states.shape[1]:
            raise ValueError('Position outside input sequence')
        if value.unique().numel() != value.numel():
            raise ValueError('Duplicate input positions')
        positions.append(value)
    if torch.isin(positions[0], positions[1]).any():
        raise ValueError('Visual and task positions overlap')
    if not torch.isfinite(hidden_states).all():
        raise ValueError('Nonfinite hidden states')
    with torch.no_grad():
        def norm(x):
            x = x.float()
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        patches = norm(hidden_states[:, visual_positions])
        task = norm(hidden_states[:, task_positions])
        queries = norm(F.scaled_dot_product_attention(patches, task, task))
        winners = []
        for q in queries.split(query_chunk, dim=1):
            best = torch.full(q.shape[:2], -torch.inf, device=q.device)
            indices = torch.zeros(q.shape[:2], dtype=torch.long, device=q.device)
            for start in range(0, patches.shape[1], key_chunk):
                score = q @ patches[:, start:start+key_chunk].transpose(-2, -1)
                score = score / math.sqrt(hidden_states.shape[-1])
                values, local = score.max(dim=-1)
                replace = values > best  # Preserve argmax's first index on ties.
                best = torch.where(replace, values, best)
                indices = torch.where(replace, local + start, indices)
            winners.append(indices.flatten())
        selected = visual_positions[torch.cat(winners).unique()]
        # Preserve every nonvisual token; no positional assumption about CLS.
        mask = torch.ones(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
        mask[visual_positions] = False
        mask[selected] = True
        return mask.nonzero().flatten()
