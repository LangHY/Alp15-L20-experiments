"""MLP-only low rank recovery, with differentiable whole-student hidden/KV KD."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RecoveryLinear(nn.Module):
    def __init__(self, base, rank=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.scale = 1.0
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        update = F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype))
        return self.base(x) + update * self.scale

    @torch.no_grad()
    def merged(self):
        self.base.weight.copy_((self.base.weight.float() + self.scale * (self.b @ self.a)).to(self.base.weight.dtype))
        return self.base


def install_adapters(model, rank=16, checkpoint_mlp=True):
    model.requires_grad_(False)
    adapters, originals = [], []
    for layer in model.vlm.model.language_model.layers:
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            adapter = RecoveryLinear(getattr(layer.mlp, name), rank)
            setattr(layer.mlp, name, adapter)
            adapters.append(adapter)
        if checkpoint_mlp:
            original = layer.mlp.forward
            originals.append((layer.mlp, original))
            def wrapped(x, original=original):
                return checkpoint(original, x, use_reentrant=False)
            layer.mlp.forward = wrapped
    return adapters, originals


def merge_adapters(model, originals):
    for mlp, forward in originals:
        mlp.forward = forward
    for layer in model.vlm.model.language_model.layers:
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(layer.mlp, name, getattr(layer.mlp, name).merged())


def interface_targets(output, positions):
    return {'hidden': output.last_hidden_state[:, positions],
            'keys': [kv[0][:, :, positions, :] for kv in output.past_key_values],
            'values': [kv[1][:, :, positions, :] for kv in output.past_key_values]}


def relative_mse(student, teacher):
    target = teacher.to(student.device).float()
    return (student.float() - target).square().mean() / target.square().mean().clamp_min(1e-8)


def interface_loss(student, target):
    hidden = relative_mse(student['hidden'], target['hidden'])
    keys = torch.stack([relative_mse(s, t) for s, t in zip(student['keys'], target['keys'], strict=True)]).mean()
    values = torch.stack([relative_mse(s, t) for s, t in zip(student['values'], target['values'], strict=True)]).mean()
    return hidden + (keys + values) / 2, {'hidden_nmse': hidden, 'key_nmse': keys, 'value_nmse': values}
