"""MLP-only low rank recovery, with differentiable whole-student hidden/KV KD."""
from __future__ import annotations

import math
from contextlib import contextmanager
import hashlib
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
        self.merged_for_eval = False
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        if self.merged_for_eval:
            return self.base(x)
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


@contextmanager
def exported_weights(adapters, base_weights):
    """Temporarily evaluate actual merged export weights, then restore training."""
    if any(m.merged_for_eval for m in adapters):
        raise RuntimeError('Nested export evaluation is unsupported')
    try:
        with torch.no_grad():
            for module, original in zip(adapters, base_weights, strict=True):
                merged = original.to(module.base.weight.device).float() + module.scale * (module.b @ module.a)
                module.base.weight.copy_(merged.to(module.base.weight.dtype))
                module.merged_for_eval = True
        yield
    finally:
        with torch.no_grad():
            for module, original in zip(adapters, base_weights, strict=True):
                module.base.weight.copy_(original)
                module.merged_for_eval = False


def mlp_tensor_digest(model):
    digest = hashlib.sha256()
    for index, layer in enumerate(model.vlm.model.language_model.layers):
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            module = getattr(layer.mlp, name)
            weight = module.base.weight if isinstance(module, RecoveryLinear) else module.weight
            digest.update(f'{index}.{name}'.encode())
            digest.update(str(tuple(weight.shape)).encode())
            digest.update(weight.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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


def logits_kl(student, teacher):
    """Exact next-token distribution KL, averaged over recorded prefix positions."""
    student_logp = F.log_softmax(student.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher.to(student.device).float(), dim=-1)
    return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1).mean()
