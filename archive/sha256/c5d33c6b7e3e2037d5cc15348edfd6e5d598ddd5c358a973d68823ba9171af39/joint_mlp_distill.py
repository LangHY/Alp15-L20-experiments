"""Local cached-teacher MLP distillation of all linked projections.

This optimizes one MLP on dense teacher inputs/outputs. It is not end-to-end KD
and does not resolve limited calibration coverage or upstream student state shift.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .mlp_pruning import _tokens, _validate_mlp
except ImportError:
    from mlp_pruning import _tokens, _validate_mlp


def train_joint_mlp(
    mlp: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor,
    val_x: torch.Tensor, val_y: torch.Tensor, *, steps: int = 300,
    batch_size: int = 128, lr: float = 3e-5, seed: int = 0,
    eval_every: int = 20, eps: float = 1e-12,
) -> dict[str, Any]:
    """Train gate/up/down in FP32; choose by actual original-dtype export NMSE.

    Production input MLPs are BF16, so validation casts all three matrices to
    BF16 and executes all three matmuls at BF16. The untrained BF16 baseline is
    included in selection. On success the best exported matrices are restored;
    any exception/nonfinite loss, gradient, or weight restores the original
    baseline. All parameters are frozen with no gradients on exit. Module mode
    and original dtype are restored (including FP32 for small CPU fixtures).
    """
    hidden, _ = _validate_mlp(mlp)
    tx, ty = _tokens(train_x, hidden, 'train_x'), _tokens(train_y, hidden, 'train_y')
    vx, vy = _tokens(val_x, hidden, 'val_x'), _tokens(val_y, hidden, 'val_y')
    if tx.shape[0] != ty.shape[0] or vx.shape[0] != vy.shape[0]:
        raise ValueError('Input and teacher target token counts differ')
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (steps, batch_size, eval_every)):
        raise ValueError('steps, batch_size and eval_every must be integers')
    if steps < 0 or batch_size < 1 or eval_every < 1:
        raise ValueError('Invalid steps, batch_size or eval_every')
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError('lr and eps must be finite and positive')
    names = ('gate_proj', 'up_proj', 'down_proj')
    original_dtype = mlp.gate_proj.weight.dtype
    device = mlp.gate_proj.weight.device
    original_mode = mlp.training
    baseline = {name: getattr(mlp, name).weight.detach().cpu().clone() for name in names}
    best = baseline
    successful = False
    best_step, completed_steps = 0, 0
    history: list[dict[str, Any]] = []

    @torch.no_grad()
    def measure(x: torch.Tensor, y: torch.Tensor, candidate: dict[str, torch.Tensor]) -> float:
        exported = {name: weight.detach().to(device=device, dtype=original_dtype)
                    for name, weight in candidate.items()}
        error_sum, energy_sum = 0.0, 0.0
        with torch.autocast(device_type=device.type, enabled=False):
            for first in range(0, x.shape[0], batch_size):
                inputs = x[first:first + batch_size].to(device=device, dtype=original_dtype)
                activ = mlp.act_fn(F.linear(inputs, exported['gate_proj'])) * F.linear(inputs, exported['up_proj'])
                prediction = F.linear(activ, exported['down_proj']).float()
                target = y[first:first + batch_size].to(device=device, dtype=torch.float32)
                error_sum += (prediction - target).square().sum().item()
                energy_sum += target.square().sum().item()
        value = error_sum / max(energy_sum, eps * y.numel())
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite exported-dtype normalized MSE')
        return value

    try:
        mlp.eval()
        for parameter in mlp.parameters():
            parameter.requires_grad_(False)
        baseline_val = measure(vx, vy, baseline)
        baseline_train = measure(tx, ty, baseline)
        best_val = baseline_val
        history.append({'step': 0, 'val_nmse': baseline_val})
        parameters = []
        for name in names:
            projection = getattr(mlp, name)
            projection.to(dtype=torch.float32)
            projection.weight.requires_grad_(True)
            parameters.append(projection.weight)
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0.0)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        order = torch.randperm(tx.shape[0], generator=generator)
        offset = 0
        for step in range(1, steps + 1):
            if offset >= order.numel():
                order = torch.randperm(tx.shape[0], generator=generator)
                offset = 0
            ids = order[offset:offset + batch_size]
            offset += ids.numel()
            inputs = tx.index_select(0, ids.to(tx.device)).to(device=device, dtype=torch.float32)
            target = ty.index_select(0, ids.to(ty.device)).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=False):
                activ = mlp.act_fn(mlp.gate_proj(inputs)) * mlp.up_proj(inputs)
                prediction = mlp.down_proj(activ)
                loss = (prediction - target).square().mean() / target.square().mean().clamp_min(eps)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite joint training loss at step {step}')
            loss.backward()
            try:
                nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            except RuntimeError as exc:
                if 'non-finite' in str(exc).lower() or 'nonfinite' in str(exc).lower():
                    raise FloatingPointError(f'Nonfinite joint gradient at step {step}') from exc
                raise
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise FloatingPointError(f'Nonfinite joint weight at step {step}')
            completed_steps = step
            if step % eval_every == 0 or step == steps:
                candidate = {name: getattr(mlp, name).weight for name in names}
                value = measure(vx, vy, candidate)
                history.append({'step': step, 'val_nmse': value, 'train_batch_nmse': loss.item()})
                if value < best_val:
                    best_val, best_step = value, step
                    best = {name: weight.detach().to(dtype=original_dtype).cpu().clone()
                            for name, weight in candidate.items()}
        final_val = measure(vx, vy, best)
        final_train = measure(tx, ty, best)
        if final_val > baseline_val:
            raise RuntimeError('Best exported candidate exceeds untrained baseline validation NMSE')
        report = {
            'baseline_val_nmse': baseline_val, 'final_val_nmse': final_val,
            'baseline_train_nmse': baseline_train, 'final_train_nmse': final_train,
            'baseline_loss': baseline_val, 'post_loss': final_val,
            'best_step': best_step, 'completed_steps': completed_steps, 'history': history,
            'export_dtype': str(original_dtype), 'optimizer_dtype': 'torch.float32',
            'trained_projections': list(names), 'gradient_clip_norm': 1.0,
            'distillation_scope': 'Local MLP output MSE on cached dense teacher states; not end-to-end KD',
            'limitations': ['Calibration/validation coverage is limited to the supplied token caches',
                            'Teacher input states do not model upstream student distribution shift'],
        }
        successful = True
        return report
    finally:
        restored = best if successful else baseline
        with torch.no_grad():
            for name in names:
                projection = getattr(mlp, name)
                projection.to(dtype=original_dtype)
                projection.weight.copy_(restored[name].to(device=device))
        for parameter in mlp.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        mlp.train(original_mode)


