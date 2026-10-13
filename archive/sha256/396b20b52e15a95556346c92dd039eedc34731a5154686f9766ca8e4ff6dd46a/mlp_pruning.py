"""Isolated linked-channel MLP pruning and local projection distillation.

Nothing here edits model config, attention, vision, or the trajectory expert.
Teacher targets must be collected BEFORE pruning, with independent train/val tokens.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _validate_mlp(mlp: nn.Module) -> tuple[int, int]:
    projections = [getattr(mlp, name, None) for name in
                   ("gate_proj", "up_proj", "down_proj")]
    if not all(isinstance(p, nn.Linear) for p in projections):
        raise TypeError("Expected gate_proj/up_proj/down_proj nn.Linear modules")
    gate, up, down = projections
    if any(p.bias is not None for p in projections):
        raise ValueError("Only bias-free linked MLPs are supported")
    hidden, intermediate = gate.in_features, gate.out_features
    if (up.in_features, up.out_features) != (hidden, intermediate):
        raise ValueError("gate_proj and up_proj shapes differ")
    if (down.in_features, down.out_features) != (intermediate, hidden):
        raise ValueError("down_proj is not linked to the gate/up intermediate channels")
    if not callable(getattr(mlp, "act_fn", None)):
        raise TypeError("Expected callable act_fn")
    if len({(p.weight.device, p.weight.dtype) for p in projections}) != 1:
        raise ValueError("All projections must start on the same device and dtype")
    return hidden, intermediate


def _tokens(tensor: torch.Tensor, width: int, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
        raise ValueError(f"{name} must have token and feature dimensions")
    if tensor.shape[-1] != width or tensor.numel() == 0:
        raise ValueError(f"{name} must be nonempty with final dimension {width}")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    return tensor.detach().reshape(-1, width)


def _features(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    weight = mlp.gate_proj.weight
    x = x.to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        return mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)


@torch.no_grad()
def compute_activation_scores(mlp: nn.Module, x: torch.Tensor,
                              batch_size: int = 128) -> torch.Tensor:
    """Mean squared gated activation times squared down-projection column norm."""
    hidden, intermediate = _validate_mlp(mlp)
    tokens = _tokens(x, hidden, "x")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sums = torch.zeros(intermediate, dtype=torch.float64, device="cpu")
    for chunk in tokens.split(batch_size):
        activ = _features(mlp, chunk).float()
        sums += activ.square().sum(dim=0).double().cpu()
    column_norm = mlp.down_proj.weight.detach().float().square().sum(dim=0)
    scores = sums / tokens.shape[0] * column_norm.double().cpu()
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite activation scores")
    return scores


def select_channels(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Stable ties prefer lower channel IDs; return IDs in original channel order."""
    scores = scores.detach().flatten().cpu()
    if scores.numel() == 0 or not torch.isfinite(scores).all():
        raise ValueError("scores must be finite and nonempty")
    if isinstance(keep_count, bool) or not isinstance(keep_count, int):
        raise ValueError("keep_count must be an integer")
    if not 1 <= keep_count <= scores.numel():
        raise ValueError("keep_count outside channel count")
    ranked = torch.argsort(scores, descending=True, stable=True)
    return ranked[:keep_count].sort().values


def _channel_ids(kept_channels: Any, intermediate: int) -> torch.Tensor:
    ids = torch.as_tensor(kept_channels, device="cpu")
    if ids.ndim != 1 or ids.numel() == 0 or ids.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("kept_channels must be a nonempty integer vector")
    ids = ids.long()
    if ids.min() < 0 or ids.max() >= intermediate:
        raise ValueError("Channel ID outside original intermediate dimension")
    if not torch.all(ids[1:] > ids[:-1]):
        raise ValueError("Channel IDs must be unique and sorted")
    return ids


@torch.no_grad()
def prune_mlp(mlp: nn.Module, kept_channels: Any) -> dict[str, Any]:
    """Physically slice linked gate/up rows and down columns in one MLP."""
    hidden, intermediate = _validate_mlp(mlp)
    ids = _channel_ids(kept_channels, intermediate)
    for name, dimension in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
        original = getattr(mlp, name)
        selected = original.weight.index_select(dimension, ids.to(original.weight.device))
        replacement = nn.Linear(selected.shape[1], selected.shape[0], bias=False,
                                device=original.weight.device, dtype=original.weight.dtype)
        replacement.weight.copy_(selected)
        replacement.weight.requires_grad_(original.weight.requires_grad)
        replacement.train(original.training)
        setattr(mlp, name, replacement)
    mlp.intermediate_size = ids.numel()
    # The input/output hidden dimension and act_fn object are preserved.
    return {"hidden_size": hidden, "original_intermediate_size": intermediate,
            "intermediate_size": ids.numel(), "kept_channels": ids.tolist()}


@torch.no_grad()
def cache_teacher_io(mlp: nn.Module, x: torch.Tensor, batch_size: int = 128
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache inputs and dense teacher outputs on CPU before modifying the MLP."""
    hidden, _ = _validate_mlp(mlp)
    tokens = _tokens(x, hidden, "x")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    weight = mlp.gate_proj.weight
    outputs = [mlp(chunk.to(weight.device, weight.dtype)).detach().cpu()
               for chunk in tokens.split(batch_size)]
    return tokens.cpu().clone(), torch.cat(outputs, dim=0)


def train_down_projection(
    mlp: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor,
    val_x: torch.Tensor, val_y: torch.Tensor, *, steps: int = 60,
    batch_size: int = 128, lr: float = 3e-4, seed: int = 0,
    eval_every: int = 10, grad_clip: float | None = 1.0, eps: float = 1e-12,
) -> dict[str, Any]:
    """Distill only down_proj; select best by actual exported-dtype validation.

    gate/up compute at their original dtype without autograd. Only down_proj is
    optimized in FP32. Every validation candidate is cast back to the original
    dtype and its matmul executes at that dtype, including the baseline candidate.
    All parameters are frozen on exit. The original dtype and module mode return.
    """
    hidden, _ = _validate_mlp(mlp)
    tx, ty = _tokens(train_x, hidden, "train_x"), _tokens(train_y, hidden, "train_y")
    vx, vy = _tokens(val_x, hidden, "val_x"), _tokens(val_y, hidden, "val_y")
    if tx.shape[0] != ty.shape[0] or vx.shape[0] != vy.shape[0]:
        raise ValueError("Input/teacher target token counts differ")
    if steps < 0 or batch_size < 1 or lr <= 0 or eval_every < 1 or eps <= 0:
        raise ValueError("Invalid training hyperparameters")
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError("grad_clip must be positive or None")
    original_dtype = mlp.down_proj.weight.dtype
    device = mlp.down_proj.weight.device
    original_mode = mlp.training
    best_weight = mlp.down_proj.weight.detach().cpu().clone()
    best_step, completed_steps = 0, 0
    history: list[dict[str, Any]] = []

    @torch.no_grad()
    def measure(x: torch.Tensor, y: torch.Tensor, candidate: torch.Tensor) -> float:
        exported = candidate.detach().to(device=device, dtype=original_dtype)
        error, target_energy = 0.0, 0.0
        for first in range(0, x.shape[0], batch_size):
            activ = _features(mlp, x[first:first + batch_size])
            prediction = F.linear(activ, exported).float()
            target = y[first:first + batch_size].to(device=device, dtype=torch.float32)
            error += (prediction - target).square().sum().item()
            target_energy += target.square().sum().item()
        value = error / max(target_energy, eps * y.numel())
        if not torch.isfinite(torch.tensor(value)):
            raise FloatingPointError("Nonfinite normalized validation MSE")
        return value

    try:
        mlp.eval()
        for parameter in mlp.parameters():
            parameter.requires_grad_(False)
        baseline_val = measure(vx, vy, best_weight)
        baseline_train = measure(tx, ty, best_weight)
        best_val = baseline_val
        history.append({"step": 0, "val_nmse": baseline_val})
        mlp.down_proj.to(dtype=torch.float32)
        mlp.down_proj.weight.requires_grad_(True)
        optimizer = torch.optim.Adam([mlp.down_proj.weight], lr=lr, weight_decay=0.0)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(tx.shape[0], generator=generator)
        offset = 0
        for step in range(1, steps + 1):
            if offset >= order.numel():
                order = torch.randperm(tx.shape[0], generator=generator)
                offset = 0
            ids = order[offset:offset + batch_size]
            offset += ids.numel()
            activ = _features(mlp, tx.index_select(0, ids.to(tx.device))).float()
            target = ty.index_select(0, ids.to(ty.device)).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction = mlp.down_proj(activ)
            loss = (prediction - target).square().mean() / target.square().mean().clamp_min(eps)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_([mlp.down_proj.weight], grad_clip, error_if_nonfinite=True)
            elif not torch.isfinite(mlp.down_proj.weight.grad).all():
                raise FloatingPointError(f"Nonfinite gradient at step {step}")
            optimizer.step()
            completed_steps = step
            if step % eval_every == 0 or step == steps:
                value = measure(vx, vy, mlp.down_proj.weight)
                history.append({"step": step, "val_nmse": value, "train_batch_nmse": loss.item()})
                if value < best_val:
                    best_val, best_step = value, step
                    best_weight = mlp.down_proj.weight.detach().to(dtype=original_dtype).cpu().clone()
        mlp.down_proj.to(dtype=original_dtype)
        with torch.no_grad():
            mlp.down_proj.weight.copy_(best_weight.to(device))
        final_val = measure(vx, vy, mlp.down_proj.weight)
        final_train = measure(tx, ty, mlp.down_proj.weight)
        return {"baseline_val_nmse": baseline_val, "final_val_nmse": final_val,
                "baseline_train_nmse": baseline_train, "final_train_nmse": final_train,
                "baseline_loss": baseline_val, "post_loss": final_val,
                "best_step": best_step, "completed_steps": completed_steps,
                "history": history, "export_dtype": str(original_dtype)}
    finally:
        mlp.down_proj.to(dtype=original_dtype)
        with torch.no_grad():
            mlp.down_proj.weight.copy_(best_weight.to(device))
        for parameter in mlp.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        mlp.train(original_mode)


@torch.no_grad()
def load_pruned_mlp_state(mlp: nn.Module, state: Mapping[str, torch.Tensor],
                          kept_channels: Any) -> dict[str, Any]:
    """Resize using recorded channel IDs, then strictly load projection weights."""
    hidden, intermediate = _validate_mlp(mlp)
    keys = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    if set(state) != keys:
        raise ValueError("Overlay must contain exactly gate/up/down projection weights")
    ids = _channel_ids(kept_channels, intermediate)
    expected = {"gate_proj.weight": (ids.numel(), hidden),
                "up_proj.weight": (ids.numel(), hidden),
                "down_proj.weight": (hidden, ids.numel())}
    for key, shape in expected.items():
        if tuple(state[key].shape) != shape or not torch.isfinite(state[key]).all():
            raise ValueError(f"Invalid overlay tensor: {key}")
    report = prune_mlp(mlp, ids)
    for key, tensor in state.items():
        name = key.split(".")[0]
        getattr(mlp, name).weight.copy_(tensor)
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_overlay(mlps: Mapping[int, nn.Module], reports: Mapping[int, Mapping[str, Any]],
                 directory: str | Path, *, base_sha256: str) -> Path:
    """Write one safetensors file per layer; record independently obtained base SHA."""
    from safetensors.torch import save_file
    if len(base_sha256) != 64 or any(c not in "0123456789abcdef" for c in base_sha256.lower()):
        raise ValueError("base_sha256 must be the independently verified checkpoint SHA256")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    layers = []
    for layer_id, mlp in sorted(mlps.items()):
        _validate_mlp(mlp)
        report = dict(reports[layer_id])
        if len(report["kept_channels"]) != mlp.down_proj.in_features:
            raise ValueError("Report channel count differs from MLP")
        filename = f"layer_{layer_id:03d}.safetensors"
        tensors = {f"{name}.weight": getattr(mlp, name).weight.detach().cpu().contiguous()
                   for name in ("gate_proj", "up_proj", "down_proj")}
        save_file(tensors, str(directory / filename))
        layers.append({**report, "layer_id": layer_id, "file": filename,
                       "sha256": _sha256(directory / filename)})
    manifest = {"format_version": 1, "base_checkpoint_sha256": base_sha256.lower(),
                "layers": layers}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def apply_overlay(mlps: Mapping[int, nn.Module], manifest_path: str | Path,
                  *, expected_base_sha256: str) -> list[dict[str, Any]]:
    """Apply overlay to a fresh dense base, checking independently supplied base SHA."""
    from safetensors.torch import load_file
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported overlay format")
    if manifest.get("base_checkpoint_sha256") != expected_base_sha256.lower():
        raise ValueError("Overlay base checkpoint SHA256 mismatch")
    checked = []
    seen = set()
    for item in manifest["layers"]:
        layer_id = item["layer_id"]
        if layer_id in seen or layer_id not in mlps:
            raise ValueError("Duplicate or unavailable layer in manifest")
        seen.add(layer_id)
        tensor_path = (path.parent / item["file"]).resolve()
        if tensor_path.parent != path.parent.resolve():
            raise ValueError("Overlay file must remain within manifest directory")
        if _sha256(tensor_path) != item["sha256"]:
            raise ValueError(f"Overlay file SHA256 mismatch for layer {layer_id}")
        hidden, intermediate = _validate_mlp(mlps[layer_id])
        if hidden != item["hidden_size"] or intermediate != item["original_intermediate_size"]:
            raise ValueError("Overlay is not being applied to the expected dense MLP")
        ids = _channel_ids(item["kept_channels"], intermediate)
        if ids.numel() != item["intermediate_size"]:
            raise ValueError("Manifest intermediate size does not match channel IDs")
        checked.append((layer_id, tensor_path, ids))
    applied = []
    for layer_id, tensor_path, ids in checked:
        state = load_file(str(tensor_path), device="cpu")
        applied.append({"layer_id": layer_id,
                        **load_pruned_mlp_state(mlps[layer_id], state, ids)})
    return applied
