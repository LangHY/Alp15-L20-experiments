"""Restore recorded recovery adapters for ADE re-evaluation, not training resume."""
import torch


def restore_adapter_checkpoint(adapters, path, parent_manifest_sha256, module_order):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['parent_overlay_manifest_sha256'] != parent_manifest_sha256:
        raise ValueError('Checkpoint parent identity mismatch')
    if saved['module_order'] != module_order or len(saved['adapters']) != len(adapters):
        raise ValueError('Checkpoint module order mismatch')
    for module, (a, b) in zip(adapters, saved['adapters'], strict=True):
        if (module.rank != saved['rank'] or module.scale != saved['scale'] or
                module.a.shape != a.shape or module.b.shape != b.shape or
                not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise ValueError('Checkpoint adapter shape/rank/scale/finite check failed')
    with torch.no_grad():
        for module, (a, b) in zip(adapters, saved['adapters'], strict=True):
            module.a.copy_(a)
            module.b.copy_(b)
    return saved['step']
