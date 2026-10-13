"""Calibration-only, measured-error allocation of linked MLP widths.

The objective is a local sensitivity proxy, never an end-to-end importance claim.
Multiple-choice dynamic programming solves the measured grid under a hard budget.
"""
from __future__ import annotations

import math


def allocate_widths(curves, *, channel_budget, minimum_width=2560,
                    protected_layers=(), protected_minimum=5376, quantum=128):
    if not curves or channel_budget <= 0 or quantum <= 0:
        raise ValueError('Invalid allocation inputs')
    protected = set(protected_layers)
    if not protected.issubset(range(len(curves))):
        raise ValueError('Protected layer outside curves')
    choices = []
    for index, curve in enumerate(curves):
        floor = max(minimum_width, protected_minimum if index in protected else minimum_width)
        layer = []
        seen = set()
        for row in curve:
            width, cost = row['width'], row['calibration_nmse']
            if type(width) is not int or width <= 0 or width % quantum or width in seen:
                raise ValueError('Invalid or duplicate measured width')
            if not math.isfinite(cost) or cost < 0:
                raise ValueError('Invalid measured error')
            seen.add(width)
            if width >= floor:
                layer.append((width // quantum, cost))
        if not layer:
            raise ValueError('No measured width satisfies the layer floor')
        choices.append(sorted(layer))
    cap = channel_budget // quantum
    if sum(min(w for w, _ in layer) for layer in choices) > cap:
        raise ValueError('Budget is infeasible with the declared layer floors')
    states = {0: (0.0, ())}
    for layer in choices:
        next_states = {}
        for used, (cost, widths) in states.items():
            for width, error in layer:
                total = used + width
                if total > cap:
                    continue
                candidate = (cost + error, widths + (width * quantum,))
                if total not in next_states or candidate < next_states[total]:
                    next_states[total] = candidate
        states = next_states
    if not states:
        raise ValueError('No feasible allocation')
    used, (cost, widths) = min(states.items(), key=lambda item: (item[1][0], item[0], item[1][1]))
    return {'layer_widths': list(widths), 'sum_widths': used * quantum,
            'calibration_nmse_sum': cost, 'channel_budget': channel_budget,
            'unused_channels': channel_budget - used * quantum,
            'minimum_width': minimum_width, 'protected_layers': sorted(protected),
            'protected_minimum': protected_minimum,
            'objective': 'Sum of measured, untrained local calibration MLP NMSE; not trajectory ADE'}


def validate_layer_widths(widths, count=36, original=12288):
    if not isinstance(widths, list) or len(widths) != count:
        raise ValueError('Allocation must have exactly one width per layer')
    if any(type(w) is not int or not 128 <= w <= original or w % 128 for w in widths):
        raise ValueError('Each layer width must be a multiple of 128 within the original width')
    return widths


def overlay_widths(manifest, selected, count=36):
    """Validate both legacy uniform overlays and explicit layerwise mappings."""
    if manifest['keep_width'] != selected['keep_width']:
        raise ValueError('Selected width label differs from overlay')
    expected = selected.get('layer_widths')
    if expected is None:
        expected = [selected['keep_width']] * count
    validate_layer_widths(expected, count)
    if manifest.get('layer_widths', expected) != expected:
        raise ValueError('Selected layer widths differ from manifest')
    if [r['layer'] for r in manifest['layers']] != list(range(count)):
        raise ValueError('Overlay must contain all ordered layers')
    if any(r['original_width'] != 12288 or r['keep_width'] != expected[i]
           for i, r in enumerate(manifest['layers'])):
        raise ValueError('Per-layer overlay dimensions differ from allocation')
    return expected
