"""Deterministic probe allocation; example identity is stable across shuffles."""


def expert_probe_index(step, n_examples, example_index, schedule='epoch'):
    if step < 1 or n_examples < 1 or not 0 <= example_index < n_examples:
        raise ValueError('Invalid probe schedule inputs')
    epoch = (step - 1) // n_examples
    if schedule == 'epoch':
        return epoch % 4
    if schedule == 'balanced':
        return (example_index + epoch) % 4
    raise ValueError('Unknown expert probe schedule')
