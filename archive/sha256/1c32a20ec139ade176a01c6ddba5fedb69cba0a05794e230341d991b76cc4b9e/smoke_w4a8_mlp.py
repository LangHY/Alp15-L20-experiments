"""Real 8B MLP dimensions and weights; no complete model quantization claim."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from w4a8 import PackedW4A8Linear, quantize_activation_reference, unpack_int4


@torch.no_grad()
def gpu_integer_reference(module, x):
    # Signed int8/int4 products and a 128-term sum fit exactly within FP32's
    # integer range. Disable TF32 and use groupwise FP32 matmul only for this
    # diagnostic reference, never for the inference forward path.
    # CUDA torch division by a host scalar can multiply by its reciprocal,
    # unlike Triton's correctly rounded div_rn. Quantize on CPU so boundary
    # ties follow the same diagnostic reference used by the unit tests.
    qa, sa = quantize_activation_reference(x.detach().cpu())
    qa, sa = qa.to(x.device), sa.to(x.device)
    qw = unpack_int4(module.qweight).reshape(module.out_features, -1, 128)
    out = torch.zeros(x.shape[0], module.out_features, device=x.device)
    for group in range(module.in_features // 128):
        dot = F.linear(qa[:, group].float(), qw[:, group].float())
        out += (dot * sa[:, group, None]) * module.scales[:, group].float()[None, :]
    if module.bias is not None:
        out += module.bias.float()
    return out.to(module.original_dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--teacher-experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite a smoke result directory')
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(8)
    state = load_file(str(args.overlay / 'mlp_00.safetensors'))
    cache = torch.load(args.teacher_experiment / 'teacher_cache/train_00.pt',
                       weights_only=True, map_location='cpu')['x']
    records = []
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        weight = state[name + '.weight'].to('cuda')
        n, k = weight.shape
        linear = torch.nn.Linear(k, n, bias=False, dtype=weight.dtype, device='cuda')
        linear.weight.copy_(weight)
        packed = PackedW4A8Linear.from_linear(linear)
        path = args.output / (name + '.safetensors')
        packed.save(path)
        reloaded = PackedW4A8Linear.load(path, device='cuda')
        for m in (1, 256):
            if name == 'down_proj':
                x0 = cache[:m].to('cuda')
                x = F.silu(F.linear(x0, state['gate_proj.weight'].to('cuda'))) * F.linear(
                    x0, state['up_proj.weight'].to('cuda'))
            else:
                x = cache[:m].to('cuda')
            actual = packed(x)
            reference = gpu_integer_reference(packed, x)
            reload_output = reloaded(x)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(actual, reload_output, rtol=0, atol=0)
            dense = linear(x).float()
            error = actual.float() - dense
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(5):
                packed(x)
            torch.cuda.synchronize()
            records.append({'projection': name, 'M': m, 'N': n, 'K': k,
                'integer_reference_max_abs': (actual.float() - reference.float()).abs().max().item(),
                'dense_nmse': error.square().mean().item() / max(dense.square().mean().item(), 1e-30),
                'packed_bytes': packed.storage_bytes(), 'bf16_bytes': weight.numel() * 2,
                'same_process_save_reload_identical': True,
                'diagnostic_ms_shared_gpu': (time.perf_counter() - started) * 1000 / 5})
            (args.output / 'progress.json').write_text(json.dumps(records, indent=2) + '\n')
            print('REAL_MLP_SMOKE', name, m, n, k, records[-1], flush=True)
        del packed, reloaded, linear, weight, actual, reference, reload_output, dense, x
        torch.cuda.empty_cache()
    (args.output / 'result.json').write_text(json.dumps({'status': 'passed', 'records': records,
        'method': 'Independent Triton RTN signed W4 / dynamic group INT8',
        'limits': ['Single MLP block, no end-to-end model quality claim',
                   'Concurrent shared GPU timing; no deployment speed claim',
                   'Same-process serialization test, not a fresh-process reload']}, indent=2) + '\n')


if __name__ == '__main__':
    with torch.no_grad():
        main()


