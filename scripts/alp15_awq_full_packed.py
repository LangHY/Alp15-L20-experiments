"""保存 ModelOpt AWQ 的 INT4、实际 amax 及输入 pre_scale；诊断反量化内核。"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from alp15_packed_w4a16 import pack_signed_int4, unpack_signed_int4

FORMAT = 'alp15-modelopt039-awq-signed-int4-group128-v1'


class PackedAWQLinear(nn.Module):
    def __init__(self, qweight, amax, pre_scale, bias=None):
        super().__init__()
        if qweight.dtype != torch.uint8 or qweight.ndim != 2 or min(qweight.shape) <= 0:
            raise ValueError('qweight 须为非空二维 uint8')
        self.out_features, half_input = qweight.shape
        self.in_features = half_input * 2
        if self.in_features % 128:
            raise ValueError('只支持 group128，不隐式 padding')
        if amax.shape != (self.out_features, self.in_features // 128):
            raise ValueError('amax 须为 output×groups')
        if pre_scale.shape != (self.in_features,):
            raise ValueError('pre_scale 须为逐 input channel')
        for name, value in [('amax', amax), ('pre_scale', pre_scale), ('bias', bias)]:
            if value is None:
                continue
            if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(name + ' 存储dtype不符')
            if value.device != qweight.device or not torch.isfinite(value).all().item():
                raise ValueError(name + ' 非有限或设备不符')
        if not (amax >= 0).all().item() or not (pre_scale > 0).all().item():
            raise ValueError('amax须非负且pre_scale须正')
        if bias is not None and bias.shape != (self.out_features,):
            raise ValueError('bias shape不符')
        unpack_signed_int4(qweight)
        self.register_buffer('qweight', qweight.detach().contiguous())
        self.register_buffer('amax', amax.detach().contiguous())
        self.register_buffer('pre_scale', pre_scale.detach().contiguous())
        self.register_buffer('bias', None if bias is None else bias.detach().contiguous())

    @classmethod
    @torch.no_grad()
    def from_quantized(cls, fake_weight, amax, pre_scale, bias=None):
        """使用已校准 fake-quant 权重，不重新计算 absmax/不覆盖搜索结果。"""
        if fake_weight.ndim != 2 or fake_weight.shape[1] % 128:
            raise ValueError('权重 shape 不符')
        if not torch.isfinite(fake_weight).all().item():
            raise ValueError('fake_weight 非有限')
        rows, channels = fake_weight.shape
        if amax.numel() != rows * channels // 128:
            raise ValueError('amax group 数量不符')
        maxima = amax.detach().reshape(rows, channels // 128).clone()
        grouped = fake_weight.float().reshape(rows, -1, 128)
        denominator = maxima.float().unsqueeze(-1)
        # 固定 ModelOpt0.39 fake_tensor_quant：FP32 中 q/(7/amax)，零amax→零输出。
        inverse_step = torch.where(denominator == 0, torch.zeros_like(denominator), 7. / denominator)
        integer = torch.round(grouped * inverse_step).clamp(-7, 7).to(torch.int8)
        module = cls(pack_signed_int4(integer.flatten(1)), maxima, pre_scale.detach().clone(),
                     None if bias is None else bias.detach().clone())
        reconstructed = module.dequantize(fake_weight.dtype)
        if not torch.equal(reconstructed, fake_weight):
            error = (reconstructed.float() - fake_weight.float()).abs().max().item()
            raise ValueError(f'packed反量化与实际ModelOpt fake_weight不完全一致: {error}')
        return module

    def dequantize(self, dtype=None):
        dtype = self.amax.dtype if dtype is None else dtype
        integers = unpack_signed_int4(self.qweight).float().reshape(self.out_features, -1, 128)
        maxima = self.amax.float().unsqueeze(-1)
        inverse_step = torch.where(maxima == 0, torch.ones_like(maxima), 7. / maxima)
        weight = torch.where(maxima == 0, torch.zeros_like(integers), integers / inverse_step)
        return weight.flatten(1).to(dtype)

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features or inputs.device != self.qweight.device:
            raise ValueError('输入shape/device不符')
        # 不隐式改变 stored pre_scale 精度，保持 ModelOpt 的 inputs * pre_quant_scale 语义。
        scaled = inputs * self.pre_scale
        bias = None if self.bias is None else self.bias.to(scaled.dtype)
        return F.linear(scaled, self.dequantize(scaled.dtype), bias)

    def storage_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.buffers())

    def save(self, path):
        from safetensors.torch import save_file
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        save_file({name: value.detach().cpu().contiguous() for name, value in self.named_buffers()},
                  str(path), metadata={'format': FORMAT, 'group_size': '128',
                                       'in_features': str(self.in_features),
                                       'out_features': str(self.out_features)})

    @classmethod
    def load(cls, path, device='cpu'):
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            metadata = handle.metadata() or {}
        if metadata.get('format') != FORMAT or metadata.get('group_size') != '128':
            raise ValueError('不是固定AWQpacked格式')
        state = load_file(str(path), device='cpu')
        if set(state) not in ({'qweight', 'amax', 'pre_scale'}, {'qweight', 'amax', 'pre_scale', 'bias'}):
            raise ValueError('packed state keys不符')
        model = cls(**state)
        if (str(model.in_features) != metadata.get('in_features') or
                str(model.out_features) != metadata.get('out_features')):
            raise ValueError('metadata shape不符')
        return model.to(device=device)
