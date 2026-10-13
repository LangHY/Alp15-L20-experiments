"""真实 packed W4A16 RTN 诊断载体，非 AWQ、GPTQ 或 ModelOpt。

按输入维度每 128 个权重分一组，signed INT4 范围为 [-7, 7]；
FP32 计算 absmax/7，舍入到实际保存的 FP16/BF16 scale 后再 round-to-nearest。
全零组 scale=1；非零组 scale 下溢时夹到存储 dtype 的最小正规值。
不量化激活，不保留原 dense 权重，不进行误差补偿或训练。
forward 临时反量化后调用 F.linear，仅供离线质量下限筛查；它没有生产
INT4 kernel，不能据其耗时宣称量化加速。可按输出行分块降低临时显存。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


GROUP_SIZE = 128
FORMAT = "alp15-rtn-signed-int4-group128-v1"
STORAGE_DTYPES = (torch.float16, torch.bfloat16)


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} 包含非有限值")


def pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    """二补码 nibble：偶数位置为低四位，奇数位置为高四位。"""
    if values.ndim < 1 or values.shape[-1] % 2 or values.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("打包输入须为末维偶数的整数 tensor")
    if not ((values >= -7) & (values <= 7)).all().item():
        raise ValueError("signed INT4 RTN 仅支持 [-7,7]")
    nibble = values.to(torch.int8).bitwise_and(15).to(torch.uint8)
    return (nibble[..., 0::2] | (nibble[..., 1::2] << 4)).contiguous()


def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.ndim < 1:
        raise ValueError("packed 权重须为 UINT8 tensor")
    low = (packed & 15).to(torch.int8)
    high = (packed >> 4).to(torch.int8)
    values = torch.stack((low, high), dim=-1).flatten(-2)
    values = torch.where(values >= 8, values - 16, values)
    if (values == -8).any().item():
        raise ValueError("发现不属于 [-7,7] 的 -8 nibble")
    return values


class PackedW4A16Linear(nn.Module):
    """只保存 qweight、scales、可选 bias 的诊断 Linear。"""

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor,
                 bias: torch.Tensor | None = None, *, group_size: int = GROUP_SIZE,
                 tile_out_features: int | None = None):
        super().__init__()
        if group_size != GROUP_SIZE or isinstance(group_size, bool):
            raise ValueError("当前格式限定 group_size=128")
        if qweight.ndim != 2 or min(qweight.shape) <= 0 or qweight.dtype != torch.uint8:
            raise ValueError("qweight 须为非空二维 UINT8")
        out_features, half_in_features = qweight.shape
        in_features = half_in_features * 2
        if in_features % GROUP_SIZE:
            raise ValueError("in_features 必须整除 128，不隐含 padding")
        if scales.shape != (out_features, in_features // GROUP_SIZE):
            raise ValueError("scales 尺寸不符")
        if scales.dtype not in STORAGE_DTYPES:
            raise ValueError("scales 必须保存为 FP16 或 BF16")
        if scales.device != qweight.device:
            raise ValueError("qweight 与 scales 须位于同一设备")
        _finite(scales, "scales")
        if not (scales > 0).all().item():
            raise ValueError("scales 必须为正")
        # 验证格式不构造完整 INT8/dense 权重。
        if ((qweight & 15) == 8).any().item() or ((qweight >> 4) == 8).any().item():
            raise ValueError("qweight 含非法 -8 nibble")
        if bias is not None:
            if bias.shape != (out_features,) or bias.dtype != scales.dtype or bias.device != scales.device:
                raise ValueError("bias 的尺寸、存储 dtype 或设备不符")
            _finite(bias, "bias")
        if tile_out_features is not None:
            _positive(tile_out_features, "tile_out_features")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = GROUP_SIZE
        self.tile_out_features = tile_out_features
        self.register_buffer("qweight", qweight.detach().contiguous())
        self.register_buffer("scales", scales.detach().contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().contiguous())

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, *, group_size: int = GROUP_SIZE,
                    storage_dtype: torch.dtype = torch.float16,
                    tile_out_features: int | None = None) -> "PackedW4A16Linear":
        if not isinstance(linear, nn.Linear) or storage_dtype not in STORAGE_DTYPES:
            raise ValueError("输入须为 nn.Linear，存储 dtype 须为 FP16/BF16")
        if group_size != GROUP_SIZE or isinstance(group_size, bool) or linear.in_features % GROUP_SIZE:
            raise ValueError("当前格式限定 group128，in_features 须整除 128")
        _finite(linear.weight, "weight")
        weight = linear.weight.detach().float().reshape(linear.out_features, -1, GROUP_SIZE)
        _finite(weight, "FP32 weight")
        maxima = weight.abs().amax(dim=-1)
        scale = torch.where(maxima == 0, torch.ones_like(maxima), maxima / 7)
        scale = scale.clamp_min(torch.finfo(storage_dtype).tiny).to(storage_dtype)
        _finite(scale, "存储 scale")
        integers = torch.round(weight / scale.float().unsqueeze(-1)).clamp(-7, 7).to(torch.int8)
        qweight = pack_signed_int4(integers.reshape(linear.out_features, linear.in_features))
        bias = None
        if linear.bias is not None:
            _finite(linear.bias, "bias")
            bias = linear.bias.detach().to(storage_dtype).clone()
            _finite(bias, "存储 bias")
        return cls(qweight, scale, bias, group_size=group_size, tile_out_features=tile_out_features)

    def _apply(self, fn, recurse=True):
        # Module.to(dtype=...) 本就保留整数 buffer；另外拒绝误转与 scale 损坏。
        qweight = fn(self.qweight)
        scales = fn(self.scales)
        if qweight.dtype != torch.uint8 or scales.dtype not in STORAGE_DTYPES:
            raise ValueError("转换须保留 UINT8 qweight 与 FP16/BF16 scales")
        _finite(scales, "转换后 scales")
        if not (scales > 0).all().item():
            raise ValueError("转换后 scales 下溢为零")
        bias = None if self.bias is None else fn(self.bias)
        if bias is not None:
            _finite(bias, "转换后 bias")
        # 已完成无副作用预检，避免 fn 再次复制同一批 buffer。
        self._buffers.update(qweight=qweight, scales=scales, bias=bias)
        return self

    def _dequantize_rows(self, start: int, stop: int, dtype: torch.dtype) -> torch.Tensor:
        integers = unpack_signed_int4(self.qweight[start:stop])
        groups = integers.reshape(stop - start, -1, GROUP_SIZE).to(torch.float32)
        weight = groups * self.scales[start:stop].float().unsqueeze(-1)
        return weight.reshape(stop - start, self.in_features).to(dtype)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """每次新建临时 dense 权重；不在模块内缓存。"""
        dtype = self.scales.dtype if dtype is None else dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("反量化 dtype 须为浮点")
        return self._dequantize_rows(0, self.out_features, dtype)

    def _check_inputs(self, inputs: torch.Tensor) -> None:
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features or not inputs.is_floating_point():
            raise ValueError("激活末维或 dtype 不符")
        if inputs.device != self.qweight.device:
            raise ValueError("激活与 packed 权重须位于同一设备")

    def reference(self, inputs: torch.Tensor) -> torch.Tensor:
        """完整反量化 F.linear 参考；FP32 CPU 输入亦可用于数学验证。"""
        self._check_inputs(inputs)
        bias = None if self.bias is None else self.bias.to(inputs.dtype)
        return F.linear(inputs, self.dequantize(inputs.dtype), bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._check_inputs(inputs)
        if self.tile_out_features is None:
            return self.reference(inputs)
        outputs = []
        for start in range(0, self.out_features, self.tile_out_features):
            stop = min(start + self.tile_out_features, self.out_features)
            bias = None if self.bias is None else self.bias[start:stop].to(inputs.dtype)
            outputs.append(F.linear(inputs, self._dequantize_rows(start, stop, inputs.dtype), bias))
        return torch.cat(outputs, dim=-1)

    def storage_bytes(self) -> int:
        """真实持久 tensor 字节；不计 Python 开销、文件头及临时反量化。"""
        return sum(t.numel() * t.element_size() for t in self.buffers())

    def storage_report(self) -> dict:
        return {"format": FORMAT, "in_features": self.in_features,
                "out_features": self.out_features, "group_size": GROUP_SIZE,
                "scale_dtype": str(self.scales.dtype), "bias": self.bias is not None,
                "persistent_tensor_bytes": self.storage_bytes(),
                "qweight_bytes": self.qweight.numel(),
                "scale_bytes": self.scales.numel() * self.scales.element_size(),
                "bias_bytes": 0 if self.bias is None else self.bias.numel() * self.bias.element_size(),
                "diagnostic_dequantize_linear": True, "activation_quantized": False}

    def save(self, path: str | Path) -> None:
        """只用 safetensors，不接受 pickle/任意代码反序列化。"""
        from safetensors.torch import save_file
        tensors = {name: value.detach().cpu().contiguous() for name, value in self.named_buffers()}
        metadata = {"format": FORMAT, "group_size": str(GROUP_SIZE),
                    "in_features": str(self.in_features), "out_features": str(self.out_features),
                    "tile_out_features": json.dumps(self.tile_out_features)}
        save_file(tensors, str(path), metadata=metadata)

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "PackedW4A16Linear":
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if metadata.get("format") != FORMAT or metadata.get("group_size") != str(GROUP_SIZE):
            raise ValueError("非支持的 packed W4A16 safetensors 格式")
        tensors = load_file(str(path), device="cpu")
        if set(tensors) not in ({"qweight", "scales"}, {"qweight", "scales", "bias"}):
            raise ValueError("packed 文件须且只能保存 qweight/scales/可选 bias")
        try:
            tile = json.loads(metadata["tile_out_features"])
            model = cls(tensors["qweight"], tensors["scales"], tensors.get("bias"), tile_out_features=tile)
            if (model.in_features != int(metadata["in_features"]) or
                    model.out_features != int(metadata["out_features"])):
                raise ValueError("metadata 维度与 tensor 不一致")
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("packed metadata 非法") from error
        return model.to(device=device)
