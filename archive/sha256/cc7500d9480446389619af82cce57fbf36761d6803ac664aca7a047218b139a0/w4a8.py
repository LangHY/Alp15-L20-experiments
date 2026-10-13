"""RTN packed W4 / dynamic per-row, per-K-group A8 inference.

This is a minimal native Triton implementation, not ParoQuant/FlashDrive.
GPU throughput must be measured: the group loop and on-the-fly activation
quantization intentionally prioritize a transparent, verifiable baseline.
No complete floating-point weight is retained or produced by GPU forward.

INT4 uses two's-complement nibbles, even K in the low nibble, values [-7, 7].
Weights: one BF16 scale per output/K-group of 128. Activations: one FP32
dynamic scale per row/K-group of 128, signed INT8 [-127, 127]. Each group's
INT32 dot is scaled and accumulated in FP32; output uses the source dtype.
Nonfinite weights/bias are rejected; nonfinite input rows yield NaN rows.

API references verified against upstream documentation/source:
https://triton-lang.org/main/python-api/generated/triton.language.dot.html
https://raw.githubusercontent.com/triton-lang/triton/main/third_party/nvidia/language/cuda/libdevice.py
https://docs.nvidia.com/cuda/archive/12.8.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

GROUP_SIZE = 128
FORMAT = "packed_signed_int4_dynamic_group_int8_v1"
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
           "float32": torch.float32}


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed integer values [-7, 7] along the last dimension."""
    if values.ndim < 1 or values.shape[-1] % 2:
        raise ValueError("INT4 packing requires an even last dimension")
    if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("INT4 packing requires a signed integer tensor")
    if bool(((values < -7) | (values > 7)).any()):
        raise ValueError("Symmetric INT4 values must lie in [-7, 7]")
    nibble = values.to(torch.int16) & 15
    return (nibble[..., 0::2] | (nibble[..., 1::2] << 4)).to(torch.uint8).contiguous()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Debug/reference only: expand UINT8 bytes to signed INT8 values."""
    if packed.dtype != torch.uint8 or packed.ndim < 1:
        raise TypeError("Packed INT4 must be a UINT8 tensor")
    byte = packed.to(torch.int16)
    low, high = byte & 15, (byte >> 4) & 15
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack((low, high), dim=-1).flatten(-2).to(torch.int8)


def quantize_activation_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Slow debug quantizer returning INT8 [M,G,128] and FP32 [M,G] scales.

    Nonfinite elements are replaced by zero for quantization. The caller must
    propagate an invalid-row mask, as ``reference`` and the GPU kernel do.
    """
    if x.ndim != 2 or x.shape[1] % GROUP_SIZE:
        raise ValueError("Reference activation must be [M,K], K divisible by 128")
    grouped = x.float().reshape(x.shape[0], x.shape[1] // GROUP_SIZE, GROUP_SIZE)
    grouped = torch.where(torch.isfinite(grouped), grouped, 0.0)
    amax = grouped.abs().amax(-1)
    scales = (amax / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
    scales = torch.where(amax > 0, scales, torch.ones_like(scales))
    values = torch.round(grouped / scales.unsqueeze(-1)).clamp(-127, 127)
    return values.to(torch.int8), scales


if triton is not None:
    @triton.jit
    def _w4a8_kernel(X, QW, SW, BIAS, Y, M: tl.constexpr, N: tl.constexpr,
                     K: tl.constexpr, HAS_BIAS: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     GROUP: tl.constexpr):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        group_k = tl.arange(0, GROUP)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        invalid = tl.full((BLOCK_M,), False, tl.int1)
        for group in range(K // GROUP):
            ks = group * GROUP + group_k
            x = tl.load(X + rows[:, None] * K + ks[None, :],
                        mask=rows[:, None] < M, other=0).to(tl.float32)
            finite = tl.abs(x) <= 3.4028234663852886e38
            invalid = invalid | (tl.sum((~finite).to(tl.int32), 1) > 0)
            x = tl.where(finite, x, 0.0)
            amax = tl.max(tl.abs(x), 1)
            sx = tl.maximum(tl.div_rn(amax, 127.0), 1.1754943508222875e-38)
            sx = tl.where(amax > 0, sx, 1.0)
            qx = libdevice.nearbyint(tl.div_rn(x, sx[:, None]))
            qx = tl.minimum(tl.maximum(qx, -127.0), 127.0).to(tl.int8)
            # [K,N] weight tile is unpacked directly into INT8 registers.
            byte = tl.load(QW + cols[None, :] * (K // 2) + (ks[:, None] // 2),
                           mask=cols[None, :] < N, other=0).to(tl.int32)
            nibble = (byte >> ((ks[:, None] % 2) * 4)) & 15
            qw = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.int8)
            dot = tl.dot(qx, qw, out_dtype=tl.int32)
            sw = tl.load(SW + cols * (K // GROUP) + group,
                         mask=cols < N, other=0).to(tl.float32)
            # Keep multiplication order equal to the explicit reference.
            acc = acc + (dot.to(tl.float32) * sx[:, None]) * sw[None, :]
        if HAS_BIAS:
            bias = tl.load(BIAS + cols, mask=cols < N, other=0).to(tl.float32)
            acc = acc + bias[None, :]
        acc = tl.where(invalid[:, None], float("nan"), acc)
        tl.store(Y + rows[:, None] * N + cols[None, :], acc,
                 mask=(rows[:, None] < M) & (cols[None, :] < N))


class PackedW4A8Linear(nn.Module):
    """Inference-only packed linear. CPU callers must explicitly use reference.

    Construct with ``from_linear`` for RTN conversion; dimensions and dtype
    are explicit in ``metadata`` for safely rebuilding from state buffers.
    ``state_dict`` contains tensors only, suitable for safetensors overlays.
    Input/output feature sizes require K % 128 == 0 and N % 64 == 0.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 original_dtype: torch.dtype = torch.bfloat16, device=None):
        super().__init__()
        if in_features <= 0 or in_features % GROUP_SIZE:
            raise ValueError("in_features must be a positive multiple of 128")
        if out_features <= 0 or out_features % 64:
            raise ValueError("out_features must be a positive multiple of 64")
        if original_dtype not in _DTYPES.values():
            raise ValueError("Source dtype must be BF16, FP16, or FP32")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = GROUP_SIZE
        self.original_dtype = original_dtype
        self.register_buffer("qweight", torch.zeros((out_features, in_features // 2),
                                                    dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.ones((out_features, in_features // GROUP_SIZE),
                                                  dtype=torch.bfloat16, device=device))
        self.register_buffer("bias", torch.zeros(out_features, dtype=original_dtype,
                                                device=device) if bias else None)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear) -> "PackedW4A8Linear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("from_linear expects torch.nn.Linear")
        if linear.weight.device.type == "meta":
            raise ValueError("Cannot quantize a meta-device weight")
        if not bool(torch.isfinite(linear.weight).all()):
            raise ValueError("Cannot quantize nonfinite weights")
        if linear.bias is not None and not bool(torch.isfinite(linear.bias).all()):
            raise ValueError("Cannot quantize a nonfinite bias")
        result = cls(linear.in_features, linear.out_features, linear.bias is not None,
                     linear.weight.dtype, linear.weight.device)
        grouped = linear.weight.detach().float().reshape(
            linear.out_features, linear.in_features // GROUP_SIZE, GROUP_SIZE)
        amax = grouped.abs().amax(-1)
        scale = (amax / 7.0).clamp_min(torch.finfo(torch.bfloat16).tiny)
        scale = torch.where(amax > 0, scale, torch.ones_like(scale)).to(torch.bfloat16)
        # Quantize against the actual stored BF16 scale, not an FP32 precursor.
        quantized = torch.round(grouped / scale.float().unsqueeze(-1)).clamp(-7, 7)
        result.qweight.copy_(pack_int4(quantized.to(torch.int8).flatten(1)))
        result.scales.copy_(scale)
        if result.bias is not None:
            result.bias.copy_(linear.bias.detach())
        result.eval()
        return result

    def _check_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last input dimension {self.in_features}")
        if x.dtype not in _DTYPES.values():
            raise TypeError("Input must be BF16, FP16, or FP32")
        if x.device != self.qweight.device:
            raise ValueError("Input and packed weights must share a device")
        if self.qweight.dtype != torch.uint8 or self.scales.dtype != torch.bfloat16:
            raise TypeError("Packed buffers require UINT8 qweight and BF16 scales; use .to(device) without dtype")
        return x.contiguous().reshape(-1, self.in_features)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self._check_input(x)
        if flat.device.type != "cuda":
            raise RuntimeError("Packed W4A8 forward requires CUDA; call .reference(x) explicitly for slow CPU testing")
        if triton is None:
            raise RuntimeError("CUDA packed W4A8 requires Triton; no dense fallback is used")
        if torch.version.hip is not None:
            raise RuntimeError("This baseline targets NVIDIA CUDA libdevice")
        output = torch.empty((flat.shape[0], self.out_features), device=x.device,
                             dtype=self.original_dtype)
        if flat.shape[0]:
            _w4a8_kernel[(triton.cdiv(flat.shape[0], 16), triton.cdiv(self.out_features, 64))](
                flat, self.qweight, self.scales,
                self.bias if self.bias is not None else self.scales, output,
                flat.shape[0], self.out_features, self.in_features,
                self.bias is not None, BLOCK_M=16, BLOCK_N=64,
                GROUP=GROUP_SIZE, num_warps=4, num_stages=2, enable_fp_fusion=False)
        return output.reshape(*x.shape[:-1], self.out_features)

    @torch.no_grad()
    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """Explicit slow integer reference, never selected by GPU forward.

        Integer matrix multiply uses CPU INT32 even when x is on CUDA, since
        public torch INT32 matmul is not consistently available on CUDA.
        Copies/full INT8 unpacking here are diagnostic-only and temporary.
        """
        flat = self._check_input(x)
        cpu_x = flat.detach().cpu()
        qa, sa = quantize_activation_reference(cpu_x)
        qw = unpack_int4(self.qweight.cpu()).reshape(
            self.out_features, self.in_features // GROUP_SIZE, GROUP_SIZE)
        sw = self.scales.cpu().float()
        acc = torch.zeros((flat.shape[0], self.out_features), dtype=torch.float32)
        for group in range(self.in_features // GROUP_SIZE):
            dot = qa[:, group].to(torch.int32) @ qw[:, group].to(torch.int32).T
            acc.add_((dot.float() * sa[:, group, None]) * sw[None, :, group])
        if self.bias is not None:
            acc.add_(self.bias.cpu().float())
        invalid = ~torch.isfinite(cpu_x).all(-1)
        acc[invalid] = float("nan")
        return acc.to(device=x.device, dtype=self.original_dtype).reshape(
            *x.shape[:-1], self.out_features)

    @torch.no_grad()
    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Debug-only full floating weight; returned tensor is never cached."""
        quantized = unpack_int4(self.qweight).reshape(
            self.out_features, self.in_features // GROUP_SIZE, GROUP_SIZE)
        return (quantized.float() * self.scales.float().unsqueeze(-1)).flatten(1).to(
            self.original_dtype if dtype is None else dtype)

    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffers())

    def metadata(self) -> dict[str, Any]:
        return {"format": FORMAT, "in_features": self.in_features,
                "out_features": self.out_features, "group_size": GROUP_SIZE,
                "bias": self.bias is not None,
                "original_dtype": str(self.original_dtype).removeprefix("torch.")}

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], device=None) -> "PackedW4A8Linear":
        if metadata.get("format") != FORMAT or metadata.get("group_size") != GROUP_SIZE:
            raise ValueError("Unsupported packed quantization format/group size")
        if metadata.get("original_dtype") not in _DTYPES or type(metadata.get("bias")) is not bool:
            raise ValueError("Invalid packed quantization metadata")
        if type(metadata.get("in_features")) is not int or type(metadata.get("out_features")) is not int:
            raise ValueError("Packed dimensions must be explicit integers")
        return cls(metadata["in_features"], metadata["out_features"], metadata["bias"],
                   _DTYPES[metadata["original_dtype"]], device=device)

    def save(self, path: str | Path) -> None:
        """Save tensor buffers and explicit dimensions to one safetensors file."""
        from safetensors.torch import save_file
        tensors = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()}
        save_file(tensors, str(path), metadata={"w4a8": json.dumps(self.metadata())})

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "PackedW4A8Linear":
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = (handle.metadata() or {}).get("w4a8")
        if metadata is None:
            raise ValueError("Missing explicit w4a8 safetensors metadata")
        result = cls.from_metadata(json.loads(metadata), device=device)
        tensors = load_file(str(path), device=str(device))
        expected = result.state_dict()
        if set(tensors) != set(expected):
            raise ValueError("Packed safetensors buffer names do not match metadata")
        for name, tensor in tensors.items():
            if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
                raise ValueError(f"Invalid packed buffer shape/dtype: {name}")
        if not bool(torch.isfinite(tensors["scales"]).all()) or bool((tensors["scales"] <= 0).any()):
            raise ValueError("Packed scales must be finite and positive")
        packed = tensors["qweight"]
        if bool((((packed & 15) == 8) | ((packed >> 4) == 8)).any()):
            raise ValueError("Packed symmetric INT4 cannot contain the reserved -8 value")
        if "bias" in tensors and not bool(torch.isfinite(tensors["bias"]).all()):
            raise ValueError("Packed bias must be finite")
        result.load_state_dict(tensors, strict=True)
        return result.eval()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, W4A8, group_size={GROUP_SIZE}, "
                f"original_dtype={self.original_dtype}")


