"""隔离 Triton signed INT4 AWQ 候选；不自动替换完整模型。

仅支持已由 PackedAWQLinear 验证的连续 CUDA BF16 group128 数据。
amax/pre_scale 保持 BF16，CPU import 无 Triton 副作用，forward 无 host 同步。
"""
import torch
from torch import nn

from alp15_awq_full_packed import PackedAWQLinear

_KERNELS = None


class FusedAWQLinear(nn.Module):
    """共享冻结 packed buffers，不建立完整 dense 权重缓存。"""
    def __init__(self, reference):
        super().__init__()
        if not isinstance(reference, PackedAWQLinear):
            raise TypeError("必须提供已经验真的 PackedAWQLinear")
        self.in_features = reference.in_features
        self.out_features = reference.out_features
        for name in ("qweight", "amax", "pre_scale", "bias"):
            self.register_buffer(name, getattr(reference, name))
        self._check_storage()

    @classmethod
    def from_reference(cls, reference):
        return cls(reference)

    def _check_storage(self):
        if self.qweight.dtype != torch.uint8 or self.qweight.shape != (self.out_features, self.in_features // 2):
            raise ValueError("qweight shape/dtype 不符")
        if self.in_features % 128 or self.amax.shape != (self.out_features, self.in_features // 128):
            raise ValueError("只支持 group128")
        if self.pre_scale.shape != (self.in_features,):
            raise ValueError("pre_scale shape 不符")
        for value in (self.amax, self.pre_scale, self.bias):
            if value is not None and (value.dtype != torch.bfloat16 or value.device != self.qweight.device):
                raise ValueError("候选仅支持同设备 BF16 amax/pre_scale/bias，不隐式转换")
        if self.bias is not None and self.bias.shape != (self.out_features,):
            raise ValueError("bias shape 不符")
        if any(not value.is_contiguous() for value in self.buffers()):
            raise ValueError("packed buffers 必须连续")

    def forward(self, inputs):
        self._check_storage()
        if inputs.device.type != "cuda" or self.qweight.device.type != "cuda":
            raise RuntimeError("候选需要实际 CUDA；不回退 reference 或 CPU")
        if inputs.device != self.qweight.device or inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError("输入 shape/device 不符")
        if inputs.dtype != torch.bfloat16 or not inputs.is_contiguous() or inputs.requires_grad:
            raise ValueError("候选只支持连续、无梯度 BF16 输入")
        flat = inputs.view(-1, self.in_features)
        m = flat.shape[0]
        if m == 0:
            raise ValueError("输入不得为空")
        with torch.cuda.device(inputs.device):
            if torch.cuda.get_device_capability(inputs.device)[0] < 8:
                raise RuntimeError("BF16 候选需要 NVIDIA compute capability >=8")
            gemv, gemm, _ = _load_kernels()
            output = torch.empty((m, self.out_features), device=inputs.device, dtype=torch.bfloat16)
            bias = self.bias if self.bias is not None else self.pre_scale
            if m <= 8:
                gemv[((self.out_features + 7) // 8, m)](
                    flat, self.qweight, self.amax, self.pre_scale, bias, output,
                    self.out_features, self.in_features, HAS_BIAS=self.bias is not None,
                    BN=8, BK=128, num_warps=4, enable_fp_fusion=False)
            else:
                gemm[((m + 31) // 32, (self.out_features + 63) // 64)](
                    flat, self.qweight, self.amax, self.pre_scale, bias, output,
                    m, self.out_features, self.in_features, HAS_BIAS=self.bias is not None,
                    BM=32, BN=64, BK=128, num_warps=4, enable_fp_fusion=False)
            return output.view(*inputs.shape[:-1], self.out_features)

    def dequantize_for_validation(self):
        """显式验证工具临时输出单矩阵 BF16；forward 永不调用或缓存。"""
        self._check_storage()
        if self.qweight.device.type != "cuda":
            raise RuntimeError("反量化桥接需要实际 CUDA")
        with torch.cuda.device(self.qweight.device):
            _, _, dequant = _load_kernels()
            result = torch.empty((self.out_features, self.in_features),
                                 device=self.qweight.device, dtype=torch.bfloat16)
            dequant[((self.out_features + 7) // 8, self.in_features // 128)](
                self.qweight, self.amax, result, self.out_features, self.in_features,
                BN=8, BK=128, num_warps=4, enable_fp_fusion=False)
            return result


def _load_kernels():
    global _KERNELS, triton, tl
    if _KERNELS is not None:
        return _KERNELS
    try:
        import triton
        import triton.language as tl
    except ImportError as error:
        raise RuntimeError("CUDA 候选需要已经安装的 Triton；不自动下载或fallback") from error

    @triton.jit
    def weight_block(Q, A, ns, ks, group, N: tl.constexpr, K: tl.constexpr):
        packed = tl.load(Q + ns[:, None] * (K // 2) + ks[None, :] // 2,
                         mask=(ns[:, None] < N) & (ks[None, :] < K), other=0).to(tl.int32)
        nibble = (packed >> ((ks[None, :] % 2) * 4)) & 15
        integer = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
        maxima = tl.load(A + ns * (K // 128) + group, mask=ns < N, other=0).to(tl.float32)
        # Torch `7. / tensor` 实际 aten.reciprocal -> mul，严格保留两次FP32舍入。
        inverse = tl.div_rn(1.0, tl.where(maxima == 0, 1.0, maxima)) * 7.0
        weight = tl.where(maxima[:, None] == 0, 0.0, tl.div_rn(integer, inverse[:, None]))
        return weight.to(tl.bfloat16)

    @triton.jit
    def gemv(X, Q, A, P, B, Y, N: tl.constexpr, K: tl.constexpr,
             HAS_BIAS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        ns = tl.program_id(0) * BN + tl.arange(0, BN)
        m = tl.program_id(1)
        ks0 = tl.arange(0, BK)
        accumulator = tl.zeros((BN, BK), dtype=tl.float32)
        for start in range(0, K, BK):
            ks = start + ks0
            x = tl.load(X + m * K + ks).to(tl.float32)
            p = tl.load(P + ks).to(tl.float32)
            scaled = (x * p).to(tl.bfloat16)
            weight = weight_block(Q, A, ns, ks, start // 128, N, K)
            accumulator += scaled[None, :].to(tl.float32) * weight.to(tl.float32)
        total = tl.sum(accumulator, axis=1)
        if HAS_BIAS:
            total += tl.load(B + ns, mask=ns < N, other=0).to(tl.float32)
        tl.store(Y + m * N + ns, total.to(tl.bfloat16), mask=ns < N)

    @triton.jit
    def gemm(X, Q, A, P, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        ms = tl.program_id(0) * BM + tl.arange(0, BM)
        ns = tl.program_id(1) * BN + tl.arange(0, BN)
        ks0 = tl.arange(0, BK)
        accumulator = tl.zeros((BM, BN), dtype=tl.float32)
        for start in range(0, K, BK):
            ks = start + ks0
            x = tl.load(X + ms[:, None] * K + ks[None, :], mask=ms[:, None] < M, other=0).to(tl.float32)
            p = tl.load(P + ks).to(tl.float32)
            scaled = (x * p[None, :]).to(tl.bfloat16)
            weight = weight_block(Q, A, ns, ks, start // 128, N, K)
            accumulator = tl.dot(scaled, tl.trans(weight), accumulator)
        if HAS_BIAS:
            accumulator += tl.load(B + ns, mask=ns < N, other=0).to(tl.float32)[None, :]
        tl.store(Y + ms[:, None] * N + ns[None, :], accumulator.to(tl.bfloat16),
                 mask=(ms[:, None] < M) & (ns[None, :] < N))

    @triton.jit
    def dequant(Q, A, Y, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        ns = tl.program_id(0) * BN + tl.arange(0, BN)
        ks = tl.program_id(1) * BK + tl.arange(0, BK)
        weight = weight_block(Q, A, ns, ks, tl.program_id(1), N, K)
        tl.store(Y + ns[:, None] * K + ks[None, :], weight,
                 mask=(ns[:, None] < N) & (ks[None, :] < K))

    _KERNELS = gemv, gemm, dequant
    return _KERNELS
