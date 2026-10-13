# Alpamayo 1.5 L20 BF16 离线兼容基准

## 结论

2026-09-20 在 NVIDIA L20 上完成 Alpamayo 1.5 10B BF16 **离线兼容性端到端推理与性能基准**。该结论不等同于“NVIDIA 官方原始入口跑通”；实际入口是自定义 `scripts/benchmark_alpamayo15_offline.py`，其记录 SHA-256 为 `861ff3d8a60c420681fd5eeb069a405d5159bf4dc4a6743fe84a3855937593b0`。

原始结构化结果：[alp15_l20_baseline_20260920_101943.json](alp15_l20_baseline_20260920_101943.json)

## 固定输入与环境

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA L20，Compute Capability 8.9，BF16 支持 |
| 系统 | Ubuntu 22.04，Linux x86_64 |
| Python | 3.12.14 |
| PyTorch / CUDA build | 2.8.0+cu128 / 12.8 |
| FlashAttention | 2.8.3 |
| Transformers | 4.57.1 |
| physical-ai-av | 0.2.0 |
| 官方源码 HEAD | `36aeb4c5938cbc2eb2aed33b22434773da4ab639`，工作树 clean |
| clip | `030c760c-ae38-49aa-9ad8-f5650a545d26` |
| `t0_us` / `chunk_id` | `5100000` / `3119` |
| 输入 | 4 摄像头 × 4 帧，`(4, 4, 3, 1080, 1920)` |
| 生成参数 | seed=42，top_p=0.98，temperature=0.6，1 条轨迹，最大 256 token |

## 结果摘要

| 指标 | 结果 |
| --- | ---: |
| 模型参数量 | 11.0785B |
| BF16 参数存储 | 20.64 GiB |
| 模型加载 | 5304.86 ms |
| 冷推理 | 2384.64 ms |
| 稳态平均延迟（5 次） | 1615.68 ms |
| 稳态 P95 | 1619.31 ms |
| 稳态标准差 | 2.74 ms |
| VLM generate | 1364.01 ms |
| Diffusion sample | 252.78 ms |
| 峰值 allocated / reserved | 22126 / 22502 MiB |
| minADE | 0.371870 m |
| `pred_xyz` | `(1, 1, 1, 64, 3)` |
| `pred_rot` | `(1, 1, 1, 64, 3, 3)` |

五次稳态运行的 CoC、输出形状和 minADE 完全一致。生成的 CoC 为：

> Nudge to the left to clear the construction equipment blocking the right side of our lane

## 证据边界

- JSON 文件 SHA-256：`3c9fe0d17a38c73fe1dd8229df117928be4425b5c439c5e299a1953eb5f919f6`。
- JSON 内部五次统计、形状、源码 HEAD 与 clean 状态已在本机复核一致。
- JSON 没有单独的退出码或完整日志路径字段；随附 PDF 手册记录端到端兼容推理 `exit_code=0`。
- 基准脚本及控制台完整日志尚未同步到本地工作区，因此当前只能核对记录的脚本摘要，不能在本机重算该脚本的 SHA-256。
- NVIDIA 官方原始入口仍需单独执行并保存结果；兼容入口成功不能替代官方原始入口验收。
