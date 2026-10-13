# 智行·端析 / Alpamayo 1.5：当前进度

更新时间：2026-09-20 10:58 CST。本文是**状态快照**；进行中的下载或构建须以完成后的退出码和验收记录更新，不能按“已完成”理解。

## 当前结论

官方源码、10B checkpoint、冻结测试 clip、Cosmos/Qwen 非权重配置和 Linux/amd64 uv 依赖已在 L20 完成加载与离线兼容推理。2026-09-20 的结构化证据证明 **L20 BF16 离线兼容性端到端基准已经跑通**；五次稳态运行平均 1615.68 ms，峰值 allocated 显存 22126 MiB，minADE 为 0.371870 m。当前仍未完成 NVIDIA 官方原始入口的独立验收；兼容基准不得命名为官方原始入口跑通。没有进行剪枝、蒸馏、量化、ONNX 或 TensorRT 工作。

| 项目 | 当前状态 | 已有证据／边界 |
| --- | --- | --- |
| NVIDIA 官方源码 | 已核对 | `alpamayo1.5/` HEAD=`36aeb4c5938cbc2eb2aed33b22434773da4ab639`，工作树 clean；原始 `test_inference.py` SHA256=`dc69646feed09f92defa00a19ae6f2fc2a37a10a70ec35678f87946fbe6fe8e8`，未改动。 |
| Alpamayo 1.5 10B 权重 | L20 加载成功 | 五个 safetensors 分片、`config.json` 和索引已在 L20 校验并以 BF16 完整加载；实测总参数量 11.0785B，参数存储约 20.64 GiB。 |
| 一个测试 clip | L20 端到端使用成功 | 冻结 `clip_id=030c760c-ae38-49aa-9ad8-f5650a545d26`、`t0_us=5100000`、`chunk_id=3119`；四路摄像头各四帧成功进入模型，输入张量为 `(4, 4, 3, 1080, 1920)`。文件验收见 [ALP15-CLIP-ACCEPT-001.md](ALP15-CLIP-ACCEPT-001.md)。 |
| 离线兼容推理代码 | 已实现、本机测试通过 | [脚本](../../scripts/test_alpamayo15_offline_compat.py)显式使用本地权重、clip、Cosmos/Qwen 配置，模型配置仅在内存中重映射，并阻止 Python socket 外联；项目 7 项本机测试通过。它不等于 NVIDIA 官方原始入口跑通，详见 [ALP15-OFFLINE-INFER-001.md](ALP15-OFFLINE-INFER-001.md)。 |
| Qwen 处理器配置 | 九项已验收并封装 | 冻结 revision `89644892e4d85e24eaac8bacfd4f463576704203` 的九个非权重文件与 HF blob 身份已校验，并以真实文件写入配置交付物；全局缓存保留。 |
| Cosmos 处理器／VLM 配置 | 九项已验收并封装 | ModelScope 传输的九个非权重文件已按大小与 Git blob SHA-1 匹配冻结 Hugging Face revision `a9fae2cf89dc64db96b12860417f0eb403013bb9`，且未发现权重；原 staging 已移入废纸篓。配置交付目录、归档与 SHA-256 见 [ALP15-CONFIG-DELIVERY-001.md](ALP15-CONFIG-DELIVERY-001.md)。**不需要 Cosmos 权重**。 |
| Linux/amd64 uv 环境 | L20 验证成功 | L20 实际环境为 Python 3.12.14、PyTorch 2.8.0+cu128、FlashAttention 2.8.3、Transformers 4.57.1、`physical-ai-av==0.2.0`；CUDA 可用且 BF16 支持。离线依赖构建记录见 [ALP15-UV-ENV-001.md](ALP15-UV-ENV-001.md)。 |
| 上传归档 | 已生成并本机校验 | 源码、模型、单 clip、配置和 Linux/amd64 uv 环境五份归档均已生成；每份带相邻 SHA-256，并汇总到 `alpamayo15-upload-manifest-20260919.sha256`。详情见 [ALP15-UPLOAD-ARCHIVES-001.md](ALP15-UPLOAD-ARCHIVES-001.md)。 |
| L20 兼容推理 | 已跑通 | 五次稳态平均 1615.68 ms、P95 1619.31 ms、标准差 2.74 ms；CoC、形状和 minADE 五次一致。结构化结果见 [测试结果](../../results/baselines/l20-bf16/2026-09-20/README.md)，操作见 [运行手册](../runbooks/ALP15-L20-OFFLINE-COMPAT.md)。 |
| NVIDIA 官方原始入口 | 未验收 | 当前成功入口是离线兼容脚本与自定义基准脚本；尚无官方原始入口的独立退出码、完整日志与结果记录。 |

## Linux/amd64 环境准备结果

原有 `ubuntu:22.04` 实际是 `linux/arm64`，没有用于目标 L20 依赖。另一个本地 `alpamayo:latest`（Ubuntu 22.04 / amd64 / CUDA 12.4.1）缺少 `nvcc`，其中 PyTorch 导入报 `libcusparseLt.so.0` 缺失，也未用作最终基线。

最终使用 NVIDIA 官方 `nvidia/cuda:12.8.1-devel-ubuntu22.04` 的 Linux/amd64 镜像，digest=`sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97`。在 `--network none` 新容器中，以导出的 uv cache 和 uv 管理 Python 完成全新环境同步；后续禁网导入验证退出码 0。首次裸 `ldd` 因未加入 PyTorch 的 `torch/lib` 以退出码 3 停止；保留失败日志后按实际运行时库路径复验，结果为 `ldd_missing=none`。验证 venv 的 Python 是指向 `/offline/uv-python` 的绝对链接，因此仅作为证据；L20 应从交付缓存重新创建环境。

## 下一步与验收门槛

1. 从 L20 同步 `scripts/benchmark_alpamayo15_offline.py` 和完整控制台日志，核对记录的脚本 SHA-256=`861ff3d8a60c420681fd5eeb069a405d5159bf4dc4a6743fe84a3855937593b0`，并补齐独立退出码证据。
2. 按 [AGENTS.md](../../AGENTS.md) 单独运行 NVIDIA 官方原始入口，保存实际命令、完整日志、退出码、输出和资源观测。若官方入口受阻，记录可复现阻塞；**不能用已经成功的兼容入口替代官方原始入口验收**。

本文件不包含 token、密码、SSH 地址、GPU UUID 或带签名下载 URL；完整过程输出以当前任务工具调用记录及后续单独的构建日志为准。
