# ALP15-UV-ENV-001：Linux/amd64 uv 离线环境决策与执行回报

## 决策单

- 日期：2026-09-19；决策者：GPT-5.6 Sol high；风险等级：高。
- 对应里程碑：跑通 NVIDIA 官方原始 Alpamayo 1.5 模型测试并建立可复现基线。
- 目标：在本机 Docker 中准备能传给无外网 L20、并可从冻结 `uv.lock` 重建环境的 Linux/amd64 uv 缓存与 Python 运行时。
- 明确不做：不下载或加载模型权重，不运行推理，不修改 NVIDIA 官方源码、`pyproject.toml` 或 `uv.lock`，不实施剪枝、蒸馏、量化、ONNX 或 TensorRT。

事实依据：官方源码 HEAD=`36aeb4c5938cbc2eb2aed33b22434773da4ab639`；`uv.lock` SHA256=`6abf4ccba882e0b7849880ccaef794cbeecc36a43a2a12007e6b30632ae4e8b1`；项目要求 Python 3.12、torch 2.8.0、transformers 4.57.1、physical-ai-av 0.2.0 和 flash-attn 2.8.3。原有 `ubuntu:22.04` 镜像是 arm64，不能作为 L20 的 amd64 依赖来源；原有 `alpamayo:latest` 缺 `nvcc` 且 torch 因 `libcusparseLt.so.0` 缺失无法导入。

允许使用经官方来源核对的 `nvidia/cuda:12.8.1-devel-ubuntu22.04` amd64 镜像；先在联网环境填充缓存并构建 flash-attn，再导出 uv、uv 管理 Python 和缓存，在全新 `--network none` 容器中执行 `uv sync --frozen --offline`。验收要求为同步退出码 0、核心包版本匹配、torch/flash-attn/Alpamayo 导入成功、动态库检查无缺失、再次离线 dry-run 无变更。停止条件为架构或哈希不符、锁文件变化、禁网同步请求联网、核心导入失败或动态库缺失。

## 执行回报

- 执行者与复核者：GPT-5.6 Sol high；执行日期：2026-09-19。
- 准备机：macOS / arm64；Docker Desktop 29.6.1，Docker 后端 Linux/arm64，通过 `--platform linux/amd64` 执行目标容器。Mac Docker 未提供 NVIDIA GPU。
- 镜像：`nvidia/cuda:12.8.1-devel-ubuntu22.04`，amd64 digest=`sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97`；容器内 Ubuntu 22.04.5、CUDA 12.8.1、nvcc build `cuda_12.8.r12.8/compiler.35583870_0`。
- 交付目录：`/Users/USER/Downloads/alpamayo15-linux-amd64-uv-20260919`；含 uv 0.12.15（SHA256=`5d59bc45431db192c0a49a01c517041c3bd8778adb3fbc5195aaa025eec19e23`）、CPython 3.12.14、uv cache、验证 venv、冻结配置副本、日志和 `validation.json`。
- 上传归档：`/Users/USER/Downloads/alpamayo15-linux-amd64-uv-20260919.tar.zst`，大小 `4,032,080,617` bytes，SHA256=`5fa01ec33ecf1e019b8235018040e086ba6d566d2eed3e23d24686eb62c12552`；相邻 `.sha256` 文件可用于接收端校验。`zstd -t` 通过，显式 `zstd -dc | tar -tf -` 完整列表检查退出码 0。
- 全新禁网同步：`docker run --rm --network none --platform linux/amd64 ... uv sync --frozen --offline`，退出码 0，安装 107 个包。关键版本：torch 2.8.0+cu128、torchvision 0.23.0、transformers 4.57.1、physical-ai-av 0.2.0、flash-attn 2.8.3、accelerate 1.12.0。
- flash-attn 本地构建 wheel SHA256=`f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb`。这与上游 2.8.3、PyTorch 2.8、CPython 3.12、Linux x86_64、CXX11 ABI TRUE 的发布 wheel哈希一致。
- 导入验证：torch、flash-attn、Alpamayo 类均成功；torch build CUDA=12.8。再次 `uv sync --frozen --offline --dry-run` 检查 107 包且无变更。
- 失败尝试：第一次裸 `ldd flash_attn_2_cuda...so` 未将 venv 的 `torch/lib` 加入搜索路径，报告五个 torch 动态库缺失并退出 3；Python 实际导入已成功。随后设置 `LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64`，`ldd_missing=none`，核心导入和 dry-run 退出 0。两份日志均保留。
- 归档检查偏差：macOS `bsdtar --use-compress-program zstd -tf` 报子进程 broken pipe；归档自身此前已通过 `zstd -t`。改用等价的显式流 `set -o pipefail; zstd -dc ... | tar -tf - >/dev/null` 后完整检查退出码 0。

## 验收结论与限制

本机已达到“Linux/amd64 uv 离线依赖交付物可在匹配容器中从零重建”的验收标准。验证 venv 含 `/offline/uv-python` 绝对链接，不能作为任意路径下的原生可搬运 venv；L20 必须按交付 README 从缓存重新创建 `.venv`。

本次没有 GPU，因此没有验证 `torch.cuda.is_available()`、CUDA kernel、flash-attn GPU执行、模型权重加载或端到端推理。只有 L20 断网重建、GPU 导入检查及官方入口实测完成后，才能进入对应模型基线验收。
