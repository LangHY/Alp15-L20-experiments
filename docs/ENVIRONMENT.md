# 环境复现与打包边界

本仓库是既有实验脚本归档。环境配方依据源项目的官方依赖声明与已有 L20 报告整理；本次没有执行新的模型、剪枝、量化、训练或推理任务，也不改变原项目的里程碑验收状态。

## 受限决策单 ENV-EXPORT-20261013-01

- 日期：2026-10-13；决策与配置执行者：环境打包智能体，按主任务受限授权执行；风险：环境配置中高。
- 目标：为用户授权的 `LangHY/Alp15-L20-experiments` 导出提供 uv 初始化和 Docker 配方。只写新仓库的环境文件，不修改源项目、服务器环境或历史结果。
- 来源：源项目 `alpamayo1.5/pyproject.toml`、`alpamayo1.5/uv.lock`，以及 `docs/baselines/ALP15-UV-ENV-001.md`、`ALP15-AWQ-FULL-20261008.md`、`ALP15-PRECISION-HOLDOUT-PREP-REMOTE-EXECUTION-20261010.md`。这些是归档时核对的本地证据，不是本轮远端环境复测。
- NVIDIA 官方源码提交：`36aeb4c5938cbc2eb2aed33b22434773da4ab639`；原始 checkpoint revision：`7aba8293c09993f2e125c6819df05d7fa3e873ea`。源码与 checkpoint 均外置，不将第三方源码声明为本仓库自写代码。
- 文件所有权：本执行者独占 `pyproject.toml`、`.python-version`、`Dockerfile`、`.dockerignore`、`compose.yaml`、`uv.lock`、本说明。README、脚本和来源清单由主任务写入。
- 允许：静态读取、创建上述配置、仅元数据依赖解析、默认轻量工具同步与导入检查。Hydra 所需 `antlr4-python3-runtime==4.9.3` 没有 wheel，允许该小型纯 Python 包的元数据构建；禁止 GPU 包构建。禁止：大 GPU wheel/镜像/模型下载、Docker build/run、GPU任务、修改原 venv、自动提交/推送。
- 停止/升级：依赖冲突无法在授权组边界内解决、元数据解析要求大型 wheel 或 GPU 编译、来源版本不一致时停止并回报主任务。新仓库配置可直接由 Git review 撤销；源项目始终无本执行者写入。
- 验收：配置解析通过、依赖组和平台边界明确；如生成 lock，须用 `uv lock --check` 校验。Docker/FlashAttention/模型端到端留待相应目标机器验证，不能用静态检查替代。

## uv 分组

要求 Python 3.12。使用 uv 0.11.11 编写和检查配置；`.python-version` 选择 3.12 系列，历史 L20 使用 Python 3.12.14。

| 用途 | 命令 | 范围 |
| --- | --- | --- |
| 默认下载工具 | `uv sync --frozen` | 默认 download 组，Hub 1.31.0；不安装 torch |
| CPU 元数据/报表 | `uv sync --frozen --extra analysis` | download + numpy/pandas/pyarrow/matplotlib/safetensors |
| L20 推理工具 | `uv sync --frozen --no-default-groups --extra l20` | 仅 Linux x86_64，torch 2.8.0 cu128 与 Transformers 4.57.1 |
| 历史 AWQ 工具 | `uv sync --frozen --no-default-groups --extra l20 --extra awq` | 上组加 ModelOpt 0.39.0；安装不授权启动实验 |

Transformers 4.57.1 要求 `huggingface-hub>=0.34.0,<1.0`，所以 GPU 组固定官方 uv.lock 中的 Hub 0.36.0，下载组固定已有下载包的 Hub 1.31.0。uv conflicts 会拒绝同时启用两个组；GPU 命令必须加 `--no-default-groups`。`awq` 只是增量组，须同时选择 `l20`。切换组会修改新仓库自己的 `.venv`；需要并存时使用 `UV_PROJECT_ENVIRONMENT=.venv-download` / `.venv-l20` 分别同步。

`torch`/`torchvision` 使用显式 PyTorch cu128 索引；其它包使用 PyPI，不把 PyTorch 索引当作所有依赖的来源。Linux GPU 依赖均带 x86_64 条件；Mac 和 Windows 上选择 GPU extra 不会安装 GPU 栈，请勿将命令退出成功当作 GPU 环境具备。

脚本归档不是安装型 Python 包，`tool.uv.package=false`。运行历史脚本时常需 `PYTHONPATH="$PWD/scripts:$PWD/compression:$PWD/external/alpamayo1.5/src"`；脚本所需输入、manifest 与输出参数仍以各脚本 `--help` 和原执行报告为准。不要直接对整个 tests 目录做“所有测试应通过”的推断，其中包含 GPU/模型/历史路径依赖。

## 外部官方源码与 FlashAttention

在有网络的目标机按需获取源码，并核对固定提交；这些命令是用户手动复现步骤，本次未运行：

```sh
git clone https://github.com/NVlabs/alpamayo1.5.git external/alpamayo1.5
git -C external/alpamayo1.5 checkout --detach 36aeb4c5938cbc2eb2aed33b22434773da4ab639
git -C external/alpamayo1.5 rev-parse HEAD
```

官方源码不纳入默认依赖解析，以免其 pyproject 自动触发 FlashAttention 构建或替换现有固定版本。通过上面的 PYTHONPATH 使用源码；该方式仍需核对与历史实验的适配差异，不能直接宣称完成官方入口基线。

FlashAttention 不在 `uv.lock`：它的构建依赖已经安装的匹配 torch/CUDA，不能在 Mac 或普通 CPU CI 上自动构建。官方入口及部分历史脚本使用 `flash_attention_2`；仅安装 l20 extra 尚不具备完整模型运行环境。不能擅自切为 SDPA 并称原后端复现。

在 Linux x86_64 CUDA 12.8 开发环境完成 l20 同步后，用户可显式安装：

```sh
MAX_JOBS=4 uv pip install --python .venv/bin/python flash-attn==2.8.3 --no-build-isolation
uv pip check --python .venv/bin/python
```

这是锁外安装步骤。记录所得 wheel SHA256、nvcc/torch/Python 版本与构建日志，再做目标 GPU 导入/小型数值检查。后续 `uv sync` 默认会删除未锁定包；重建时重新执行本步骤，或仅在确认所选组一致后用 `uv sync --inexact ...` 保留锁外包。ModelOpt CUDA extension 的加载与历史 legacy quant 后端也需单独验收，安装 Ninja 不代表历史结果可直接替换。

## Docker（待目标机验证）

Docker 默认 target 是 CPU；本仓库锁文件覆盖 Linux x86_64，CPU 与 L20 容器均按 `linux/amd64` 构建和运行。在 Mac ARM 上需显式指定该平台，Compose 两个服务也已固定 amd64；Mac 本机直接使用 uv 的下载和分析工具仍受支持。CPU 和 L20 构建都会联网获取镜像与依赖。L20 CUDA image 与 torch wheel 为 GB 级，本次不执行构建/拉取。Dockerfile 使用固定版本标签，未把动态标签伪装为已冻结 digest；首次实际构建须保存镜像 RepoDigest、构建日志和 `uv pip freeze`。

```sh
docker build --platform linux/amd64 --target cpu -t alp15-tools:cpu .
docker run --rm --platform linux/amd64 -v "$PWD:/workspace:ro" alp15-tools:cpu python --version

# Linux x86_64 NVIDIA 主机，已安装 NVIDIA Container Toolkit。
docker build --platform linux/amd64 --target l20 -t alp15-tools:l20 .
# 需要完整 flash2 入口时再选择安装，可能耗时并占用大量 RAM/磁盘。
docker build --platform linux/amd64 --target l20 \
  --build-arg INSTALL_FLASH_ATTN=1 --build-arg MAX_JOBS=4 \
  -t alp15-tools:l20-flash2 .
docker run --rm --gpus all -v "$PWD:/workspace:ro" alp15-tools:l20-flash2 \
  python -c 'import torch, flash_attn; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), flash_attn.__version__)'
```

Compose 提供 `cpu` / `l20` profiles。默认命令只打印 Python 版本，不自动运行下载、训练或推理。模型、数据、官方源码与结果不写入镜像；按运行脚本需要显式挂载既有目录，官方源码可放新仓库 `external/alpamayo1.5` 后通过只读工作区挂载进入容器。输入目录只读、输出写 `/outputs`；不要把 token 写进 Docker ARG、镜像层、仓库或 compose 文件。

Docker 中 venv 位于 `/opt/venv`，工作区挂载不会遮住依赖。`.dockerignore` 排除 `.venv*`（含 `.venv-l20`、`.venv-download`）及嵌套环境、常见凭据/token/SSH 私钥文件名，避免复制进镜像层。CPU target 支持 amd64 容器中的普通工具链；Windows 下载脚本仍使用其 PowerShell 环境。L20 配方不支持 Orin/Jetson ARM64，不能用 Mac Docker 或开发机结果替代实机验证。

## 本次执行回报

本次仅配置写入与轻量验证；实际命令与结果由下方验收记录补充。没有修改源仓库或远端 venv；没有模型/checkpoint 输入，没有 GPU 输出，没有 Docker 镜像拉取或运行。原 L20 报告的 torch 2.8.0+cu128、CUDA12.8、Transformers4.57.1、PhysicalAI0.2.0、safetensors0.7.0 与 ModelOpt0.39.0 用作版本依据，未在本轮远端重新测量。

### 2026-10-13 本地验收

准备与执行时间约 14:47–15:10 CST；实际准备机 macOS 27.2 / Darwin arm64，uv 0.11.11，本次 `.venv` Python 3.12.10。不是 L20/Jetson 环境；CUDA/驱动/GPU 本轮未测量。锁文件固定的是归档时解析出的完整依赖树，核心实验版本沿用记录，未冻结过的传递依赖按当前索引解析；不能直接称它与历史 L20 全部已安装包逐项相同。

变更文件为决策单列出的配置与本说明，另加 `.github/workflows/ci.yml`。新 `.venv` 为本地验证产物，不入 Git。CI 仅同步默认工具、检查依赖与 Hub 导入、编译 `scripts compression tests`，不自动执行模型测试或历史实验。

| 实际命令或核验 | 退出码 | 结果 |
| --- | ---: | --- |
| `python3 -c 'import tomllib; ...'` | 0 | pyproject 与 lock 可解析，Python 要求为 3.12 系列 |
| `uv lock --offline --no-build --no-python-downloads` | 1 | 本机缓存缺 matplotlib 等索引，安全停止，无下载 |
| `UV_HTTP_TIMEOUT=15 UV_HTTP_RETRIES=1 uv lock --no-build --no-python-downloads` | 2 | PyPI numpy 请求超时，未生成 lock |
| 加 `--system-certs` 且较低并发的元数据重试 | 2 | PyPI matplotlib TLS handshake EOF，未生成 lock |
| 命令级既有代理加 `uv lock --no-build --no-python-downloads` | 1 | 网络通过，但 antlr4 4.9.* 无 wheel，被 no-build 正确拒绝 |
| 命令级代理加 `uv lock --no-build-package torch --no-build-package torchvision --no-build-package flash-attn --no-python-downloads` | 0 | 2.49s，解析 91 个包；只允许小型纯 Python 元数据构建 |
| `uv lock --check --offline --no-python-downloads` | 0 | 已冻结的 91 个包与配置一致 |
| 全新临时空缓存下相同 offline check | 0 | 18ms，不依赖已有索引缓存进行 lock 一致性检查 |
| `uv sync --frozen --dry-run --offline --no-python-downloads` | 0 | 默认只计划 15 个包，均为下载工具依赖，无 torch |
| 命令级代理加 `uv sync --frozen --no-build --no-python-downloads` | 0 | 实际安装 15 个轻量包；唯一显式下载进度为 hf-xet 3.4MiB，无 CUDA wheel |
| `uv pip check` | 0 | 15 个默认包兼容 |
| `uv run --frozen --offline python -c '...'` | 0 | Hub=1.31.0，`find_spec("torch") is None`，DEFAULT_NO_TORCH_PASS |
| `uv run --frozen --offline python -m compileall -q scripts compression tests` | 0 | 当时归档的 137 个文件集合语法编译通过，不执行脚本 |
| `uv sync --frozen --no-default-groups --extra l20 --extra awq --dry-run --python-platform x86_64-unknown-linux-gnu --offline --no-python-downloads` | 0 | Linux 目标仅计划安装 75 个包，torch 2.8.0+cu128、torchvision 0.23.0+cu128、Triton 3.4.0；未安装 |
| `uv sync --frozen --extra l20 --dry-run --offline --no-python-downloads` | 2 | download 默认组与 l20 的冲突按设计拒绝 |
| `docker compose config --quiet` | 0 | Compose 静态解析通过，不连接或启动容器 |

元数据联网前已读取 web-access skill；浏览器前置检查 Node22.23.0 通过、Chrome 未连接。本轮公共索引使用静态读取，不操作登录态或用户浏览器。PyTorch 页面含 `data-dist-info-metadata`，核验的 Linux CPython3.12 torch wheel SHA 为 `4354fc05bb79b208d6995a04ca1ceef6a9547b1c4334435574353d381c55087c`、metadata SHA 为 `aad6db3dd1aeb1e9a59ab1143a372adba214646e6e8eb91c9e594d25b34c858c`；本轮没有下载该 wheel。代理仅为命令环境变量 `HTTP_PROXY`/`HTTPS_PROXY` 选择本机现有地址，未写仓库或改系统设置。

首次配置验收时 SHA256（后续审查修订见下节）：

```text
pyproject.toml ca0adef97a52021549bf389044582ddaad3c7a46293a41fc50ba83d9185aa8be
uv.lock        7adb56e6cdfcc19724ffa0bf0c69f007e8a9bc1c6aa5e8f0fe2c099788c87165
Dockerfile     dd1a3110d2b9fd32215a3bb0b320db07ac7efbb73ae29b23f01225de86f1c790
compose.yaml   07ca50a2a2a2679e717560477bcf636e7c0e2e412ba354b60c8510335c84a687
```

完整终端命令与输出留在本任务工具记录；本表保留实际命令、退出码与关键输出。本次环境配置和默认工具验收通过；Docker 构建、Linux GPU 同步/FlashAttention 构建、官方源码导入、模型端到端、Windows PowerShell 与 Orin 实机验收仍未执行，交目标平台独立验证。

### 最终配置审查修订

按主任务的最终审查，`.dockerignore` 由精确 `.venv` 改为根目录和嵌套 `.venv*`，并补充 `credentials*`、`token`、`stored_tokens`、`id_rsa*`、`id_ed25519*` 排除。CPU Compose 服务与手动 Docker 示例明确 `linux/amd64`，匹配现有 Linux lock 平台范围；Mac 本机 uv 使用方式不变。不扩大锁文件平台、不重新解析依赖、不修改 GPU Dockerfile 或官方源码。此修订仅做 Compose 静态解析及文件核对，没有联网、容器构建/启动或包下载。

`docker compose --profile '*' config --quiet` 及 JSON 平台断言均退出 0，两个服务的平台均为 `linux/amd64`。修订后 `compose.yaml` SHA256 为 `562a66cb58b5b10c642f765559be4f46a756b80c246c9f23f4a0020869fe05f9`，`.dockerignore` 为 `e88e079216ef612c1f04a9026c3d1251d3dacd966deb9ab4895aa26a8ecdb662`；`uv.lock` 与 `Dockerfile` SHA256 均与首次验收相同。
