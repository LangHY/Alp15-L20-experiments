# Alp15 L20 Experiments

Alpamayo 1.5 离线实验脚本归档：从早期 FFN/MLP 剪枝、蒸馏与恢复，到低秩敏感性、
RTN/AWQ、混合内核、KV/少步采样、精度审计，以及数据下载和 Mac→L20 流水线。

本仓库只整理**本机已有源码和 2026-10-04 交付包**，没有重新获取 L20 上可能存在的更新。
998 条来源记录按 SHA-256 去重为 **223 个历史源码版本**；当前入口/测试/辅助程序145个。
原工作区和原始交付包不修改。权重、原始传感器数据、PT缓存、预测结果和实验日志不上传。

## 1. uv 初始化

需要 uv 0.11.11 和 Python 3.12；`.python-version` 与 `uv.lock` 固定版本。

```bash
git clone https://github.com/LangHY/Alp15-L20-experiments.git
cd Alp15-L20-experiments
uv sync --frozen
uv run --frozen python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

默认环境只用于下载工具，不安装 GPU/PyTorch。额外 CPU 数据分析依赖：

```bash
uv sync --frozen --extra analysis
```

**下载与推理必须使用不同环境。** 下载器固定 Hub 1.31.0，Transformers 4.57.1 则要求
Hub <1.0；配置明确禁止同时选择，不能用一个pip升级命令覆盖原推理环境。

### Linux x86_64 / L20 CUDA 12.8 环境

```bash
UV_PROJECT_ENVIRONMENT=.venv-l20 uv sync --frozen --no-default-groups --extra l20 --extra awq
UV_PROJECT_ENVIRONMENT=.venv-l20 uv run --no-sync python -c \
  "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

GPU依赖使用显式 PyTorch cu128 索引：torch2.8.0、torchvision0.23.0、Transformers4.57.1、
physical-ai-av0.2.0、safetensors0.7.0；AWQ额外使用ModelOpt0.39.0。
这是一份冻结环境配方，不代表已经在新环境重跑全部模型实验。

官方 Alpamayo 源码和 FlashAttention 必须单独准备：

```bash
git clone https://github.com/NVlabs/alpamayo1.5.git external/alpamayo1.5
git -C external/alpamayo1.5 checkout --detach 36aeb4c5938cbc2eb2aed33b22434773da4ab639
MAX_JOBS=4 uv pip install --python .venv-l20/bin/python flash-attn==2.8.3 --no-build-isolation
export PYTHONPATH="$PWD/scripts:$PWD/compression:$PWD/external/alpamayo1.5/src"
```

FlashAttention需要CUDA开发工具链及编译资源；命令由用户执行。后续使用`uv run --no-sync`
避免默认同步移除手动安装的FlashAttention。不能把改为SDPA当成同口径结果。
ARM64/Jetson不在本环境的已支持GPU平台范围内。

## 2. Docker

提供CPU和L20两种build target，默认不启动模型、不下载权重、不做数据实验。
构建需要下载较大的基础镜像/GPU依赖，以下命令仅由用户显式执行：

```bash
docker build --platform linux/amd64 --target cpu -t alp15-tools .
docker run --rm --platform linux/amd64 alp15-tools

docker build --platform linux/amd64 --target l20 -t alp15-l20 .
docker run --rm --gpus all --shm-size=8g alp15-l20
```

GPU宿主须已安装驱动和NVIDIA Container Toolkit。
CPU容器同样使用已锁定的linux/amd64平台；Mac ARM可用Docker的amd64仿真，
这与直接在Mac运行默认uv下载环境不同。
需要FlashAttention时显式添加`--build-arg INSTALL_FLASH_ATTN=1`，默认不编译它。
Compose入口：`docker compose --profile cpu run --rm cpu`或`--profile l20 run --rm l20`。
模型、数据和旧实验缓存需另行挂载到脚本要求的位置，不能烘焙认证token或权重进镜像。
Dockerfile/Compose通过静态检查，不宣称GPU镜像或Orin实机已构建运行。

## 3. 目录与实验阶段

|目录|用途|
|---|---|
|`compression/`|早期剪枝、蒸馏、参数分配、恢复、资源审计及启动脚本|
|`scripts/`|近期低秩、量化、KV、混合内核、精度及下载工具|
|`tests/`|当前脚本的测试；部分GPU/历史输入测试需另行准备环境|
|`dashboard/`、`handover/`|早期监控和交付验证辅助程序|
|`archive/sha256/`|所有去重后的原字节历史源码，不默认为可运行入口|
|`archive/SOURCE_INDEX.json`|所有原始位置、版本SHA与当前选择的对应关系|
|`archive/framework_sources/`|FlashDrive/LightVLA/vLLM研究快照来源记录|
|`deliverables/alp15_windows_949/`|Windows下载入口及固定的小型校验清单|
|`docs/ENVIRONMENT.md`|环境设计、验证记录与限制|

主要阶段：

1. `compression/run_mlp_experiment.py`：剪枝基线；`run_interface_recovery.py`等恢复/蒸馏。
2. `scripts/alp15_whitened_*`、`alp15_lowrank_pilot.py`：低秩与FFN敏感性测量。
3. `alp15_quant_route_pilot.py`、`alp15_awq_full.py`：RTN与真实驾驶数据AWQ。
4. `alp15_awq_kernel.py`、`alp15_awq_hybrid.py`、`alp15_kv_steps_probe.py`：内核/KV/少步采样。
5. `audit_alp15_precision.py`、`prepare_alp15_precision_holdout.py`：审计与候选测试数据准备。
6. `download_alp15_mac_only.py`和`download_alp15_mac_l20_pipeline.py`：仅下载与下载/上传流水线。

多数GPU入口需要`--root`和`--output`，历史模型/配置/数据/teacher-cache/overlay/packed权重
并不随Git仓库分发。旧shell启动器保留历史绝对路径，不建议直接运行；先读源码/`--help`，
用uv环境显式调用对应Python入口。不能只凭`uv sync`成功就声称历史实验已复现。

两个严格来源检查文件（`run_mlp_experiment.py`、`mlp_pruning.py`）选用后续full20证据的
原字节版本；交付包中末尾换行不同的版本也完整保留在archive，不擅自归一。

## 4. 数据下载

在Mac下载固定949候选clip约54.38GB；默认仅展示计划，无网络或下载：

```bash
uv run --frozen python scripts/download_alp15_mac_only.py
uv run --frozen python scripts/download_alp15_mac_only.py \
  --download --confirm-bytes 54382445102 --output "$PWD/datasets/alp15-small-test-949"
```

流水线入口：

```bash
uv run --frozen python scripts/download_alp15_mac_l20_pipeline.py \
  --run --confirm-bytes 54382445102 \
  --local-dir "$PWD/datasets/alp15-small-test-949" \
  --host user@your-l20-host --port 22 --expected-hostname your-l20-hostname
```

源版本远端接收器仍保护原服务器用户/独立数据根；迁移到新服务器须先审阅并显式配置这些值，
不能把本例hostname/用户名替换当作已经完成移植。该流水线为Mac/Linux入口，Windows另用PS1。
SSH使用严格主机密钥检查；认证在用户终端/官方HF工具中完成，不把密码或token写入脚本。
下载、上传都支持保留缓存/partial、有界重试、已有文件真实SHA校验与冲突拒绝。
949只是候选数据，不能称独立精度已经验证；完整大池脚本默认仍不下载。

## 5. 验证与研究边界

本仓库提供源码、环境配方和离线测试，不承诺无输入即可重跑所有实验。
不要把兼容离线基线、重复使用开发集、算子计时或mock通过改称官方原样入口、
独立精度、闭环仿真或Orin NX16GB实机验收。失败/未完成的研究版本同样被归档。
FlashDrive/LightVLA/vLLM快照不完整，仅供研究阅读，不加入默认运行依赖。

上游版权和许可证见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
所有凭据、大文件和运行产物由`.gitignore`与`.dockerignore`排除。
