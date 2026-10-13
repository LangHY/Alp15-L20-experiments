# ALP15-OFFLINE-ASSET-001：Alpamayo 1.5 离线资产准备决策单

## 决策信息

- 决策编号：`ALP15-OFFLINE-ASSET-001`
- 日期：2026-09-16
- 决策者：GPT-5.6 Sol high
- 执行者：GPT-5.6 Terra high
- 对应里程碑：NVIDIA 官方原始 Alpamayo 1.5 10B 测试与可复现基线。
- 风险等级：高。涉及官方来源、checkpoint、数据集、依赖环境和官方推理链路；必须遵循 Sol 决策、Terra 执行、Sol 验收。

目标是在有外网的准备机上生成一个可搬运到无外网 L20 的离线包。此决策不授权安装、推理、量化、剪枝、ONNX 或 TensorRT；脚本默认 dry-run，只有显式 `--execute` 才下载或写入资产。

## 冻结事实与三类资产

| 类别 | 官方来源与冻结版本 | 离线包位置 | 本决策要求 |
| --- | --- | --- | --- |
| 官方源码与推理代码 | `https://github.com/NVlabs/alpamayo1.5.git` @ `36aeb4c5938cbc2eb2aed33b22434773da4ab639` | `source/alpamayo1.5/` | detached checkout；官方入口完整路径 `src/alpamayo1_5/test_inference.py`，SHA256 必须为 `dc69646feed09f92defa00a19ae6f2fc2a37a10a70ec35678f87946fbe6fe8e8`。 |
| 10B checkpoint | `nvidia/Alpamayo-1.5-10B` @ `7aba8293c09993f2e125c6819df05d7fa3e873ea` | `models/Alpamayo-1.5-10B/` | 仅下载 `config.json`、`model.safetensors.index.json` 与五个指定 safetensors；两份 JSON 和每片均需固定 size+SHA256。 |
| 一个官方数据 clip | `nvidia/PhysicalAI-Autonomous-Vehicles` @ `33f9bf447ed3bcb7d545ce13f4226f824214fafb`；`physical_ai_av==0.2.0` 源码提交 `59d578b53ef186ffc3f1de1d1b2598717a982d64` | `datasets/physical-ai-av/` | 只通过官方 Python API 预取 `clip_id=030c760c-ae38-49aa-9ad8-f5650a545d26`；`t0_us=5100000`；EGOMOTION 和四个冻结 CAMERA 属性。 |

额外的非权重配置依赖为 `nvidia/Cosmos-Reason2-8B` @ `a9fae2cf89dc64db96b12860417f0eb403013bb9` 与 `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203`，分别置于 `dependencies/configs/`。**不得下载 Cosmos 或 Qwen 权重。**

Alpamayo 模型白名单严格限定为 `config.json`、`model.safetensors.index.json` 与五个指定 safetensors；不得以 processor/tokenizer 或额外 safetensors 替代。Cosmos/Qwen 两项依赖才允许下载以下明确的非权重配置：`config.json`、`generation_config.json`、`preprocessor_config.json`、`video_preprocessor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.json`、`merges.txt`、`vocab.json`。ModelScope 模型传输另允许 `LICENSE`、`README.md`，但适用后文更严格的 R1 白名单。任何允许文件缺失均应记录为 `MISSING` 并停止验收，不能用额外权重或未冻结文件替代。

五个模型分片的固定 size/SHA256 依次为：`4928204944/537259bb56815f9dfdcb028d5606e84dff5ab0c8e0e559782d63d91b6671d15c`、`4915963032/18841e5049b7836c16c034123ae6d40b2ff0414f9f77008bc7ca72490302a47b`、`4983071160/e8953b27fe827a60604a006b07ab1216fe766375f9926d1c3bd5ecee84ecef78`、`4980341192/604f0c0f19986f363a23a1e0b5736eedadd231ca285beab28d5db968a1d37602`、`2349614196/9d889c09634e5a21b4c957957ede2cdab4418c268ee1d870c1f942d04242adf6`。`config.json` 必须为 `3058/824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e`，index 必须为 `104778/b899e51816e15a96edc7cd57a01cd36c5e39fe1bcbf1537c4ac914f0cad43a0d`。

单 clip 脚本在导入包前关闭字节码写入，并以 `importlib.metadata.version('physical-ai-av')` 严格核验 `0.2.0`。它只能从实例化后的 `interface.features` 精确读取 `LABELS.EGOMOTION`、`CAMERA.CAMERA_CROSS_LEFT_120FOV`、`CAMERA.CAMERA_FRONT_WIDE_120FOV`、`CAMERA.CAMERA_CROSS_RIGHT_120FOV`、`CAMERA.CAMERA_FRONT_TELE_30FOV`，不得枚举整个 CAMERA 集合。执行时必须以 `interface.get_clip_chunk(clip_id)` 得到非负整数 `chunk_id`（允许 `0`），再以 `features.get_chunk_feature_filename(chunk_id, feature)` 固定五条文件路径。`dataset-manifest.json` 顶层记录 `chunk_id`、库 version 与 commit；五项 feature 各自记录精确 feature/path/size/SHA256/chunk，metadata 独立记录 `features.csv`、`clip_index.parquet`、`metadata/feature_presence.parquet`、`metadata/data_collection.parquet` 的 path/size/SHA256。根据后续决策 `ALP15-CLIP-RESUME-001`，实际 manifest 路径统一加固定前缀 `datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/<冻结 revision>/`，交付保留对应 `blobs/` 与安全的相对链接；clip 下载缓存位于 bundle 外并在失败时保留。

## 来源与认证边界

`prepare_alpamayo15_offline_bundle.sh` 将模型与数据集来源分别设为显式参数，默认 `--model-source hf --dataset-source hf`。脚本不接收 token 参数、不回显 token，并仅依赖准备机已有的认证状态。

若 checkpoint 或数据集受 gated 协议限制，只允许在有外网的 **staging 准备机** 通过其交互式客户端登录并确认已获访问授权；token 绝不作为脚本参数、日志、manifest 或离线包内容。L20 只接收经验证的离线资产，不需要、也不得为本流程登录 Hugging Face 或 ModelScope。

Sol R1 核验后，模型可以使用 `--model-source modelscope` 从唯一的传输源 `nv-community/Alpamayo-1.5-10B` @ `d26524f2d3bd005149d7f23e2af7c3ea10123df3` 获取。该分支只通过 ModelScope SDK `snapshot_download` 接收 `LICENSE`、`README.md`、`config.json`、`model.safetensors.index.json` 和五个指定分片；拒绝 `configuration.json` 与 `.gitattributes`。其中 `config.json` 必须为 3058 bytes、SHA256 `824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e`，index 必须为 104778 bytes、SHA256 `b899e51816e15a96edc7cd57a01cd36c5e39fe1bcbf1537c4ac914f0cad43a0d`；`LICENSE` 与 `README.md` 的 SHA256 分别为 `2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df` 和 `ab3a10f3abd36624b55d3f511ca6ac957a2a18a922ca6a27c4b0844b5f925240`。

HF 与 ModelScope 都必须先下载到 bundle 外 `mktemp -d` staging。SDK/客户端产生的 `.cache` 等隐藏元数据只可留在 staging；通过完整 payload 校验后，HF 仅复制七个核心文件，ModelScope 仅复制九个固定文件到此前不存在的最终模型目录，随后再次校验。staging 由退出 trap 清理，不能进入离线包或全文件 manifest。

Cosmos/Qwen 配置也先进入各自包外 staging；确认九个冻结文件齐全后才逐个复制到 bundle，下载客户端生成的 `.cache`、锁文件及隐藏元数据不得进入交付物。

ModelScope 是传输来源，不改变权威来源。模型 manifest 必须同时记录 `authority_source=huggingface:nvidia/Alpamayo-1.5-10B`、`authority_revision=7aba8293c09993f2e125c6819df05d7fa3e873ea` 和实际的 `transport_source`、`transport_revision`。数据集的 `--dataset-source modelscope` 仍立即拒绝：已核验的 `nv-community/PhysicalAI-Autonomous-Vehicles` 仅含 calibration，缺少 metadata、egomotion 和四路 camera chunk，不能替代官方数据集或触发下载。无论传输来源是什么，都必须以本决策冻结 SHA256、revision manifest 和校验结果验收。

## Linux/amd64 依赖边界

可交付物是 **Linux/x86_64（amd64）的 `uv` cache/wheels**，只能通过 `--prepare-linux-amd64-deps --execute` 准备；其输入必须是已冻结源码中的 `uv.lock`。`uv sync --frozen` 为准备命令，运行时 venv 通过 bundle 外 `mktemp -d` 创建，并由退出 trap 清理；它绝不是可搬运交付物，也不会进入全文件 manifest。接收端必须在干净的 Linux/amd64 环境以 `uv sync --frozen --offline` 验证缓存可离线解析。macOS/arm64 会被脚本拒绝，绝不生成可交付 venv。

当前已知准备机约有 152 GiB 可用空间；这只是观测，不构成完整资产或依赖所需空间已满足的结论。Docker daemon 当前未运行，`ubuntu:22.04` 镜像的 OS/架构也未核验，不能据此认定可用作 Linux/amd64 依赖构建环境。脚本会只读报告磁盘、工具和可见 Docker 镜像架构；实际依赖构建前应在确认的 Linux/amd64 环境重新记录。

## 执行顺序与文件所有者

| 顺序 | 执行内容 | 允许命令/边界 | 文件所有者 |
| --- | --- | --- | --- |
| 1 | 在有外网准备机做 dry-run 预检 | `bash scripts/prepare_alpamayo15_offline_bundle.sh ...`；不写入 | Sol high |
| 2 | 获取源码、10B checkpoint、配置依赖 | 显式 `--execute`；源码必须 detached；模型先下载至 bundle 外 staging，完整校验后仅复制固定白名单文件进 bundle | Sol high |
| 3 | 获取单 clip | 显式 `--with-clip --execute`；交付目录必须为空；持久缓存位于 bundle 外；调用 `get_clip_chunk`、`get_chunk_feature_filename` 与按五条冻结路径逐项 `download_file`（见 `ALP15-CLIP-XET-001`）；允许官方 chunk ZIP，不制作 `.pt` 或自定义 ZIP | Sol high |
| 4 | 准备依赖（如获执行许可） | 仅确认的 Linux/amd64，`uv sync --frozen`；交付 uv cache/wheels，不把构建 venv 视为可搬运产物 | Sol high |
| 5 | 只读验收 | `bash scripts/verify_alpamayo15_offline_bundle.sh --bundle-root ...`；不联网、不安装、不运行模型 | Sol high |
| 6 | 上传至 L20 并尝试官方入口 | 另行记录传输 hash、L20 环境与完整日志 | Sol high |

预期 bundle 结构：

```text
alpamayo15-offline-bundle/
├── source/alpamayo1.5/
├── models/Alpamayo-1.5-10B/
├── datasets/physical-ai-av/
├── dependencies/configs/{Cosmos-Reason2-8B,Qwen3-VL-2B-Instruct}/
├── dependencies/linux-amd64/uv-cache/
└── manifests/
```

## 完全离线官方入口的阻塞与命名规则

现阶段**未验证**冻结官方 `test_inference.py` 在完全断网的 L20 上不会触发任何网络访问，也未证明离线资产和 Linux/amd64 依赖完整。因此，在 L20 实际以 `--offline` 等官方支持方式完成原始入口、官方样例、权重加载、预处理和端到端输出前，不得称为“完全离线官方入口跑通”。

若官方原始入口因断网、缺失的官方受限资产、依赖解析或上游 API 行为失败，必须记录完整命令、退出码、环境、输入与原始日志，并命名为：`官方原始入口完全离线阻塞`。这不是模型成功，也不是失败归因的终点，须升级 Sol。

只有在该阻塞被 Sol 事前批准后，才可做最小兼容性测试；其名称必须包含 `兼容性测试`（例如 `ALP15 离线兼容性测试`），并明确列出与官方入口的差异。兼容性测试不得被表述为“官方测试跑通”或“完全离线官方入口跑通”。

## 验收、日志与停止条件

`verify_alpamayo15_offline_bundle.sh` 必须只读检查：源码 HEAD/status 与 `src/alpamayo1_5/test_inference.py` 哈希；模型 snapshot revision、五片和 config/index 的固定大小/SHA256、index `weight_map` 只引用五片且无额外 safetensors；Cosmos/Qwen 配置与 revision；dataset manifest 的版本/commit、非负 `chunk_id`、五条精确 feature 和四条 metadata 文件路径、大小、SHA256、chunk 字段；`uv.lock`；精确敏感文件名；以及全文件 SHA256 manifest。脚本不运行模型。

默认校验仅表示上述三类资产的文件校验通过，不表示 L20 离线环境可运行。使用 `--require-runtime` 时还要求非空 Linux/amd64 `uv-cache` 与 `manifests/linux-amd64-offline-validation.json`；后者须记录干净 Linux/amd64 环境执行 `uv sync --frozen --offline` 的退出码 0 和日志位置。即使这两项通过，也必须在 L20 实机断网运行并验收输出后，才能称为离线推理跑通。

应保留：dry-run 和 execute 命令、完整 stdout/stderr、准备机 OS/架构/磁盘/工具版本、下载资产 manifest、全文件 manifest、验收退出码和传输后 L20 复验日志。以下任一项是停止并升级 Sol 的条件：冻结 revision、size 或 SHA 不匹配；index/feature/metadata 集合不符；允许文件缺失；下载到 Cosmos/Qwen 权重；`physical_ai_av` API/版本变化；ModelScope authority/transport 不匹配；非 Linux/amd64 依赖产物；疑似凭据文件；或离线官方入口网络阻塞。
