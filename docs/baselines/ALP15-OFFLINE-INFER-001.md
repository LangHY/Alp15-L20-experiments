# ALP15-OFFLINE-INFER-001：Alpamayo 1.5 离线兼容性测试

## 决策单

- 日期：2026-09-17；决策者与执行者：GPT-5.6 Sol high。
- 当前里程碑：NVIDIA 官方原始 Alpamayo 1.5 测试与可复现基线。
- 风险：高。只允许新增仓库顶层的独立离线兼容入口及测试，不修改已冻结的 NVIDIA 官方源码、checkpoint、数据集或模型结构。
- 输入：官方源码提交 `36aeb4c5938cbc2eb2aed33b22434773da4ab639`；官方 `src/alpamayo1_5/test_inference.py` SHA256 `dc69646feed09f92defa00a19ae6f2fc2a37a10a70ec35678f87946fbe6fe8e8`；Alpamayo checkpoint 的冻结 `config.json` SHA256 `824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e`；数据集 revision `33f9bf447ed3bcb7d545ce13f4226f824214fafb`，`physical_ai_av==0.2.0`。

离线兼容入口保留官方样例的 clip、时间戳、四路相机、处理器调用、采样参数与 minADE 计算口径。仅作下列运行时适配：

| 阻塞 | 兼容处理 |
| --- | --- |
| 官方入口从远端模型 ID 加载 | 从显式本地 Alpamayo 权重目录加载，模型 `config.json` 只读；仅在内存中将其 `vlm_name_or_path` 指向本地 Cosmos 非权重配置。 |
| 官方 helper 固定从远端 Qwen ID 加载处理器 | 从显式本地 Qwen 非权重配置目录加载处理器，再使用实际模型 tokenizer。 |
| `physical_ai_av==0.2.0` 初始化无条件调用 `HfApi.file_exists` | 仅在初始化范围内对冻结的本地 `feature_presence.parquet` 做精确离线存在性响应；clip 仍通过顶层 `PhysicalAIAVDatasetInterface.get_clip_feature`，且 `maybe_stream=False`。 |

不下载 Cosmos/Qwen 权重，不修改 checkpoint JSON，不改变网络不可用的 L20 环境，不执行剪枝、蒸馏、量化、ONNX 或 TensorRT。所有输入路径必须显式指定。缺文件、来源或哈希不符、试图访问网络、输出口径异常时停止，不将失败的兼容测试写成官方入口成功。

## 资产与环境前提

现有本机权重位于仓库 `models/`；单 clip 交付位于 `/Users/USER/Downloads/alp15-test-clip-delivery-20260917`。这两项的文件完整性已分别核查，但还没有 L20 加载证据。本机 Hugging Face 缓存中可见 Qwen 冻结 revision `89644892e4d85e24eaac8bacfd4f463576704203` 的九个非权重文件；截至本决策检查，项目目录和已检查缓存中没有 Cosmos 冻结 revision `a9fae2cf89dc64db96b12860417f0eb403013bb9` 的非权重配置。若只交付配置目录，应将其复制为**真实文件**并附上来源 manifest；若交付完整 Hugging Face cache，则必须保留 snapshot 相对符号链接和对应 blobs，不能只复制孤立链接。

允许的两组配置文件名仅为：`config.json`、`generation_config.json`、`preprocessor_config.json`、`video_preprocessor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.json`、`merges.txt`、`vocab.json`。其冻结来源、打包限制与 Linux/amd64 离线依赖要求见 [ALP15-OFFLINE-ASSET-001.md](ALP15-OFFLINE-ASSET-001.md)。现有 Ubuntu 22.04 Docker 镜像不等于这些 Python/CUDA 依赖已安装或已验证。

如需只补两组配置而不重下 10B 权重，可在**有网准备机**运行下列命令；本决策未启动下载。`hf download` 的九个位置参数均是明确文件名，不包含权重。复制到 L20 时必须保留 `alp15-config-cache` 下完整的 `models--.../snapshots/<revision>/` 与 `blobs/` 结构：

```bash
hf download nvidia/Cosmos-Reason2-8B \
  config.json generation_config.json preprocessor_config.json video_preprocessor_config.json \
  tokenizer.json tokenizer_config.json chat_template.json merges.txt vocab.json \
  --revision a9fae2cf89dc64db96b12860417f0eb403013bb9 \
  --cache-dir /transfer/alp15-config-cache

hf download Qwen/Qwen3-VL-2B-Instruct \
  config.json generation_config.json preprocessor_config.json video_preprocessor_config.json \
  tokenizer.json tokenizer_config.json chat_template.json merges.txt vocab.json \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --cache-dir /transfer/alp15-config-cache
```

## L20 运行顺序

先在有网准备机补齐 Cosmos/Qwen 非权重配置，连同已校验的模型目录、完整 clip 交付目录、冻结官方源码、兼容脚本及匹配的 Linux/amd64 Python 依赖传至 L20。上传后须重新核对文件哈希、符号链接目标和依赖版本；不得在 L20 上触发下载。

在 L20 的 Python 3.12 环境中，从本仓库根目录先做只读预检。以下路径是示例，须替换为服务器上的实际绝对路径：

```bash
python -B scripts/test_alpamayo15_offline_compat.py \
  --model-dir /srv/alpamayo/models \
  --dataset-dir /srv/alpamayo/alp15-test-clip-delivery-20260917 \
  --cosmos-config-dir /srv/alpamayo/dependencies/configs/Cosmos-Reason2-8B \
  --qwen-config-dir /srv/alpamayo/dependencies/configs/Qwen3-VL-2B-Instruct \
  --config-manifest-dir /srv/alpamayo/manifests \
  --preflight-only
```

上例是复制后的配置目录，因此必须提供 `cosmos-reason2-source.json` 与 `qwen3-vl-source.json` 所在的 `--config-manifest-dir`。只有当两组配置均位于各自冻结 revision 的完整 Hugging Face snapshot/cache 结构中，才能省略该参数。

若使用上文单独下载的 `alp15-config-cache`，则把两项配置路径分别指向 `/srv/alpamayo/alp15-config-cache/models--nvidia--Cosmos-Reason2-8B/snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9` 和 `/srv/alpamayo/alp15-config-cache/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203`，不传 `--config-manifest-dir`。

预检通过后，先按项目门禁原样尝试冻结官方入口并记录断网阻塞的完整命令、环境、退出码和日志；本兼容入口不能取代这一步。确认该阻塞后，在 L20 实际断网条件下运行同一兼容命令但去掉 `--preflight-only`，保留 stdout/stderr、退出码、环境清单、显存与内存观测、端到端耗时和输出产物。可用 `tee` 记录日志，但须同时保留管道中的 Python 退出码；不要将日志中的认证信息或主机标识对外发送。

## 验收与命名

本机无网单测、语法检查和资产预检只证明脚本行为及已检查文件，不证明 L20 可运行。只有 L20 在物理或网络策略断网条件下完成模型加载、四路相机解码、预处理与端到端采样，进程退出码为 0，输出包含 CoC、预测张量形状及 minADE，才可记录为“ALP15 离线兼容性测试通过”。即使通过，也**不能**称为“官方原始入口跑通”；官方入口和 Jetson 门禁仍按 [AGENTS.md](../../AGENTS.md) 独立验收。

## 本机执行回报（截至 2026-09-17）

- 决策编号：`ALP15-OFFLINE-INFER-001`；执行者与复核者：GPT-5.6 Sol high；执行时段：2026-09-17 当日，复核截至 19:18 CST。
- 环境：准备机 macOS 27.0 / Darwin arm64；单测解释器 Python 3.14.6；真实数据接口烟测解释器 Python 3.12.10、`physical-ai-av==0.2.0`。本机未使用 CUDA、未加载模型；L20 的驱动、CUDA、torch、flash-attn 等环境尚未核验。
- 文件变更：新增 `scripts/test_alpamayo15_offline_compat.py`、`tests/test_alpamayo15_offline_compat.py` 与本记录；更新 `docs/baselines/README.md` 链接。官方 `alpamayo1.5/` 工作树 clean，HEAD 为冻结提交，入口 SHA256 仍为 `dc69646feed09f92defa00a19ae6f2fc2a37a10a70ec35678f87946fbe6fe8e8`。
- 验证命令：`python3 -B -m unittest discover -s tests -p 'test_*.py' -v`，退出码 `0`，全仓库 7 项测试通过（其中兼容入口 6 项）；`ruff check scripts/test_alpamayo15_offline_compat.py tests/test_alpamayo15_offline_compat.py` 与 `ruff format --check scripts/test_alpamayo15_offline_compat.py tests/test_alpamayo15_offline_compat.py` 均退出码 `0`；AST 解析通过。完整命令与 stdout/stderr 在本任务工具调用记录，本机未另存日志文件。
- 在本机 clip venv（`physical-ai-av==0.2.0`，未安装 torch）以 `HF_HUB_OFFLINE=1` 并启用脚本的 socket 外联拦截，实测 `PhysicalAIAVDatasetInterface` 从交付缓存初始化、返回 `chunk_id=3119`，随后 `get_clip_feature(...EGOMOTION, maybe_stream=False)` 返回 `Interpolator`，退出码 `0`。
- 未执行：Cosmos 配置下载、Linux/amd64 离线依赖安装、L20 权重加载、四路相机解码和端到端推理；因此当前不能给出兼容测试通过或官方入口通过的结论。
