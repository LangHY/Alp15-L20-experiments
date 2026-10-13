# ALP15-COSMOS-CONFIG-001：Cosmos 非权重配置验收

## 决策单

- 日期：2026-09-19；决策者、执行者与验收者：GPT-5.6 Sol high。
- 当前里程碑：跑通 NVIDIA 官方原始 Alpamayo 1.5 测试并建立可复现基线。
- 目标：补齐 `nvidia/Cosmos-Reason2-8B` 的九个非权重配置，不下载 Cosmos 权重，并证明 ModelScope 传输结果与冻结 Hugging Face revision 的文件内容一致。
- 权威来源：`nvidia/Cosmos-Reason2-8B` @ `a9fae2cf89dc64db96b12860417f0eb403013bb9`。
- 传输来源：`modelscope:nv-community/Cosmos-Reason2-8B` @ `master`；原下载目录 `/Users/USER/Downloads/alp15-cosmos-config-20260919`，验收封装后已按清理要求移入废纸篓。
- 范围仅限：`config.json`、`generation_config.json`、`preprocessor_config.json`、`video_preprocessor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.json`、`merges.txt`、`vocab.json`。
- 停止条件：任一文件缺失、存在额外顶层 payload、存在权重文件，或大小／Git blob SHA-1 与冻结权威 revision 不符。

## 执行与验收回报

- 执行环境：macOS / arm64 准备机；未加载模型、未使用 GPU、未修改 NVIDIA 官方源码或 checkpoint。
- ModelScope 仓库公开文件树包含上述九个文件；其 `master` 不是权威 revision，因此未仅凭文件名或大小声明等价。
- 对下载目录的九个文件逐一计算 `SHA-256`，并按 Git blob 规则计算 `SHA-1("blob <size>\\0" + content)`；随后与 Hugging Face 冻结 revision 公开文件树的 `size` 和 `oid` 逐项比对。
- 结果：九项大小与 Git blob SHA-1 全部一致，顶层 payload 恰好为九项，未发现 `*.safetensors`、`*.bin`、`*.pt`、`*.pth` 或 `*.ckpt`。
- 原下载目录中的 `.cache/huggingface` 是先前失败的 Hugging Face 下载尝试留下的客户端元数据，不属于九项 payload。最终只复制白名单九项和来源 manifest 至 `/Users/USER/Downloads/alp15-configs-delivery-20260919`；完整 staging 随后移动到废纸篓，见 [ALP15-CONFIG-DELIVERY-001.md](ALP15-CONFIG-DELIVERY-001.md)。
- 来源清单：[cosmos-reason2-source.json](../../manifests/cosmos-reason2-source.json)。该清单同时记录权威 Hugging Face repo/revision、ModelScope 传输来源以及九项大小、SHA-256 和 Git blob SHA-1，可供离线兼容入口的 `--config-manifest-dir` 使用。

## 结论与边界

Cosmos 九个非权重配置验收通过，可进入离线交付封装。此结论不代表 L20 权重加载、官方入口或离线兼容端到端推理通过；也不授权下载或使用独立 Cosmos 权重。
