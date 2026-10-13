# ALP15-VLM-EXTRACT-001：从 Alpamayo 1.5 提取独立 VLM checkpoint

## 决策单

- 日期：2026-09-22；决策者、执行者与验收者：GPT-5.6 Sol high。
- 对应基线：已完成的 Alpamayo 1.5 L20 BF16 离线兼容基准。
- 目标：在 L20 既有推理环境中，从冻结的 Alpamayo 1.5 10B checkpoint 提取 `vlm.*` 权重，生成可由 `Qwen3VLForConditionalGeneration.from_pretrained(..., local_files_only=True)` 独立加载的 Qwen3-VL checkpoint。
- 事实依据：源索引包含 1159 个 tensor，其中 `vlm.*` 750 个、8,797,933,808 个参数、17,595,867,616 bytes BF16 权重；VLM 横跨源分片 1–4，源分片 4 同时包含 expert 权重，不能直接复制整片。移除 `vlm.` 后的键与 Qwen3-VL 模型键对应；embedding 与 lm head 均为 `[155697, 4096]`。
- 处理器依据：Alpamayo 官方初始化逻辑以 Cosmos-Reason2-8B 的 Qwen3-VL 配置为基础，增加 4000 个离散轨迹 token 和项目特殊 token，最终词表大小为 155697。提取器使用已验收的 Cosmos 非权重配置和冻结来源清单重建 processor，并逐项核对轨迹 token id。
- 环境边界：脚本默认要求服务器推理环境中的 Python 3.12、PyTorch 2.8.0、Transformers 4.57.1、safetensors 0.7.0；只在 CPU 上读取和重写权重，不占用 L20 显存。
- 明确不做：不修改源 checkpoint，不提取 diffusion expert/action projection，不执行剪枝、量化、蒸馏、ModelOpt 转换或性能结论。
- 风险等级：高。源分片 4 是 VLM/expert 混合分片；词表配置与 processor 若不同步，独立 checkpoint 会加载失败或 token 语义错位。
- 文件所有者：`scripts/extract_alpamayo15_vlm_checkpoint.py` 与 `tests/test_extract_alpamayo15_vlm_checkpoint.py` 由本决策单独占用。
- 停止条件：源模型或 Processor SHA-256 不匹配、关键 VLM 权重缺失、去前缀后键冲突、词表大小/token id 不一致、输出目录已存在、剩余空间少于 VLM 权重大小加 1 GiB，或输出结构复验失败。

## 运行方式

在 L20 服务器现有项目根目录执行：

```bash
cd /home/USER/lang/Lighten_VLA
source alpamayo1.5/.venv/bin/activate
```

先执行只读 dry-run。正式记录中不要加入跳过哈希或放宽版本的参数：

```bash
python -B scripts/extract_alpamayo15_vlm_checkpoint.py \
  --model-dir "$PWD/models" \
  --processor-source-dir "$PWD/alp15-configs-delivery-20260919/dependencies/configs/Cosmos-Reason2-8B" \
  --processor-source-manifest "$PWD/alp15-configs-delivery-20260919/manifests/cosmos-reason2-source.json" \
  --output-dir "$PWD/outputs/alpamayo15-vlm-standalone" \
  --dry-run
```

dry-run 通过后执行正式提取和 CPU 完整加载复验：

```bash
set -o pipefail
mkdir -p logs

python -B scripts/extract_alpamayo15_vlm_checkpoint.py \
  --model-dir "$PWD/models" \
  --processor-source-dir "$PWD/alp15-configs-delivery-20260919/dependencies/configs/Cosmos-Reason2-8B" \
  --processor-source-manifest "$PWD/alp15-configs-delivery-20260919/manifests/cosmos-reason2-source.json" \
  --output-dir "$PWD/outputs/alpamayo15-vlm-standalone" \
  --verify-load \
  2>&1 | tee "logs/alp15_vlm_extract_$(date +%Y%m%d_%H%M%S).log"

exit_code=${PIPESTATUS[0]}
echo "exit_code=$exit_code"
test "$exit_code" -eq 0
```

输出目录必须事前不存在。脚本使用同一父目录下的隐藏临时目录，只有全部结构校验和可选完整加载通过后才原子改名为最终目录；失败时保留 partial 目录供排查，不覆盖任何旧结果。

## 输出和验收

输出至少包括：

- `config.json`：Qwen3-VL 配置，text/top-level vocab 均为 155697；
- 扩展后的 tokenizer、processor、video processor、chat template 和 generation config；
- `model-xxxxx-of-yyyyy.safetensors`：只含去掉 `vlm.` 前缀后的 VLM 权重；
- `model.safetensors.index.json`：独立权重索引；
- `extraction-manifest.json`：源摘要、环境版本、参数量、权重大小、processor token id、输出文件大小与 SHA-256。

验收要求：

1. dry-run 报告 `selected_tensors=750`、`parameters=8797933808`、`weight_bytes=17595867616`、`vocab_size=155697`；
2. 正式运行退出码为 0，源 checkpoint 的文件大小和 SHA-256 全部匹配；
3. 输出不含 `expert.*`、`action_in_proj.*`、`action_out_proj.*`，也不再含 `vlm.` 前缀；
4. 每个输出 tensor 的 shape/dtype 与源 tensor 一致，并逐项通过 `torch.equal` 精确值比对；
5. processor 的轨迹 token 起点及六个固定 token id 与 Alpamayo config 一致；
6. `--verify-load` 完成 CPU 独立加载；
7. 保存完整日志、退出码、输出 manifest 和输出目录全文件 SHA-256。

## 本机代码验收

- 真实源索引 dry-run 通过：750 个 tensor、8.7979B 参数、16.3874 GiB BF16 权重、155697 词表、默认规划 11 个输出分片；未写入权重。
- 使用真实 Cosmos 配置重建 processor 通过，最终词表及六个轨迹 token id 与 Alpamayo config 完全一致。
- 小型 safetensors 测试证明：混合分片中的 expert 权重不会进入输出，`vlm.` 前缀正确移除，输出 shape/dtype 正确，源分片字节未改变。
- L20 正式提取尚未执行；不得把本机 dry-run 记为独立 VLM checkpoint 已生成。
