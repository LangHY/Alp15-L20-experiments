# ALP15-CLIP-XET-001：单 clip 的 Xet 进度接口兼容修复

- 日期：2026-09-17
- 决策与执行模型：GPT-5.6 Sol high
- 里程碑：NVIDIA 官方原始 Alpamayo 1.5 10B 单 clip 资产准备
- 范围：只修改本项目的单 clip 预取脚本与相应说明；不修改 `physical-ai-av`、Hugging Face 依赖或冻结资产口径。

## 事实与决策

本机环境为 `physical-ai-av==0.2.0`、`huggingface-hub==1.31.0`、`hf-xet==1.6.0`。官方库的 `download_clip_features` 经 `download_files` 向 Hub 传入只有 `update` 的 `_AggregatedTqdm`；当前 Hub Xet 进度报告还调用 `set_postfix_str`，产生用户日志中的 `AttributeError`。向该批量方法传 `tqdm_class` 会与其内部显式参数重复，不能作为修复。

已直接核对官方固定源码：[NVIDIA 顶层导出](https://github.com/NVlabs/physical_ai_av/blob/59d578b53ef186ffc3f1de1d1b2598717a982d64/src/physical_ai_av/__init__.py)、[NVIDIA `dataset.py`](https://github.com/NVlabs/physical_ai_av/blob/59d578b53ef186ffc3f1de1d1b2598717a982d64/src/physical_ai_av/dataset.py)、[NVIDIA `hf_interface.py`](https://github.com/NVlabs/physical_ai_av/blob/59d578b53ef186ffc3f1de1d1b2598717a982d64/src/physical_ai_av/utils/hf_interface.py) 与 [Hugging Face Hub v1.31.0 Xet 进度代码](https://github.com/huggingface/huggingface_hub/blob/v1.31.0/src/huggingface_hub/utils/_xet_progress_reporting.py)。本机安装版本的对应调用路径也一致。

Sol high 决定：保留 `get_clip_chunk`、五个冻结 feature 的 `get_chunk_feature_filename` 和精确路径核对；随后按固定顺序对五条路径逐个调用 `PhysicalAIAVDatasetInterface.download_file(relative)`。这是 `physical_ai_av==0.2.0` 顶层导出接口实例的公开方法，使用同一固定 revision 和持久 cache_dir，并绕过有问题的批量进度适配。已完成的缓存文件可命中；不禁用 Xet，不更换依赖版本。

冻结来源：`nvidia/PhysicalAI-Autonomous-Vehicles` @ `33f9bf447ed3bcb7d545ce13f4226f824214fafb`；clip `030c760c-ae38-49aa-9ad8-f5650a545d26`；五个特征与四个元数据不变。

## 验收与局限

- 无网络测试需确认只调用五条冻结路径的 `download_file`，不再调用 `download_clip_features`；缓存和交付目录参数不变。
- Python 语法检查通过。真实运行后复核九项 snapshot/blob、文件身份、大小、SHA-256、manifest 与进程退出码；都成功后才可安排休眠。
- 逐文件下载改为串行。当前用户日志停在 `1/5`，没有最终退出码；不能据此判断此前整次下载的最终状态。当前只确认一个 egomotion 特征已经缓存、四个未完成临时文件存在、交付目录为空。
- 单文件失败时上游客户端可能清理该文件的临时数据；本项目脚本不清理已有持久缓存。文件完整性通过不代表 L20 原样官方入口断网可用。

## 执行回报

- 执行者：GPT-5.6 Sol high；修改：`scripts/prefetch_physical_ai_clip.py`，并新增无网络夹具 `tests/test_prefetch_physical_ai_clip.py`。
- 夹具使用固定五条 feature 路径，核对逐项 `download_file` 调用顺序、旧缓存复用、未调用批量 `download_clip_features`，以及 5 项 feature + 4 项 metadata manifest。命令：`/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B -m unittest discover -s tests -p 'test_prefetch_physical_ai_clip.py' -v`；结果：1 测试通过，退出码 0。
- Python AST 检查与真实包 dry-run 退出码均为 0；未启动实际下载，用户当前缓存与交付目录保持原样。真实下载后的最终资产校验仍待执行。
- 在 NVIDIA 与 Hugging Face 的上述固定版本官方源码完成逐项只读核对后，Sol high 对代码和测试复核通过；此结论仅证明修复符合接口调用路径，不代表实际远端传输已成功。
