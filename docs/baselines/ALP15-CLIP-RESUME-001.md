# ALP15-CLIP-RESUME-001：单 clip 缓存续用修复

- 日期：2026-09-17
- 决策者：GPT-5.6 Sol high
- 里程碑：NVIDIA 官方原始 Alpamayo 1.5 10B 测试数据准备
- 范围：修复单 clip 预取脚本的类型兼容和 Hugging Face 缓存布局；不启动下载或推理。

## 事实与故障

- `physical-ai-av==0.2.0` 的 `get_clip_chunk` 从 pandas 索引返回 `np.int64(3119)`；旧脚本仅接受 Python 原生 `int`，日志为 `get_clip_chunk 必须返回非负整数`。
- 现有 `/Users/USER/Downloads/alp15-test-clip-20260917` 缓存约 32 MB，含冻结 revision `33f9bf447ed3bcb7d545ce13f4226f824214fafb` 下的四个元数据文件；尚无五个特征文件和 `dataset-manifest.json`。
- 官方库使用 `datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/<revision>/` 与 `blobs/` 缓存布局。旧脚本在输出根目录查找数据文件，即使修复整数类型也会失败。
- 安装版官方库初始化时调用 `HfApi.file_exists`，会访问网络；资产文件校验通过不等于 L20 完全离线官方入口跑通。

## 决策与边界

1. 排除 `bool`，接受整数标量并转换为内建 `int`，随后验证非负。
2. 下载使用独立持久 `--cache-dir`，可复用既有 32 MB 缓存；失败时不自动清理缓存。交付用全新 `--output-dir`，不得覆盖非空目录。
3. 交付目录只复制固定 revision 下五个特征、四个元数据的 snapshot 链接及对应 blob，并生成 manifest；链接必须在交付目录内解析，保留 Hugging Face 缓存布局。五项特征的下载调用方式由后续 `ALP15-CLIP-XET-001` 修订为逐项 `download_file`。
4. 校验器固定九条 snapshot 路径，复算大小和 SHA-256，校验来源、clip、chunk、链接和 blob 集合。
5. `prepare_alpamayo15_offline_bundle.sh` 的 clip 缓存必须位于 bundle 外，并允许显式指定缓存以支持重试。

冻结 clip ID：`030c760c-ae38-49aa-9ad8-f5650a545d26`；t0：`5100000` μs。禁止改动冻结来源、revision、五个特征和四个元数据的口径；禁止把缓存中的 `.locks`、`.DS_Store` 等杂项复制进交付目录。

## 验收与停止条件

- 无网络夹具测试接受 `np.int64(3119)`，拒绝布尔值、负数和越界链接，检查九个 snapshot 链接及 blob；脚本语法检查通过。
- 真实下载后，九个文件、manifest 和 bundle 校验器全部通过，才可称单 clip 资产文件完整。
- 官方 API、版本或固定路径变化，来源冲突、认证/网络错误、必需文件缺失或哈希不匹配时停止并保留缓存及日志。
- 本决策不宣称已完成下载或原样官方入口在 L20 完全离线运行。

## 执行与验收回报

- 执行者：GPT-5.6 Sol high；执行日期：2026-09-17；环境：macOS arm64，`physical-ai-av==0.2.0` 的 Python 3.12 虚拟环境。未启动真实下载。
- 修改文件：`scripts/prefetch_physical_ai_clip.py`、`scripts/verify_alpamayo15_offline_bundle.sh`、`scripts/prepare_alpamayo15_offline_bundle.sh`，以及本决策和原资产基线文档。
- `bash -n` 两份 Shell 脚本退出码 0；Python AST 解析退出码 0；预取脚本指向既有 32 MB 缓存与新交付目录的 dry-run 退出码 0，交付目录未创建。
- 无网络夹具覆盖 `np.int64(3119)`、九个 snapshot 相对链接及 blob、manifest、40 位 Git blob SHA-1 与 64 位 SHA-256；篡改缓存或交付 blob 的反例被拒绝。现有缓存中的四个 metadata blob 内容摘要与文件名匹配。
- 最终 Sol high 复核通过，可以提供单 clip 续传命令。真实五项 feature 下载、完整包验收与 L20 断网官方入口测试均未执行。
