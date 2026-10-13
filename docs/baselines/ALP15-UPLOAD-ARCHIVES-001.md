# ALP15-UPLOAD-ARCHIVES-001：L20 上传归档

## 决策单

- 日期：2026-09-19；决策者、执行者与验收者：GPT-5.6 Sol high。
- 当前里程碑：跑通 NVIDIA 官方原始 Alpamayo 1.5 测试并建立可复现基线。
- 目标：在不重复归档已完成 Cosmos/Qwen 配置包的前提下，为剩余必传资产生成三份可校验归档：源码与运行代码、Alpamayo checkpoint、单 clip 数据；最后生成总上传 SHA-256 清单。
- 输入与边界：官方源码 HEAD `36aeb4c5938cbc2eb2aed33b22434773da4ab639`；模型仅允许 `LICENSE`、`README.md`、`config.json`、`model.safetensors.index.json` 和五个冻结分片；clip 必须保留完整 cache snapshot、内部 blobs、九个相对符号链接和 `dataset-manifest.json`。
- 明确排除：源码中的不可搬运 `alpamayo1.5/.venv`；项目根 `.git`、`.venv`、`.DS_Store`、`.ruff_cache`；模型 `.gitattributes` 与 `configuration.json`；下载缓存和日志；已单独归档的 Cosmos/Qwen 配置目录。
- 产物：每份 `.tar.zst` 及相邻 `.sha256`；总清单包含三份新归档、现有配置归档和现有 Linux/amd64 uv 环境归档。
- 验收标准：归档目标事前不存在；`zstd -t`、相邻 SHA-256 和 `zstd -dc | tar -tf -` 全部通过；源码归档含官方 `.git` 而不含 `.venv`；模型归档成员集合恰好为九项；clip 归档保留九个符号链接；项目测试通过；原始资产不删除。
- 停止条件：可用空间不足 29 GiB、任一输入集合或哈希异常、归档覆盖已有文件、归档流失败或成员集合偏离。执行采用直接 tar→zstd 流，不创建大型中间副本。

## 执行回报

- 模型归档首次在写入前停止：历史决策记录中的 `config.json` SHA-256 只有 63 位，属于无效摘要；未创建模型归档。随后以冻结 Hugging Face revision 的 Git blob 和 ModelScope 固定传输 revision 的 `X-Linked-Etag` 双重核对，本机 3058-byte 文件的正确 SHA-256 为 `824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e`。相关脚本与文档已修正后再继续。
- 执行环境：macOS 开发机；归档工具为系统 `tar` 与 Homebrew `zstd`；采用 tar 到 zstd 的直接流式写入，没有创建未压缩中间副本。
- 源码与运行代码：`/Users/USER/Downloads/alpamayo15-source-runner-20260919.tar.zst`，497803 bytes，SHA-256=`2a7a1c1ac9a0748d7db4964eaf9bd58615f7b371165272f7b150d3ed15f0c7d2`。修正摘要前的旧包已移入废纸篓并保留可恢复副本；新包共 143 个成员，包含官方仓库 `.git/HEAD` 和离线兼容入口，不含 `.venv` 或 `.DS_Store`。
- Alpamayo 模型：`/Users/USER/Downloads/alpamayo15-model-20260919.tar.zst`，17324735980 bytes，SHA-256=`004232f3c2cc26b7cfd7982e2ac608804a687594b0cb80b70c940b2537d9bc26`。归档前九项输入逐一核验，归档成员集合恰好为九项。
- 单 clip 数据：`/Users/USER/Downloads/alp15-test-clip-delivery-20260917.tar.zst`，6541940065 bytes，SHA-256=`28aa4d3099f6ba2b3ecd8b59a541f1832cb24632fb5ea28cb679d9ff04495bfe`。`dataset-manifest.json` 中 5 项 feature 和 4 项 metadata 的大小及 SHA-256 均通过，归档共 32 个成员并保留九个符号链接。
- 已有配置包没有重复归档：`/Users/USER/Downloads/alp15-configs-delivery-20260919.tar.zst`，5320084 bytes，SHA-256=`35673de77b02c5a3e66d8a6fbab12c25095ae319aa8b9b001f9fe723ef5d2fa4`。
- 已有 Linux/amd64 uv 环境包纳入总清单：`/Users/USER/Downloads/alpamayo15-linux-amd64-uv-20260919.tar.zst`，4032080617 bytes，SHA-256=`5fa01ec33ecf1e019b8235018040e086ba6d566d2eed3e23d24686eb62c12552`。
- 总上传校验清单：`/Users/USER/Downloads/alpamayo15-upload-manifest-20260919.sha256`。五份归档合计 27904574549 bytes；每份归档同时保留相邻 `.sha256`。
- 验收完成时间：2026-09-19 22:52 CST。三份新归档均通过 `zstd -t`、相邻 SHA-256 与完整成员列表读取；对总清单执行 `shasum -a 256 -c alpamayo15-upload-manifest-20260919.sha256` 后五份归档全部返回 `OK`。项目 7 项测试通过，`git diff --check` 通过，官方嵌套仓库 HEAD=`36aeb4c5938cbc2eb2aed33b22434773da4ab639` 且工作树 clean。所有原始模型、clip、源码、配置和 uv 环境目录均保留，未删除；本机尚未执行 L20 推理。
