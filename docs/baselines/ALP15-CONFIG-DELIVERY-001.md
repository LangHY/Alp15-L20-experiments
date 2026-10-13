# ALP15-CONFIG-DELIVERY-001：Cosmos/Qwen 离线配置交付

## 决策单

- 日期：2026-09-19；决策者、执行者与验收者：GPT-5.6 Sol high。
- 当前里程碑：跑通 NVIDIA 官方原始 Alpamayo 1.5 测试并建立可复现基线。
- 目标：将两组已验收的非权重配置整理为可传输至断网 L20 的干净交付物，并在完成验证后清理本次中间 staging。
- 输入：Cosmos `nvidia/Cosmos-Reason2-8B` @ `a9fae2cf89dc64db96b12860417f0eb403013bb9`；Qwen `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203`。
- 明确不做：不下载或复制 Cosmos/Qwen 权重；不清理全局 Hugging Face Qwen 缓存；不修改官方源码、Alpamayo checkpoint、clip 或 uv 环境。
- 验收标准：两组目录各恰好九项、全部为真实文件、两份 manifest 的大小与哈希全部匹配、无权重；全文件校验表通过；归档通过 Zstandard、SHA-256 与 tar 流验证；仅在上述条件满足后清理 staging。

## 交付结果

- 目录：`/Users/USER/Downloads/alp15-configs-delivery-20260919`，约 22 MiB，共 22 个文件。
- 归档：`/Users/USER/Downloads/alp15-configs-delivery-20260919.tar.zst`，5,320,084 bytes。
- 归档 SHA-256：`35673de77b02c5a3e66d8a6fbab12c25095ae319aa8b9b001f9fe723ef5d2fa4`。
- 校验文件：`/Users/USER/Downloads/alp15-configs-delivery-20260919.tar.zst.sha256`。
- 内容：Cosmos 九项、Qwen 九项、两份来源 manifest、`README.md`、`SHA256SUMS`；无符号链接、无权重。
- L20 使用路径已写入交付物 `README.md`；解包后应先在交付目录运行 `sha256sum -c SHA256SUMS`。

## 执行与验收

- Qwen snapshot 的九个符号链接均解析至同一 Hugging Face cache 的 `blobs/`，并按大小与 Git blob SHA-1 匹配冻结 revision；交付时解引用复制为真实文件，避免跨主机断链。
- Cosmos/Qwen 两组交付目录分别与 `cosmos-reason2-source.json`、`qwen3-vl-source.json` 逐项核对，结果均为九项通过；交付树无额外 payload、无符号链接、无权重。
- `SHA256SUMS` 共 21 项并通过逐项校验。首次生成时 zsh 循环变量误用保留数组名 `path`，覆盖了子 shell 的 `PATH`，导致 `shasum` 未找到；资产未受影响。改用 `relative_file` 和 `/usr/bin/shasum` 后重新生成并通过。
- 归档通过 `zstd -t`、相邻 SHA-256 校验及 `zstd -dc | tar -tf -` 完整流枚举。
- 项目七项单元测试在整理后重新运行并通过；官方嵌套仓库保持 clean。

## 清理结果与恢复边界

- 原 Cosmos staging `/Users/USER/Downloads/alp15-cosmos-config-20260919` 已在全部验收通过后移动到 `/Users/USER/.Trash/alp15-cosmos-config-20260919-staging-20260919-2208`，约 11 MiB，可从废纸篓恢复；只有清空废纸篓后才释放这部分磁盘空间。
- 本次公开网页抓取临时文件 `/tmp/modelscope-cosmos-page.html` 已删除。
- 用于生成交付 `README.md` 的项目内临时模板已删除；正式 README 已包含在交付目录与归档中。
- 全局 Qwen Hugging Face cache 可能被其他任务复用，未视作本次中间冗余，也未删除。

## 结论与边界

配置交付封装通过，可以上传 L20。该结论只覆盖离线配置资产的身份和传输完整性，不代表 L20 模型加载、官方入口或兼容入口端到端推理通过。
