# ALP15-CLIP-ACCEPT-001：单 clip 资产文件验收

- 日期：2026-09-17 17:27 CST
- 验收者：GPT-5.6 Sol high
- 环境：本机 macOS 27.0；只读验收，不运行模型
- 输入：`/Users/USER/Downloads/alp15-test-clip-delivery-20260917`
- 下载日志：`/Users/USER/Downloads/alp15-test-clip-resume-20260917.log`
- 验收命令：`python3 -B - <<'PY'` 内联只读校验（完整命令与输出位于本任务的 exec 工具调用记录）；另以 `rg -n '^(已生成 dataset manifest：|单 clip 九项文件校验通过$)' /Users/USER/Downloads/alp15-test-clip-resume-20260917.log` 定位本轮结束标记。

## 验收结果

- 独立校验退出码：`0`；输出：`PASS: 5 features, 4 metadata, 9 links, 9 blobs, 6562843701 bytes`。
- `dataset-manifest.json` 的 repo、revision、`physical-ai-av` 版本与源码提交、clip、t0、chunk 均与冻结值一致；chunk 为 `3119`。
- 五项 feature 和四项 metadata 的路径集合精确匹配固定 chunk 的九条路径。
- 九条 snapshot 均为相对符号链接，目标都位于交付目录的 `blobs/`。逐项流式复算大小和 SHA-256 与 manifest 相同；40 位 blob 名匹配 Git blob SHA-1，64 位 blob 名匹配 SHA-256。
- 全目录叶节点仅有九条 snapshot 链接、九个 blob 与一个 manifest，无额外或缺失文件。交付目录约 6.1 GiB；资产数据字节数为 `6,562,843,701`。
- 追加日志第 87、88 行依次为新一轮生成 manifest、`单 clip 九项文件校验通过`。该日志此前保留历史 CAS 报错，且未单独记录这次 Shell 管道的最终退出码；以现存交付资产及独立只读校验退出码 `0` 作为文件验收依据。

结论：冻结的**单 clip 测试数据资产文件完整**。这不代表模型权重、Linux/amd64 离线依赖、L20 实机官方入口或端到端推理已验收。日志含历史带签名下载 URL，保留在本机，不作为公开附件分发。
