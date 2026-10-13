# 完整候选池报价与用户下载准备执行回报

- 日期2026-10-11；执行者GPT-6.1 Sol high主任务，对应D-ALP15-PRECISION-FULL-POOL-DOWNLOAD-20261011；主任务独占新增脚本、测试、本回报及本次证据目录。独立验收者只写acceptance。
- 完整集合是正式v3 metadata test候选快照：61301唯一clip、633块、每块egomotion+四相机，共3165 ZIP。**不是已验收独立最终测试集**，coverage/drive/QC仍未通过。
- 本轮只做服务器CPU文本导出、本地小文件传输、官方size/SHA API查询、合成CPU测试；未下载ZIP、模型、视频或其它大资产，未执行推理/训练/解压，未安装或变更依赖。

## 1. 实际来源与导出

正式CSV `eligible-candidates.csv` SHA `cf620cf9b278bacd149378591d356e30a8efa3679e2696850a5bc743c717d8e7`；原 `candidate-file-plan.csv` SHA `b76b687b43f3d9923c1a213e2b34ad3a3e64936d2c6c90a31470f7cf83a1b0ac`；ledger SHA `0697cec28cd36ca841fe75ddeb95f1037269d69871bc2bda59ae831947e870d1`。固定revision `33f9bf447ed3bcb7d545ce13f4226f824214fafb`。

服务器实际CPU命令：既有 `/home/USER/lang/Lighten_VLA/alpamayo1.5/.venv/bin/python -B` 执行主任务新 `export_alp15_test_candidate_snapshot.py`（在既有补充证据目录内），读取原三文本并前后验SHA。只筛split=test；clip_id去重61301、chunk去重633，与ledger旧排除块无交集；从原file-plan实际行提取五feature路径，非按UUID或编号区间猜测。原件未改。新目录 `experiments/precision_full_pool_download_20261011_v1` 排他创建、产物open-x拒覆盖。

实际远端export exit0（5f2b19），随后使用既有SSH严格known_hosts/ControlPath、SCP旧协议 `-O` 取回两份小文本（666388 exit0）；本地SHA与远端stdout声明逐一相等（12bc8b）。没有复制原45MB候选全表或46MBledger。

|归档产物（evidence/full_pool_download_20261011）|SHA256|
|---|---|
|candidate-snapshot.json|e75bb6813da2f51767e701ea1f7fc9f920da8e943abb89bcb18f06a27b1b2096|
|test-candidates.csv|ad03761a64e92503b6a1bd6954b04898b7d57cb7badb45d194e9f9f7611d248a|
|download-plan.json|2a2352b5e8d690763c7d66c40d3baed2154095987b64bd33c8206614ac93e8f0|

## 2. 本日国内源小查询

主任务完整读web-access技能并前置检查（a522a7）：Node22.23.0可用，Chrome未连接；本轮使用公开小API/HEAD及既有官方认证的只读文件信息，不操作浏览器、不发送token给第三方。

- 魔搭同名仓固定提交与master顶层树均Code200、TotalCount4，仅calibration/.gitattributes/LICENSE.pdf/README.md（ebf5b9、73c10f）。无所需camera/labels原始视频树，未据同名仓宣称相同完整数据。后续master查询只打印安全路径/计数。
- HF-Mirror固定revision的front-wide示例ZIP入口HTTP308，Location为同路径的huggingface.co（8b1564）。只请求HEAD，不下载响应文件。故它不是本轮已验证可用的国内大文件传输端。
- 尚无可验收完整国内源；本次可执行传输仍是NVIDIA官方HF备用，需要用户自己的gated许可与登录，未代接受许可。

以上是本日真实查询，不复述20261009观察为新证据；也未对互联网上所有镜像穷尽搜索。公开入口：[魔搭树](https://modelscope.cn/api/v1/datasets/nv-community/PhysicalAI-Autonomous-Vehicles/repo/tree?Revision=master&Recursive=false&PageSize=100)、[HF-Mirror固定文件入口](https://hf-mirror.com/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/resolve/33f9bf447ed3bcb7d545ce13f4226f824214fafb/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0001.zip)。

## 3. 全量实际报价与用户确认

本机既有下载venv `/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3`，未安装/升级。实际查询命令：

```bash
cd /Users/USER/Documents/ChatGPT/Lighten_VLA
/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B \
 scripts/download_alp15_full_test_candidates.py --quote
```

仅 `HfApi.dataset_info`/`get_paths_info`小型元信息API；先实证API签名（341220）、固定revision首个候选文件size/LFS SHA（5f6fc7），再每100个路径查询一次全3165项。每次actual path集合精确匹配，正bytes、完整LFS SHA、无重复，最后求和，不用单chunk估算。quote成功exit0（b2a5f5），完整stdout在工具会话；报价JSON以open-x存新本地文件。

**真实总ZIP容量3500560168089 B = 3.500560168089 TB = 3.1837409261142966 TiB。** 用户初次下载时工具另要求5%空闲余量，约3.68TB可用空间；解压/其它缓存/模型/工作空间不包括在这里，不自动解压或复制第二份完整ZIP。ZIP可能附带非test clip，实际模型输入须用61,301-ID名单并最终审定，不能跑所有ZIP成员。

用户尚未在看到3.50TB实价后确认全池成本与目标位置，**因此主任务不执行下载分支**。所给下载命令中的明确plan SHA/confirm-bytes参数，是让用户在知晓规模后自己启动的确认；不是代理已取得“自动下载TB”的授权。用户可不运行或先提供其他输出目录；不得擅自删大块变成便利小集。

## 4. 下载工具边界与验证

- 默认无参数只读本地快照/报价、打印规模，不联网或下载（dcb57e）；help也不联网（1f0209）。`--quote`只查询小元信息；`--download`必须显式给完整plan SHA、确认总bytes、独立绝对输出目录。
- 固定官方endpoint；不读取/打印token，不将既有认证交给第三方镜像。启动前设HF_HUB_DISABLE_XET=1，沿HF自己的partial缓存续传，不删除任何缓存。
- 已存在目标必须bytes和SHA匹配才跳过，否则停并保留，拒symlink/逃逸/排除块交集/重复或缺失路径/未知SHA。磁盘余量不足在下载前停止。逐文件下载后流式校验bytes/SHA，成功写追加receipt；失败文件保留不覆盖/删除。只有全部3165项通过才写completion，仍标independence_verified=False/input_qc_verified=False。
- 本地标准库合成测试7项全部通过（c471e1 exit0）：完整集合、缺失/重复、路径逃逸、旧块重叠、遮蔽SHA/错误总量、默认无网络、缺少下载确认时拒绝且不创建输出。测试未跑真实quote或ZIP分支。`git diff --check` exit0（82dfe4）；本次新脚本尚待独立Sol复核，不把测试数量当验收。

冻结代码SHA：download脚本 `82634fc340517de475cd19d94e2f18ba9019f4eb0dbecc15a54dc856bc9d3f18`；export脚本 `805593baaf559f54b90a7dc6c7bb6cf8f640a6ce0c246683a9e941c82de9b1ef`；tests `516ae873c88748dcbe3f88c5d2b2860a16b7aeb51350fd80418d7cc8c0907244`（3579bc）。旧用户未提交改动、旧census/结果/模型/数据全部保留。

## 5. 当前结论

完整候选池的ID/块/五feature路径和全量真实报价已准备；用户下载命令只针对这3165ZIP。尚未获取新ZIP，没有新精度、完整独立性、drive、输入QC、Orin16GB/300ms、闭环或官方原入口通过。全池获取完成后仍须usage范围/QC/planning排除及最终名单冻结，不能因为下载完整就称“完整独立测试集已经交付”。

## 6. 独立复核发现后的输出安全修订

独立验收者发现首版只校验ZIP输出路径，未覆盖HF local_dir缓存与receipt路径，且未拒绝输出路径含`..`别名。主任务在任何真实ZIP下载前修复：输出必须无`..`、规范化且不是根/home；拒输出祖先、整个已有.cache目录树及两个receipt的symlink；已有缓存类型冲突停，已有receipt须绑定同一plan SHA及文件/总量，不向未知日志追加。保留合法HF partial缓存，不删除任何用户内容。成功receipt新加入plan_sha256。

新增两项合成负例验证目录别名、缓存symlink、receipt symlink/异计划日志拒绝且外部目标原文不变。最终9tests/0.198s/exit0（014bbc）。最终下载脚本SHA `93911fd3542bfbac7f9a14a79b4062873aacc446d6cd5510ecb191b2e3d9c8aa`；最终tests SHA `bfacf13ebdd4580166abad56bcce4920d026a9d3700d93585db5d1a80e1f13a0`（16d9ff）。先前第4节SHA/7tests为被独立复核的首版历史，不冒称当前冻结版本。quote快照/CSV/3165文件报价及plan SHA未改；修复未运行下载分支。
