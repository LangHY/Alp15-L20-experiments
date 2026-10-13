# 小型测试候选集：2026-10-11

## 决策 D-20261011-SMALL-CANDIDATE-01

决策者：6.1 Sol high 子智能体 small_test_decision；执行者及文件所有者：主智能体。范围仅限数据准备，不改变原始模型或 Jetson 验收门禁，不启动大文件下载。

数据源为 NVIDIA PhysicalAI-Autonomous-Vehicles，revision 为
`33f9bf447ed3bcb7d545ce13f4226f824214fafb`。
复用已核对的完整池报价，SHA256 为
`2a2352b5e8d690763c7d66c40d3baed2154095987b64bd33c8206614ac93e8f0`。

在已排除21个已知使用块的633个候选块中，数值排序后用 Python
`random.Random(20261011).sample(chunks, 10)` 随机无放回抽样。
冻结块为337、339、355、1468、2212、2220、2232、2653、2665、2669。
共有949个候选clip，50个ZIP（四路相机和egomotion），
54,382,445,102 bytes = 54.38 GB / 50.65 GiB。
首次下载至少需要约57.11 GB余量；解压与额外缓存另算，建议预留120 GB。

不是官方发布的独立“小版本”，而是固定revision的可复现抽样。
只有10个抽样块，不能把949 clips直接当作949个独立样本。
drive独立性与历史使用审计未完成，正式样本量功效未计算。
适合阶段性配对评估，不构成“最大精度已经充分验证”。
后续固定输入与模型版本，比较ADE/FDE、置信区间、长尾退化、失败率，
按drive检查泄漏、相关性与聚类；不要按优化模型结果换样本。

## 执行与验证

新文件：`scripts/download_alp15_small_test_candidates.py`。原完整池脚本及资产不改。
读取并绑定原始plan、snapshot、clip名单哈希；默认只打印计划。
用户明确确认字节数后才调用官方HF接口；复用原目录安全检查，
保存子集plan与949ID名单，逐文件校验大小/SHA，已有冲突保留并停止。
不自动解压、运行模型、删除缓存或休眠。

环境：MacBook本地已有Python虚拟环境；无GPU操作。
默认CLI运行退出码0，输出949/50/54,382,445,102及PLAN ONLY。
新增3项有界检查通过：固定选择、默认无输出目录操作、缺确认不创建目录。
复用组件9项unittest通过；`git diff --check`退出码0。
本次未调用任何ZIP下载，不声称网络下载、续传或payload已验。
6.1 Sol high独立只读验收通过：范围仅限冻结计划与下载工具，实际传输未验。
子集plan SHA为`7a2356ea86921d535a96ef84e49f1018f8209dd31f7011423c6574372703471f`，
949ID CSV SHA为`253d480bfeccc82ecd1c815f9f731eeff11458c68c0b4b50b9f3c4846a037e96`。
脚本不写单独completion manifest；完成时打印50ZIP verified，保存逐文件receipt。
国内来源沿用完整池本轮核查记录，暂未验证完整可用国内镜像，不擅自替换认证端点。

## 用户启动命令

```bash
cd /Users/USER/Documents/ChatGPT/Lighten_VLA
HF_HUB_DISABLE_XET=1 \
/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B \
scripts/download_alp15_small_test_candidates.py \
  --download \
  --output /Users/USER/Downloads/alp15-small-test-949-20261011 \
  --confirm-bytes 54382445102
```

需已有官方数据仓访问权限。输出目录可改为足够容量硬盘上的独立绝对路径。
下载后评估只能读取`test-candidates.csv`中的949个ID，不能把ZIP内其他成员加入测试。
