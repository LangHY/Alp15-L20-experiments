# 完整候选池用户下载命令

61,301个metadata test候选clip；633个块；3,165个ZIP。实际总量 **3,500,560,168,089字节（3.50TB / 3.18TiB）**，初次下载至少约3.68TB空闲；解压空间另算。只含四相机和egomotion，不下载模型/其它传感器。

目前未验证可用的完整国内传输源：魔搭同名仓缺少camera/labels，HF-Mirror所核文件入口308跳回官方。因此以下是官方备用命令，不冒称国内镜像。需要已有NVIDIA数据许可与HF登录，脚本不打印token或将token交第三方。

**这是完整候选池，不是已验收独立最终测试集。** 历史曝光审定、输入QC和最终名单仍待完成；ZIP可能附带额外非test clip，不能将所有成员自动纳入评测。

主任务没有启动任何ZIP下载。以下命令由用户在明确接受3.50TB下载成本后自己运行；可先把`--output`替换为具备足够空间的独立绝对目录。缺空间则脚本在下载前拒绝，不会删除已有文件腾空间。

```bash
cd /Users/USER/Documents/ChatGPT/Lighten_VLA
HF_HUB_DISABLE_XET=1 \
 /Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B \
 scripts/download_alp15_full_test_candidates.py --download \
 --output /Users/USER/Downloads/alp15-full-test-candidates-20261011 \
 --plan-sha256 2a2352b5e8d690763c7d66c40d3baed2154095987b64bd33c8206614ac93e8f0 \
 --confirm-bytes 3500560168089
```

中断后原命令重跑，沿HF缓存继续；不会清空旧缓存。已完整文件只有bytes/SHA一致才跳过；已有冲突文件或缓存/日志symlink会停止并保留。逐文件流式SHA校验，全部通过才追加download-complete.jsonl，独立性/QC状态仍False；不自动解压、休眠或启动模型。

仅查看本地计划、**不联网/下载**：

```bash
cd /Users/USER/Documents/ChatGPT/Lighten_VLA
/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B \
 scripts/download_alp15_full_test_candidates.py
```

完整路径/size/官方SHA在 `evidence/full_pool_download_20261011/download-plan.json`；实际test ID和块映射在同目录test-candidates.csv。不要改这些文件、revision、SHA或删大块来绕过门禁。若需要较小集合，应另定概率抽样/统计规划，不称为全池。
