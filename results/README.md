# 各阶段实验结果总览

整理日期：2026-10-13。本次只整理本机已有记录与 2026-10-04 交付包，不访问服务器、不重跑实验。
表内精度与性能取自既有报告，具体输入、失败尝试、环境、退出码及证据边界以对应报告为准。

## 阶段与主要结论

| 阶段 | 主要指标/结果 | 口径与状态 | 详细记录 |
| --- | --- | --- | --- |
| L20 BF16 离线基准（9月20日） | 11.0785B；单 clip minADE 0.371870m；稳态5次平均1615.68ms；峰值 allocated 22126MiB | 自定义离线兼容入口；不能与后续20clip平均ADE直接比较；不是官方原入口验收 | [基准报告](baselines/l20-bf16/2026-09-20/README.md) |
| FFN剪枝、恢复、蒸馏（至10月4日） | 冻结20clip参考：教师ADE1.232399m，最佳8B 1.299637m，7B 1.329894m，6.797B 1.428902m | 恢复/层间接口/重要性实验完整历史记录在交付包结果归档；6.797B相对教师+15.95%；反复用于选型的开发集 | [路线与冻结参考](../docs/baselines/ALP15-ROUTE-COMPARISON-20261008.md) |
| 全36层FFN低秩敏感性（10月7日） | 324配置；72组配对；能量NMSE改善52/72，34/72组尾部误差恶化 | 局部重建测量，不是完整6B模型ADE；停止统一低rank直接组装路线 | [敏感性报告](../docs/baselines/ALP15-FFN-SENSITIVITY-RESULTS-20261007.md) |
| W4A16 RTN（10月8日） | 8B ADE1.442645m、FDE4.101184m、持久storage9.564990GiB；原始结构RTN ADE1.403494m、FDE3.989972m、storage11.033313GiB | 20clip×3seed；相对各自原精度+11.00%/+13.88%；临时反量化，不是高效部署内核 | [RTN完整结果](../docs/baselines/ALP15-ROUTE-COMPARISON-20261008.md) |
| ModelOpt驾驶AWQ（10月8日） | 8B ADE1.282724m、FDE3.653804m；相8B BF16 -1.3013%，相教师+4.0835% | 真实252层INT4保存/重载；20clip开发结果，不是独立泛化或Orin验收 | [AWQ报告](../docs/baselines/ALP15-AWQ-FULL-20261008.md) |
| 混合M1内核（10月9日） | ADE1.282724m、FDE3.653804m；L20请求mean3204.169ms/max3716.489ms；相配对reference60/60预测坐标一致 | 共享L20描述性计时；60/60请求超300ms；不是Orin或闭环仿真结果 | [Hybrid完整20clip结果](../docs/baselines/ALP15-HYBRID-FULL20-RESULTS-20261009.md) |
| KV/少步采样（10月9日） | KV4 ADE1.245025m、FDE3.595224m、相教师+1.0245%；mean5260.32ms/max6495.00ms | 20clip×3seed×5variant；减少步骤未消除语言生成瓶颈，均未达300ms | [KV完整结果](../docs/baselines/ALP15-KV-FULL20-RESULTS-20261009.md) |
| 精度审计与独立集准备（10月9–13日） | Hybrid/KV4均有7/20clip相对教师退化超15%；60轨迹坐标/来源审计通过；949候选clip下载计划 | 审计通过≠独立精度通过；949为候选计划，不声称下载或模型评测完成 | [收工总结](../docs/baselines/ALP15-DAY-END-SUMMARY-20261010.md)、[独立集状态](../docs/baselines/ALP15-PRECISION-FULL-POOL-DOWNLOAD-ACCEPTANCE-20261011.md) |
| FlashDrive/LightVLA、AlpaSim、Orin | 有研究/适配决策与预检记录，暂无完整独立精度、闭环仿真或Orin实机合格结果 | 不能将论文速度、单算子速度或L20显存外推为Orin部署达标 | [适配研究](../docs/baselines/ALP15-FLASHDRIVE-LIGHTVLA-ADAPTATION-20261008.md)、[仿真预检](../docs/baselines/ALP15-ALPASIM-PREFLIGHT-20261008.md) |

## 查看与还原原始结果的公开副本

阶段报告可直接在 `docs/baselines/` 阅读。结果 JSON、CSV、资源与训练曲线文本按内容去重、gzip压缩分片存放在 `results/packed_evidence/`。

```bash
uv run --frozen python tools/restore_experiment_results.py
```

还原目录：

- `results/restored/local/docs/baselines/evidence/`：近期结果、逐clip/seed指标、资源记录及审计资料。
- `results/restored/local/results/baselines/`：9月20日基准JSON。
- `results/restored/handoff_20261004/compression/results/`：早期剪枝/恢复/蒸馏历史结果及曲线。

所有还原文件逐个核对发布 SHA-256，已有不同内容会拒绝覆盖。压缩包整体也校验 SHA-256。
公开副本删除个人账户路径、IP、邮箱和认证信息；索引同时保留原始与公开副本 SHA，并标识变更。
原始文件不修改。模型权重、张量缓存、原始数据、二进制图表、运行日志和代码快照不在本结果包范围；代码已在此前源码归档。
失败与中间结果保留，不用最优结果掩盖失败；未完成/未验证状态以各阶段记录为准。

## 当前总体判断

目前只有开发集和离线L20证据，尚不能证明“最大精度＋Orin NX16GB合理内存＋完整请求≤300ms”目标达成。
3个随机seed是同一clip的重复测量，20clip不能算作60个独立场景；单clip基准与开发集多阶段计时也不可直接横向比较。
