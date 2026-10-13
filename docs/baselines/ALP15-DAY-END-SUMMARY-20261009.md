# 2026-10-09 工作收尾与续接

用户要求“总结一下今天就到这里”。长期目标已设为paused：Jetson Orin NX16GB硬件约束、完整请求延迟不大于300ms、相对教师精度损失在15%以内。已中断本任务执行/准备子智能体；不再启动实验、下载或优化。未改为完成，未停止其它服务器任务。

## 今天确认的结果

以现有20clip开发集、每clip三个轨迹种子为口径；多种子是重复测量，不是60个独立场景。

|候选|mean ADE m|相对教师ADE损失|逐clip退化超过15%|
|---|---:|---:|---:|
|教师离线参考|1.232399479|—|—|
|Hybrid fullprefill10|1.282724220|+4.0835%|7/20|
|actualKV4|1.245025108|+1.0245%|7/20|

Hybrid与同权重参考的60条预测坐标逐元素exact，五路来源/保护摘要/实际252模块分派已核查。它证明该实现对这个参考的等价性，不证明相对教师的独立精度或泛化。两个候选分别评测，未组合实施。学生FDE有旧记录，教师FDE与GT未保存在输入JSON，不能从中独立重算GT误差。

Hybrid在共享L20上的请求耗时mean3204.169ms、p503156.781ms、max3716.489ms；60/60请求超过300ms。该结果不能外推Orin，也没有目标设备16GB统一内存与300ms的实机验收。因此长期目标未达成。

## 工具与数据准备

- 精度审计修复了跨Python求和、records结构错误失败留痕和超大整数异常。停止时主任务既有测试进程已正常结束：Python3.9.6，18tests，18.438s，exit0（session31352、chunk943738）；没有终止或重跑该已完成测试。
- v4正式审计目录已存在，run.log记录completed_development_audit、diagnostics=[]、exit0。v4独立验收尚未完成，执行回报收尾尚未确认；不能把工具测试通过写成完整精度通过。v1/v2/v3与原模型输入保留。
- 未见数据准备脚本及测试已落盘，但执行者在用户收尾时被中断；未完成验收、远端CPU正式census或新名单冻结。当前代码是待续接状态，不能直接当作可执行交付。
- 服务器已有306152clip的全局元数据，不需重复下载。使用历史须覆盖experiments、datasets、compression及logs/旧交付等显式目录；尚未完成内容级台账，不能把官方test标签直接当成独立测试资格。
- 未启动新大文件下载、新模型训练、FlashDrive/LightVLA实施或Orin测试。FlashDrive/LightVLA适配继续等待精度优先阶段。

## 停止时文件身份

|文件|实际SHA256|
|---|---|
|scripts/audit_alp15_precision.py|85d978f4b9ebbadb55c168a9539af84316d95d2c1521305c6f5d9af3e34c6341|
|tests/test_audit_alp15_precision.py|dac4285a8f0f4874cde9b172deb03e65b119d78e2b96b0af58443e2ddea9e78a|
|scripts/prepare_alp15_precision_holdout.py|957e11d47f973437886636811728c8435d275c7ca0a04ac1b0c25c616783a7fe|
|tests/test_prepare_alp15_precision_holdout.py|66968679a0fc982e68e08aa0c62d946dc0f6584d1311251b4f50920fdcc93821|

SHA为停止时实际读取，不替代执行回报、完整产物manifest或后续独立验收。

## 下次续接顺序

用户明确恢复后，先检查代码/进程和SHA，而不是重启旧实验：完成v4执行回报与独立复验；审查并完成准备脚本CPU测试，在既有L20环境运行只读census；审定历史覆盖范围、冻结未见名单与采集前统计口径；核查实际下载容量并给用户命令（不代下载大文件）；对新raw输入分别运行教师与候选自主生成，保存GT/hash和预测轨迹，核算ADE/FDE、尾部/分场景/失败与不确定性。之后才决定是否继续FlashDrive/LightVLA。闭环和Orin实机验收仍须独立完成。

证据入口：ALP15-HYBRID-FULL20-RESULTS-20261009.md、ALP15-PRECISION-AUDIT-EXECUTION-20261009.md、ALP15-PRECISION-AUDIT-V2-ACCEPTANCE-20261009.md、ALP15-PRECISION-PRIORITY-STATUS-20261009.md、evidence/precision_20261009/development_audit_v4/。
