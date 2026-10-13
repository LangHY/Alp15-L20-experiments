# 混合M1：20段完整严格配对终态

2026-10-09，主任务唯一写入。对应D-ALP15-HYBRID-M1-20261008；仅同冻结code_v3/fullprefill10，不启用FlashDrive速度缓存或LightVLA筛选。用户当前要求先完整验证精度。

## 终态与证据

服务器pipeline记录 `TERMINAL hybrid20_v3 2026-10-09T05:17:11Z exit=0`；重新SSH读取旧launcher2084459/reference2084464，ps无该PID。中途旧SSH socket/临时trust丢失，未重启GPU流程；重新取得的ED25519指纹与用户已确认指纹一致，使用新隔离known_hosts并保持strict checking，不改系统known_hosts。

本地两个完整小JSON均可解析，status=`completed_awq_hybrid_development_eval`、exit_code=0、每侧60条completed。文件SHA：

- [reference20 result](evidence/jetson_300ms_20261009/hybrid_reference20_v3/result.json)：`e83ef209381c6e166e862795e65fc21681528fe77d90c3bf48bd4d9e048e3b59`。
- [hybrid20 result](evidence/jetson_300ms_20261009/hybrid_full20_v3/result.json)：`a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6`。
- [教师开发集原结果](evidence/precision_20261009/teacher_development_result.json)：`67b328a1ecac9de414b47680ef6476c96f6cb30dab0c736a500a902970cebaa0`，已独立核对等于hybrid结果中绑定的教师input SHA。

参考JSON SCP尾部及服务器再次逐条重算/SHA查询曾暂未返回；以上是本地实际SHA与生产结果/现场pipeline终态，不冒称新远端查询已完成。后续继续读取原观察handle，不因超时重跑。

随后原观察handle40874返回exit0：实际服务器冻结v3 `strict_pair_gate` 重算60/60通过且60/60坐标差为0；`assert_same_provenance`通过，四个前后保护摘要相等、每请求252模块真实M1分派检查为True。服务器三文件SHA与上述本地SHA逐一相同。SCP原handle23804也正常退出0，没有重跑或覆盖。

## 本地逐请求复核

以stdlib只读两个实际JSON，重新检查每侧20clip×seed42/43/44=60唯一键、completed、64×3/finite、exit0；code/runtime/input/steps/request_route/text_seed/trajectory_seeds完全一致。逐条比对noise/prefix/generated_token_ids/future_start/prefix_length/decision/steps，全部相同。**60/60预测坐标逐元素完全一致**，不只是读取保存的strict passed字段。生产strict门禁60/60通过，保护摘要前后一致；来源/实际M1dispatch还由独立验收者及现场绑定器继续复核，不用本次字节比较替代全资产验证。

| 指标 | reference | hybrid |
| --- | --- | --- |
| mean ADE m | 1.282724220 | 1.282724220 |
| 同clip/seed教师 mean ADE m | 1.232399479 | 1.232399479 |
| 相对教师 ADE变化 | +4.0835% | +4.0835% |
| mean FDE m | 3.653804187 | 3.653804187 |
| clip ADE p95 / max m | 2.504728 / 3.317172 | 2.504728 / 3.317172 |
| clip优于/差于教师 | 10 / 10 | 10 / 10 |
| 请求耗时mean / p50 / max ms | 6385.769 / 6519.787 / 7581.773 | 3204.169 / 3156.781 / 3716.489 |
| 请求超过300ms | 60/60 | 60/60 |

共享L20先reference后hybrid，耗时是描述性观测，非随机交叉无干扰速度结论；计时磁盘输入等另列，不能外推Orin完整请求。

## 当前结论

混合M1实现的20段开发回归未增加该批请求的精度损失；当前学生开发集meanADE损失4.08%在15%以内。**这不是“完整精度验收”：**20段反复参与选择，3seed不是新增独立场景；独立未见测试集、尾部/分场景/失败核查、统计不确定性和实际目标设备验证仍需执行。FlashDrive/LightVLA新机制暂不实施，候选权重与当前接口冻结，先完成独立精度方案和数据准备。
