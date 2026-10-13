# M1混合解码：252实际矩阵桥接结果

日期2026-10-09，Asia/Shanghai；主任务独占。对应D-ALP15-HYBRID-M1-20261008。目标仍为Orin NX16GB、完整请求max≤300ms、教师meanADE损失≤15%，本报告只验收L20算子桥接，不能代替目标。

## 已完成证据

新reference2_v3成功退出0，6条完整轨迹，原helper严格门禁6/6通过。结果SHA256 `02a163ddbadf9aec6c778d36a35e2e7b76515d17b8a384a35e55a9ace75d2356`；本机与服务器一致。原字节动态源码快照、virtual命名空间真实实现身份及reference_final_receipt已在生成进程存活时保存，旧v2临时源码缺失失败不补造、不覆盖。

服务器算子输出：`experiments/jetson_300ms_20261009_hybrid252_v3/result.json`。
本地证据：[result.json](evidence/jetson_300ms_20261009/hybrid252_v3/result.json)。SHA256 `3a26316c87508541b69b96a6571def779ed7cdb088b325a02fa25341c4efe4f3`，服务器/本机一致。

终态 `passed_all252_cuda_m1_probe_with_actual`，exit0，coverage=252/252、actual_activation_coverage=252/252。PID2068050已结束，现场GPU只余原项目PID453201，随后才启动本项目下一作业。完整argv/环境/源码/资产/参考绑定与逐矩阵误差、first/warm/CUDA event、allocated/reserved及dispatch在JSON。参数cuda:0、warmup3、repeats10；来源均新code_v3/reference2_v3，不复用旧v2失效捕获。

## 数值与存储

| 检查 | 实际结果 |
| --- | --- |
| 反量化BF16权重逐元素完全一致 | 252/252，最大绝对误差0 |
| 真实M1输出逐元素完全一致 | 187/252 |
| 真实M1算子容差通过 | 252/252，rtol0.01/atol0.02，超容差元素0 |
| 真实M1最大绝对输出误差 | 0.00390625 |
| 禁用路径逐元素等于原reference | 252/252 |
| M2原reference路径逐元素一致 | 252/252 |
| 逐模块persistent bytes前后不变 | 252/252 |

65个真实M1输出并非bitwise完全一致；算子容差不自动证明驾驶精度或prefix一致。只允许M==1融合，M>1走原reference，不强开大M慢kernel，也未改变packed格式、amax/prescale或保护权重。

## 性能口径

对每矩阵真实activation取10次同步wall中位数，再对252矩阵汇总：原reference矩阵中位数1.143401ms，hybrid中位数0.150979ms；逐矩阵reference/hybrid中位数比值的中位数6.7893倍。这是异形矩阵的描述性算子统计，不是完整请求加速比；不要相加stage峰值、乘252或论文倍数推断Orin延迟。共享L20有原项目训练，首调用/JIT另记，非独立无干扰硬件实验。

## 已启动而未验收

主任务串行启动新 `hybrid_smoke2_v3`，Python PID2071254。命令同code_v3入口，--awq-forward hybrid_m1 --clips2 --steps10 --matrix-probe hybrid252_v3 --paired-reference hybrid_reference2_v3。输出`experiments/jetson_300ms_20261009_hybrid_smoke2_v3`，日志`code_v3/hybrid_smoke2.log`；已收到PID活跃输出，后续SSH读取出现延迟，不据观察超时重启、覆盖或宣称进程失败。仍需当前终态result、exit、6条strict gate/生成token/prefix/noise/64×3/实际M1dispatch和保护前后SHA复核，才能扩评。

Orin实机、全系统统一内存/冷加载、独立测试集、FlashDrive/LightVLA联合适配与完整请求max300ms均未通过；目标保持active。低峰值loader仅资产headers通过，meta尚有能力查询适配待执行，未实体加载。

## 后续终态复核：strict smoke2_v3 已通过

2026-10-09 主任务重新读取原PID2071254及其结果；进程已结束，result状态 `completed_awq_hybrid_development_eval`、exit_code=0。未因之前SSH读取延迟重启。结果SHA256 `4aaa6d68fcfa0ef327542e797ef6dcb52a75dbdbd765d1128973069b747db6cb`，小文件取回 [result.json](evidence/jetson_300ms_20261009/hybrid_smoke2_v3/result.json)，与服务器SHA一致。

实际服务器以冻结v3源码调用 `validate_smoke(d,d,d['all_frozen_clips'])`，重算6条64×3配对及capture/probe/来源绑定，输出 `STRICT_SMOKE_RECOMPUTED_PASS`。strict与original helper均6/6，所有轨迹NMSE/ADE差/max坐标差均0，生成token、noise、prefix及future_start完全配对；每请求252模块均有实际M1分派，保护摘要前后均 `94b2256c7e277915c3b570c6ba941aaf49b56dfa4094b2adac06b70a394093bb`。persistent storage仍10,272,486,512B。只余其它项目PID453201占15428MiB，本任务未停止它。

两clip×3seed均值ADE/FDE与reference2_v3相同：1.232741m / 3.205859m。E2E mean2995.606ms、p50 2916.832ms、max3377.170ms、6/6超过300ms；reference对应mean5503.289ms。这是共享L20的先reference后hybrid描述性对照，非随机交叉无干扰测速，也非Orin结果。计时范围为rawimages→processor/historyfuse→generate→fullprefill→Expert→轨迹，磁盘读取另记且不包含，最终部署还须计入必要输入解码/传输。

已准备 `scripts/run_alp15_hybrid_full20_v3_20261009.sh`，bash语法检查exit0；此记录时尚未启动，待独立验收。用户进一步明确要求开始FlashDrive/LightVLA适配；新决策将分离两个机制侧并锁定联合消融，不将本次smoke精度等价宣称满足全部性能目标。

随后独立验收者批准原决策范围内的20段配对。2026-10-09T04:59:51Z（12:59:51 CST）已启动新launcher2084459、reference Python2084464；40秒后 `ps` 仍分别S/Rl，现场日志已写START。新launcher SHA `38a99baaec69ed70adc681a9e26ba92e0677e2cd78617043302c61a1c37784cd`，远端与本地源码冻结身份核对一致。远端 `code_v3/full20_pipeline.log` 顺序保留reference20及hybrid20退出码；同冻结10步/fullprefill、相同20段/3seed，reference失败不会启动hybrid。当前只证明已启动且进程live，不能提前宣称60/60通过。源码及产物使用新路径，不覆盖旧smoke/参考/算子证据。
