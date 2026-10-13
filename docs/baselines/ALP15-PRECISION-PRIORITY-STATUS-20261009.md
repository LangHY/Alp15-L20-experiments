# 精度优先：当前执行状态与数据边界

2026-10-09，主任务唯一写入。用户先要求FlashDrive/LightVLA适配，随后明确“先完全验证精度”；最新要求优先。最终Orin NX16GB、完整请求≤300ms、教师meanADE损失≤15%不变，目标仍active。

## 当前执行

保留已启动的冻结v3 fullprefill10严格配对，不修改运行源码。2026-10-09T04:59:51Z启动launcher2084459、reference Python2084464；2分11秒时实际ps显示S/Rl，日志已出现首条reference轨迹。输出为 `experiments/jetson_300ms_20261009_hybrid_reference20_v3`，参考完成且exit0后才会启动 `jetson_300ms_20261009_hybrid_full20_v3`；完整命令在 `scripts/run_alp15_hybrid_full20_v3_20261009.sh`，远端日志 `jetson_300ms_20261009_hybrid_m1_code_v3/full20_pipeline.log`。此时不是终态验收。

混合smoke2_v3的6条strict已现场重算，轨迹/前缀/token/noise一致；这只支持实现等价，不能证明泛化。独立验收在 `ALP15-HYBRID-SMOKE2-ACCEPTANCE-20261009.md`。

暂停新FlashDrive/LightVLA机制的实施和GPU评测，仅保留适配事实与候选决策。低峰值loaderv4经独立审查有输入字节预估和冷入口门禁缺口，未启动真实meta、未实体加载；证据保留。优先安排精度审计及独立评测方案，不为了新机制继续改当前候选。

## 实际数据盘点

实际服务器以既有venv的stdlib JSON/ZIP与pyarrow读取元数据，不加载模型、不下载数据：

- `datasets/physicalai_chunk0000_20261002_165238` 为8.0G，5个ZIP覆盖100clip（egomotion100成员、每camera300成员）。
- `training_partition.json` 明确80training、20monitor_validation，seed20261002、t0_us5100000、固定dataset revision `33f9bf447ed3bcb7d545ce13f4226f824214fafb`。这批训练/监控数据不能重命名为独立测试集。
- 原开发集teacher_prefix_cache实际20文件，已反复用于候选选择；其3seed产生60请求不是60独立场景。
- 原恢复目录的旧split含4calibration、2validation、4holdout，注明chunk3119内部划分；仅文件名holdout不能证明历史从未参与选择。
- `metadata/data_collection.parquet`字段为country、month、hour_of_day、platform_class、radar_config、clip_id，没有drive/session字段。按clip分离不等于已证实按行程独立；现阶段必须报告这一限制，不能猜行程标识。

后续应冻结现候选、输入/教师/采样口径，先完整检查开发集逐请求与逐clip误差，再选择未参与训练、校准、监控或方案筛选的新数据。独立测试样本名单不得看候选误差后调整；大数据仍先核查国内来源并给用户下载命令，主任务不自行下载。样本量与置信区间及失败处理由新精度决策事先锁定，不凭固定数量宣称“完全验证”。

## 后续全局元数据核验：无需再次下载

主任务继续以服务器既有venv/pyarrow只读三个文件的parquet row count、实际文件长度及SHA256，命令exit0。路径前缀为 `datasets/physicalai_chunk0000_20261002_165238/files/`；虽然交付的ZIP只含100clip，但三个全局parquet实际各有**306152行**，不是100行子集。

| 文件 | bytes | 实际SHA256 |
| --- | ---: | --- |
| clip_index.parquet | 11132300 | 9630a3de5ee057ded194deaa94d3b9b42af6f18c301785706e50141314a6935c |
| metadata/feature_presence.parquet | 11209157 | a9946feb269124e3a7cbfe22d97669585735134f634c2ca97332e7abc4314790 |
| metadata/data_collection.parquet | 11339400 | 9624750cadabf8028264cc1404e75e81777654be9def06b00e35c9f834b173d5 |

三项实际字节SHA与固定官方revision文件信息逐一相同（见 `ALP15-PRECISION-DATA-DOWNLOAD-SOURCES-20261009.md`）。因此下一步可在原服务器复用现元数据开展完整usage排重和新名单筛选，不要求用户再次下载这约33MB元数据。视频五ZIP仍未选择或下载，必须先冻结名单再查询实际大小并给用户命令。元数据完整不等于usage账本完整，更不等于已经有未见输入窗口；drive/session字段缺失限制仍保留。

随后资格清点命令exit0：原clip_index及feature_presence索引唯一，clip_is_valid=True且egomotion/四camera均存在的clip为306152。**仅排除已知chunk0/3119后**为305952clip、3144chunk；实际官方split值及计数为train153425、val90928、test61599。这是eligibility census，不是完整usage排重或抽样结果。已有split字段可用于优先规划官方metadata test子集；它不是已证明符合Alpamayo专门评测协议，也不补足缺失drive ID。任何实际名单仍须完整历史审计、输入窗口QC和采集前决策，不能把上述余额直接称独立数据已就绪。

新决策 `D-ALP15-PRECISION-VALIDATION-20261009-01` 已完整读完，文件 `ALP15-PRECISION-VALIDATION-DECISION-20261009.md` SHA为 `989b64ac1adfa3979556db82d035f71230d91132454b8eb35029eafacf0280db`。纯stdlib四JSON审计执行者已接收三个新文件所有权，正在实现；用户新机制保持暂停。本轮不计算既有数据的验证性p/CI，教师缺GT/FDE明确保持缺失，不制造指标。

## 后续更新：开发证据审计与未见数据准备

以下为后续状态，保留上文各次观测的历史时点，不把早期运行中状态当成现在仍在运行。full20参考及hybrid流程已结束、退出0，实际60条配对轨迹逐坐标exact；详情见 `ALP15-HYBRID-FULL20-RESULTS-20261009.md`。这仅支持该实现对既有参考的等价性，不证明教师相对精度泛化。

正式JSON审计v1保留；主任务在Python3.9独立回归发现NMSE求和舍入差，v2仅改用math.fsum，未放宽模型门禁或修改输入。冻结脚本SHA `bb4ca945da96918ccf8604003e4973c5cf669b10b96c768db232b3100f5f6ea7`，v2审计JSON SHA `d8d3ba55842659fd7a986c26ba2d98ebc3a80183ff3f6336b72a3226cf35d6fc`。主任务实际命令 `/usr/bin/python3 -m unittest discover -s tests -p test_audit_alp15_precision.py -v`，16tests、9.671s、exit0；工具session7214、最终chunk7da45b。执行者另在本机Python3.12.10测试/实际CLI通过，独立验收正在复核；不是远端模型重跑。产物见 `evidence/precision_20261009/development_audit_v2/`，完整失败与修复历史见精度审计执行回报。

已有20clip开发证据：教师meanADE1.2323994789272548m；hybrid/fullprefill10 mean1.282724219933152m（+4.0835%），KV4 mean1.2450251077612242m（+1.0245%）。两者均7/20clip相对ADE退化超过15%，作为尾部诊断，不把逐clip阈值替换整体主门禁。每个候选60请求为20clip×3seed，不能当作60独立样本；多候选表480请求/160clip条目也不是480/160独立数据。没有GT、教师FDE或未见数据，不能宣布完全精度验收通过。

未见数据资格准备v1的内部Features调用在实施前被发现违反用户顶层API约束，原决策冻结不改；v2修订SHA `32b534ea574677928fca9c6e9277498807f00839fe8de852f14f01d7a9e7a23b`已由主任务完整阅读。当前仅批准脚本实现与CPU测试：使用校验后的CSV原始chunk_path做纯数据解析，明确没有调用官方API，不导入内部类或伪造联网接口。扫描显式历史roots、保守排除所有引用clip的整chunk、记录未解决资产，coverage_complete保持False待独立范围验收；不读取PTpayload、不安装环境、不启动下载或GPU测试。执行者拥有准备脚本、对应测试与新执行回报，不拥有本状态文档或已冻结审计源码。

后续完整验证还必须先审定usage范围及新名单、固定采集前统计口径，然后对新raw输入分别运行教师与候选自主生成，并保存GT及其hash、教师/候选预测与ADE/FDE。已有runner的GT来源为data['ego_future_xyz']，当前结果只保存预测和误差，不能从两个预测反推出GT。新名单输入窗口QC、下载规模、统计样本量与闭环/Orin验证均未完成；新FlashDrive/LightVLA实施继续等待精度优先阶段。

### v2独立验收后续发现：工具失败留痕待修复

独立验收者复跑双Python均16tests通过、真实冻结输入CLI通过，但新增结构错误探针发现teacher.records=null（对应输入SHA同步更新）会在len(None)后进入失败库存处理，又因enumerate(None)二次异常导致输出目录为空。虽然退出1，未达到失败报告留痕契约，因此不能宣布v2工具全部验收通过。独立验收工具日志chunkd49140；无真实模型输入变更。

主任务已明确授权唯一审计执行者做v3最小修复：先真实CLI结构错误red，再增加records列表结构检查与安全失败库存，双环境回归与新v3产物。v1/v2保持原样，阈值/模型/源数据不动。准备脚本暂停在读取决策与包身份检查边界，三个准备文件尚未创建，无下载或远端操作。v3修复及独立复验后继续准备census。另跨Python statistics.stdev存在约2.22e-16的描述SD末位差，不宣称全部CSV跨版本字节exact；该差不影响冻结NMSE门禁、均值/ratio或配对坐标。

## 下一轮只读服务器范围预检

通过已批准SSH连接在指定服务器运行既有venv的`python -B -c`，仅Path目录/JSON文件计数及importlib.metadata发行版查询；工具chunka76036/d7c917，exit0，无GPU/模型导入/文件payload/下载/安装。实际pandas2.3.3、pyarrow22.0.0、physical-ai-av0.2.0；本机测试依赖版本不能替代这些远端实际版本。

显式范围预检存在：experiments（1152 JSON）、datasets（3）、docs/baselines（0）、scripts（0）、compression（7）；还发现logs（19）、incoming（0）、alp15-test-clip-delivery-20260917（1）、alp15_l20_audit_20260920_094859（0）、Self-realization（0）。完整路径都在`/home/USER/lang/Lighten_VLA/`内。JSON为0不等于没有引用，仍需按已批准文字类型扫描；目录存在与计数不是usage内容或coverage完成证明。后续census应将这些补充路径显式纳入或逐一记录覆盖缺口，不能只扫原五root就宣布所有历史已排除。尚未读取这些JSON内容或开始census。

### 工具v3与文件所有权后续调整

v3解决了四输入×六records错误结构的24个实际CLI负例，双环境17tests通过且坏null输入能保留六失败产物。v3脚本SHA `c3f34af61db9b201ff074c5be09c4df78da450302906b1c410ff28ac21a40faa`、正式audit SHA `072b98a033c01e4e8def97472804a0248d813c53d812ebea15f04e537b34dfa2`保留，不静默覆盖。随后主任务CPU最小探针`number(10**400,...)`确认math.isfinite抛OverflowError，日志chunk10cc76、exit1；已批准v4只局部将不能表示为有限浮点的数值转为AuditError，并新增同步SHA实际CLI负例。模型正常结果、四真实输入SHA、精度门限不变；v4待冻结与独立复验。

数据准备与审计修复按文件边界并行：precision_audit_execution_20261009只拥有审计script/tests/执行回报；原prep三文件仍未创建时，主任务已撤销其prep所有权，转交precision_decision_fresh_20261009实现准备script/tests/独立执行回报。该执行者仅本机CPU合成parquet回归，无网络/服务器/GPU/PTpayload/安装；主任务以后再上传冻结代码进行远端CPU验证。独立验收者等待v4冻结，不能把正在变化的源码当作旧版本再验收。任何通过声明仍限工具契约，不等于完全精度通过。

远端既有venv包身份另以importlib.metadata定位、只读源码字节/AST核验（chunk9b54b8、exit0，不import执行physical_ai_av）：version0.2.0；dataset.py SHA `b35f5f1bbb42ba92c83c835f7525c33e8ee915eb5ef071ac3dd972d91b0d9560`；__init__.py SHA `40f7c62c601c74f9b44460745a137db8d4fafdff7d9719b402e962e6d97c22f0`；__all__确为HfRepoInterface、PhysicalAIAVDatasetInterface，不包含Features。与v2准备决策来源一致。
