# ALP15 精度验证决策单

- 决策编号：D-ALP15-PRECISION-VALIDATION-20261009-01。
- 日期：2026-10-09（Asia/Shanghai）；决策者：GPT-6.1 Sol high，按用户当前模型安排执行。
- 当前授权：用户最新要求“先完全验证精度”，暂停 FlashDrive/LightVLA 新实施。既有优化授权与离线参考例外见 `ALP15-AWQ-FULL-20261008.md`，不据此改写 NVIDIA 官方原入口或 Jetson 门禁。
- 目标：先审计已有精度证据、冻结候选与评测口径，再准备未参与选择的新数据。最终目标仍为 Orin NX 16GB、完整请求最大耗时 ≤300 ms、相对教师 mean ADE 损失 ≤15%；这些目标当前未完成。
- 风险：精度结论与数据选择为高风险；本轮批准的执行仅为本地 CPU 标准库只读审计，不涉及模型运行、SSH、网络、权重转换或新训练。
- 文件所有者：本决策仅由精度决策者写入；脚本、测试、执行回报由主任务另分配同一执行者；同文件不得并发写入。

## 1. 已核对的事实与证据范围

已完整读取 `AGENTS.md`、`ALP15-PRECISION-PRIORITY-STATUS-20261009.md`、`ALP15-HYBRID-FULL20-RESULTS-20261009.md`、`ALP15-AWQ-FULL-20261008.md`，并补读 `ALP15-KV-FULL20-RESULTS-20261009.md`。实验设计与统计分析技能及其计划/报告模板已完整读取。本决策不用历史结果伪造采集前预注册。

官方身份沿已记录证据冻结：Alpamayo 1.5 源码提交 `36aeb4c5938cbc2eb2aed33b22434773da4ab639`，checkpoint revision `7aba8293c09993f2e125c6819df05d7fa3e873ea`，PhysicalAI 数据 revision `33f9bf447ed3bcb7d545ce13f4226f824214fafb`。来源可从 NVIDIA [Alpamayo 官方仓库](https://github.com/NVlabs/alpamayo)及本地结果的 `source`、`code_sha256`、`runtime_source_sha256`、输入摘要追溯；本轮不新增联网核验。既有运行 `source.dirty_status` 有兼容修改记录，不能称 NVIDIA 官方原入口原样验收。

以下 SHA256 已于本决策编写时以本地实际文件字节重算。路径均相对仓库根目录：

| 输入 | SHA256 |
| --- | --- |
| `docs/baselines/evidence/precision_20261009/teacher_development_result.json` | `67b328a1ecac9de414b47680ef6476c96f6cb30dab0c736a500a902970cebaa0` |
| `docs/baselines/evidence/jetson_300ms_20261008/kv_candidate_full20_v1/result.json` | `db04c2ed3226e2c3c988c3f6ebd9218bd6150759d3334d25d7e193c7693ef84e` |
| `docs/baselines/evidence/jetson_300ms_20261009/hybrid_reference20_v3/result.json` | `e83ef209381c6e166e862795e65fc21681528fe77d90c3bf48bd4d9e048e3b59` |
| `docs/baselines/evidence/jetson_300ms_20261009/hybrid_full20_v3/result.json` | `a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6` |

教师 JSON 共 240 条，四组 `(model,prefix)` 各 60 条：`(dense,teacher)`、`(student8,teacher)`、`(student8,student8)`、`(dense,student8)`。唯一教师比较组明确为 **`model=dense AND prefix=teacher`**；不能把使用学生前缀的 dense 组混入教师。该教师使用当时冻结的教师生成前缀/full-prefill，不等同于新测试的双方自主生成设计。

现报告给出的开发集 teacher mean ADE 为 1.2323994789272548 m，fullprefill10 与 hybrid 为 1.282724219933152 m（+4.0834763294%），actualKV4 为约 1.245025 m（+1.0245%）。hybrid/reference 60/60 预测坐标逐元素一致。以上仍须由下面的独立审计从实际记录复算；决策单不把报告数字当审计替代。

教师旧 JSON 没有 GT 轨迹与 FDE 字段；现 JSON 审计只能核验已有 ADE 字段绑定和数值完整性，不能独立重算对 GT 的 ADE/FDE。学生 FDE 可描述，教师 FDE、相对教师 FDE 必须输出 `null` 与缺失原因。不能把学生—教师轨迹终点距离命名为 ground-truth FDE。

## 2. 本轮立即批准：已有开发证据的只读审计

### 2.1 执行边界与文件所有权

主任务现在可分派一名 GPT-6.1 Sol high 执行者，实现：

- `scripts/audit_alp15_precision.py`：纯 Python 标准库，禁止导入 torch/numpy、加载 pickle/PT、模型、网络或执行子进程远程访问。
- `tests/test_audit_alp15_precision.py`：标准库 unittest，测试输入校验与公开 CLI 行为。
- `docs/baselines/ALP15-PRECISION-AUDIT-EXECUTION-20261009.md`：独立执行回报。该路径不存在时创建；若已存在则先升级主任务选择新回报路径，禁止覆盖。
- 新输出目录 `docs/baselines/evidence/precision_20261009/development_audit_v1/`：不存在才能创建；存在即 CLI 失败，不以 `--force`、自动续写或清空目录规避。输出不会覆盖输入，输入 SHA 审计前后必须相同。

允许读取上述四份 JSON、相关本地冻结脚本和报告；不得改写既有证据、模型运行脚本、候选、缓存、数据名单、venv。审计输出标记 `analysis_scope=exploratory_reused_development`；本轮不执行 p 值、CI、功效分析或事后“预注册”。

### 2.2 CLI 契约与冻结命令

CLI 必须显式接受 `--teacher`、`--kv`、`--reference`、`--hybrid` 及各自的 `--*-sha256`，另接受 `--output`。不得从任意文件发现后自动挑选“最好”结果；本命令的四个路径和哈希为冻结输入。脚本须保存完整 argv、UTC 起止时间、Python/平台、脚本和本决策 SHA，以及各输入解析前后 SHA。

```sh
python3 scripts/audit_alp15_precision.py \
  --teacher docs/baselines/evidence/precision_20261009/teacher_development_result.json \
  --teacher-sha256 67b328a1ecac9de414b47680ef6476c96f6cb30dab0c736a500a902970cebaa0 \
  --kv docs/baselines/evidence/jetson_300ms_20261008/kv_candidate_full20_v1/result.json \
  --kv-sha256 db04c2ed3226e2c3c988c3f6ebd9218bd6150759d3334d25d7e193c7693ef84e \
  --reference docs/baselines/evidence/jetson_300ms_20261009/hybrid_reference20_v3/result.json \
  --reference-sha256 e83ef209381c6e166e862795e65fc21681528fe77d90c3bf48bd4d9e048e3b59 \
  --hybrid docs/baselines/evidence/jetson_300ms_20261009/hybrid_full20_v3/result.json \
  --hybrid-sha256 a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6 \
  --output docs/baselines/evidence/precision_20261009/development_audit_v1

python3 -m unittest discover -s tests -p test_audit_alp15_precision.py -v
python3 -m py_compile scripts/audit_alp15_precision.py tests/test_audit_alp15_precision.py
```

正式运行执行者先跑测试，再运行审计；执行回报保存实际命令、退出码和完整日志。失败时保存诊断，不修改输入凑数。日志可由脚本输出到新目录，也可由主任务保存工具会话；不要在运行前用 shell 重定向创建 output 目录破坏拒覆盖契约。

### 2.3 必须实现的完整性与同源检查

1. 拒绝非法 JSON、重复对象键、NaN/Infinity、缺失必填字段、bool 冒充数字、非 finite 或负误差。每份输入 SHA 必须精确匹配预期。
2. 教师四组身份必须各为 20 clip × seed 42/43/44 的 60 唯一键。只取 dense/teacher 做教师绑定；其他三组保留计数和排除原因。教师无需不存在的 `exit_code` 或每行 `status`，不能补造；顶层实际 `status=completed`，缺失退出码另记证据限制。
3. KV 顶层 `status=completed_kv_steps_development_probe`；300 条必须精确覆盖五个 variant：`fullprefill_10steps`、`actualKV_10steps`、`actualKV_6steps`、`actualKV_4steps`、`actualKV_2steps`，各 60 条、每行 completed、steps 与 variant 相符。KV JSON 不含 exit_code，单列 `not_embedded_in_json`；报告/pipeline 的退出证据不能伪填 JSON。
4. reference/hybrid 顶层均 completed、exit_code=0，各 60 条、每行 completed、相同 20 clip ×三 seed、steps=10、request_route=fullprefill。clip 集合与教师、KV 必须一致；以完整 `(clip_id,seed,variant)` 或教师 `(model,prefix,clip_id,seed)` 键索引，拒绝重复/缺失/额外键。
5. 所有预测 `pred_xyz` 是 64 行、每行恰好 3 个 finite 数。ADE/FDE/e2e 等可用标量须 finite、范围合理（误差和时间非负）；逐条绑定 `teacher_ADE_m` 到 dense/teacher 同 clip/seed `ADE_m`，精确相等，不能仅检查整体教师均值。初始噪声 SHA 同键配对；教师前缀与学生自主前缀可以不同，不把它们不同误判为来源失败。
6. reference/hybrid 逐键独立比较 `pred_xyz` 全部数值、ADE/FDE、noise、prefix_sha256、prefix_length、generated_token_ids、future_start_token_id、decision_id、steps。坐标 exact 必须来自实际数组比较，不只信 `pred_xyz_sha256` 或 strict_pair_gates 保存的 passed。生成 token IDs 为非空整数列表，边界/prefix 字段必须存在并类型正确。记录每键的 exact、最大绝对坐标差和配对字段结果。
7. 核验 reference/hybrid 的共同冻结 `code_sha256`、`runtime_source_sha256`、`input_sha256`、源码/checkpoint/dataset 身份、text_seed/trajectory_seeds、保护前后 SHA 四值相等。动态临时源的路径差异依照 `stable_identity_sha256` 与实际源 hash 比较；真实不同的代码/输入 hash 不能以 basename 或路径规范化吞掉。输入摘要必须实际绑定教师文件 SHA；hybrid 的 reference gate provenance 必须绑定本次 reference JSON SHA。
8. hybrid_install coverage=252/252，252 唯一 target_names、enabled=True、persistent_dense=False；每个请求 dispatch.module_count=252，modules 的目标集合与 install 一致，每个模块 m1_fused>0；计数为非负整数，totals 须按 modules 实际求和吻合。reference enabled=False，目标完整，dispatch 实际 reference 路径可检查但不要求 m1_fused>0。不能只读一个保存的 True；也不能把当前本地 JSON 来源一致当作对远端全部权重重新字节核验。
9. 对已有 strict_pair_gates 与 original_helper_gates 做数量、键、布尔和有限数值核验，并与实际比较交叉核对。actualKV 的数值差异是已有候选特征，不能沿用 hybrid 的 exact 验收豁免、也不能因其非 exact 就丢弃不利记录。

### 2.4 聚合、异常与输出

每候选每 clip 先对三个 seed 取算术均值：`S_i=mean_s ADE_student(i,s)`、`T_i=mean_s ADE_teacher(i,s)`。整体 `S=mean_i S_i`、`T=mean_i T_i`，唯一主要相对损失为 `R=S/T-1`。**禁止把单 clip ratio 的平均值作为整体 ADE 相对损失。** 若 T_i=0，逐clip ratio 输出 null并说明；若整体 T=0 则 ratio 无定义，审计失败。所有20clip等权，3seed 是重复测量而非60独立场景。

输出 `audit.json`、`per_request.csv`、`per_clip.csv`、`report.md`、`input_manifest.json`，必须列完整所有候选/clip/seed，不只列优胜或最差项。per_clip 含 teacher/student ADE、绝对差、ratio、3seed 范围/SD、学生 FDE、失败/配对状态。report 给各候选 mean/SD/median/IQR/p95/max（clip层级），ADE 改善/持平/退化数、超过15%的clip数量及完整退化表、请求/clip失败率和分母、非finite/生成边界/缺失/重复等诊断；p95 使用排序后 `(n-1)*q` 线性插值。

尾部比较列教师与学生 clip ADE p95/max及其差；学生 FDE mean/median/p95/max仅描述，teacher FDE=null。如提供分场景表，只能使用实际 metadata 字段，并列样本数；无 metadata 时标记缺失，不能从 clip 名称或误差猜场景。输入失败、异常和尾部样本全部保留，不能删去离群值改善均值。完整性门禁失败应输出失败审计与非零退出码；失败数据的部分描述可列但不得给达标结论。

轨迹间距离可选作为额外诊断，必须命名为学生—教师预测差，并说明没有GT；不是ADE/FDE重算。本轮不得输出验证性 p/CI，不得称泛化、安全或“完全精度验收”。

### 2.5 最低测试与验收

真实冻结四文件审计必须成功且输入前后字节 SHA 不变；报告可复算上文 teacher/hybrid 均值并验证 60pair exact。负例至少覆盖：输入 SHA 不符、重复 JSON key、同variant重复请求/缺一个 seed、错教师组、teacher_ADE错绑但整体均值相同、64×3错误、NaN/Inf、非法数字类型、prefix/noise/token差、保护/代码来源变化、漏一个dispatch模块或伪totals、输出目录已存在、整体ratio误用平均clipratio。合成测试须与真实文件测试区分，不用仅镜像实现的测试宣称已审计。

执行回报记录 CPU/OS/Python、无GPU运行、实际变更文件及其SHA、四输入SHA、决策SHA、命令/退出码/日志/输出产物SHA、逐门禁结果、失败尝试和限制。由独立 Sol 验收者复核脚本、负例与实际输出后，才可称“已有开发证据审计通过”。

## 3. 下一阶段冻结候选与数据：允许准备，不自动启动GPU

候选分别冻结为：A=`actualKV_4steps`（既有KV v3路线），B=`hybrid_m1/fullprefill/10steps`（冻结hybrid code_v3），另保留对应 fullprefill10 packed reference。A 和 B **尚未组合实现或联合验收**；不能把 KV4 精度、hybrid速度与低峰值loader拼成一个已验证系统。冻结包含源码/运行依赖/候选权重manifest及各tensor payload/数据revision/预处理/文本生成参数/采样参数，不换后端、不重新AWQ校准、不继续训练。

chunk0000 已知80 training+20 monitor_validation；旧20开发clip反复选型；旧chunk3119也历史使用。以上及其校准/派生缓存全部排除新独立测试资格，不能重命名 holdout。主任务先构建使用记录台账：训练、校准、监控、开发、调参、冒烟、仿真、历史holdout实际使用；不能只按某份split文件名判断“未见”。

新名单只按冻结metadata与历史使用记录筛选，在看候选误差前归档。metadata目前仅 country/month/hour_of_day/platform_class/radar_config/clip_id，缺drive/session ID；可称“未参与模型选择的clip集合”，不能声称行程独立。尽量跨未用chunk、已有元数据时段/国家/平台覆盖，同时检查相邻片段或重复输入；unknown关联必须披露。数据不可用/损坏须在模型误差不可见时按预定QC与候补顺序处理并留痕，不能按误差挑替补。

主任务于本决策编写期间另行只读核验：服务器 `datasets/physicalai_chunk0000_20261002_165238/files` 中已有三份**全量**metadata，各306152行，并已核验大小/SHA对应官方冻结数据revision；不是100clip子集。可立即复用这些现存文件准备 usage ledger/候选名单，无需下载metadata；完整SHA由主任务在准备执行回报登记，不从本决策收到的短摘要补造完整值。`clip_index` 实际列为 clip_is_valid(bool)、chunk(int64)、split(string)、clip_id(string)；只选择 clip_is_valid=True 且 `feature_presence` 五项 egomotion、camera_cross_left_120fov、camera_cross_right_120fov、camera_front_tele_30fov、camera_front_wide_120fov 完整的clip。保留官方split真实值，不能猜其名为validation或val。

服务器 experiments 已知约1152份JSON；usage ledger须显式覆盖合理范围内的历史结果、split/manifest/receipt和派生缓存索引，并逐clip映射至chunk，记录扫描范围、未解析资产及覆盖缺口。已知旧3119和chunk0000使用不是完整历史证明，不能仅排除这两个chunk便写 coverage_complete=True。此项与新名单目前标为pending，属于后续独立测试准备决策；它不扩大本轮stdlib四JSON审计的必完成范围。仅之后缺失的大ZIP按用户下载边界提供命令。

初始100个未用clip仅是新链路可行性阶段的目标规模，**没有功效依据，不是充分性证明**。可行性阶段用于核验GT、预处理、双方自主生成、完整记录、资源与失败行为；若其结果用于调整候选或统计方案，这批立即成为开发/计划数据，不得继续作为最终验证集。大资产仍先查国内来源并提供用户下载命令，主任务不得自行下载；本决策不批准新增网络或下载操作。

## 4. 新测试采集前必须完成的统计计划

当前预锁科学问题为同一GT口径下学生 mean ADE 是否超过教师15%。教师为冻结dense原结构；双方从同一raw input各自自主生成前缀，使用相同noise seed 42/43/44、64步×3坐标、相同GT与有效时间戳/坐标系。不能强制学生使用教师前缀来替代生产质量测试；双方前缀可以不同，rawinput/GT/noise及生成设置必须配对。教师求解步数按冻结教师定义固定，学生A/B分别固定其步数，不能测试后改成有利组合。

主指标仍为 clip 先seed均值后的整体 ADE ratio `R=S/T-1`，预锁15%为非劣界限。最终通过至少要求点估计 R≤0.15，且预先指定并合理校准的单侧置信上界≤0.15；不能仅因点估计落在门限内就说精度通过。统计计划必须在最终新测试误差不可见时完成并归档SHA，明确：

- 推断总体、真正独立重复层级、数据抽样/分层与相关性处理。若缺drive信息，clip bootstrap等默认独立方法只能报告“条件于clip近似独立假设”的CI，并安排已知chunk/时间分组等敏感性检查；这不能制造行程独立证明。
- 单侧置信水平（原则为95%）、CI方法/算法/随机seed/重采样次数、两候选比较时的检验族与校正，或者在独立结果不可见时明确一个主候选、另一个探索候选。不能两个未校正CI择优后报告单一“通过”。
- 样本量依据、方差/相关性来源、与15%界限的预期余量、固定N或预定序贯停止规则。开发集描述可以用于保守规划但必须披露选择偏倚；不做事后功效来证明100足够，不边看CI边任意加样本直到通过。
- 故障、缺失、GT不合法、生成边界丢失、OOM/timeout、非finite与重试规则。模型失败不得从主要分母静默剔除；完整性和失败门禁单列，不能用成功样本均值掩盖失败。
- 尾部与分场景指标、失败率及其预定验收口径。没有用户阈值的尾部字段先描述，不臆造安全阈值；尾部未解释或关键场景无覆盖不能称“完全验证”。

本文件是审计授权与后续计划约束，不是假装已经完成上述完整统计计划。完成数据资格审计、候选冻结和可行性方案后，Sol另出采集前最终决策单，明确统计方法、样本量与采集命令，再由主任务排队新GPU测试；不得绕过该技术门禁。用户已经授权的精度工作可自主准备，不要求再次泛泛确认。

## 5. 暂停、升级与最终证据边界

未完成精度验证前，暂停新增FlashDrive/LightVLA机制、训练、量化/剪枝候选变化、KV4与hybrid组合、低峰值loader实体运行、ONNX/TensorRT与性能机制扩展。已有终态证据保留；本决策不停止其它项目进程，不重跑已完成GPU流程。

遇到文件SHA变化、事实冲突、无法逐键绑定教师、来源/保护摘要不符、派发覆盖不全、名单使用历史不清、GT口径缺失或行程依赖未解决，停在只读安全边界，保存错误与失败输出并升级主任务/Sol；不得换文件、剔除clip、补写FDE/退出码或自行换模型绕过。

本轮验收终点：已有开发证据可复核、候选和未见数据条件明确。独立精度门禁、闭环质量、Orin ARM64/统一内存/完整请求max≤300ms、NVIDIA官方原入口门禁均各自保留，不能用这份本地审计互相替代。只有完成预先冻结的新测试、全部完整性与统计/失败门禁且Sol独立复核，才可报告相应范围的精度验收；“完全”必须同时声明抽样覆盖、行程关联和设备/闭环的实际限制。
