# ALP15 采集前独立精度草案：独立技术审查

- 审查编号：R-ALP15-PRECISION-INDEPENDENT-EVAL-20261010-01。
- 日期：2026-10-10（Asia/Shanghai）；独立审查者：GPT-6.1 Sol high，依主任务分派执行。
- 被审查版本：`ALP15-PRECISION-INDEPENDENT-EVAL-DRAFT-20261010.md`，实际 SHA256 `6730a018d4c25cf0c4b777d5d44aa65774984050c565cbfe65621490dd4cdf82`，206 行，全文读取。
- 结论：**保留为 draft，当前不得进入 ready。** 抽样设计、推断目标与主 CI 之间尚有实质错配风险，且实际 census、样本量/覆盖率规划、GT 与运行资产门禁未完成。本审查不是采集批准，也不是独立精度验收。
- 写入边界：仅新增本文；未修改草案、决策、源码、测试、模型或正式证据。全程本地文本/源码只读与一个标准库精确枚举；无网络、SSH、下载、GPU、模型运行、payload 或依赖安装，不使用 superpowers。
- 使用技能：完整读取 `/Users/USER/.agents/skills/experiment-design/SKILL.md` 与 `templates/preregistration.md`，用于核对“分析匹配设计”、真实重复层级、样本量依据及采集前冻结。技能没有要求本轮另行征求权限；pending 由以下实际缺口产生。

## 1. 已正确保留的设计约束

草案第 54–70、118–128、134、148、164–172 行具有以下正确边界，修订时应保留：

- 教师 T 与候选 A/B 分别从同 raw input 自主生成，教师不使用学生前缀或 packed reference；A/B 分别评价，不拼成组合系统。
- 同 clip 三个轨迹 seed 是重复测量，64 点和 token 不是独立重复。先 seed 均值，再 clip 等权；主指标 `R_k=S_k/T−1`，不使用平均 clip ratio。
- 两候选共同构成预先指定 family；单侧 family α=.05，Bonferroni 每候选 α=.025。对同教师、同数据的依赖不使 Bonferroni 失效；同时抽三臂保留配对。
- 模型失败不触发输入替补；原 planned 请求、首次失败和故障分母保留。成功子集不能代替主要验证分母。
- planning pilot 只用于规划；一旦用于候选/QC/CI/N 等调整，全体转 dev 并排除关联 chunk。固定 N、不按有利 CI 早停或加样的规则正确。
- GT 必须实际保存，ADE/FDE 独立复算；未知单位/时间/坐标不能补造。闭环、设备、官方入口与开环精度分别验收。

这些是可继续完善草案的基础，尚不是主 CI 或采集 ready 的证据。

## 2. 会影响统计结论的修订项

### R1：统一估计目标、clip 抽样设计与 chunk CI

位置：草案第 84–86、136–140 行。优先抽样为从有限资格池 E 对 clip 简单随机无放回；主 CI 却均匀有放回抽取观察到的 G 个 chunk，并完整复制每个 chunk 中已选 clip。该重采样对应一套观察块模型，不能自动当作有限 E 上 clip SRS 的设计型抽样分布。第 84 行还同时称主要效应描述“冻结测试集合”，第 138 行称 CI 条件于“冻结抽样框”；固定评测集合、有限框总体与假设下的块超总体是不同对象，应先指定哪个接受推断。

设有限 E 共 M 个 clip，chunk g 有 M_g 个，抽 N 个 clip。chunk 被观察到的概率为：

`π_g = 1 − C(M−M_g,N)/C(M,N)`（M−M_g<N 时 π_g=1）。

所有 clip 的纳入概率相同 N/M，但 chunk 的纳入概率通常不同；观察 chunk 也不是被等概率整群抽来的完整原始 chunk。举例 M=10、N=4，大小为 1/9 的两块被观察概率分别为 .4/1。这一事实不能被“均匀抽观察 chunk”或“块数够多”消除。预算若改为两阶段/分层抽样，还会产生新的纳入概率，草案第 86 行正确要求另行冻结。

本审查**没有认定大小不同必然改变当前 ratio 点估计**：草案复制所有已选 clip 后按 clip 等权求比，未直接变成平均 chunk ratio。问题是置信上界对应的随机性/目标总体与覆盖率未经论证，而非必须把每块 ratio 平均或机械加逆概率权重。没有 census/实际块大小/目标模型，不应指定新的权重公式。

仅为区分设计的精确代数例：普查一个 10-clip 有限框，两块大小 9/1，所有 T_i=1，两个块 S_i 分别为 1/2，则已知全框 R=.1，有限框抽样误差为 0。均匀重抽两个块得到 R* 为 0/.1/.1/1，仍有非零分布。这证明普通整群 bootstrap 不是该有限框普查的设计型分布；不是用 G=2 判断一般 CI 好坏，也不是模型数据或功效模拟。

进入 ready 前须由最终 Sol 决策明确选择并论证：

1. 若目标仅是冻结已评测集合，R 是该集合的描述量；CI 不得模糊改成未知总体保证，需另定它表达的随机性。
2. 若目标是冻结 QC 合格有限框的 clip 等权效应，应使推断匹配实际无放回/分层/两阶段设计，记录纳入概率与抽样比例，并处理或说明有限总体因素。固定框内响应作为固定值的设计型推断与道路超总体推断应分开；前者不直接要求 clip 响应相互独立，后者不能因此获证。
3. 若保留观察 chunk 的条件模型 CI，应写明完整的块模型、目标效应、为什么 clip SRS 形成的部分块允许此重采样、块大小与误差/教师均值相关时的处理，以及跨 chunk 独立/可交换假设与范围；用实际设计的规划模拟校准。不能仅将第 138 行改名视为论证已完成。

R1 是 ready 阻断项；本审查不自行替换已批准的主 ratio 或设计路线。

### R2：97.5% percentile 与 Bonferroni 需要覆盖率校准

位置：第 134–147、168–170、191、196 行。Bonferroni 的 family 控制以前提“每个边际方法确实满足所声明误差水平”为基础。把重采样 ratio 的 .975 分位数命名为单侧 97.5% 上界，只规定了算法，没有证明该方法在拟采用的非劣边界、有限框/块模型、块大小失衡与 ratio 分母情景下具有 97.5%覆盖。

B=50000 主要降低对所选重采样分布的 Monte Carlo 误差；它不能修复抽样设计错配、少量有效块、重尾、教师分母不稳定或未知跨 chunk 相关。G 的数量和单块占比也不能单独证明块独立或 family 错误率。chunk 主 CI 亦不保证在每份数据上都比 clip CI 更保守；第 144 行应使用“按所声明相关结构选定的主分析”，避免把“较保守”当未经验证的性质。

草案已将模拟/接受条件标 pending，方向正确；需进一步将第 136 行的“预锁主算法”表述为待 R1/校准后审定的候选算法。正式冻结必须预定模拟生成机制/假设范围、边界 R=.15 的误通过率或覆盖率、MC 不确定性及接受标准，并包含不等块大小、大小与 margin 相关、教师均值接近零、重尾、有限抽样比例与声明的跨块相关情景。模拟能评估已指定假设，不能从模拟制造真实 drive 独立证据。

两个候选的合成数据应共享教师/配对结构，校验实际 family 错误或至少校准各边际错误后应用 Bonferroni。若 R1 修订设计/方法，重新校准并冻结，不根据最终 test 的通过与否选择分析。

R2 是 ready 阻断项；不在本轮猜 B/G 阈值、覆盖率或功效。

### R3：QC 后的有效抽样框与替补选择必须明示

位置：第 76、84、90–94 行。E 当前仅由 metadata/feature/usage 资格产生；窗口 QC 还会排除解码/时间覆盖/重复输入等 clip。随机顺序扫描至 N 个 QC 合格 clip，在预定、确定且误差不可见的 QC 条件下，可对应从 QC 合格子集抽样；它不继续代表所有原 metadata 合格 E。

应区分 `E_metadata`、经规划冻结规则定义的 `E_QC` 与最终样本，记录排除/关联去重的规则、未知覆盖、候补消耗和每个 clip 纳入依据。若只给样本 QC 而不知道全池合格率，应明确推断条件和 frame 大小未知，不能伪填全框 QC 计数或抽样比例。因 payload 无法获得而按便宜 chunk 替换，属于抽样/目标改变，不是普通 QC 替补。候选误差不可见的 QC 也会改变适用总体，故不得省略这一范围限制。

R3 是目标/抽样一致性的 ready 阻断项，需与 R1 一并修订。

### R4：预测时间轴必须与 GT 时间轴实际配对

位置：第 103–110、120、154、194 行。草案已有 GT 时间/坐标记录，但未显式要求每臂实际 action_space 的 `class/config/dt/n_waypoints`、预测首点偏移与预测时间轴逐点绑定 GT。64×3 形状相同不证明相同时间；diffusion Euler4/10 的积分步骤不是道路轨迹 dt。

本地官方 loader 第 102–117 行生成 t0+0.1…6.4s，128–144 行变换到 t0 ego frame，208–219 行 return 没有 future_timestamps。官方 `UnicycleAccelCurvatureActionSpace` 第 49–50、89–90 行仅给默认 dt=.1/n_waypoints=64；307–389 行 `action_to_traj` 使用实际 `self.dt` 累积速度/位移，返回累计未来点。默认源码不是实际模型实例配置证据。

最终 manifest 应保存每臂 action_space 类型/源码/实际配置、loader history/future dt 与姿态/时间身份，逐点验证预测对应 t0+j×dt 与同一个 GT 时间轴（j=1…64，含终点），无 t0 首点混入、无一帧偏移。frame/原点/朝向/米制证据也应随运行资产绑定，不只依靠全局文档常数。参数/单位无法实证时不能计算有效主 ADE/FDE；舒适性 dt 同步使用该契约，不默认硬编码替代核对。

R4 是实际 runner/GT 数值验收的 ready 阻断项；本轮没有读取新 payload、加载模型或证明时间/单位运行值。

### R5：功效成功事件与失败分母须在最终 N 前固定

位置：第 124–128、140、164–172 行。草案已有失败留痕和固定 N，但功效“对哪个候选/最差臂”仍不足以确定 family 层面的资源目标：每臂各 .80、至少一个通过概率 .80、两个都通过概率 .80 是不同规划事件，不能互相替代。技术完整性要求全 planned 请求有效，规划还须考虑故障使验证 incomplete 的概率，不能只给条件于全部成功的统计功效。

明确按候选 k 的请求分母为 N×3，其配对依赖同 N×3 个教师请求；整个实验唯一 planned 请求为 N×3×3，统计 table 中重复绑定教师不能冒称额外运行。教师失败影响所有共享它的比较，A 失败使 A nonpass，不应静默删除该 clip 后算 A 的成功子集；B 是否仍可单独判定须由共同资产/技术停止规则决定。无论 A 失败/撤销，已冻结两候选 family 不事后缩成单候选 α=.05 来放宽 B。

正式输出对 incomplete 比较的主 R/U/pass 应为 null/false（并附原因），成功子集 R 另具显式 descriptive 字段。输入 QC 替补、attempt 状态和最多一次模型开始前基础设施重启要保留唯一 request 键及 attempt_id，不能将 attempt 数当 planned 分母。pilot/QC/timeout/数值容差/资产变更的转 dev 规则须覆盖全部受影响名单；固定 N 后不得用模型失败再补量追求 pass。

R5 是 N/功效与判定契约的 ready 待填项。未测失败率、成本、ICC 或方差均保持 pending；不从旧开发集估计值制造保证。

## 3. 本地来源核对与当前外部阻塞

本轮命令均 exit 0，除不存在 review 的预检成功外，没有修改/运行被审查程序。源码仅文本读取，未 import torch/numpy。

|依据|实际 SHA256 /身份|核对内容|
|---|---|---|
|`alpamayo1.5` HEAD/status|`36aeb4c5938cbc2eb2aed33b22434773da4ab639`；status 空|仅本地 checkout，不代替远端身份|
|`scripts/alp15_kv_steps_probe.py`|`74886a867a8aae480dd848a6d068d2e3f817674d891774bcdabeb57f412292f8`|161–212 raw/生成；215–236 sample/XY误差|
|`scripts/run_alp15_awq_hybrid_eval.py`|`fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd`|452–494 旧 raw/自主生成/旧教师 ADE 绑定|
|`alpamayo1.5/src/alpamayo1_5/load_physical_aiavdataset.py`|`92bd5359a6583226d62f3165fbce4f63e3ec3e573442e2a572171dd70a900b8a`|全文；真实时间、t0坐标、return字段|
|官方 `diffusion/flow_matching.py`|`57775891e6e0c047c3f203b7b141ffd6034b13969076926200f76404e48058d7`|32–110、138–185；Euler/噪声/参数|
|官方 `models/alpamayo1_5.py`|`87b23954570f4d466a6d0dd64090f779af635433b61106eb3b028d8d802ff1e8`|244–437；默认6样本、rollout、action_to_traj|
|官方 `action_space/unicycle_accel_curvature.py`|`1e8035a3f2456b82219e96f7058fe2c0d94410a60367fe7395df3f22b4bedf9e`|40–105、300–389；实际dt与未来点积分|
|旧 `controlled_8b_diagnostic.py`|`2d7049d548bce8ee0410e89a68c51750dcb6ca5a734ab34afa10b7d62afc3508`|32–83；文本seed/单条噪声/旧XY ADE|
|`ALP15-PRECISION-RESUME-20261010.md`|`eeefa523b5b8d5d522785fd222e2cf5a3c5b8289684c5b7d735d762b43df69b9`|全文；最新CPU准备/SSH阻塞边界|

官方短路径均相对 `alpamayo1.5/src/alpamayo1_5/`；旧 helper 位于 `docs/baselines/evidence/quant_route_20261008/full20/code/`。被审查草案原 hash 表与本轮实际源码 SHA 相符。本轮进一步发现的预测时间契约是本地源码推导，未将源码默认值当作 checkpoint runtime 已测结果。

完整工具证据：skill/定位 chunk `47e844`，模板/草案/holdout决策 chunks `bd9eb5`/`1a57bf`/`ecf543`；草案后半与 loader/runner chunks `ca0b87`/`21c9ee`；本地 git与七文件 SHA chunks `33fcd1`/`5e26a9`/`45a4e7`；官方模型/flow chunks `d10a5e`/`9cc226`；actionspace/续接/hash chunk `ee9381`；精确枚举 chunk `379afe`。采用 `/usr/bin/python3 -B` 的枚举仅用 itertools/math：计算上述组合概率与两块四种重抽 ratio，exit 0。它不是功效/CI覆盖率模拟，不产生任何模型数据或证据文件。只读时钟为 `2026-10-10 15:38:53 UTC`，不是新模型采集时间。

据主任务最新续接记录，SSH `REDACTED_IP:2222` Connection refused、exit255，远端字段/空值预检没有执行。本审查未尝试连接或替换主机。既有 parquet 字节/历史行数记录和合成准备工具测试不能证明本次真实 census 已完成；真实字段归属、空值策略、usage、资格池/块大小、QC与预算都 pending。

## 4. ready 判定与后续边界

当前判定为 `draft_reviewed_not_ready`。R1/R3 首先决定可解释的估计目标与抽样设计，R2 再校准并冻结对应的主算法，R5 用同一成功事件完成 N/资源规划，R4 随 planning 与新 runner 的数值验收核对实际 GT/预测契约。上述项可以在最终误差不可见时修订草案；本审查不改文档或实现，交主任务/设计者处理。

同时仍须完成实际 census/独立 usage receipt、输入QC与候补、下载报价/payload receipt、planning/最终test排斥、教师与 A/B 完整资产冻结、独立 runner/分析脚本验证、运行顺序/timeout/数值容差与最终可执行决策。未取得事实时只保留 pending，不靠 B=50000、G 数量、official test 标签或旧开发均值称 ready。

本文没有提供新 GT 误差、CI、精度通过或设备结论。Orin NX16GB/统一内存/ARM64、完整请求 max≤300ms、闭环碰撞/违规/TTC/舒适安全、NVIDIA 官方原入口均继续未测或未验收；声明范围内的离线非劣结论也要等待新的完整采集与独立复核。
