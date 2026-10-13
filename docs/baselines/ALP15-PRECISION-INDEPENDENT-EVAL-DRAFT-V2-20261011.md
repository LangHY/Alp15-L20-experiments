# ALP15 采集前独立精度设计草案 v2：有限框、配对 margin 与实际时间契约

- 编号：P-ALP15-PRECISION-INDEPENDENT-EVAL-DRAFT-20261011-02；日期：2026-10-11（Asia/Shanghai）；设计者：GPT-6.1 Sol high。
- 状态：`draft_v2_pending_scope_qc_budget_design_calibration_and_runtime_freeze`，**仍未 ready**。本文件是针对审查 R1–R5 的独立新设计，不修改 20261010 草案、审查及其当时 census/SSH 阻塞事实。
- 唯一写入：本文件；无新 SSH、GPU、模型、安装、训练、权重转换或 payload 读取，无大资产下载，不使用 superpowers。
- 长期目标保持：Jetson Orin NX 16GB、完整请求 max≤300ms、相对教师 mean ADE 损失≤15%。本设计是其中的新数据开环精度子门禁，不能以减少覆盖/便宜 chunk 取代最终目标，也不替代闭环、Orin 或 NVIDIA 官方原入口验收。
- 已完整重读 experiment-design 技能、预注册模板、AGENTS.md、旧草案/审查；使用技能落实“分析匹配设计”，而非把已有误差改写为采集前预注册。

## 1. 更新的事实、身份与范围

以下是本轮阅读本地归档证据后的当前事实，不是本轮远端重跑。

| 已归档证据 | 当前可支持的事实 | 本轮实际 SHA256 |
| --- | --- | --- |
| 原 `ALP15-PRECISION-INDEPENDENT-EVAL-DRAFT-20261010.md` | 保留旧设计与历史状态 | `6730a018d4c25cf0c4b777d5d44aa65774984050c565cbfe65621490dd4cdf82` |
| 原 `ALP15-PRECISION-INDEPENDENT-EVAL-REVIEW-20261010.md` | R1–R5未ready的独立审查 | `b8652a5b21cd83eaf97f3ee2d5e9246709ac7638351ffdf7a9829103dbb13133` |
| `evidence/holdout_prep_remote_20261010_v3/run-manifest.json` | census已 completed、exit0；元数据前后SHA稳定；coverage=False | `022ae526670625ff44c1d69b7507bb1925e0a91cedd25442573cb9ccd29c3ff2` |
| `ALP15-PRECISION-ACTIVATION-LINEAGE-ACCEPTANCE-20261011.md` | 544个历史路径的来源链窄域通过；不验payload/全coverage | `774a8d400ab408629e6bc235c566289860b92f17ff1e58fef1eea832837bd7e2` |
| `ALP15-PRECISION-TEACHER-CACHE-LINEAGE-EXECUTION-20261011.md` | 700个teacher路径的同一100clip有限曝光候选，待独立验收 | `ef6682d62fd4437fbfc03236c819cab37457867bd969cb5dc31c5de8675cd1f0` |

路径相对 `docs/baselines/`。正式census原UTC起止为2026-10-10 15:47:18.150641至15:49:47.818446，解释器Python3.12.14、Linux6.8.0-87-generic、pandas2.3.3、pyarrow22.0.0、physical-ai-av0.2.0，offline/空CUDA。`ALP15-PRECISION-HOLDOUT-PREP-REMOTE-EXECUTION-20261010.md:56–84`记录终态及主任务复核；本轮按manifest字段读取，不把早期“执行中”段落当现状。

全metadata306152clip/3146chunk；usage保守引用219clip、排除21chunk后为304075clip/3125chunk，其中test **61301clip/633chunk**。`coverage_complete=False`、`drive_disjoint_verified=False`仍成立；radar_config缺失在test池为29771，保留未知，不影响已经规定的四camera/egomotion资格，也不证明场景覆盖充分。实际metadata前后固定SHA为：

- clip_index.parquet：`9630a3de5ee057ded194deaa94d3b9b42af6f18c301785706e50141314a6935c`。
- feature_presence.parquet：`a9946feb269124e3a7cbfe22d97669585735134f634c2ca97332e7abc4314790`。
- data_collection.parquet：`9624750cadabf8028264cc1404e75e81777654be9def06b00e35c9f834b173d5`。

544是路径数，非独立clip数；来源已归因不等于payload被读验。700有限曝光候选也不是700clip或完整训练祖先范围。它尚不能把原1892匿名记录的剩余数更新为正式已验收结果；coverage不能因两个窄域进展改True。原21块只保留或增加，不释放。其它匿名缓存、adapter/checkpoint祖先、外部路径、revision冲突、symlink和未纳入文本仍待完整范围审定。

数据revision为`33f9bf447ed3bcb7d545ce13f4226f824214fafb`；官方Alpamayo源码身份沿冻结提交`36aeb4c5938cbc2eb2aed33b22434773da4ab639`与checkpoint revision`7aba8293c09993f2e125c6819df05d7fa3e873ea`，新运行仍须实证实际资产manifest/hash，不用历史字符串替代payload/运行源冻结。用户离线教师参考例外沿精度验证决策及`ALP15-AWQ-FULL-20261008.md:7`，不会补造官方原入口原样验收。

## 2. R1/R3：先区分四个对象，再讨论置信度

| 对象 | 定义与当前状态 | 允许的结论 |
| --- | --- | --- |
| `E_metadata` | 固定revision、feature资格、usage排除后的metadata池；当前61301是候选数，coverage未审定 | 清点/文件报价规划，尚不能称独立test就绪 |
| `E_QC` | 完成usage审定后，对整个metadata候选框施加**冻结、确定、模型误差不可见**的输入QC/去重规则所得有限框；大小记`M_Q` | 设计型概率抽样可推断这个有限框，不自动代表被QC排除者 |
| `s_final` | 按最终批准的概率设计取得的n个QC合格clip；固定seed/config下实际三臂响应 | 该集合ratio是描述量；与E_QC的抽样误差须另算 |
| 道路/行程/随机seed超总体 | 用户最终目标涉及的新行程、场景与运行条件，而非一份固定revision框 | 无drive/session及外部覆盖证据时，不从有限框CI获得此保证 |

本轮没有全框raw payload QC，`M_Q`未知；61301绝不能填作`M_Q`。官方test标签不是Alpamayo专门benchmark协议。chunk是文件打包标签，不是drive。缺drive不阻止明确有限框设计的代数推导，也不解除道路泛化/数据曝光/最终设备门禁。

响应定义为在冻结模型/环境/算法与text seed42、trajectory seeds42/43/44下，每clip三个轨迹ADE的算术均值。三个seed是重复测量，不是3倍独立n；本有限框分析不推断所有随机seed的期望质量。若运行存在不可忽略非确定性，须在planning阶段冻结其处理与目标随机性，不能在最终test按最有利一次实现当固定响应。

### 2.1 可评审选项 C1：先形成完整QC框，再对clip SRSWOR

在全usage与输入QC完成后，对`E_QC`全部clip_id排序，先锁n，再用归档Python版本及独立`random.Random(20261011)`产生均匀无放回clip样本；不按chunk便利、模型误差或结果调顺序。理想设计每个n子集的概率是`1/C(M_Q,n)`：

`π_i=n/M_Q`；`π_ij=n(n−1)/(M_Q(M_Q−1))`（i≠j）。

抽样对象是clip，方差来自这个无放回设计。已选样本内观察到的chunk数不构成一个新的一级SRS；**否决旧观察chunk bootstrap作为C1主CI**。不同chunk大小及块内响应相关作为有限框固定响应的分布特征已进入真实有限框方差，无需假设固定clip响应相互独立；这不等于未知道路超总体的drive独立性被验证。

此选项的难点是完整QC/获取成本，当前未选择。不能为了廉价而只QC少数便利chunk，随后宣称完成61301框QC或仍在该框SRS。

### 2.2 可评审选项 C2：隐式QC框，随机顺序扫描至固定n个合格clip

若不事先QC全框，可在批准前冻结对**所有**metadata clip均定义的确定QC谓词Q、完整metadata随机排列、n和获取/扫描规则；顺序扫描直到n个Q=True，输出全部已检查clip、QC结果、未检查集合与停止位置。条件于真实但未知的固定`E_QC={i:Q(i)=True}`，随机排列中前n个合格clip是它的SRSWOR：对称性使每个合格n子集同概率。该论证不需要先观察所有QC结果，但要求所有未检查clip也存在同一确定Q值且可按设计获取，不受候选误差影响。

`M_Q`仍未知，`π_i=n/M_Q`只能符号记录，**不得填入61301或用已检查部分的QC比例当已知真值**。若用`v0=s_D²/n`忽略FPC，它的期望不小于真实设计方差`(1−n/M_Q)S_D²/n`；这是variance层面的保守关系，不能据此称任意t/normal CI已经保守或exact。C2的近似覆盖率仍须对未知`M_Q`、QC选择结构与重尾做规划校准。

因无下载额度/文件暂时无法获得而跳过、达到费用上限后只从便宜块补足、人工挑替补、随机样本不足即改n，都不符合上述固定QC SRS论证。若预定扫描尚未完成便遇预算/资源阻塞，保留`acquisition_censored/incomplete`，不得以便利数据完成n后使用C2推断。若要把“可得数据”纳入Q，必须采集前明确定义新的有限框和其范围，不以行政缺失伪装传感器QC。

QC去重也必须确定且排序无关：例如事前固定等价类与canonical代表ID，而非“先遇到哪个就保留哪个”导致随机顺序改变E_QC。相邻clip无法证明同drive，不自行删除；保留可测hash/关联标签及其规则。QC变化若利用planning数据，则planning全体及关联chunk转dev、剔除最终框；正式模型误差可见后不得调整Q。

### 2.3 预算需要整群/两阶段时：这是不同设计 C3

只有预算事实表明确要求，才考虑**事前真正抽chunk**，随后QC/clip抽样，明确目标及概率；不能重抽观察chunk来追认C1为C3。示例设计须先已知全QC框H个chunk及各`M_g`，一级SRSWOR抽G个chunk，二级在每个入选chunk从其完整QC集合SRSWOR抽预定`n_g`。对其中clip：

`π_g=G/H`；`π_i=(G/H)(n_g/M_g)`；同chunk的`π_ij=(G/H)n_g(n_g−1)/(M_g(M_g−1))`；不同chunk则为`G(G−1)/(H(H−1))×(n_g/M_g)(n_h/M_h)`。

这是为有限框clip等权总体估计而给样本不同权重，**不是改变用户15%界限**。若不加权，少clip的chunk可能被过度代表；不允许把平均chunk ratio当总体ratio。以margin为例，`dhat_g=M_g×mean_{sample g}D_i`，`tauhat_D=(H/G)sum_{g in sample}dhat_g`，`muhat_D=tauhat_D/M_Q`，并同法估计T和S总量、报告`tauhat_S/tauhat_T−1`。样本自身未加权ratio另作描述，不能代替目标参数估计。

两级SRS下对应variance estimator为：

`vhat(tauhat_D)=H(H−G)s_dhat²/G + (H/G)sum_{g in sample} M_g(M_g−n_g)s_Dg²/n_g`，`vhat(muhat_D)=vhat(tauhat_D)/M_Q²`。

需要`G≥2`且抽样部分的`n_g≥2`才能按此式估计相关样本variance；这些只是公式可计算条件，**不是充分功效/覆盖阈值**。普查chunk的二级variance项为0；完全普查时抽样variance为0。此示例依据已读[PSU STAT506 Lesson09 §9.1](https://online.stat.psu.edu/stat506/Lesson09)，记号作本项目重映射。

如果未选chunk的QC数量未知、`M_Q`未知，不能直接套上述权重/分母。可另设计一级metadata框概率抽样、入选chunk全QC、对QC合格域的HT总量/ratio域推断，但QC域分母估计与二级自适应抽样需单独推导校准；**该扩展本草案未批准、未ready**。PPS、分层或限每块数也必须预锁π/联合π、权重和variance，不混用C1公式。

## 3. R1的精确代数检查：抽样分布必须与设计相同

仅用本地CPU标准库`itertools`和`fractions.Fraction`，不读取模型数据。构造有限框M=10：9个clip的T=1/S=1、另1个T=1/S=2；人为chunk大小9/1。对用户1.15界限，9个D=−.15、1个D=.85，框均值D=−.05、框ratio=.10、有限总体`S_D²=.10`。

| 同一固定框的实际设计 | 精确枚举结果 |
| --- | --- |
| clip SRSWOR n=4，全部210子集 | `E(Dbar)=−1/20`；`Var(Dbar)=3/200=.015`；`E[(1−n/M)s_D²/n]=3/200` |
| n=4时那1个clip所属chunk被观察概率 | `.4`；9clip chunk被观察概率为1；观察chunk非等概率一级SRS |
| 全框普查n=10，唯一子集 | Dbar=−.05，真实抽样variance及FPC估计皆0 |
| 对普查后两个观察chunk均匀有放回重抽两次 | Dbar*为−.15/−.05/−.05/.85，仍非零分布，故不是该普查设计分布 |

另枚举metadata5个clip的120个随机排列，固定QC通过集合{0,2,4}，扫描至2个通过者：三个可能样本{0,2}/{0,4}/{2,4}各40次，验证C2的对称性小例。它不证明真实QC谓词确定、全框已QC或预算能完成。

另构造C3的H=3块、大小2/3/4，响应分别[−1,0]/[1,2,3]/[4,5,6,7]，一级G=2、二级各抽2，精确枚举所有两级概率：框mean=3，HTmean期望=3，真实设计variance与上述vhat期望都为`883/324`。此处响应只是合成数字，不称真实ADE。

这些检查解决旧R1的**抽样随机性错配**：C1匹配clip SRS公式；C3匹配真正两阶段概率。它们只验证期望/variance代数，**没有验证t/normal CI覆盖、功效或真实独立性**，不能使本文件ready。

## 4. R2：配对 margin、15%等价条件与待校准CI

教师T为原始dense官方checkpoint的用户离线参考口径，自己从新raw输入自主生成；A=actualKV4，B=hybrid_m1/fullprefill10，分别冻结、未组合。保留相同raw/GT/initial-noise配对，禁止共享教师continuation/KV或把packed reference称教师。固定text seed42、轨迹seeds42/43/44及generation/processor/tokenizer/history fuse等身份，任何新训练/量化/后端/权重配置属于新候选。

每请求仍用同GT、64步局部XY米制`ADE=mean_t ||pred_xy−GT_xy||₂`、`FDE=最后一点距离`；保存XYZ但Z不入主ADE。对clip先三个seed均值，记T_i/S_ki；**主ratio报告仍为均值的比**，不取平均clip ratio。

有限QC框参数：`mu_T=mean_E_QC T_i`，`mu_Sk=mean_E_QC S_ki`，`R_k=mu_Sk/mu_T−1`；定义有符号配对margin `D_ki=S_ki−1.15T_i`，`mu_Dk=mu_Sk−1.15mu_T`。当且仅当`mu_T>0`时，`R_k≤.15`与`mu_Dk≤0`等价。因此D在0处的非劣检验保持用户15%界限，**不是另订绝对米制容忍阈值**。

保留固定family{A,B}，单侧family α=.05，各比较Bonferroni α=.025。T共享一次，配对三个臂不能拆开；A故障/撤销后也不将B放宽到α=.05。C1候选近似上界为：

`Dbar_k + c×sqrt((1−n/M_Q)s_Dk²/n)`；初始候选`c=t_(n−1,.975)`或有限总体normal近似的校准临界值；C2若M_Q未知只候选使用`s_D²/n`而不伪造FPC。C3使用对应设计variance与一级自由度/校准临界值，**不能直接用clip n−1追认**。

这些是**待批准、待规划校准的近似方法**，未固定为生产主算法。SRS期望与variance公式是设计恒等式；任意固定有限框的studentized统计量不因此exact t。已读[PSU STAT506 Lesson01 §1.3/1.4](https://online.stat.psu.edu/stat506/Lesson01)提供SRS/FPC与近似CI依据；课程简化的样本数建议不作为本项目重尾/非劣边界覆盖定理。旧B=50000/.975 chunk percentile不再列为预锁主CI；增加模拟次数只控制MC误差。

最终决策需明确是否用margin上界≤0作为等价非劣门禁，或要求给ratio上界：后者应由**同一设计**下对`D(r)=S−(1+r)T`的检验反演构造ratio confidence set（记录协方差和分母条件），不能拿`U_D/Tbar`充当合法ratio上界。反演可出现无界/不连通集合，原样报告；没有有限上界时不声称ratio CI通过。本草案不默默撤销上游上界门禁，须由Sol在最终决策审定这项等价实现。

教师分母必须实际记录/复算而非旧常数。sample teacher≤0、非finite、无法可靠区分数值0/近0，或预锁分母诊断给出跨0/近0区间时：ratio与其CI为null，比较`incomplete/nonpass`；即使D计算出数值也不据它单独宣称“相对精度通过”。分母正性/数值稳定性判定、ε和数值容差必须由单位/计算精度/planning冻结，不根据最终test挑阈值。分母附加门禁是并列技术解释条件，不为候选增加宽松的误通过空间。

### 4.1 校准规格必须在最终test前填写

规划以明确固定合成有限框为母体，按**实际C1/C2/C3设计**反复抽样；在`mu_D=0`（R=.15且mu_T>0）及更不利null下验证单侧误通过与family误通过。两候选共享教师及raw结构；包含一个null/一个alternative和两个null，检验strong family控制。不能只检查两个边际power就声称family保证。

情景至少包含：有限抽样比例n/M_Q的范围、QC与响应相关/去重、大小不等chunk、chunk大小与D/T相关、重尾/稀有大误差、近零教师、成对teacher-student协方差、未知QC框大小、适用的两阶段π/成本结构。对道路超总体若另声明关联模型，须独立设定跨chunk关联/drive假设，不把它们混作有限框设计需要iid的理由。

每个情景锁定生成机制/参数来源、模拟seed、次数、MC精度/置信方法及多个情景的MC不确定性处理。建议审定的定量接受标准是：各候选null误通过率的**预定MC单侧上界≤.025**且family相应上界≤.05；临界值需如有必要在planning提高。该标准在有限假设网格上的意义必须说明，不能推广为所有未指定分布保证。MC置信度、情景数的校正、次数及允许网格范围目前pending，不能用一次无误通过的短模拟代替。

若近似未满足校准、样本过小/重尾破坏声明条件，否决该生产CI；可重规划方法/n、考虑有真正事前有效响应范围依据的设计型界限或扩展普查，不能事后裁剪ADE取得有限范围。全框普查仅消除该有限框抽样误差，不消除测量、曝光、失效或道路泛化风险。

## 5. R4：GT与每臂实际action_space逐点时间契约

本轮文本核对官方`load_physical_aiavdataset.py:27–37,102–154,208–219`：默认history16/future64/dt.1，GT未来时间t0+.1…6.4s，转换到t0 ego frame；return没有future_timestamps。官方`action_space/unicycle_accel_curvature.py:38–102,307–389`：源码默认dt=.1/n_waypoints64，但实际`self.dt`参与累计速度、角度、位移，输出从第一段未来累计位移开始。其实际文本SHA为`1e8035a3f2456b82219e96f7058fe2c0d94410a60367fe7395df3f22b4bedf9e`。

运行时须对T/A/B分别保存：真实action_space完整class/module、实际import源码SHA、构造配置/最终实例的dt与n_waypoints、初始状态/历史输入的frame/时间身份、是否包装/替换、action_to_traj函数来源。不能用checkpoint名称、源码默认值、64×3形状或diffusion Euler4/10判断道路轨迹dt。

必须保存并核验`pred_timestamps[a,j]=t0+j×actual_dt[a]`（j=1…64）与**实际GT采样时间**逐点相同：首点不能是t0、不得错位一帧，终点须对应同一6.4s时刻；history时间轴同源用于状态估计。预测是否确实具有该时间含义由实际class/config与action_to_traj源码/数值夹具验收，不允许仅将GT时间数组复制到预测字段声称验证。

GT记录包括实际ego_future_xyz64×3、tensor bytes SHA/dtype/shape/字节序与规范化数组SHA，egomotion/raw文件/成员SHA、t0 pose/变换receipt、米制依据、loader参数与source、真实解码帧timestamps、插值覆盖/QC证据。预测每臂保存pred_xyz/rot及hash，绑定同一raw/GT/noise身份。新runner将官方loader的真实future时间记录导出/核对，不改变其数值；本轮未实施runner、未读取新payload/实例配置。

任何时间、frame、单位、GT有限性/形状或action_space身份不满足，主ADE/FDE判为无效并标incomplete；不得靠插值、缩短64步或重命名时间轴绕过。实际同时间轴的XY误差才可独立CPU复算，容差在planning冻结。舒适性速度/加速度/jerk只有在此契约通过后按实际时间差求导，不能硬编码.1替代核对；无阈值依据只描述。无障碍物/道路几何/车辆尺寸/交互演化和闭环证据，TTC、碰撞、违规、闭环成功一律not_measured。

## 6. R5：planned分母、失败与功效成功事件

最终样本n个clip、每臂3seed，共**唯一planned请求9n**：教师3n，A3n，B3n。每候选比较依赖自身3n加共享教师3n，不能把统计表重复绑定教师算额外运行；attempt_id仅表示一次planned key的重启尝试，不增加n或planned分母。clip/seed/token/waypoint/chunk数量均分别报告。

建议状态契约：先列全部planned key `(eval,clip,t0,seed,arm,candidate_hash)`，分别保存attempt、started/completed/finite/paired/runtime_time_gate、错误、UTC/exit与原始日志。首次模型OOM/timeout/前缀边界/shape/nonfinite失败保留，不能换clip/seed补足n、只取成功子集、无限重试。最多一次模型尚未开始且无输出可见的基础设施重启需预锁理由/日志与同key；模型开始后修复改变路径，受影响最终名单转dev、另出新test决策。

T任一planned请求失败使两个共享比较incomplete。仅A模型局部故障可使A nonpass而B仍待核，但须证明不是共同源码/资产/GT/环境失效；共同身份漂移或时间/GT故障影响两臂并触发全局技术停止。B即便可独立完成仍保持α=.025。缺失比较输出`primary_R=null`、`primary_U_D=null`（或U_R=null）、`pass=false`及原因；成功子集ratio另存`descriptive_success_subset_R`，不能出现在主通过字段。

完整性门禁要求该比较所有planned请求有效；这是当前无失败插补的技术契约，不是驾驶安全“零失败”保证。报告每臂planned/attempted/complete/valid、失败clip/seed/理由，共享T故障影响、QC纳入/排除与acquisition_censored分母。真finite但大误差保留，不截尾/winsorize、不按结果删离群值。

资源规划事件必须先选，不能混淆：

- `P(pass_k | 技术全部有效)`：单候选条件统计power。
- `P(至少一个候选有效且非劣通过)`：含共享教师故障的无条件family成功事件，符合最终需有可用系统的候选选择问题。
- `P(A和B均有效且非劣通过)`：更强的另一资源目标。

本草案**推荐将第二项作为资源规划主事件，目标暂拟≥.80，待最终Sol决策审定**；同时报告两臂各自条件/无条件power及两者都通过概率，不偷换为每臂.80。若最终要求两者都通过，则重新规划n/成本，不把已有第二项功效直接沿用。当前没有真实方差/失败率/成本/预算，以上尚不能给数值n或达标保证。

planning须包括共享T故障、候选局部/共同环境故障、clip内3seed相关与非均匀场景故障，给预锁失败情景网格而非假定9n请求独立。举例若任一候选与共享教师都必须完整，真正相互独立、每请求成功概率p时完整概率是p^(6n)，但真实相关故障不满足该假设；该式仅说明大n与全有效门禁的取舍，不能估计实际成功率或建议降低用户质量门禁。

## 7. 样本量、QC/预算分支与停止规则

没有用户选择下载额度、没有已审定planning方差/失败成本，**不指定100clip、G数量或n即可充分**。planning数据一旦用于候选、QC、方差/相关性、CI、临界值/n与timeout等调整，全体转dev，关联chunk保守剔除，不能混回finaltest。planning原始输入与final框排斥名单须各自hash归档。

对C1 signed margin，若预期余量`delta=−mu_D>0`、规划finite方差为S_D²，近似初值满足`delta/sqrt((1−n/M_Q)S_D²/n)≈c+z_power`；令`A=(c+z_power)²S_D²/delta²`，可得`n≈A/(1+A/M_Q)`。这是**采集前近似规划初值**，c随n/方法校准、方差/余量来自独立planning或披露偏倚的旧dev，不能后验用test估delta证明足够。C2未知M_Q不套伪FPC，C3按π/块大小及两级variance另规划。最终以第4节覆盖校准与第6节无条件成功事件联合选择n/成本。

| 事前事实/偏好 | 决策分支 | 否决条件 |
| --- | --- | --- |
| 可承受全框QC与SRS的去重五ZIP费用 | C1；已知M_Q、固定n与FPC、配对D候选CI | 全QC未完成/usage未审定、CI未校准、成本无法完成随机样本 |
| 不能先QC全池但能按随机顺序获取至固定n | C2；隐式QC框、M_Q未知、相应候选variance/近似校准 | 跳便宜chunk、成本删选、未知Q规则、扫描被censor而仍推断完成 |
| 大文件费用需要真正抽少数chunk | 提交C3或其它明确两阶段设计，先定π/目标/权重/variance | 使用观察chunk bootstrap追认；M_g/QC域未知却套已知框公式；不加权仍称clip总体 |
| 预算仅允许便利小集或未定 | 只保留规划/可行性；最终目标仍待完成 | 将小集描述称全部精度、将目标缩成少数便宜chunk即可验收 |

先报价固定revision的去重chunk五ZIP实际bytes/SHA，给用户规模及其自行执行的下载命令；不能由本草案代下载。metadata/小证据已有不等于payload已有。用户预算不足不授权缩小最终目标；只允许明确留下待完成的大范围/设备门禁。

最终采集前锁n、框/随机设计、运行顺序、候补QC规则与获取边界。固定n后不看R/U加样至pass、不有利早停、不因失败改样本；新序贯方案只有采集前另定look/最大n/α spending才可用。技术停止保留完整日志，结果是incomplete，不是重抽有利样本的理由。

## 8. 其它指标、归档与最终判定

FDE仍为次要GT结局，教师与候选实际保存并复算mean/median/IQR/p95/max及差；无用户FDE硬阈值不套15%拒绝界限。ADE尾部、逐clip退化>15%数量/完整退化表、真实metadata分场景与失败率报告，不从粗标签猜天气/路口/drive。尾部p99在小n下不稳定；异常未解释/关键场景未覆盖必须作为未完成质量维度说明，不用主mean通过概括所有场景。

最终归档应包含本草案及审查SHA、独立usage scope receipt、E_metadata/E_QC定义/去重/未知计数、planning转dev/关联chunk、设计π和随机化输出、预算/下载/输入QC receipt、planned与attempt key、实际runtime/source/权重与action_space/time gate、GT/预测/noise/prefix/tokens完整记录、分析脚本及合成数学验收、CI规划校准和MC不确定性、n/功效成功事件/失败率情景、原始日志/环境/退出码、全部per-request/per-clip/per-chunk/分场景表与产物SHA。输出新建拒覆盖。

候选可报告“声明有限框下非劣通过”的必要条件：最终设计已批准且按设计完成、比较全planned有效、teacher分母/GT/时间/资产门禁通过、ratio点估计≤.15、审定的同设计margin上界≤0或合法ratio上界≤.15、相应校准与family误差控制证据满足、Sol独立复核通过。incomplete/null不能算pass；A/B单独判定不推出组合系统。

## 9. R1–R5落实状态与 ready 审定

| 审查项 | v2已根本调整 | 进入ready仍需要 |
| --- | --- | --- |
| R1设计/总体错配 | 区分四对象；C1 SRS方差、C3真实两级π/HTvariance；否决观察chunk bootstrap；精确枚举核代数 | 选择设计与推断目标、实际框/QC/预算、代码验收 |
| R2覆盖/ratio分母 | 配对D0界限与ratio15%等价；CI候选非exact；明确校准/MC/family与teacher不稳定处理 | 最终方法/定量校准/MC标准与次数/数值容差、上游门禁实现审定 |
| R3QC/替补 | E_QC与样本区分；C2对称性、未知M_Q不能伪填FPC；获取censor与排序无关去重 | 全QC或隐式Q可获取性证据、候补/未检查范围、全usage验收 |
| R4预测时间 | 每臂实际class/config/dt/n_waypoints、首点/终点逐点GT绑定；Euler步数非轨迹dt | 新runner与真实实例/GT/单位/时间数值验收，不只看源码默认 |
| R5power/失败分母 | 9n唯一请求、T共享、primary null规则、局部/共同故障、无条件family资源事件选项 | power主事件审定、方差/失败/成本规划、固定n/timeout/运行顺序 |

设计推导可继续审查，scope coverage=False不妨碍它们；但**没有完成以下证据前否决ready**：完整usage、实际QC框或有效隐式设计、下载预算/获取规则、教师及A/B真实资产冻结、新runner/分析算法验收、行动空间时间/GT/坐标/单位、近似CI校准、无条件功效和固定n，以及最终Sol采集决策。当前没有新模型精度、CI、闭环或Orin结果。

## 10. 本轮工具留痕与来源获取限制

本地只读 `nl/sed/rg/jq/shasum`核对上述文档/manifest/源码与身份。两组标准库Fraction精确枚举工具输出分别为`67f31d`（SRS/普查/错误chunk重抽）与`156932`（隐式QC/两阶段），exit0，无文件产物；它们仅验证本草案小例代数，不是覆盖率模拟。manifest选择字段读取`ba9639`；身份核对`277c16`。唯一新增文件为本文，自身SHA在停止写入后由主任务登记。

联网前已完整读取web-access技能并运行前置检查（`4a7ef7`）：Node22.23.0可用、Chrome未连接；未开启CDP或操作用户tab。按技能公开静态URL路径，以搜索定位PSU，web open对Lesson01/06 timeout、Lesson07 Bad Gateway；这些失败保留，不假称正文已读。随后公开curl+stdlib HTMLParser成功读取Lesson01完整正文（session4459，`a93e88` exit0）与Lesson09完整正文（session51598，`a50b0d` exit0）；未持久化额外文件。只采用其SRS/FPC与两阶段设计公式作一手依据，不把课程简化t/CLT表述当任意有限框exact覆盖保证。

本轮只对正文已获取的[PSU Lesson01](https://online.stat.psu.edu/stat506/Lesson01)和[PSU Lesson09](https://online.stat.psu.edu/stat506/Lesson09)作统计依据；Lesson06仍仅定位/获取失败，未用于本文公式。所有候选方法还需按本项目实际条件审定。原草案/审查、census/usage原证据未改写。
