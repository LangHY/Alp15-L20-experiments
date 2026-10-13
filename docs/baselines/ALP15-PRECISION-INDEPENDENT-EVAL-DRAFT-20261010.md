# ALP15 独立精度评测：采集前设计草案

- 文档编号：P-ALP15-PRECISION-INDEPENDENT-EVAL-DRAFT-20261010-01。
- 编写日期：2026-10-10（Asia/Shanghai）；设计者：GPT-6.1 Sol high。
- 状态：`draft_pending_census_qc_budget_sample_size_candidate_freeze`。可对本草案归档 SHA，但这不表示 `ready`、最终采集决策已批准或精度已通过。
- 唯一文件所有者：本设计执行者；只新建本文件，不改模型、runner、权重、旧决策或旧结果。
- 当前目的：按用户“先完全验证精度”的要求，定义新数据资格、双方自主生成、GT 记录、统计分析与证据门禁。长期目标仍为 Jetson Orin NX 16GB、完整请求最大耗时 ≤300 ms、相对教师 mean ADE 损失 ≤15%；本草案不把它们标成完成。
- 本轮实际动作：本地只读源码/小文档与编写草案；无模型执行、SSH、联网、下载、训练、依赖安装或大二进制读取。未使用 superpowers。

## 1. 依据、授权与结论边界

上游是 `D-ALP15-PRECISION-VALIDATION-20261009-01`，以及 holdout 准备 v1/v2（冲突按 v2）。已完整读取 experiment-design 技能和 `templates/preregistration.md`；下面按问题、单元、响应、干扰因素、样本量、分析、QC、归档填写。现有开发数据已经被看过，不将它们事后登记为验证性预注册。

用户的离线参考例外已记录在 `ALP15-AWQ-FULL-20261008.md:7`，精度决策第 5、14、25–29、94–120 行进一步限制了参考与候选口径。教师是已有离线参考定义的原始 dense 结构与官方 checkpoint；新测试须从新 raw 输入生成其自己的前缀、采用冻结的 full-prefill/10-step Euler 比较口径。不得加载学生 overlay 或 packed 权重充当教师，不混入旧 `(dense,student8)` 组，不沿用旧教师逐行 ADE 充当新测试结果。

这是一份离线开环轨迹质量设计，不补造 NVIDIA 官方原入口验收。AGENTS.md 第 78–104 行的官方入口与 Jetson 实机门禁仍独立存在。离线 ADE/FDE 不能替代闭环、安全性、Orin ARM64、16GB 统一内存或完整请求 max≤300ms 的实机证据；用户已授权精度准备无需再次泛泛确认，但新模型采集仍须待本文件的技术待定项完成后，由 Sol 出最终可执行决策。

### 1.1 本轮实际核对的本地身份

本地 `git -C alpamayo1.5 rev-parse HEAD` 为 `36aeb4c5938cbc2eb2aed33b22434773da4ab639`，本轮 `git -C alpamayo1.5 status --short` 输出为空。这只描述本地 checkout，不证明远端已运行代码洁净。官方 checkpoint revision 与数据 revision 延用上游已记录身份，尚未在本轮读取 checkpoint 大文件核验。

| 文件或身份 | SHA256 / revision | 本轮核验范围 |
| --- | --- | --- |
| 精度验证决策 | `989b64ac1adfa3979556db82d035f71230d91132454b8eb35029eafacf0280db` | 本地实际文件 SHA |
| holdout 准备 v1 | `3efbfea76b5b336d5199c93337863bff49c120f2b3bd92c9919d6f26e979297c` | 本地实际文件 SHA |
| holdout 准备 v2 | `32b534ea574677928fca9c6e9277498807f00839fe8de852f14f01d7a9e7a23b` | 本地实际文件 SHA |
| `scripts/alp15_kv_steps_probe.py` | `74886a867a8aae480dd848a6d068d2e3f817674d891774bcdabeb57f412292f8` | 本地实际文件 SHA、全文只读 |
| `scripts/run_alp15_awq_hybrid_eval.py` | `fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd` | 本地实际文件 SHA、全文只读 |
| 官方本地 `src/alpamayo1_5/load_physical_aiavdataset.py` | `92bd5359a6583226d62f3165fbce4f63e3ec3e573442e2a572171dd70a900b8a` | 本地实际文件 SHA、全文只读 |
| 官方本地 `src/alpamayo1_5/diffusion/flow_matching.py` | `57775891e6e0c047c3f203b7b141ffd6034b13969076926200f76404e48058d7` | 本地实际 SHA、sample/Euler 来源 |
| 官方本地 `src/alpamayo1_5/models/alpamayo1_5.py` | `87b23954570f4d466a6d0dd64090f779af635433b61106eb3b028d8d802ff1e8` | 本地实际 SHA、rollout/sample 来源 |
| `evidence/quant_route_20261008/full20/code/controlled_8b_diagnostic.py` | `2d7049d548bce8ee0410e89a68c51750dcb6ca5a734ab34afa10b7d62afc3508` | 本地实际 SHA、生成/采样 helper 来源 |
| 官方 checkpoint | `7aba8293c09993f2e125c6819df05d7fa3e873ea` | 上游身份；新运行前须实证 manifest 与 payload 绑定 |
| PhysicalAI 数据 | `33f9bf447ed3bcb7d545ce13f4226f824214fafb` | 上游固定 revision；新 raw payload 尚待获取/QC |

表中官方源码路径相对 `alpamayo1.5/`，证据路径相对 `docs/baselines/`。本轮命令与工具输出可由当前会话追溯；未来正式执行另保存完整 run manifest，不把本轮 hash 表当作远端候选冻结证明。

### 1.2 新 runner 必须解决的真实缺口

| 已读位置 | 真实行为 | 新评测约束 |
| --- | --- | --- |
| KV runner 第 161–170、185–212 行 | raw 图像经 processor 与 history fuse；文本 seed42；自主生成，检查 future-start 与 KV 长度 | 保留原始输入生成路径；教师/学生分别生成，不能共享教师 continuation |
| KV runner 第 229–236 行 | 用 `data['ego_future_xyz'][0,0,:,:2]` 计算 64 步 XY ADE/FDE，只存预测和误差 | 保存实际 GT 数组、其身份及时间轴，离线独立复算双方误差 |
| KV runner 第 294–318、411–417、453–471 行 | 绑定旧教师/AWQ 20 clip、旧前缀 cache、旧噪声 SHA | 旧 cache/旧教师 ADE 不得作为新测试依赖；新 manifest 决定名单和配对 |
| Hybrid runner 第 301–311、407–410、452–494 行 | 仍绑定旧20clip；每请求自主生成后 full-prefill、10步采样；教师 ADE 来自旧 JSON | 候选 B 路径保持冻结，教师 ADE/FDE 必须来自本轮同 raw/GT/seed 的教师输出 |
| 官方 loader 第 27–37、102–154、161–218 行 | 默认 t0=5.1s、16历史点、64未来点、0.1s；future 为 t0+0.1…6.4s；轨迹转换到 t0 ego frame | 记录实际 loader 参数、采样时间与变换；其 return 没有 future_timestamps 字段，不得假称已保存 |
| 官方 `FlowMatching` 第 32–50、53–110、171–177 行 | 默认 Euler10，公开 inference_step 覆盖，初始 randn；采样参数影响结果 | 教师10、A4、B10固定；记录实际 noise 字节 SHA、dtype/shape、采样配置 |
| 官方模型第 244–331、389–437 行 | VLM rollout 后 diffusion 与 action_to_traj；官方接口默认6条轨迹样本 | 本方案冻结离线单轨迹重复seed口径；不能称为原样官方6样本/minADE benchmark |

旧 helper `controlled_8b_diagnostic.py:32–83` 同样固定文本 seed42、单条生成与独立轨迹 RNG，计算 XY ADE但未存 GT/FDE。本草案依据源码而非模型命名推断数据形状或流程。

## 2. 科学问题、对照与实验单元

主问题：在预先冻结、没有参与训练/校准/监控/候选选择的新 raw clip 集合上，候选 A 或 B 自主生成的平均 XY ADE 相对离线 dense 教师是否至多增加15%。这是待完成前置门禁后开展的验证性非劣效比较；旧20clip及 planning pilot 均是探索/规划数据。

每个候选 k 的主效应为 `R_k = S_k/T - 1`，其中各 clip 先对轨迹 seeds42/43/44取均值，再对 clip 等权取均值。`H0_k: R_k >= 0.15`；`H1_k: R_k < 0.15`。通过时另要求点估计 `R_k <= 0.15`，并满足第6节预锁单侧上界与技术完整性门禁。15%是用户要求的实际界限，不使用默认 Cohen medium 效应量替代。

实验是同 raw 输入、同 GT、同轨迹初始噪声配对的重复测量设计。clip 是记录与汇总单元；三个 seed 是同一 clip 的重复测量，不能把 n_clip×3 当独立 n，也不以生成 token、64未来点或多候选条目扩充 n。metadata 无 drive/session ID，因此真正独立行程层级未知。chunk 是已知打包/关联标签，绝非已证实 drive；`drive_disjoint_verified=False` 必须保留。

### 2.1 冻结实验矩阵

| 臂 | 原始结构/资产 | 自主文本与请求路径 | diffusion | 地位 |
| --- | --- | --- | --- | --- |
| T | 原始 dense、官方 checkpoint、用户离线参考口径 | 同新raw输入，自己的自主前缀；冻结 full-prefill 比较路径 | Euler10 | 唯一教师；本轮新输出 |
| A | 已冻结 AWQ/恢复父资产，actualKV 候选 | 自主生成的实际 KV | Euler4 | 固定验证候选 A |
| B | 同冻结候选资产，hybrid_m1 既有 code_v3 | 自主生成后 full-prefill | Euler10 | 固定验证候选 B |

候选 A、B分别评价、未组合。fullprefill10 packed reference只作 B 的实现等价诊断，不能冒充 dense 教师。若需要新数据 reference/hybrid 等价检查，必须放入最终决策的额外诊断矩阵；不能据诊断择优调整候选。任何新组合、重校准、训练、权重/后端/精度/loader改动均形成新候选，原测试不再自动覆盖。

冻结文本配置建议延续已核对的离线口径：text_seed42、top_p0.98、temperature0.6、top_k=None、do_sample=True、num_return_sequences1、max_new_tokens256、pad token及停止/LogitsProcessor身份。教师和候选前缀允许不同；强制前缀相等会改变科学问题。对应 tokenizer、processor、chat template、history fuse、special token IDs、模型 generation_config 和采样 helper 都要进入冻结 manifest。本草案不声称新 dense raw→generation 入口已经实现或验证。

## 3. 数据资格、采样与泄漏控制

### 3.1 先完成 census，再决定最终名单

沿 holdoutprep v1/v2：复用已有固定 metadata 三 parquet 与经 blob 校验的 features.csv；资格为 clip_is_valid=True 且 egomotion/四camera完整。服务器既有306152行是历史记录，正式 census 应重新验证 bytes/SHA/唯一键/计数。`split=test`只作为优先候选池，不表示它是 Alpamayo 专门协议或独立行程测试集。

usage台账必须覆盖显式范围内 experiments、datasets、docs/baselines、scripts、compression，以及已发现 logs、incoming、旧clip delivery、旧audit、Self-realization 等目录；扫描失败、外部引用、匿名缓存映射与未索引资产逐项审定。所有历史引用 clip 的整 chunk 保守排除，包括0/3119，不能只排两个已知 chunk 就宣布覆盖完成。保持脚本 `coverage_complete=False`，由独立 Sol receipt 审定明确范围；未知历史风险仍须披露。具体范围依据状态文件第59–71行与准备决策，不猜外部目录。

旧20clip及三个 seed的开发结果只用于诊断和保守 planning，不进入最终检验。旧80training/20monitor_validation、校准、冒烟、候选选择及任何用于调整本方案的 planning数据均进入dev；其相关chunk继续排除，不能混回最终test。测试误差揭盲后不得重命名不利样本为dev以保留其余结果作首次验证。

### 3.2 抽样算法草案与预算分支

在usage审定、planning集合排除及最终 N 确定后，以剩余固定 revision 的官方 test 资格池 `E` 为有限抽样框。优先方案为对 clip_id 排序后，使用独立 `random.Random(20261010)` 无放回产生完整随机顺序，取前 N，剩余顺序作为仅用于输入QC的候补；记录Python版本、算法、输入集合SHA、名单SHA及chunk/metadata覆盖。所有clip等权，主要效应描述这份冻结测试集合，并在推断时明确抽样框/关联假设。

此草案算法有可能产生较多chunk、增加五ZIP下载量。最终预算尚未确定：先用候选文件计划和固定revision逐路径实际 bytes/SHA 报价，再决定是否执行上述抽样。若预算需要两阶段chunk抽样、限制每chunk clip数或分层不等比例配额，须在任何最终测试误差不可见时另行修订并冻结抽样概率、目标总体与加权分析；不能采集后压缩到便宜chunk却保留“全池代表性”声明。主门禁的等clip平均不暗换成权重门禁；若改变估计目标，需新决策明确该取舍。

不能为coverage选取看起来表现好的clip。真实country/month/hour_of_day/platform_class/radar_config可做预先覆盖盘点；没有这些字段不能猜天气、路口、车流或行程标签。若关键使用场景需额外标签，标签来源、编码者盲法和纳入标准应在最终采集前确定。粗粒度元数据有覆盖仍不证明困难场景覆盖。

### 3.3 输入窗口 QC 与替补

先在模型输出不可见阶段，用相同 loader与固定参数检查：五feature来源与revision、raw文件完整性、16历史点及64未来点时间覆盖、四camera各4帧、真实解码时间戳、camera ID/顺序、历史/GT finite、64×3 GT形状、插值和坐标变换来源、重复/相邻或完全相同输入hash。由于尚未读取新payload，解码偏差容忍度与egomotion是否超界插值的判定目前是待定技术项，须由官方数据API源码与planning QC确定，不自行臆造10ms等阈值。

QC失败只允许按预定同一候补顺序替补，记录原因、身份与时间，且发生在该clip任何教师/候选误差揭盲前。重复输入保留审计但不作为额外独立clip；其排除/关联方式在采集前冻结。QC后生成最终输入manifest并锁 N、chunk数、metadata覆盖、所有候补消耗。模型生成失败不能反过来触发输入替补。

## 4. 新采集记录契约

新 runner须另出实现决策并验收，不直接把现有2/20clip入口改参数当作独立测试入口。先保存planned requests `N×3×(T,A,B)`，每个键为 `(evaluation_id, clip_id, t0_us, trajectory_seed, arm, frozen_candidate_id)`；允许同一raw/GT副本用于三臂，禁止复用另一臂生成的continuation/KV。

### 4.1 输入/GT每clip必存

1. clip/chunk/官方split和真实metadata字段；数据revision；raw egomotion和每camera文件/成员身份与SHA；完整下载receipt及大小/校验来源。没有实际SHA则标pending，不用路径代替内容身份。
2. 实际loader参数：t0_us5100000、history16、future64、time_step0.1、camera_features、num_frames4、maybe_stream=False及API版本/源码SHA。保存历史xyz/rot与camera帧tensor SHA、dtype/shape、camera IDs、图像实际absolute/relative timestamps；这些供重复/配对与QC审计，不是模型输出。
3. `ego_future_xyz[0,0,:,:]` 的实际64×3 float数组，另保存对应GT tensor bytes SHA、shape、dtype、字节序、布局和规范化数组SHA；JSON hash须固定UTF-8、排序key、finite、数字编码。双方都绑定同一gt_id/gt_sha256，不能从两份预测反推出GT。
4. 保存实际 future时间轴。官方loader内变量为 `t0_us + [100000,…,6400000]`（102–117行），当前return未提供它，未来实现应在不改变数值的记录层显式导出/核对；记录实际egomotion覆盖与插值来源，不能仅凭64行默认时间成立。
5. 坐标语义为t0 ego frame：官方loader128–144行以 `R_t0^-1(x_world−x_t0)` 转换。记录frame_id、原点/朝向来源与可获取的实际t0 pose或变换receipt；不能缺值时编造零位姿。未来实现须核对米制依据（API/字段定义与源码身份）并存`units=m`证据，不能仅因字段名ADE_m就假定单位。坐标/单位身份未审定即阻止主门禁。

### 4.2 每请求必存

保存臂/候选/源码/权重manifest身份、开始/结束UTC、实际exit/status、text_seed与trajectory_seed、原始prompt身份、生成token IDs/数量、prefix length/SHA、future_start_token_id与边界检查、KV长度/route、noise SHA/dtype/shape、采样步数及Euler/温度/实际CFG参数、pred_xyz64×3和其tensor/数组SHA，pred_rot若实际可用则保存其有限性诊断。前缀不同不是配对失败；raw/GT/noise/生成配置错配是失败。

教师与每候选各自记录同GT的ADE和FDE以及64点距离序列；以独立CPU分析从保存数组复算，比较runner保存数值并采用预定浮点容忍度（待planning数值验收冻结）。teacher_ADE/FDE不得使用旧常数或引用旧JSON。保留生成失败token边界/错误类型及完整日志，敏感信息只保存安全定位，禁止抄入报告。

保存环境：指定L20主机身份、OS/Python/torch/transformers/CUDA/驱动、实际依赖/source/dirty状态、TF32及dtype设置、GPU进程/共享状态、loader与预处理身份、资源日志。按批次分别进程加载可行，但批次/顺序须配平：以冻结seed20261010生成clip块顺序与T/A/B批次臂序排列，避免整天教师先跑、候选另一天的固定顺序混杂；重置RNG与cache。不能为配平同时加载三模型造成额外内存需求。规划pilot需决定块大小并冻结，正式N/运行次序文件独立归档。

计时和资源只作本轮L20描述；磁盘解码、冷加载、raw→processor→生成→expert→轨迹、审计分别记录，e2e范围写清。不根据L20数据宣称Orin300ms或16GB通过，不把辅助reference的速度混入A/B。

## 5. 响应、失效与报告分母

对任意臂 a、clip i、seed s，用同GT局部坐标XY距离 `d[a,i,s,t]=sqrt((pred_x−gt_x)^2+(pred_y−gt_y)^2)`，t=1…64。`ADE[a,i,s]=mean_t d`；`FDE[a,i,s]=d[t=64]`；单位米。Z保存用于完整性，不纳入本次主指标。3D ADE、minADE或择优seed均不替代主指标。

定义 `T_i=mean_s ADE[T,i,s]`、`S_ki=mean_s ADE[k,i,s]`，`T=mean_i T_i`、`S_k=mean_i S_ki`。主ratio是 `S_k/T−1`，不是逐clip ratio的平均。整体T≤0、非finite或GT错配使主效应无定义，输出null与原因；T_i=0时逐clip ratio为null，同时保留绝对差。

所有planned请求先登记状态，不能只写成功项。模型OOM、timeout、future边界缺失、KV不符、shape错误、非finite、source变化、未完成/重复键等均保留；教师失败使该配对不可解释，候选失败使该候选完整性门禁不通过。仍可给成功子集的明确描述，但不得从主要分母静默删除失败、插补零误差、换种子或以成功子集ratio宣称通过。

主验证的技术完整性要求是冻结请求全覆盖、全部可配对且有效；这不是新造“安全零失败率”阈值。正式runner必须预锁timeout与资源边界（待planning），模型开始后的失败不以更换输入/种子重试取最好结果。只允许最多一次有日志证据的、模型尚未开始且没有输出可见的基础设施重启，沿同请求键/资产/seed，记录原尝试及复跑；任何模型执行后故障均保留首次失败并升级，修复若改变执行路径应撤销该批验证资格、转dev并使用新最终test决策。

报告分别列输入QC排除/候补分母、planned/attempted/completed/valid paired请求、受影响clip和chunk、各故障类型及所有缺失；不能把因chunk×seed×arm展开的条目数叫独立样本量。

## 6. 采集前锁定的统计分析

### 6.1 主分析与两候选检验族

固定family为{A actualKV4，B hybrid fullprefill10}，两个对同教师的非劣效结论，family单侧α=0.05。采用Bonferroni，各候选α=0.025、单侧97.5%上置信界；两个候选全部报告，不从未校正95%CI中择优。后续若新增候选不得并入同一已冻结测试后继续择优。

草案预锁主CI算法为配对chunk整群percentile bootstrap：先形成每clip的(T_i,S_Ai,S_Bi)，使用实际chunk label分组，G个观察chunk作为已知关联块；按升序chunk ID及clip ID固定输入顺序，独立`random.Random(20261010)`有放回抽G个chunk，完整复制每抽到chunk的所有已选clip及其seed汇总值，三臂共同抽样。每次按复制后的clip等权重新计算ratio；B=50000，使用排序后(n−1)q线性插值取q=.975。T*≤0/非finite的重复数须报告并视CI算法未满足门禁，不静默丢弃重采样抽到有效为止。分析脚本/版本/seed/算法和合成数值验收需在最终测试前冻结。

采用chunk整群重采样保留已知chunk内关联；它要求用于CI的chunk块近似独立、可交换且块数/大小支持该近似。metadata没有drive标识，这一假设未经证明。CI只能命名为“条件于所声明chunk关联假设和冻结抽样框的单侧上界”，不能推广为无限道路/行程总体保证。G不足、单chunk、极端不平衡或跨chunk强关联未知时，可报告观察集合ratio，但不能强行给可靠CI或完全验证通过。最终G/单块占比与CI数值稳定性接受条件必须用census/pilot规划冻结；本草案不拍脑袋设一个G即可充分。

候选k通过的必要条件是：技术完整性门禁全部通过，R_k≤.15且预锁U_k≤.15，已批准覆盖/关联适用范围满足，以及Sol独立复核。U_k>.15表示未能证明该范围内非劣；即使R_k≤.15也不能通过。不因两者中较好者通过就称组合系统通过。

### 6.2 敏感性与次要分析

- 配对clip bootstrap（同B、seed与上界水平）仅作为clip近似独立条件下的敏感性结果，不能覆盖或替代较保守的主块分析；重复seed始终随clip一起保留。
- leave-one-chunk-out ratio与最大单chunk样本占比，检验一块是否支配均值；逐chunk、实际country/month/hour/platform/radar分组给完整n与mean/尾部/失败。它们是诊断，不作事后筛选。
- 对跨chunk未知关联，在planning阶段用更粗的可用真实分组（如country×month）检查能否形成足够块；若可行，最终决策在误差揭盲前固定分组算法并复算整群CI。若只有极少组则报告无法推断，不强行bootstrap。country×month也不是drive。
- 样本量规划额外给ICC假设ρ={0,.05,.10,.25,.50,1}下的设计效应/有效样本量敏感性，或保留跨chunk合并块模拟；这些ρ是明确的假设网格，不是已测相关系数。关联越强的不确定性需披露，不能只选使门禁通过的假设。
- 候选family共享教师及配对数据；bootstrap同时抽三臂保留相关性，Bonferroni无需假定两候选独立。分场景和多个尾部指标只作探索描述，不制造额外未校正验证结论。

FDE是次要结局：保存教师/候选每请求FDE、clip seed均值、整体mean及绝对/相对差、分布和同法CI描述。未有用户FDE硬阈值，不擅自套用15%作为FDE拒绝门禁；但明显尾部异常须完整报告并解释，不据单一ADE通过声称所有质量维度安全。

尾部固定报告clip seed均值层级的ADE/FDE median、IQR、p90/p95/p99、max、教师与候选的配对差、逐clipADE ratio>15%数量（T_i=0另列）、所有退化记录与完整表。分位数统一(n−1)q线性插值。超过15%的clip数是诊断，不替代整体mean门禁；少样本p99明确不稳定。只用真实标签做分场景，列小组n与未知/缺失，不用模型误差猜场景。

舒适性仅在真实时间轴、坐标/单位、预测轨迹有限且相关导数定义预先冻结后，报告XY速度、加速度与jerk的离散差分（dt=.1s；相邻一阶/二阶/三阶，不额外平滑），含导数有效时间范围、GT/教师/候选和量纲。该结果是开环运动学诊断，没有人体/车辆阈值依据时不称“舒适通过”。实际道路曲率、姿态/车辆动力学不足时，不编造横纵向舒适性安全阈值。

没有障碍物状态、几何/地图/车身、有效交互演化与闭环环境时，TTC、碰撞率、违规率、闭环成功率一律标`not_measured`与原因。两条ego预测距离不是碰撞/TTC，也不是GT FDE。

### 6.3 离群值、缺失与分析冻结

finite但大误差一律保留，不winsorize、不截尾、不按结果删clip；只有第3节事前输入QC可排除。主分析不对缺失或模型失败随意插补；完成率门禁不满足时结果为incomplete/nonpass，同时保留描述和故障分母。分析字段、精度容忍度与超时在最终决策锁定；结果揭盲后任何修订都记偏差，原计划结果也报告，修订分析标探索性。

## 7. 样本量、功效、预算与停止规则

当前N、G、输入QC脱落率、下载字节上限、GPU小时预算均待定，不能以“100clip够用”、单chunk或旧n20×3宣布完全验证。旧20clip已反复参与选择，开发均值和方差可能乐观，只能作为带选择偏倚声明的保守规划参考；不计算事后功效证明已通过。

先完成usage census与文件报价，预留从未使用的planning pilot与最终test相互排斥集合。planning pilot用于raw/GT记录、有限值/边界/失败、运行成本、分场景/相关性、 paired margin `D_ki=S_ki−1.15T_i` 的方差与教师均值规划。任何pilot结果用于调整候选、QC、CI/方差/关联或N，即全体转dev并排除关联chunk，绝不能计入最终检验。这一阶段也须另出可行性决策，不由本草案启动模型。

采集前N依据应包含：15%界限与预期余量、pilot/旧dev来源及其偏倚、两候选α=.025、目标power暂定≥.80（对哪个候选/最差规划臂明确）、方差和块关联上界、QC脱落与模型失败情景。可用配对margin正态近似 `n ≈ ((z_(1−.025)+z_power)×sd(D)/abs(E[D]))²` 作初值，仅适用于明确近似；再按chunk数量/大小、重尾、双方相关性、预算和预锁bootstrap流程做采集前Monte Carlo规划，锁模拟seed/代码/输入。skill的两独立样本ttest脚本不直接适用于此配对ratio/整群设计，不能机械套默认d。

规划同时考察多个预期真实ratio与ρ网格，给N/G和CI覆盖/功效敏感性及GPU小时/去重五ZIP真实总bytes。不能仅选能支持预算的效应量或相关性。若预算达不到预设精度与coverage，应明确缩小验证范围或只做可行性诊断，不能改用一小块数据仍称全量精度充分。

固定N设计为默认停止规则：最终名单/候补/实际QC后N与所有请求数在模型误差不可见时锁定；完成规定请求或触发技术停止即结束，不看R/U逐批加样直到pass，不提前停在有利CI。若未来选择序贯方案，需在最终采集前另冻结最大N、look时点、α spending与功效，不事后转换。技术停止（资产/source变化、GT身份冲突、OOM/资源边界、不可解释配对）留日志，不能借停止更换不利数据。

## 8. 归档、盲法与验收产物

建议内部仓库归档：最终采集决策+本草案SHA+变更差异；usage扫描manifest/独立coverage receipt；候选/教师/source/依赖manifest；census/资格池、planning排除、抽样随机顺序/seed与预算quote；输入QC/候补/最终raw-GT manifest；planned-requests与冻结运行顺序；所有stdout/stderr/UTC/exit/resource logs；每请求数组和错误；独立统计脚本/单元及合成算法验证、完整per_request/per_clip/per_chunk/per_metadata tables、分析manifest与产物SHA。正式输出新建拒覆盖，不改旧结果。

角色上，资格筛选与QC人员在最终冻结前不得使用候选误差；数据管理员可见raw/metadata，候选选择不得接触最终test误差。首次执行时允许为故障检测监控状态，但不按看到的ADE改变名单/候选/样本量。结果分析先按固定脚本生成全部臂报告，再由独立Sol核对数组、hash、GT、完整分母、CI算法与限制。盲法无法完全执行之处如实记录，不写虚假blinded。

验收命名分层：工具契约通过、usage范围审定、raw输入QC通过、离线精度在声明范围内通过、闭环通过、Orin实机通过、官方原入口通过分别记录。后几项本草案当前均未提供证据。报告“完全验证”时必须列出范围和所有未测维度，不能抹去drive关联、抽样coverage或设备/闭环缺口。

## 9. 从草案进入 ready 的必填清单

| 门禁 | 当前状态 | 最终需要的可核验项 |
| --- | --- | --- |
| 精度审计/准备工具独立验收 | 待主任务最新复核 | 冻结版本、实际CPU tests/CLI、负例与失败留痕、验收receipt；旧状态不能代替 |
| usage覆盖与未解决资产 | pending | 完整显式扫描范围、匿名缓存/外部引用审定、整chunk排除与独立coverage receipt |
| 资格池/抽样覆盖 | pending census | 实际eligible counts、test池、metadata覆盖、chunk关联/重复审查、最终抽样目标与算法 |
| 下载预算与payload | pending | 五ZIP去重实际bytes/SHA报价、可接受规模、用户执行下载命令及receipt；本任务不代下载大文件 |
| planning pilot | pending独立决策 | 与最终test排斥、用途/成本/方差/关联来源、转dev ledger |
| N/G与power/CI条件 | pending | 预期余量、方差/ICC依据、采集前模拟、固定N/G与单块占比/数值稳定性接受条件、QC补量 |
| 教师与A/B完整资产冻结 | pending | 新运行实际权重manifest/payload、overlay/packed、源码/dirty/dependency与保护SHA；A/B分别冻结 |
| 新raw→自主生成runner | 尚未实施 | 参数/入口/候选路径、teacher自主前缀、无共享continuation、GT保存及独立复算验收 |
| GT/坐标/单位/时间轴 | pending输入QC | 实际ego_future_xyz+SHA、64×3、采样时间、t0变换、米制依据、插值覆盖与有限值 |
| 超时/重试/浮点容忍度 | pending planning | 采集前冻结阈值与日志规则，不根据最终误差调节 |
| 统计实现与运行顺序 | pending | 两候选family .025、B50000 seed20261010、chunk主CI及敏感性、数值验证与随机顺序文件SHA |
| 最终采集决策 | 未批准 | Sol签发范围/命令/环境/文件所有者/停止条件；独立Sol验收计划 |

以上待定项未消除前，本文件仅为完整的采集前设计草案。不得凭本草案启动新模型、下载或宣称独立测试就绪；planning pilot也须其自己的决策和排斥最终test安排。下一步可继续已授权CPU census/工具验收，将真实资格和预算信息带回最终决策。最终目标保持，现有候选和旧报告保持原样。

## 10. 本轮执行记录

- 本轮读取开始与写作归档时点以当前工具会话UTC记录为准；写作时只读时钟观测为2026-10-10 04:44:05 UTC（12:44:05 Asia/Shanghai）。该值不是后续模型采集开始时间。
- 实际环境：本地macOS工作区 `/Users/USER/Documents/ChatGPT/Lighten_VLA`；只用shell文本读取、git身份、shasum与apply_patch，未导入模型/GPU库。
- 只读命令类型：`nl -ba`/`sed`读取上述文档与源码；`rg --files`/`rg -n`定位；`git -C alpamayo1.5 rev-parse HEAD`、`git -C alpamayo1.5 status --short`；`shasum -a 256`核对表中实际文本文件。定位时两个不存在的本地script路径与compression目录返回rg exit2，随后改读已存在evidence源码；未替换来源或执行新模型。
- 实际变更仅本新文件。文档最终SHA由主任务在文件不再写入后重算登记；不在正文嵌入自身SHA造成自引用。独立验收与最终N/GT/资产身份仍pending。
