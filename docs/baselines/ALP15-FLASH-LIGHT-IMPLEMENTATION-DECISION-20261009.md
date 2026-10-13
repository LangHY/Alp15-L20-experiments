# FlashDrive / LightVLA 初阶段实施决策

日期：2026-10-09，Asia/Shanghai。决策编号：`D-ALP15-FLASH-LIGHT-IMPL-20261009`。决策者及本文唯一写入者：`flash_light_adaptation_decision_20261009`，GPT-6.1 Sol high。风险：高；执行及独立验收使用 GPT-6.1 Sol high。目标保持 **Orin NX16GB、每条完整请求 max≤300ms、同任务教师 meanADE 损失≤15%**。

## 1. 可立即分派的决定

现在并行实现两个独立的新机制接口：FlashDrive 侧首项为**请求局部 Euler velocity cache**；LightVLA 侧首项为 **Qwen3-VL keep-all 等价接口及完整生成/Expert 紧凑位置映射**，等价门禁通过后可测试本文预锁的非学习视觉筛选。接口开发、CPU fixtures、离线 preflight 现在就做，不等待新增训练数据，不等待正在运行的 full20 结束。

FlashDrive 首项选择 velocity cache 是因为它能复用冻结 Expert 与参数存储，只改变显式求解步骤，不需要复制整个 Expert 或改其 attention/RoPE；它是新的有损因素，不能归入旧 actualKV 豁免。Expert gate/up 与 QKV 融合、静态 KV、graph、跨窗缓存、PARO/DFlash 不在本阶段执行范围，后续可另列独立决策。LightVLA 本阶段完成真实推理接口及有限固定策略的开发评测，不声称复现或达到原 LightVLA 训练策略的精度/速度。

本决定承接 `ALP15-JETSON-300MS-ROUTE-20261008.md` 及 2026-10-09 主任务收到的直接持续优化授权。用户后续明确要求尝试两者适配，覆盖旧 AGENTS 优化等待门禁的适用限制；**这不是 NVIDIA 原入口基线或 Jetson 门禁已通过**。不得使用 superpowers。当前 GPT-6.1 Sol high 角色沿用主任务记录的后续模型配置。

主任务已通知启动冻结 full20 的 reference（PID2084464）；本决定不修改、停止或重启该流程。所有服务器上传、SSH、实际模型装载及 GPU 作业仅由主任务串行安排；两名执行者本阶段不 SSH、不启动模型/训练/GPU、不下载大资产、不安装包、不执行全包 uv sync，不修改既有 venv、官方模型源码、旧 helpers、旧 runner 或历史结果。

## 2. 本次核实的事实依据

已完整读取仓库 `AGENTS.md`、`ALP15-FLASHDRIVE-LIGHTVLA-NEXT-DECISION-20261008.md`、`ALP15-FLASHDRIVE-LIGHTVLA-ADAPTATION-20261008.md`、`ALP15-KV-FULL20-RESULTS-20261009.md` 与 `ALP15-HYBRID-252-RESULTS-20261009.md`。下列来源为本次再次核实，不以论文名称猜测接口。

| 身份/事实 | 本次证据 |
| --- | --- |
| FlashDrive 原作者 commit | GitHub commit API 返回 `26265174f0413db77c8a16d99993dbeecacf458f`；[fusion.py](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/fusion.py)、[diffusion.py](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/diffusion.py)、[_backbone.py](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/_backbone.py)公开原始字节读取成功 |
| FlashDrive velocity cache 差异 | diffusion.py 使用 BF16 time_grid 和持久 `diffusion._cached_velocity`；本项目原快照 `_euler` 使用默认 FP32 noise/time_grid，保留广播 dt 与 CFG 公式。因此仅适配复用 velocity 的机制，不能直接复制原实现 |
| LightVLA 原作者 commit | GitHub commit API 返回 `a4680fda5ffe73029190ac97328aa34b0e87a45a`；[modeling_prismatic.py:49-130](https://github.com/LiAutoAD/LightVLA/blob/a4680fda5ffe73029190ac97328aa34b0e87a45a/prismatic/extern/hf/modeling_prismatic.py#L49)是 task 条件筛选，CLS/固定 patches/task 分割，下游为 PrunedLlamaModel；非 Qwen 插件 |
| HF Qwen 版本 | [v4.57.1 modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)公开 raw SHA256=`dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da`，与 smoke 实际 runtime 库存一致。固定 tag 解析 commit 已由上轮记为 `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc`，本轮以实读字节 SHA 绑定 |
| Qwen 关键接口 | text forward: 三轴 position_ids、compact cache_position、逐层全部 DeepStack 注入；top model: 完整 grid 算 mRoPE、placeholder 数与 features 核对；`prepare_inputs_for_generation` 在 super 调用后将 position_ids 清为 None，decode 去掉 pixel_values。因此只传预算 position_ids 到 generate 不足以保持位置 |
| 原 Euler/Expert | `evidence/jetson_300ms_20261008/source_snapshot/flow_matching.py:139-196`；`alpamayo1_5.py:162-239,330-381`；`evidence/quant_route_20261008/full20/code/expert_interface_distill.py:30-51`。Expert 语义位置为 `offset + rope_deltas + action_index`，mask 使用实际 KV 长度与 prefix_mask |
| NVIDIA/权重/数据身份 | smoke 记录源码 HEAD=`36aeb4c5938cbc2eb2aed33b22434773da4ab639`，base_model/delta_tokenizer dirty；checkpoint_revision=`7aba8293c09993f2e125c6819df05d7fa3e873ea`，dataset_revision=`33f9bf447ed3bcb7d545ce13f4226f824214fafb`。实际文件 SHA 仍须逐份冻结，HEAD 不意味着官方原样 |
| 当前 hybrid 实证 | `evidence/jetson_300ms_20261009/hybrid_smoke2_v3/result.json` SHA=`4aaa6d68fcfa0ef327542e797ef6dcb52a75dbdbd765d1128973069b747db6cb`，exit0、strict6/6、64×3 预测逐元素相同；E2E mean2995.606/max3377.170ms，6/6>300ms。两clip开发试验不证明泛化/Orin性能 |
| 252 实际矩阵 | HYBRID252 报告：252/252 random/actual 数值容差通过、65个输出非 bitwise；只能 M1 fusion，M>1 reference。现有 hybrid 严格配对验收仍保留 |
| KV20 开发结果 | actualKV4 meanADE=1.245025m、教师1.2323994789272548m，上限1.4172594007663428m；actualKV10 strict仅2/60，因此 actualKV4 联合仍是显式有损候选。generation 约4981.60ms，共享L20，非Orin |

冻结实际新接口复用的旧文件：`scripts/alp15_awq_hybrid.py` SHA=`026eadf0bb1924e0356993e60172d732423e27264862aa751b792c3b6e791672`；`scripts/run_alp15_awq_hybrid_eval.py` SHA=`fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd`；`scripts/alp15_kv_steps_probe.py` SHA=`74886a867a8aae480dd848a6d068d2e3f817674d891774bcdabeb57f412292f8`。执行者只读复用，先核 SHA，真实 imports/动态来源/virtual 来源/receipt 不得删减。

## 3. 文件所有权与并行边界

以下为拟新增文件。不存在即创建，已有同名文件则先核归属并升级；禁止覆盖未知来源。

| 所有者 | 唯一可写文件 | 必须交付 |
| --- | --- | --- |
| Flash 执行者 | `scripts/alp15_flashdrive_velocity_cache.py`；`scripts/run_alp15_flashdrive_eval.py`；`tests/test_alp15_flashdrive_velocity_cache.py`；`docs/baselines/ALP15-FLASH-VELOCITY-EXECUTION-20261009.md` | 请求局部 cache helper、接冻结服务器资产的新完整请求入口、CPU回归/失败记录/实际CLI |
| Light 执行者 | `scripts/alp15_lightvla_qwen_adapter.py`；`scripts/run_alp15_lightvla_eval.py`；`tests/test_alp15_lightvla_qwen_adapter.py`；`docs/baselines/ALP15-LIGHT-QWEN-EXECUTION-20261009.md` | 完整 compact/mRoPE/DeepStack/生成/Expert 接口、keep-all及固定policy开关、新入口、CPU回归/实际CLI |
| 主任务 | 后续新增 `scripts/run_alp15_flash_light_joint_eval.py`、联合launcher与验收文档，实施前明确其独占所有权 | 合并调用两个公共 helper；串行 GPU、现场 hash、同代码同资产配对及联合验收 |
| 决策者 | 本Markdown | 初阶段决策；其它文件只读 |

两侧公共协议锁为 JSON 可序列化 provenance、request_identity、variant_config、stage_times、mechanism_counters；helper 不主动 IO/网络/模型下载。runner CLI 至少接 `--root --awq-artifact --output --clips 2|20 --candidate-decision --awq-forward reference|hybrid_m1 --request-route fullprefill|actualKV --steps 10|4`，新增显式侧开关，默认关闭；输出目录排他创建。先交付可执行 `--preflight`（不加载模型）与 `--help`，不能停在接口草案。

旧 helper 的决策/10step约束不可由 monkeypatch 常量绕过。新 runner 为本决策保存新的 validator/结果schema，只读调用旧纯函数；对于 steps4、有损侧或 compact prefix 使用独立门禁，不能把旧 validator 拒绝隐藏成 passed。公共来源库存按实际 import 收集，未改的部分 SHA 完全相等；新增 helper 各自绑定 SHA。不得复制整套旧源码造成不受控双份实现，可复用来源/加载/指标纯函数；必要最小 adapter 必须说明复制行与来源，冻结 hash。

## 4. Flash 首项：请求局部 velocity cache

1. 实现 `RequestVelocityCache` 或等价请求上下文、`sample_euler_with_velocity_cache(...)`，状态仅存在于单次 sample 调用栈，不存 model/diffusion 的持久 `_cached_velocity`。每请求/每clip/每seed/每CFG分支重新建立并在 finally 释放。原 helper、FlowMatching 类与 Expert 不修改。
2. `cache_steps=[]` 必须通过**真实新 helper 路径**完成与原 `_euler` 对拍，不能简单委托原函数来掩盖新积分器差异。保持官方默认 FP32 torch.randn、temperature、FP32 torch.linspace、dt/t 的 shape/broadcast、CFG权重公式、return_all_steps 与 RNG消耗顺序。先以冻结入口当前CFG配置运行；CFG支持必须按 guided/unguided组合后缓存完整velocity，若未实现则显式拒绝CFG True，不能静默改配置。
3. cache_steps 必须严格唯一、升序、整数、0≤i<steps，禁止0；空cache等价；每个非缓存步刷新本请求velocity；缓存步仍执行原 Euler 更新，计入实际积分步数。禁止跳过动作积分、缩短time_grid或改变noise/time dtype。复制/alias需要保证前步velocity不被后续原位写覆盖，并记录额外bytes。
4. 预锁因素：10steps无缓存对照与 `[1,3,5,7,9]`（5次Expert预测）；4steps无缓存对照与 `[1,3]`（2次）。记录CFG实际Expert forward总数（开启CFG时可能翻倍），cache_hit/miss、velocity dtype/shape、noise hash、time_grid hash。保持4步与10步为不同因素；不得用4步cache对10步无cache来声称cache独立收益。
5. 新runner接旧实际packed252+hybrid_m1资产；本阶段先`fullprefill10 + cache=[]`严格配对，再`actualKV4 + cache=[]`验证接已批准路线，然后才单因素有损缓存。不得改变原生成策略/文字seed/256token上限/未来边界/64点dt0.1。GPU作业由主任务启动。
6. CPU fixtures覆盖空cache真实积分等价、逐步x/t/CFG/return_all_steps、调用5/2与10/4、非法列表、两请求两seed隔离、异常退出清理、NaN拒绝及结果失败记录。synthetic CPU不声明真实CUDA或驾驶质量通过。

## 5. Light 首项：Qwen keep-all 到 compact 接口

1. batch=1、num_return_sequences=1；先只支持冻结真实请求的 image/video组合及二维prefix mask，未支持组合/批次/beam显式拒绝。视觉encoder/merger仍完整计算。使用原processor/history fuse，保留全部非视觉token（历史、时间戳、相机/视觉起止、任务、文本、special/未来边界），只筛选 merger 后进入语言模型的视觉placeholder对应行。无ViT加速声明。
2. 在**未删token的原完整序列**与原grid上调用原 `get_rope_index`，保存完整三轴位置/原rope_delta及 next_semantic_position。先按原 `get_image_features/get_video_features/get_placeholder_mask` 验证完整特征数量和原embedding scatter；再建立同序唯一 keep-index。同步 gather input_ids、完整inputs_embeds、attention_mask、三轴position_ids、visual_pos_masks及**所有**DeepStack分支。DeepStack是仅视觉行，使用全序列索引映射出的视觉行索引，不能直接用全prompt索引gather它；混合image/video按照原源码联合顺序合并，长度/分支数不符拒绝，不截断zip。
3. `policy=keep_all` 也必须经过同一gather/位置/生成适配路径并取全部视觉，证明实现接口等价；不能直接delegate旧forward当作新接口验收。特征提取、selector、gather、prefill、decode、Expert分别计时，所有必要新开销进入完整请求。图捕获、跨请求vision/KV reuse本阶段关闭，每请求全部重建。
4. 为新请求建立显式 `CompactRequestContext`：完整prompt长度 N，compact长度 K；`delta_compact = delta_original + (N-K)`，等价于 `next_semantic_position-K`（batch1无padding冻结口径）。generated text第j个有效位置=next_semantic_position+j，三轴同步；compact物理cache slots连续0..K+j-1。保存完整序列得到的next位置，不能由keep patch最大坐标重新推断；禁止把compact IDs+旧grid再送原get_rope_index。
5. 必须处理HF4.57.1 `prepare_inputs_for_generation` 清空position_ids：新文件中用**当前模型实例、单请求作用域**的generation prepare/forward adapter（或等价显式GenerationMixin集成），保留原签名/参数校验/LogitsProcessor/StopAfterEOS与抽样顺序，实际prefill明确传compact embeds、gather后的全部DeepStack/visual mask和三轴位置到原language_model；decode明确传位置，并维护compact rope_delta。不可全局patch transformers类或import FlashDrive全包。finally恢复原实例callables与rope_deltas；异常及嵌套/并发请求拒绝或隔离。位置绝不能再次被HF默默清空。
6. generate使用compact input_ids作为真实prefix，保留生成token IDs与future_start。在StopAfterEOS后仍要求 sequences[:,:-1]末项=future_start、KV length=该prefix长度；记录compact future_offset。Expert调用原 `_build_expert_pos_ids_and_attn_mask`，参数为compact offset、compact delta、实际compact KV seq_len与compact prefix_mask。证明 `offset_compact+delta_compact = offset_original+delta_original` 对相同生成continuation成立；64 action tokens位置不变、mask宽度正确且future边界后的gap不可见。不能只改VLM位置漏掉Expert。
7. 身份映射至少 `(stream_id,clip_id,camera_id,frame_timestamp_us,merged_patch_t/h/w,original_prompt_index)`→compact_index→cache_slot→mRoPE三轴；记录grid/preprocess/prompt/history/hash、keep-set SHA、policy版本、source/overlay/packed/hash、有效长度。当前每请求重建，reset/帧乱序/不同keep-set/相机顺序/分辨率/窗口差异必须清空上下文，相同shape不是reuse依据。
8. 预锁非学习policy为 `spatial_uniform_75pct_v1`、`spatial_uniform_50pct_v1`。每个camera/frame视觉组分别保留 k=max(1,ceil(r*n)) 个merged patch；flatten原t/h/w顺序用 `floor((j+0.5)*n/k)`，j=0..k-1，取唯一升序索引；token/group边界由真实grid及placeholder映射，不硬编码patch数。比例r=0.75/0.50与公式采集前冻结，不按当前20clip择优改规则。该策略是LightVLA启发的**非学习、有损新候选**，不是其task条件训练策略。独立训练策略留后续决策，当前可完成所有接口工作。
9. CPU fixtures用至少两个frame/camera、不同视觉组长度、三轴不相等位置与多个DeepStack分支验证keep-all、两比例、全部非视觉保留、单调映射、生成prepare不覆盖位置、连续decode/Expert不变量、mask/长度错误拒绝、两request/exception恢复及来源漂移失败。测试不得只验证孤立selector，而不覆盖generation/Expert adapter。

## 6. 门禁、顺序与预锁评测

先完成两侧CPU回归、py_compile和CLI preflight；主任务固定代码目录/四类来源库存/资产/实际环境，再独立验收。旧同代码来源限制继续生效；新代码必须做新的同代码配对reference，不能拿旧smoke的decision/code SHA强塞通过。已验252探针可作资产/算子绑定证据，只要其真实来源和packed/保护身份仍满足，不把它当新请求接口质量验收。

| 顺序 | 比较/门禁 | 通过后允许 |
| --- | --- | --- |
| G0 | CPU+preflight；不加载模型；来源、排他输出、禁用开关、失败JSON及保护库存完备 | 主任务排队GPU smoke |
| G1 | 同代码2clip×3seed：新Flash cache=[]、新Light keep_all对 frozen hybrid/fullprefill10；相同token/noise/future/prefix，ADE差≤1e-4m、轨迹NMSE≤1e-3，64×3有限，保护参数逐字节不变 | 无损接口全20；不得自动放宽 |
| G2 | 新base actualKV4+hybrid_m1（两侧关闭）对相同资产已批准actualKV4；确认完整请求路由、无重复prefill、噪声与生成边界配对；旧actualKV严格失败另保留 | 开启预锁有损因素 |
| G3 | 缓存单因素10/4各对相同步数无cache；视觉两比例各对同actualKV4+hybrid基准；2clip×3seed，来源/noise/data/时间网格硬配对，无NaN/失败，smoke配对子集teacher meanADE≤1.15倍 | 各候选同代码20clip×3seed开发扩评 |
| G4 | 固定joint四组：B=actualKV4+hybrid/keep_all/cache[]；F=B+cache[1,3]；L=B+uniform75%；FL=B+cache[1,3]+uniform75%；F/L单侧门禁通过才联合。50%只作预锁次级视觉候选，不能未声明替代联合75% | 开发候选冻结；独立clip/实机决策 |

G1对所有未启用的新机制要求strict；若新接口失败，停止相应侧，保留失败，不借用actualKV损失授权、不直接进入有损扩评。G3/G4允许质量变化是本独立决策的显式授权，严格数值失败如实保存；仍要求noise同seed同shape、原输入/保护/来源不变。视觉有损可能改变生成token，**不要求同旧文字prefix**，但必须自主generate、合法future_start、完整噪声/时间grid和同任务观测/GT；不能使用教师续写或挑掉未生成future的clip。将“生成改变”与“来源/边界错误”分别记录。

独立单位为clip；seed42/43/44是同clip内重复，不增加独立n。开发20clip反复筛选已存在选择偏倚；本次所有20clip只能叫开发集，不能宣称泛化、原LightVLA达标或独立验证。全20教师meanADE固定1.2323994789272548m，上限1.4172594007663428m；2clip用对应clip/seed教师均值，不能套全20阈值。汇报clip内seed均值后跨clip均值、配对损失、FDE、clip ADE p95/max、失败率及>300ms数。不挑参数后充作test；冻结后需要独立clip和真实Orin门禁。

使用experiment-design技能采集前锁定区组与顺序：clip为区组，seed为重复，每block四组按B,F,L,FL的循环移位（block_index%4）排序；记录实际顺序、首次/稳态、其它GPU负载/温度/频率。既有full20先reference后hybrid是历史描述性试验，不回写成随机化。新variant先按组配置构建/清理上下文，权重与256token上限不变。若首次/JIT只存在某候选，单列而不剔除失败。

20clip×3seed=每variant60请求；核心四组240请求。按当前hybrid均值约3s仅作**L20排程粗估**，240请求净请求时间约12min；加加载、审计、首推、指标、共享GPU干扰可能明显更长，不是完成承诺或Orin推断。单侧smoke六请求按3s估约18s/variant，时间根据实际请求分段修正，不用论文倍数相乘。10steps cache若expert占比小只能节省该部分；视觉后筛选不省ViT且仍保留语言decode。两项目前都没有证明足以把3s变300ms。

完整计时必须含rawimages→processor/historyfuse→完整vision→selector/gather→自主generate（prefill/decode）→必要cache重建→Expert→action_to_traj及同步；磁盘decode/传输若部署必需须最终计入，可在L20另列但不得藏掉。每请求记录first/steady、wall及阶段、真实cache/selector/M1 dispatch、tokens/keep数、memory/RSS/PSS/system available；不相加RSS/CUDA峰值。低峰loaderv4仍独立验收，新侧不得把其pending写作通过。先可在冻结旧loader开发验等价，Orin部署还需低峰loader与统一内存证据。

## 7. 保护、失败与最终验收

源/权重/packed/protected SHA漂移、shape/placeholder/DeepStack不匹配、非单调keep、缺未来边界、噪声污染、wrong delta/cache slot、NaN/OOM、receipt/动态来源丢失均立即停止该新候选。源码助手文件和原模型只读；velocity/Light接口不改任何参数，因此所有保护参数应完全一致，persistent storage不新增大权重副本。增量vision/DeepStack/gather/KV/velocity内存实际计量。任何fallback均必须显式记录failed及原因，不能标completed；已有实验目录拒绝覆盖。

失败保存argv、决策/variant、环境、源码/资产hash、请求identity、输入映射/keep-set、已完成rows、traceback、exit_code与完整日志。无loss smoke未过不得进full20；有损开发meanADE>15%、边界失败或质量尾部明显退化记录为不接受候选，不靠静默换比例/省clip解决。执行者发现原接口与事实不一致先在安全边界停止并报告主任务，由Sol修订决策，禁止自行替换路线。

回滚：关闭新侧开关、退出新进程/恢复本实例原callables，重新使用冻结旧入口；不删除原资产/结果、不停止其它项目。执行回报列开始/结束、实际命令/日志/退出码、文件SHA、CPU与CUDA分开、逐门禁证据、来源偏差与已知限制。独立验收须在代码交付固定点之后，不把执行者自报当验收。

最终目标只有实际Orin NX16GB冷启动/首窗/持续稳态/reset与恢复、aarch64/JetPack/SM87实际依赖与kernel、全系统统一内存无OOM/无swap掩盖、每条完整请求max≤300ms且超限0、独立同任务teacher meanADE损失≤15%联合成立才达标。当前没有Jetson实机证据；大权重/镜像/数据继续由用户按核实来源下载，小接口源码允许定向公开读取，禁止为了适配整包安装或自动大下载。

## 8. 本次决策留痕

本机 macOS，cwd `/Users/USER/Documents/ChatGPT/Lighten_VLA`；本次只读本地Markdown/小源码/JSON，公开固定commit API/raw源码读取，`shasum -a 256`核实际字节；无SSH/模型/GPU/包安装/大资产下载。web-access完整读取，前置 `node .../check-deps.mjs`：Node22.23.0可用、Chrome未连通、exit1；依技能工具表采用公开静态curl，不操作用户浏览器。experiment-design完整读取并用于预锁四组消融、clip层级、区组顺序、失效/失败与开发属性。唯一新写入文件为本文；未声称新helper、CPU测试或GPU门禁已经通过。
