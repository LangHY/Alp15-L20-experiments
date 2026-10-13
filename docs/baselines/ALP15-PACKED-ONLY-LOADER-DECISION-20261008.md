# Alpamayo 1.5 packed-only低峰值加载正式决策

日期：2026-10-08（Asia/Shanghai）。决策编号：`D-ALP15-PACKED-LOADER-20261008`。决策者及本文独占写入者：GPT-6.1 Sol high / `flashdrive_lightvla_next_decision_v2`。风险高；执行与验收由GPT-6.1 Sol high，GPU归主任务串行调度。本决策承接用户持续优化授权与`ALP15-JETSON-300MS-ROUTE-20261008.md`，不把旧官方入口豁免记作历史验收成功。

## 决定与仍未证明的前提

批准独立实现 **原构造器的meta骨架 → 用占位模块移除252个dense投影 → 907个保护权重按≤32MiB行块读取/直装目标设备 → 逐文件装入既有packed252 → 恢复非persistent状态 → 严格hash/轨迹对拍**。第一阶段只运行无CUDA的实际meta构造与文件头部探针；**CPU-meta探针未成功前禁止实体加载**，不能因源码看起来支持meta就称loader可用。

此项解决Orin NX16GB真实冷启动峰值，保持已有AWQ数值和模型任务。它不改变推理内核、actualKV、Euler步数或图捕获，不降低完整请求max≤300ms、teacher meanADE损失≤15%、统一内存16GB的最终联合门禁。既有10.27GB storage不等于实机够用；加载成功也不代表请求时延通过。

本次只读审计未运行GPU、模型、服务器写入或包安装。Mac默认`python3`与工作区`.venv`缺少相关依赖；随后核实`/usr/bin/python3`可import torch2.8.0，但没有transformers/accelerate/safetensors/hydra，不能在这里证明完整meta构造。原服务器已有torch2.8.0+cu128/transformers4.57.1，实际其余包版本、关键源码hash需执行者在既有环境只读核实；不用本地根目录简化`uv.lock`替代服务器环境。

## 一手源码与冻结现场

NVIDIA仓库固定commit：`36aeb4c5938cbc2eb2aed33b22434773da4ab639`，checkpoint revision：`7aba8293c09993f2e125c6819df05d7fa3e873ea`。现场实际源码以smoke_v3记录及`evidence/jetson_300ms_20261008/source_snapshot/`为准：

| 文件/资产 | 本次核查与SHA256 |
| --- | --- |
| 现场base_model.py | `1f2e49a2ac62346cc61c3ea954982729da1a92674bc4ceed7bb43c87ba212cc2`；Mac源树同名SHA=`94425eef...`，不拿其替代现场文件 |
| 现场alpamayo1_5.py | `87b23954570f4d466a6d0dd64090f779af635433b61106eb3b028d8d802ff1e8` |
| config.py | `224ba379806886f3e4b122dc72fb527eaa3e11f9518576555eb0e540581dc6be`；本地/官方固定源码一致 |
| action_in_proj.py | `11a21d317c9b8df1097a00d32c8f93cc96e46eeda73f4c16948a4be84205f240`；与smoke现场记录/本次官方raw相同 |
| unicycle_accel_curvature.py | `1e8035a3f2456b82219e96f7058fe2c0d94410a60367fe7395df3f22b4bedf9e`；与smoke现场记录/本次官方raw相同 |
| models/config.json | `824fc3552466aaecb67c896a453667e1e15c5687adbb416ce42a6bda3de1e68e`，3058B |
| models/model.safetensors.index.json | `b899e51816e15a96edc7cd57a01cd36c5e39fe1bcbf1537c4ac914f0cad43a0d`，104778B |
| AWQ completed_full20_v4/result.json | 本次实际SHA=`16aa6b20915bac68597614361e3932c89b955e7c05c024f1f5025b435ddd76cc`，与既有runner常量一致 |
| 父overlay manifest | 本次已完整核查`evidence/jetson_300ms_20261008/loader_provenance/parent-overlay-manifest.json`，SHA=`d0133763c495781933cf9e5f6e32726b3c2577cb00319a27eb7c0c57bcc27d5c`；服务器来源`experiments/mlp_newdata8_recovery_20261002_165238/width_5376/distilled/manifest.json`，主任务核实服务器/本机相同 |

父manifest实际36层全original_width12288/keep_width5376；每层5376个keep_indices均为合法、唯一、升序整数；recovery为MLP LoRA merged BF16、rank32、selected_step400。row键仅layer/original_width/keep_width/keep_indices/calibration_token_count/sha256，没有保护权重的新增来源；实际文件仍须逐个核实三tensor头部及SHA。manifest的`protected_sha256=77b2770ec860d12e3c69db459a4439d4fa7cc66cbb3079c45a240b35a016f094`排除所有MLP，而AWQ guard排除全部252投影，两者口径不同，不互换。

### 实际checkpoint头部

本机五个原safetensors文件实际存在但被ignore；本次用Node只读取首8字节长度和原JSON头部，共142096B，未取权重payload。1159个键一一对应index，无重复，全部BF16。完整payload本轮未重新验hash，后续沿冻结五分片SHA严格核验。

| 分片 | header字节 | header原字节SHA256 |
| --- | ---: | --- |
| 00001 | 51368 | `49df741ba611dbe3ca4dbba7cff0b35dd73de4cb0131178f374acd89d3d6058a` |
| 00002 | 19088 | `89c487d89c1de9be5f75f7adc5842acd6fb283a1f8c78e8db7c64bf2b655bb2b` |
| 00003 | 18864 | `422596c5eae572fa8a8c4e66f711622bc47eacf8d162838c84c3f9dc63b2c48d` |
| 00004 | 28096 | `1330aad43ef7dc0a6624ed9f345a2f535dbfad936a818896faa262d7e1a54031` |
| 00005 | 24680 | `d767e19b5997d6cc48aa1aed85a7f09bbaf862dedcd0e4609328345a74ae99c8` |

排除252个语言投影weight后，907个保护weight/bias实际占8,265,517,540B；packed252占2,006,968,320B；二者之和比旧实际storage10,272,486,512B少652B，恰与下文预期注册非persistent缓冲量相符，仍需运行时逐项核实。最大`vlm.lm_head.weight`与`vlm.model.language_model.embed_tokens.weight`分别为BF16[155697,4096]，各1,275,469,824B，**仅改成逐张量get_tensor仍产生1.28GB CPU副本**。

## 实际构造链：可实施范围与meta风险

1. `source_snapshot/base_model.py:302-326,378-392`的ReasoningVLA先读取VLM配置，创建Qwen3VLForConditionalGeneration骨架，再创建trajectory tokenizer和AutoProcessor tokenizer。原helper `evidence/quant_route_20261008/full20/code/test_alpamayo15_offline_compat.py:360-389`只在内存中把VLM配置路由到本地Cosmos，再`Alpamayo1_5.from_pretrained(...dtype=BF16).to(cuda)`；该helper**不能作为packed-only加载步骤**，仅可复用其中来源验证与processor契约。
2. `source_snapshot/alpamayo1_5.py:93-139`从VLM text config复制Expert config并叠加实际expert_cfg，AutoModel.from_config构造Expert后删除embed_tokens；Hydra构造ActionSpace、FlowMatching、action_in/out；keep_same_dtype只对diffusion及动作投影转Expert dtype；最后post_init。构造器没有checkpoint读取入口，源码未见必须对参数取item的步骤，但仍需实际meta探针捕捉依赖内部行为。
3. **不能把packed meta tensors交给旧PackedAWQLinear.__init__**：`scripts/alp15_awq_full_packed.py:13-37`及`alp15_packed_w4a16.py:48-57`调用isfinite/all.item和unpack读取值。PhaseA新增只保存shape/dtype/meta buffer、forward必报错的占位模块，真实packed仍用旧类逐文件CPU验证并移到目标后替换占位；不修改旧类或绕过其数值验真。
4. 优先使用`with torch.device('meta')`覆盖整个构造器，配合已存在HF `no_init_weights()`，在独立进程中设置并finally恢复BF16 default dtype。禁止`from_pretrained_submodules`/读取任何VLM权重，禁止model.to_empty对整模型分配，也不调用全模型.to(cuda)。所有参数和预期大buffer必须meta，任何大CPU/CUDA分配失败即停止，不回退常规构造。
5. 该策略会把小型确定性buffer也置meta，必须按官方小模块构造恢复。若现有构造器对这些meta值进行数据依赖判断，先保存异常/调用栈，由决策者批准精确小buffer的CPU构造适配；执行者不得放弃meta构造改成完整CPU模型。

HF固定commit `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc`（v4.57.1），实际modeling_qwen3_vl.py hash=`dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da`与已运行smoke记录相同。[modeling_utils:2103-2112,4366-4378](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/modeling_utils.py#L4366)支持no_init_weights/meta构造。[HF integrations/accelerate:70-141](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/integrations/accelerate.py#L70)的include_buffers=False会先登记原参数再转meta，不能假设无CPU临时分配。

accelerate1.12.0固定commit `61ff83394592db1052be9e64ab46beb21d794d69`，[big_modeling:60-170](https://github.com/huggingface/accelerate/blob/61ff83394592db1052be9e64ab46beb21d794d69/src/accelerate/big_modeling.py#L60)明确include_buffers=True进入设备上下文。其历史离线环境记录不等于当前import可用，PhaseA打印真实version/path/hash。核心loader采用torch原生meta及显式叶节点赋值，不依赖新安装accelerate；不使用load_checkpoint_and_dispatch或自动device_map/offload偷偷分派保护权重。

## 离线配置、tokenizer与非checkpoint状态

本地路由必须真实由原config与已核验依赖读出，而不是根据8B/2B目录名猜架构：

- 原checkpoint config的`vlm_name_or_path=nvidia/Cosmos-Reason2-8B`在**内存副本**中只映射为`alp15-configs-delivery-20260919/dependencies/configs/Cosmos-Reason2-8B`。来源为Cosmos revision`a9fae2cf89dc64db96b12860417f0eb403013bb9`，调用既有_verify_config_dir/_verify_config_source。
- 本次实际Cosmos config：36语言层，hidden4096，head_dim128，KVheads8，intermediate12288，attention_bias=false，tie_word_embeddings=false；visual27层、hidden1152、out_hidden4096，DeepStack[8,16,24]。原checkpoint vocab改为155697，original_vocab_size仍取151936。Expert复制text config后实际override hidden2048、heads16、intermediate8256、head_dim128；KVheads由复制配置得到8，实际header k_proj=[1024,2048]支持此判断。Expert始终36层，不以目录名或2B配置替换。
- 原tokenizer构造仍调用同一ReasoningVLA._build_tokenizer：本地Cosmos AutoProcessor，再按原顺序添加4000个<i*>和SPECIAL_TOKENS，保留traj_token_start_idx=151669、traj_token_ids/特殊边界、padding_side及词表。必须对全部新增token IDs与tokenizer序列化hash/长度核验，不只测future-start。构造期任何AutoProcessor网络尝试立即停止；不重写tokenizer JSON。
- 部署processor仍从本地Qwen3-VL-2B-Instruct revision`89644892e4d85e24eaac8bacfd4f463576704203`读取原helper.MIN_PIXELS/MAX_PIXELS，并将processor.tokenizer指向model.tokenizer。同名2B仅作原processor来源，绝不用于VLM骨架。记录真实chat_template、preprocess配置及测试输入token/grid/hash。HF离线环境变量在首次导入前设，原socket禁网context覆盖所有构造与加载；每个from_pretrained的绝对本地配置路径在allowlist内，存在即读，缺失硬停。
- 不随意加载Cosmos的generation_config.json覆盖构造出来的嵌套VLM生成配置；新旧构造器vlm.generation_config完整字典逐项对拍，后续runner原top_p=.98、temperature=.6、top_k=None、max_new_tokens256与停止边界不改。外层配置/原dtype也比对。

非persistent状态不得由load_state_dict缺键逻辑忽略：

| 对象 | 一手源码、恢复规则 |
| --- | --- |
| `vlm.model.visual.rotary_pos_emb.inv_freq` | [Qwen3-VL:79-89](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L79)：官方小Rotary类在CPU重建，维度来自真实visual配置；不能把meta拷出 |
| `vlm.model.language_model.rotary_emb.inv_freq`与`expert.rotary_emb.inv_freq` | [278-297](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L278)：用各自实际text config调用官方小Rotary类；同时恢复original_inv_freq、attention_scaling、max_seq_len_cached、mrope_section等字段。whole-module仅对小Rotary允许.to(device)，保持原nn.Module.to语义：非注册original_inv_freq不由to自动移动，不伪称其仍与移动后buffer共享storage |
| `action_in_proj.sinus.{0,1}.freqs`、`timestep_fourier_encoder.freqs` | [NVIDIA action_in_proj:73-89,131-146](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/models/action_in_proj.py#L73)：官方FourierEncoderV2，真实num_fourier_feats20/max_freq100；按旧constructor defaultdtype及keep_same_dtype恢复，不运行完整MLP实体构造 |
| `action_space`四个scalar | [Unicycle:82-98](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/action_space/unicycle_accel_curvature.py#L82)：按原config官方小ActionSpace构造，64点dt0.1及所有归一化/边界/solver常量不改。记录实际dtype而非统一强cast |
| `traj_tokenizer.action_space` | [DiscreteTrajectoryTokenizer:24-40](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/action_space/discrete_action_space.py#L24)：tokenizer是普通Python对象，其ActionSpace不进入model.named_buffers/state_dict；必须显式检查并恢复，不让其残留meta。原whole-model.to不会移动此子对象，保留同口径CPU状态。hist DeltaTokenizer是Python常量，无checkpoint加载 |

预期10个注册非persistent buffer按实际类推导共652B：visual RoPE18FP32=72B；VLM/Expert RoPE各64FP32=512B；三个Fourier[1,10]BF16=60B；top ActionSpace四个BF16scalar=8B。**这是源码+storage差额相互支持的待核实清单**，执行时输出完整named_buffers含persistent flag，以及非注册对象tensor属性；任何新增状态、dtype不同或未知meta属性先停并升级。禁止手填全零inv_freq/freqs以消除meta。构造器dtype上下文finally恢复，保证后续官方Euler默认FP32 noise/time_grid不被全局BF16设置污染。

## 目标键、来源优先级、alias与dtype

必须先生成并冻结LoaderPlan，不在load过程中临时猜键：

1. `S_original`是实际五分片header与index验证出的1159键；同时检查原模型meta state_dict（持久键）恰好此集合、shape与header相同。不因base_model_prefix=vlm自动剥/添名字。
2. `Q`恰好252个全名`vlm.model.language_model.layers.{0..35}.{mlp gate/up/down, self_attn q/k/v/o}.weight`，全部bias=None。AWQ result的module name为`language_model.layers...`，只允许**显式添加一次`vlm.model.`**定位。Q全部dense实体加载禁止。
3. `P=S_original−Q`恰好907键，均由原checkpoint index指定shard读取；它们对应Expert、vision、norm、embedding/lm_head和action投影。父overlay经过旧load_overlay源码确认只含36层MLP gate/up/down，因此无P键由它覆盖；如实际overlay出现保护键，直接停止，不“overlay优先”静默装入。
4. overlay manifest完整读取与hash核定：原checkpoint_revision匹配，layer0..35按序、每层original_width12288，keep_indices唯一排序且合法，keep_width与已恢复父step400/实际packed形状对应；36个mlp文件按冻结SHA/头部验证三键，不加载其dense payload。当前结果252packed MLP全部为5376宽：gate/up[5376,4096]，down[4096,5376]；骨架相应mlp.intermediate_size设5376。原model.config/text_config.intermediate_size仍12288，如旧load_overlay行为，不擅自改全局config。
5. 每个packed目标的键集合严格`{qweight,amax,pre_scale}`（当前无bias），metadata format=`alp15-modelopt039-awq-signed-int4-group128-v1`、group128；shape按AWQ row，qweight uint8[out,in/2]、amaxBF16[out,in/128]、pre_scaleBF16[in]。每文件真实SHA与row.sha256相同。q/o[4096,4096]、k/v[1024,4096]，MLP使用上一项真实宽度。禁用旧RTN scales格式、ModelOpt重新校准、nibble重排或新rounding。
6. 替换后持久目标集合`S_target=P ∪ Q_packed`，按每module三buffers共756键，合计1663。必须检查完整集合/shape/dtype/param-or-buffer种类，不用strict=False容忍未装项。bias=None不是state键；907保护参数不会误转为buffer，packed是buffer不会误转Parameter。
7. 原Cosmos tie_word_embeddings=false，实际checkpoint独立提供embedding与lm_head；当前result logical_bytes=unique_bytes，不允许为省内存将二者强tie。PhaseA记录同对象alias与跨属性关系；持久参数本例预期无tie，任何差异停机。禁止以meta.data_ptr()==0判断它们别名。非注册RoPE original_inv_freq关联按上表单独恢复。
8. 907保护payload均BF16，packed amax/pre_scale BF16、qweight uint8；meta参数shape/dtype必须一致，浮点/整数都不默认cast。RoPE等重建小状态按旧运行dtype而非checkpoint dtype统一处理。FP16是另外质量候选，不能因设备不支持BF16在本loader偷偷转换。

## 可执行模块API与文件边界

以下为实施规格，文件尚未因本文存在；主任务派**另一名明确执行者**独占全部4个文件，不与当前hybrid执行者交叉写入旧文件：

- `scripts/alp15_packed_loader.py`：`audit_loader_assets(paths) -> LoaderPlan`；`build_meta_skeleton(plan) -> model, processor, MetaAudit`；`load_packed_only_model(plan, *, device, chunk_bytes=33554432, audit_dir) -> model, processor, LoadReport`；`protected_hashes_chunked(model, plan, chunk_bytes) -> HashReport`。metadata-only路径不检查cuda availability、不调用CUDA API。所有加载副作用与环境上下文在独立进程；CPU导入不能自动读大权重或import ModelOpt/Triton。
- `scripts/probe_alp15_packed_loader.py`：`--stage headers|meta|load --root ... --awq-artifact ... --overlay ... --output ... --device cuda:0 --chunk-bytes 33554432`，缺省headers；meta忽略GPU参数并禁止实体源payload访问，load必须携带同来源hash的通过MetaAudit。输出目录存在即拒绝，状态headers_only/meta_passed不冒称loaded。
- `scripts/run_alp15_packed_loader_eval.py`：隔离新入口`--loader bf16_reference|packed_only`，旧参考沿冻结helper，packed_only只用新API。默认reference；加载后AWQ-forward保持reference、actualKV/步数/输入/生成/seed沿冻结比较口径，第一轮不同时开启hybrid_m1/FlashDrive/LightVLA。所有import真实路径/hash留痕，CPU/GPU按主任务统一安排。
- `tests/test_alp15_packed_loader.py`：小型合成safetensors fixtures与mock构造器验证key/shape/dtype/header/alias/路径拒绝、来源优先、缺漏/重复、分块上限、hash次序、非persistent恢复、不完整load退出、不调用全load_file/get_tensor大tensor、不调用.cuda或常规from_pretrained；运行实际大型meta构造另由probe验证，不由mock宣称通过。

不修改现有AWQ kernel/packed helper、正在运行KV代码、官方源/原venv、hybrid执行者文件或原权重。计划中实际runtime使用哪个helper/hash必须由执行者冻结，不用同名导入暗中加载Mac旧代码。

### 块加载步骤与资源上限

1. 先验证源/配置/完整文件SHA及只读路径allowlist，small headers/overlay全部合法，MetaAudit通过；hash扫描读文件page cache也计入全系统观测。源读完再变更或文件mtime/size变化停止，不承认旧预检hash为后续修改文件背书。
2. 原构造器在meta生成907参数及252将被替代的投影；将252替成独立meta占位、更新每层MLP实际宽度，此时所有大参数仍meta。默认BF16只覆盖构造范围，退出恢复原defaultdtype。禁止完整CPU BF16模型、CPU完整shard字典或ModelOpt export状态。
3. 遍历P时，按计划为一个目标参数在目标设备直接一次分配完整最终tensor，替换对应叶节点参数slot（requires_grad=False）；再从`safe_open(...framework='pt',device='cpu').get_slice(key)`以dim0连续行块读取，每块字节≤32MiB，顺序copy_到目标相同行区。零维/小tensor可get_tensor；每个行块shape/dtype/finite/hash核对完成后释放，复制与释放间同步，不累积CPU词表权重。若单行大于上限当前实现必须拒绝，不静默超预算。本例所有已审计header可沿dim0分块，最大两embedding各行8192B。
4. 不用`get_tensor`读取1.28GB头、`load_file`读取原shard、全模型state_dict().cpu()或tobytes()；不先CPU构建整张目标再.to(cuda)。目标finaltensor一次分配后不得另存CPU镜像，chunk transfer/hash暂存合计受控。默认CPU暂存保护块预计≤3×32MiB；实际RSS额外预算256MiB（相对imports/config/processor完成时的阶段基线）为机械停止阈值，不能从中排除文件page cache或部署系统其他开销，后者另以system available硬门禁记录。
5. packed逐文件沿原`PackedAWQLinear.load(device='cpu')`验证，最大单投影文件约11.4MB；其unpack会生成整份INT8及临时boolean，对此按实际临时量采样。通过后单模块移动目标设备，安装并释放CPU副本，再下一文件；不一次load252到CPU或GPU双份。从factory临时占位替换时原dense本来meta，不产生实体双份。任何验证需更大临时量超过预算，先停再单独批准分块validator，不能本轮改旧helper跳过检验。
6. 按官方小类恢复上表确定性非persistent状态及普通Python子对象，检查每个parameter/buffer/已知tensor属性无meta；非persistent未写入checkpoint，不能以目标持久键1663齐全作其通过依据。eval/requires_gradFalse，无随机初始化残留；不在加载阶段capture图/分配KV或重复更改dtype。
7. 已加载参数的独立验真从GPU按同样≤32MiB readback核算，而不是复用刚读CPU块的sourcehash自证复制成功；审计readback前后记录峰值。整个LoadReport包含阶段时间/内存、bytes_read/copied、chunk最大值、已装持久字节、源/targethash、tensor种类、未装键、buffer清单和defaultdtype恢复证据。

## 严格hash与验收门禁

PhaseA headers：实际907/252/1663集合、原1159header与index、overlay/hash/36层形状、packed252真实header/文件hash、配置/tokenizer路径均核定；本文已完成父manifest结构审计，但尚未替执行者完成36个overlay文件及252个packed实际header验真。PhaseA meta：在既有L20环境独立CPU进程，无CUDA，禁网；原构造器成功，所有大参数meta，buffer待恢复清单精确，CPU峰值在预锁预算内，small rebuild恢复后无未知meta属性，tokenizer/processor契约正确。失败留traceback，停止实体load。

保护hash分两层：

- 兼容旧`protected_sha256`：复用`scripts/alp15_awq_full.py:323-333`/当前KV runner同一state_dict遍历顺序，跳过252投影下全部键；每保护键一次加入UTF8名称、`str(tuple(shape))`，随后**原连续BF16 bytes**依行块按原顺序update。不得改成字典字母排序、不加dtype字段到旧格式。907保护原sourcehash、最终target readbackhash均必须等于`94b2256c7e277915c3b570c6ba941aaf49b56dfa4094b2adac06b70a394093bb`。如果meta骨架state顺序有差异，先分析并冻结旧实际ordered_keys，不默默放宽aggregate。
- 新per-tensor报告：每键记录source file/key/offset、shape、dtype、rawbytes SHA256、target rawbytes SHA256及byte数，严格逐键相等；另加schema版manifesthash。非persistent/普通子对象tensor用单独清单逐项dtype/shape/bytes与旧loader的小官方构造路径及参考运行核对，不混入旧保护digest。packed每buffer原文件bytes与已装target bytes逐项相同；三buffer及module结构不能凭aggregate漏报。

另外父overlay旧guard排除108个MLP权重，共1051原始保护键，包含即将被packed替代的144个attention投影。其`77b2770e...`验证若执行，以**原meta骨架ordered_keys**顺序对原checkpoint源块计算，仅hash这144个attention原始payload，不将它们装入模型；等于manifest值才通过父来源门禁。最终packed模型的907键guard不能与这个1051键guard比较。两个digest来源/集合/ordered_keys清单分别留存，CPU块上限相同。

load阶段预期注册storage去重10,272,486,512B、持久1663键；同时报告不进入named_buffers的tensor空间，不能用10.27GB只统计模型掩盖它。packed252逐层核对qweight/amax/pre_scale原样，并分层受控输出/反量化桥接（每次一个矩阵临时）验证旧类行为；量化计算的完整矩阵只在桥接测试短暂存在，不在加载/模型cache持久保存。加载与桥接峰值分开。

然后同代码两clip×3seed完整reference/packed_only请求，source、token IDs/prefix、noise完全配对，保护hash与所有packedbytes前后不变，64点轨迹finite/shape正确；逐条ADE差≤1e-4m、NMSE≤1e-3、rot/边界核对通过，才扩20clip×3seed。loader本应等价，不自动申请actualKV旧有损豁免。整个评测单请求含rawimages/processor/historyfuse/generate/Expert/traj，不把三seed共同generation摊销。扩评完整退出0、所有条目保留、独立clip统计和FDE/尾部/失败率记录；不以mock、小模型、单矩阵或L20加载通过代替完整模型验收。

Orin最终还需真实冷进程加载、first valid trajectory、长时间稳态/热状态、reset/recover证据。采集型号/JetPack/L4T/CUDA/package/功耗/频率/温度、system available、tegrastats/RSS/PSS/CUDA、swap/majorfault，分阶段时间与最大峰值。统一页不得RSS+CUDA简单相加；mmap和文件page cache也不能排除。OS/相机运行空间不足、任何OOM、swap/offload掩盖或任意完整验收请求>300ms不能称通过；独立clip同任务teacher meanADE≤1.15同时成立后才称完整目标完成。ARM64/BF16/Triton实际路径仍须原决策的实机编译/算子/全链路门禁；本loader不新增依赖来规避。

## 停止、回滚及执行留痕

meta构造失败/试图实体化大权重、未知非persistent/非注册meta属性、配置路径或来源hash变化、907/252/1663键不符、overlay额外覆盖保护键、dtype/alias不符、chunk超上限、预算超限、保护hash不符、packed格式错误/漏层、非finite/OOM、复制后hash变化或轨迹strict门禁失败：停止新loader，保留阶段产物、具体tensor、命令/traceback/退出码，由决策者升级。不能自动换到CPU完整BF16、FP16或磁盘offload完成伪门禁。

回滚仅退出新进程或隔离入口选择原bf16_reference；旧helper/模型/manifest无修改，不删除旧包/失败目录，不停止其它项目。当前full20独占GPU，新CPU探针只由主任务安排独立进程，后续GPU任务串行；不是因观察超时重启旧流程。

执行回报必须列决策编号、文件所有者、开始/结束时间、实际环境/version/import路径源码SHA、命令/输入/退出码/完整日志、MetaAudit/LoaderPlan/LoadReport/逐tensor HashReport、峰值口径、所通过及未通过门禁。本次唯一写入文件为本文；其余审计只读。web-access前一任务已完整读取，本轮继续按该技能允许的静态官方固定commit读取，无浏览器会话/包安装/大资产下载。
