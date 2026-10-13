# FlashDrive / LightVLA：下一项实施决策

日期：2026-10-08（Asia/Shanghai）。决策编号：`D-ALP15-HYBRID-M1-20261008`。决策者与本文件唯一写入者：GPT-6.1 Sol high，`flashdrive_lightvla_next_decision_v2`。风险：高；执行与验收均由 GPT-6.1 Sol high，服务器 GPU 由主任务统一串行调度。

## 结论与授权

**立即实现现有 AWQ 的 M=1 混合解码路径，并完成全部 252 个真实矩阵及完整轨迹的分级验证。**不更改现有 packed 格式、amax、pre_scale 或 kernel；M>1 继续运行原 `PackedAWQLinear`。动作缓存、Expert 融合、静态 KV、LightVLA 视觉筛选分别保留为后续因素，第一轮混合路径评测不同时启用它们。

此决定服务于 Orin NX16GB、完整请求 max≤300ms、同任务教师 meanADE 损失≤15% 的原目标。现有约 4 秒 L20 请求证明还有主瓶颈，M=1 融合是可立即消除的重复反量化工作，**没有证据表明单独此项足以满足 300ms**。后续仍必须压缩视觉/语言前向工作量并联合考虑 FlashDrive 与 LightVLA；不能用单层结果或两个 clip 宣布目标完成。

本决策承接 `ALP15-JETSON-300MS-ROUTE-20261008.md` 记载的直接用户豁免与持续优化授权；不把豁免记作旧官方原入口验收成功。原 AGENTS 的历史门禁与用户后续授权须在主任务留痕；本文件不另行放宽最终硬件、精度、时延门禁。

允许主任务现在指派执行者实现独立新文件和 CPU 回归；本轮正在运行的 code_v3 smoke_v4→full20_v1 不改、不重启。GPU 探针和轨迹评估等待主任务明确安排，禁止并发启动。不得连接服务器自行写入，不下载权重/镜像/数据，不安装包、不改原 venv，不修改旧 kernel、runner、packed helper、权重、来源源码或旧实验目录。

## 当前可追溯事实

本机仓库 HEAD：`792b3d86592d7cde258e9a36c9846b8e54945ee0`，有既有改动。NVIDIA 源仓库身份为 `36aeb4c5938cbc2eb2aed33b22434773da4ab639`，现场 dirty 文件以实际 SHA256 识别，不据 HEAD 称官方原样。

| 事实 | 本次读取的证据 |
| --- | --- |
| M1 gate_proj 5376×4096 reference→fused 同步 wall 中位 1.26925→0.15014ms；临时 allocated约354MB→10752B | `evidence/jetson_300ms_20261008/kernel_probe/gate_fused_bf16_v2.json`；不是252层覆盖 |
| M17可加速，但M256与M3086变慢 | 同一探针；因此本轮只批准M1，不扩到现有kernel内部的M≤8范围 |
| 当前融合共享 packed buffers，forward不缓存完整dense | `scripts/alp15_awq_kernel.py:14-73`；SHA256 `3f09aea862990733b9bb1db536c450c79fd6e2ac93c7015eed6b1eef4f191025` |
| 原forward每次 inputs*pre_scale，再构造整个dense矩阵 | `scripts/alp15_awq_full_packed.py:75-82`；SHA256 `150e1310ad3dbcb9f82966146f9627855e87bd12a29f45f410d0b94f1a96de0a` |
| 2clip smoke_v3每variant6条，actualKV4平均 generation=3836.61ms、Expert及action_to_traj=182.97ms、E2E=4174.10ms | `evidence/jetson_300ms_20261008/kv_candidate_smoke_v3/result.json`，SHA256 `feb8b3cd08ea8bec3e11f5dd59bdc9ca529c6a0dc0d01dfb6a3662e0afd54595`；共享L20，非无干扰因果测速 |
| 同smoke_v3所有6个strict数值门禁False | 同目录 `acceptance.md`；actualKV是已批准的新质量候选，不能改称无损 |
| persistent storage=10,272,486,512B，其中Expert=4,558,477,568B | 同result中的按实际storage去重统计；不是Orin全系统内存 |
| 现场torch=2.8.0+cu128、CUDA12.8、L20 | 同result；不能外推ARM64/SM87环境 |

当前 `scripts/alp15_kv_steps_probe.py` SHA256=`74886a867a8aae480dd848a6d068d2e3f817674d891774bcdabeb57f412292f8`。该hash标识本次读取的版本，不假称已正在服务器完成full20。主任务通知本轮新流程live，仍须以终态result/退出码及来源hash验收，本文不等待或预测其结果。

## 冻结的下一项实现

拟新增接口是本决策的实施规格，以下文件尚未因本决策自动存在；执行者完成后再提供实际CLI命令。

1. `scripts/alp15_awq_hybrid.py`：提供 `HybridAWQLinear.from_reference(reference, enabled=True)` 和 `install_hybrid_awq(model, packed_manifest, enabled=True)`。`reference`必须是已验真的`PackedAWQLinear`；fused由现有`FusedAWQLinear.from_reference`建立。共享qweight/amax/pre_scale/bias的原storage，不clone、不重新pack，不存persistent dense。保留in_features/out_features与bias语义，state inventory按storage去重。
2. `forward(inputs)`：按真实输入元素数/最后维度得到M；`enabled and M==1`才调用现有fused，其余调用同一个原reference。M1只接受原kernel已支持的连续CUDA BF16无梯度输入；不偷偷cast/contiguous/CPU回退。禁用模式必须逐元素复现原reference；禁用与M>1不要求Triton导入/JIT。空输入、错误shape/device/dtype保留显式错误。接口不改变上游attention、生成参数、KV或求解器。
3. installer只替换冻结manifest恰好252个`vlm.model.language_model.layers.{0..35}`的q/k/v/o和gate/up/down投影。检查名字、shape、格式、原artifact SHA、重复/遗漏。不得替换Expert、visual、embedding、lm_head、action投影。日志分别计数M1 fused/M>1 reference调用，避免“开关已开”却未真正经过融合路径。
4. `scripts/probe_alp15_awq_hybrid.py`：已有服务器资产逐层加载，不把252矩阵全部复制成dense。每层使用固定seed、M=1随机BF16及捕获的真实decode activation（有真实输入时），验证同amax/prescale的输出；每次dense桥接立即释放，下一层再验证。记录module→shape/dtype/hash→误差→first/warm wall/CUDA event→峰值与dispatch证据。不把所有同shape权重视作一个数值测试。
5. `scripts/run_alp15_awq_hybrid_eval.py`：隔离的新完整请求评测入口。开关冻结为`--awq-forward reference|hybrid_m1`，缺省reference；候选decision必须为本编号，输出目录存在即拒绝。复用本轮runner的来源、输入、noise、future-start、保护模块与完整请求逻辑，导入混合installer；不得修改当前runner以影响已启动流程。记录新helper与所有真实import路径hash，以新的同代码reference/hybrid配对smoke为扩评依据，不能拿旧smoke的代码hash门禁强行通关。
6. `tests/test_alp15_awq_hybrid.py`：无CUDA时mock现有fused工厂，验证M1/M2/M17/M256/M3086分派、enabled=False、原buffers storage身份、manifest252边界、错误输入与无Triton导入。CPU测试不声称CUDA数值/性能通过。

文件所有权：主任务须把以上三个scripts及tests交由一名明确执行者独占；本决策者只写本Markdown。执行回报必须列实际文件与hash、命令、环境、日志、退出码、失败记录和偏差。原kernel/packed helper唯一只读。

### 从小测试到全轨迹

- 全252矩阵M1桥接：先沿现有算子门限rtol=0.01/atol=0.02，报告最大绝对/相对误差和误差分布；dequant桥接须保持BF16逐元素一致。不能由算子容差推导驾驶精度。
- 禁用及M>1走reference：必须确认原函数调用、storage未增加、结果逐元素一致。记录manifest覆盖252/252和所有保护模块hash。M1若任何shape unsupported或NaN/OOM，停止新候选，不静默跳过坏层。
- 同代码2clip×3seed×reference/hybrid（至少10步）：完整rawimages→processor/historyfuse→自主generate→Expert→64点轨迹逐请求运行，不复用教师文字。先要求生成token IDs、future-start/prefix、noise与边界一致，逐条ADE差≤1e-4m、trajectory NMSE≤1e-3。若fused数值导致prefix变化或超过strict轨迹门限，保留失败，决策者另行批准有损新候选后才能扩，不自动套用actualKV的旧豁免。
- smoke通过才同代码20clip×3seed配对；最终步数由正在进行的full20质量结果决定，10步对照保留。如果full20无actualKV候选通过15%，hybrid仍可对fullprefill10验证计算收益，但不声称已找到部署候选。
- 同一模型实例原packed artifact/protected模块固定；改variant顺序应事前冻结成区组，记录其它GPU负载。模型先首推/稳态分开，再报告每条完整请求mean/p50/p95/p99/max与>300ms数。来源哈希、finite、失败记录不能缺失。
- meanADE按独立clip聚合；20开发clip反复筛选不能证明泛化。教师全20 meanADE=1.2323994789272548m，开发上限=1.4172594007663428m；子集使用同clip/seed教师均值，FDE、clip尾部与失败率并报。冻结候选之后需独立clip和Orin实机联合验收。

## FlashDrive机制的接入边界

固定原作者提交：`26265174f0413db77c8a16d99993dbeecacf458f`。本次公开raw源码读取，未安装。以下按独立因素安排，不相乘论文加速。

**Expert融合可在M1完成后作为下一项无新大资产候选。** [fusion.py:14-57](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/fusion.py#L14)将同输入q/k/v及gate/up权重按行拼接、输出split，[_backbone.py:243-251,289-294](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/_backbone.py#L243)保留norm和gate乘法。当前Expert未量化，需现场确认每层实际类/shape/bias/dtype一致后只适配投影forward，保留attention/RoPE/KV语义。不要import全包触发全局patch。AWQ q/k/v各自pre_scale可能不同，未经逐张量核实不能把其packed rows直接拼接成共同prescale的投影；初项不做AWQ跨投影融合。Expertdense拼接的初始化会短暂双份驻留，必须逐层或loader直接目标布局，Orin不能靠先cat整个Expert再删除旧参数解决冷峰值。

**静态KV先仅请求内预分配、每请求重置，不启用跨帧shift。** [_backbone.py:30-100](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/_backbone.py#L30)按cache_position原位写KV；同文件[223-264](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/_backbone.py#L223)缓存未RoPE K。原HF路径先RoPE再缓存，不能只copy FlashDrive shift索引到现有cache。后续静态候选保留已RoPE表示；容量按实际prefill+冻结decode上限256+实际action_token数计算，另维护valid_length并掩蔽全部unused slots。Expert每步覆盖同一action尾区，prefix immutable，不用dynamic append/crop。用本地官方 `_build_expert_pos_ids_and_attn_mask` 的合法mask/位置作为基准，并扩到真实静态容量；不得把alloc长度误当valid长度。图捕获等其eager等价门禁通过后再开。

本地`expert_interface_distill.py:30-51`每步重建DynamicCache，官方快照`alpamayo1_5.py:330-381`则复用cache并crop：此处可优化对象/concat/尾区，但要确认36层KV接口、request/seed隔离。StaticCache的容量开销和attention对padding的计算可能抵消收益，必须实测。FlashDrive[_compile.py:48-94](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/_compile.py#L48)按key绑定首次closure；图registry必须含shape/求解步数/cache策略/保留布局/dtype，不把后续config变化误复用旧图。

**动作速度缓存是有损求解因素。** [diffusion.py:14-46](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/diffusion.py#L14)复用前一步velocity并禁止跳过第0步，其time_grid是BF16。当前官方快照`flow_matching.py:171-191`用默认FP32噪声/time_grid；直接复制会同时改变dtype。后续仅移植“请求内复用velocity”思想，保留本地官方noise/time/dt广播及CFG公式，cache_steps只接受唯一、升序、范围内、排除0的整数。缓存为请求局部，不挂持久`diffusion._cached_velocity`造成跨seed污染。先`cache_steps=[]`完整等价；预锁10步`[1,3,5,7,9]`与4步`[1,3]`两组，分别与相同步数无缓存比较，实际Expert调用数应为5/2。只有独立新决策才执行这两组，不能插入本轮hybrid矩阵。

[__init__.py:82-120](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/__init__.py#L82)以z-lab前缀派生PARO/DFlash并默认BF16-first；本轮不调用此loader。既有AWQ不是PARO，也没有恢复8B对应的DFlash草稿接受率/hidden接口/内存证据。PARO/DFlash均保持关闭，不自动下载；若未来研究，需独立本地artifact/接口验证与完整退回路径。

## LightVLA联合接口：先keep-all，再筛选

LightVLA固定提交`a4680fda5ffe73029190ac97328aa34b0e87a45a`。[TokenPruner:49-130](https://github.com/LiAutoAD/LightVLA/blob/a4680fda5ffe73029190ac97328aa34b0e87a45a/prismatic/extern/hf/modeling_prismatic.py#L49)用task条件评分，按CLS/固定patch/task拆分，同步筛选position和mask；[PrunedLlamaModel:192-206](https://github.com/LiAutoAD/LightVLA/blob/a4680fda5ffe73029190ac97328aa34b0e87a45a/prismatic/extern/hf/modeling_prismatic.py#L192)重建cache_position。它不是Qwen3-VL的即插接口。

本次核实HF v4.57.1 tag解析到commit `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc`，其modeling_qwen3_vl.py SHA256=`dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da`，**与smoke_v3实际运行记录逐字相同**。[官方源码:822-867](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L822)有三轴位置与DeepStack逐层注入；[987-1013、1137-1233](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L987)按完整grid计算位置、检查placeholder数量并传全部DeepStack。禁止把删token后的input_ids与旧grid重新交给原get_rope_index猜位置。

联合适配冻结为batch1、Vision merger之后筛选LLM视觉token；视觉自身仍完整计算，不能宣称ViT加速。预先在完整序列计算原mRoPE三轴position_ids；相同排序keep索引同时gather image embedding、全部DeepStack分支、input_ids、visual_pos_masks、attention_mask及位置表，所有非视觉历史/时间戳/文本/边界token保留。生成与Expert使用独立的compact cache slots。若选择后保留原位置坐标，`rope_delta_compact = next_semantic_position - compact_prefix_length`；用原完整序列的后续text位置确定next_semantic_position，不能用保留patch的最大值倒推而跳变。keep-all必须与原接口完整对拍，才可开启训练得到的筛选策略。

冻结token身份字段：`(stream_id, clip_id, camera_id, frame_timestamp_us, merged_patch_t/h/w, original_prompt_index)`；每个request保存原索引→compact index→cache_slot→mRoPE三轴映射、keep-set hash、policy版本、grid/preprocess hash。缓存reuse键还包含模型/overlay/packed/source hash、dtype、attention/cache策略、prompt/history身份和有效长度。**相同shape不等于同一保留集合。**任务条件选择改变、帧乱序/缺帧/相机顺序/分辨率/标定/窗口改变、stream reset、policy更新或原位置变化均失效重建。初期每请求全部KV重建，仅可单独缓存按稳定帧身份的视觉特征；某token隐藏态仍依赖上下文，不因patch身份相同就允许复用所有语言层KV。真实跨窗重用需单独验证未RoPE/重新RoPE、因果mask及上下文等价或作为有损候选。

后续消融固定base、FlashDrive机制侧、LightVLA侧、联合侧四组；数据/teacher/64点dt0.1/noise一致，分别统计重建、selector、processor与图管理开销。独立train/dev/test按clip分隔；现有20clip不进入新增训练，不用它挑参数后充作独立测试。动态keep数需要预锁bucket/拒绝或重建规则，不能首次capture形状后静默截断padding凑兼容。

## 低峰值部署加载与真实Orin边界

本轮混合路径可先沿现有L20 loader诊断；**部署loader必须避免CPU与CUDA任何一侧完整BF16模型驻留**。CPU43GB完整加载再转GPU不满足16GB统一内存目标。

下一阶段部署包采用meta架构+逐张量装载：所有构造器在meta上建立，Qwen配置/tokenizer只读本地；先将252目标投影替换meta packed模块，禁止其dense参数实体化；从原checkpoint的safetensors索引逐张量safe_open读取非量化保护模块，并按父overlay优先规则装载；overlay中量化目标的dense权重跳过，只读既有packed252资产。记录每个目标tensor的来源文件/hash/键/shape/dtype、alias/tie关系，检查遗漏/重复/unexpected keys/残留meta；构造期间非meta的小buffer、tokenizer/processor/trajectory模块也要审计。无需新增原始大权重副本，导出manifest和分块保护张量可在已有服务器素材生成到新目录，实际打包前冻结格式与文件所有者。

必须验证新loader与旧模型的persistent storage、packed buffers逐元素与所有保护参数hash相同，完整轨迹对拍。量化模块独立装入，read-copy-release每次受控；未完成hash验收不得删除旧包。mmap页/文件page cache也是统一内存，必须采集RSS/PSS、system available、tegrastats、majorfault/swap与CUDA观测，不把RSS和CUDA峰值相加。图workspace、静态cache、prefill dense临时矩阵、image buffers另算实际峰值；当前10.27GB persistent不是“16GB够用”证据。若prefill reference峰值超预算，需单独大M真实packed kernel方案，不能为显存把当前慢GEMM强开然后藏掉延迟。

[NVIDIA Jetson PyTorch安装指南](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)要求匹配JetPack的构建；[官方兼容表](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)把2.8.0a0+5228986c39列于JetPack6.2，不能视为现有L20 torch2.8.0+cu128原封可用；该表也标注Model Optimizer在Jetson的依赖限制，部署应读取离线packed，不在Orin重跑ModelOpt校准。表是2026-10-08读取的动态文档。

Triton固定审计commit `e50b186e8bd2d16ae3f564311ef868aa3bd45a2d`：[README:292-302](https://github.com/triton-lang/triton/blob/e50b186e8bd2d16ae3f564311ef868aa3bd45a2d/README.md#L292)列Linux/NVIDIA CC8+；[RELEASE:9-21](https://github.com/triton-lang/triton/blob/e50b186e8bd2d16ae3f564311ef868aa3bd45a2d/RELEASE.md#L9)列3.5+ AArch64 wheels，而3.4.0仅x86-64。**这不证明目标JetPack/PTX/SM87上的当前kernel可执行。**现场Triton版本未在已有probe记录中给出，执行者补记，不能假定已是ARM兼容版本。

无Jetson连接时只锁定门禁，不能宣布BF16/FP16/Orin内核可用：实际aarch64、JetPack/L4T、驱动/CUDA、Python/torch/Triton/cuBLAS/cuDNN/ptxas版本和package来源先读；在既有环境分别做BF16/FP16真实小matmul、SDPA、prescale反量化/输出桥接和当前Triton M1真实shape compile/launch，记录退出码和数值。当前kernel仅BF16，不静默FP16cast；FP16全模型转换是独立质量候选，需再次252桥接和全轨迹。若依赖不支持，保留失败并另行决策隔离环境，不能本轮uv sync/pip替换原环境。

最终Orin验收：加载无OOM、首个完整窗产出有效轨迹（FlashDrive原API首窗None不算成功）、稳态连续运行与热状态、重置/恢复，按实际部署输入计入必要decode/processor/H2D/视觉/语言/Expert/后处理和同步；每条完整请求≤300ms、>300ms为0；独立同任务teacher meanADE损失≤15%；全系统统一内存留有OS/相机空间且无swap掩盖。固定输入、64点dt0.1、功耗/频率/温度、失败率、日志及hash，三项须联合成立。

## 停止、回滚和升级

来源或packed hash改变、漏层/多层、保护参数变化、CPU测试失败、真实activation不支持、NaN/OOM、缓存污染、noise/prefix边界不符、strict桥接失败均停止新候选；完整保留产物与退出码，不删除失败，不静默fallback。不停止其它项目。不以观察超时重启现有full20。回滚是关闭新`hybrid_m1`开关/启用新入口reference或退出新进程；旧文件与资产无改动。

若全20质量过线但完整generation仍超预算，升级到LightVLA Qwen视觉适配与短/无实时CoT策略学生的独立训练决策，维持原动作接口，不能无证据继续叠加局部机制当300ms承诺。若Orin加载仍OOM或ARM内核不可用，优先低峰值loader/真实依赖桥接，不把L20成功作实机结果。PARO/DFlash、静态跨帧shift、FP16、动作速度缓存及新增训练均另列决策，未被本轮开关自动批准。

本次执行留痕：仅本地只读源码/JSON/hash核查、固定commit公开源码与NVIDIA文档读取；无GPU、无服务器连接写入、无包安装、无模型下载。web-access已完整读取；CDP前置检查显示Node22.23.0、Chrome未连通，使用其允许的公开静态读取。唯一新增文件为本文；不声称任何尚未执行的小测试、全模型、全20或Orin验收通过。
