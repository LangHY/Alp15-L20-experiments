# Jetson16GB / 300ms / 教师精度损失15%：执行与采集前计划

日期：2026-10-08（Asia/Shanghai）；主智能体独占本文。用户目标完整保留，不以L20筛查代替Jetson验收。决策依据见 `ALP15-JETSON-300MS-ROUTE-20261008.md`。

## 当前状态与授权

当前目标active，上一目标请求被中断不视为已有任务在跑。现场重新确认服务器hostname zwl-acloud、GPU只有其它项目PID453201（15428MiB），本项目上一AWQ任务已结束。不是因观察超时而重启旧任务。本轮持续优化构成新目标的实际进展，不改变旧结果。

编排：GPT-6.1 Sol high独立路线决策、两名GPT-6.1 Sol high按文件边界执行，主智能体统一服务器GPU。kernel执行者独占3个新kernel/benchmark/tests及kernel报告，KV执行者独占新KV runner/tests/report，路线决策者仅拥有路线报告；原3个AWQ源码及所有旧权重/数据/实验不可覆盖。

允许：新隔离源码/测试/结果、小型文件准备、既有数据与权重只读、L20受控实验。禁止：自动下载大模型/镜像/数据、sudo/系统变更、修改原venv、停止其它项目、删除旧权重或缓存。用户要求的大文件先核实国内来源，只提供命令。

源仓库HEAD为36aeb4c5938cbc2eb2aed33b22434773da4ab639，但现场base_model.py与delta_tokenizer.py存在既有未提交改动；保留并独立记录实际源码SHA，不能据HEAD冒称官方原样链路。小型现场快照保存 `evidence/jetson_300ms_20261008/source_snapshot/`，不修改原源码。

## 目标及统计口径

精度沿现有同任务mean ADE（对GT的误差，越低越好）作为主筛查指标；教师60条固定记录实算meanADE=1.2323994789272548m，15%上限=1.4172594007663428m。同时记录FDE、clip中位/p95/max、动态/非有限/边界失败、改善/退化比例；不使用minADE，不縮短64waypoint/dt0.1，不改噪声或剔除失败场景凑门禁。

独立重复为clip而非seed：20clip开发集×3配对seed42/43/44是探索性筛查，非独立泛化证明。沿冻结20clip列表，不事后换名单；候选冻结后需新独立测试及真实部署口径的教师对照。没有事后功效计算，不以现有20clip夸称统计功效足够。

300ms保留为完整请求上限，报告mean/p50/p95/p99/max与超时数，不擅自改成平均或p95门禁。每条部署请求从当前图像/历史可用开始计时，包含实际必要的decode/processor/H2D/视觉/语言/Expert/后处理/同步，不能将三seed共有generation摊销后冒称单次延迟。加载、首推、稳态分别记录；Jetson启动不能OOM，稳态也要通过连续运行/热状态测试。

16GB门禁是Jetson全系统统一内存，不是模型裸storage，也不能直接相加RSS/CUDA独立峰值。必须实机记录tegrastats、系统可用内存、进程、缓存/workspace、swap/majorfault、功耗与频率/温度。当前未取得Jetson连接，不宣布最终完成。

## 已批准的首轮矩阵（正式采集前锁定）

1. 算子探针：固定full20 AWQ的真实gate_proj 5376×4096、M=1/17/256/3086，固定随机activation seed20261008+M、BF16，首调用另记，warmup3/repeat10。分开测full_reference/dequant/prescale/dense_linear的同步wall与CUDA event，不相加阶段峰值。不是全模型质量或Jetson性能。候选需使用同matrix/activation及格式，不重新量化amax/prescale。
2. KV门禁：固定前2clip×3seed，原full-prefill10步对旧AWQ参考逐条ADE差<=1e-4m、noise/prefix一致；actual generated KV路径对同full-prefill逐条ADE差<=1e-4m才称无损。NMSE<=1e-3仅辅助。所有生成边界、cache长度与污染检查通过后才扩20clip。
3. 少步求解探索：对稳定的actualKV候选使用官方FlowMatching.sample的inference_step=10/6/4/2，只改变该因素。每clip与seed对配，不换token/噪声/数据。采用固定seed的区组内顺序以减少时间漂移；共享GPU干扰记录，不由时间排序直接认定Orin加速。原10步对照保留。
4. 如果KV非无损：停在门禁，保留差异，由决策者判断是否作为新部署口径继续；不默默放宽阈值。任何OOM/非有限/来源变化/缓存污染停止新任务并留痕，不动其它项目。

探索优先级由用户实际约束决定：减少重复工作→求解步数→真实packed内核→视觉token/策略学生/动作蒸馏→低峰值loader与Orin联合验收。不会将“易通过的单层内核”替代完整模型目标。

## 已执行的算子基线证据

原benchmark冻结服务器 `experiments/jetson_300ms_20261008_code_v1/scripts/benchmark_alp15_awq_kernel.py`，SHA256 ab8fe9704b0a3b08a96570f9571fbc89bd93be9822f73a0321933824680631ae。服务器4项CPU回归0skip、退出0；实际CUDA基线进程退出0，结果 `experiments/jetson_300ms_20261008_kernel_probe/gate_reference_bf16.json`，已取回本地 `evidence/jetson_300ms_20261008/kernel_probe/`。具体命令见kernel报告，执行参数与源码hash同时保存在JSON。

| M | reference wall中位ms | dequant中位ms | dense_linear中位ms | reference额外allocated字节 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.2679 | 1.2333 | 0.0332 | 353878016 |
| 17 | 1.2891 | 1.2330 | 0.0426 | 354009088 |
| 256 | 1.3590 | 1.2335 | 0.1257 | 355966976 |
| 3086 | 6.4405 | 1.6934 | 3.5016 | 379150336 |

此结果形成真实反馈环：小M主要代价在原reference完整反量化及同步，允许研发不构造整份dense的fused候选。仍需CUDA数值/实际矩阵桥接与完整驾驶验证；不是已实现加速或300ms证明。不会据此省略视觉/Expert/serial decode优化。

## 持续交付门禁

### 2026-10-09：full20终态与混合解码实际启动

此前201条/running记录为历史快照，现追加终态：full20_v1已completed，300条完整，pipeline于2026-10-08T14:22:12Z记录COMPLETE/成功退出，结果SHA256 db04c2ed3226e2c3c988c3f6ebd9218bd6150759d3334d25d7e193c7693ef84e。本机/服务器一致。结果及限制见ALP15-KV-FULL20-RESULTS-20261009.md：四步相对教师meanADE+1.0245%，L20 E2E p50=5267.97ms/max=6495.00ms，60/60超过300ms；不是Orin达标或独立泛化。actualKV严格对拍58/60失败，保持新候选身份。

混合解码v2独立复核P1/P2修复；本地及服务器30项CPU回归均通过，无skip。源码冻结code_v2，与执行报告SHA一致，旧code_v1保留。仅启动reference2及252真实激活捕获：2026-10-09T04:17:42Z，真实launcher2038610、Python2038615已观察live。日志experiments/jetson_300ms_20261008_hybrid_m1_code_v2/reference2.log，输出hybrid_reference2_v2；不自动扩评。启动脚本scripts/run_alp15_hybrid_reference2_20261009.sh。捕获成功终态/receipt后才运行252CUDA probe，再过门禁才hybrid2；失败留痕而非覆盖重启。其它项目PID453201未停止。

低峰值loader执行者连接中断后恢复原文件所有权任务，尚未实体加载。CPU-meta flash2可能调用CUDA availability必须用真实probe确认，不自行改SDPA或放宽门禁。最终三目标保持active，未取得Orin实机证据。

2026-10-09T04:21:07Z追加reference2终态：launcher日志TERMINAL exit=0，Python2038615已结束，result completed_awq_hybrid_development_eval/exit_code0、6条完整轨迹，252真实激活及reference_final_receipt已生成。结果SHA256 1b60bab73f88865e873f4bde777c3298f04a47286c6de5532a65df780b5770b8；轻量result/receipt/reference2.log/cpu_tests.log取回evidence/jetson_300ms_20261009/hybrid_reference2_v2。reference E2E均值5604.93ms、max6298.50ms，6/6>300ms；不是hybrid收益。下一阶段已串行启动probe_alp15_awq_hybrid.py，PID2042839已确认live，--device cuda:0 --warmup3 --repeats10，--paired-reference hybrid_reference2_v2和--activation-manifest绑定其真实capture，输出hybrid252_v2，日志code_v2/probe252.log。probe自行先验证成功reference/来源/252真实覆盖，任何失败停止；尚未启动hybrid2或扩20。

后续终态更正：probe PID2042839已结束，result=failed、matrices0；绑定校验 FileNotFoundError 指向reference实际import库存中的`/tmp/tmpu5d2gwvq/_remote_module_non_scriptable.py`。reference生成本身exit0，但其跨进程来源凭据不可消费，因此不批准hybrid2。主任务在服务器仅对该库存调用原verify_hash_inventory，连续两次稳定复现同缺失文件；probe失败result/日志取回evidence/jetson_300ms_20261009/hybrid252_v2。执行者恢复五文件唯一所有权做v3回归和动态来源持久化；禁止忽略所有/tmp、事后伪造旧源码、改旧v2结果或启动GPU。新reference需新code/output快照；修复未完成，目标保持active。

本次继续执行：SSH核实服务器torch/distributed/nn/jit/instantiator.py第21-23行使用TemporaryDirectory并注册atexit cleanup，SHA256 567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615；临时import消失有一手实际环境源码依据，v3采用精准动态module身份与原字节持久快照，不忽略普通tmp源码。reference2每条generated_token_ids长度10、prefix_length3096，generation约3586.50–4583.68ms；长输出文本不是此两clip的证据，不能据此承诺缩CoT足够。

低峰值loader首版PhaseA已冻结到服务器experiments/packed_loader_code_v1，服务器15项合成回归exit0。真实headers_v1在原payload读取前失败：代码期望gate_proj/up_proj/down_proj，实际mlp_00与mlp_29头部为gate_proj.weight/up_proj.weight/down_proj.weight，BF16及5376宽度一致。主任务用原read_safetensors_header/_require对mlp_00连续两次复现；失败result、traceback、headers.log/cpu_tests.log保存evidence/jetson_300ms_20261009/packed_loader_headers_v1。执行者独占五文件修v2 exact三个*.weight键契约，拒绝保护键规则不放宽。尚未跑meta或实体load。现场GPU仅原项目PID453201，未修改原环境/大权重，未执行大下载。

v2真实headers成功exit0：907保护/252packed/1663target/1159original全部核定，原checkpoint payload读取0B，planSHA256 3dcc7b4a7c0368930de87a45879a0ebf7c1fc619edb5dffc6e92b7653aa68bfc；服务器16回归exit0。LoaderPlan/result/cpu日志保留evidence/jetson_300ms_20261009/packed_loader_headers_v2。code_v2 loaderSHA d00e4a6f586ea6d8f300601d2ffd2b8c93fcc5fd5941c08b8e9f3dab60066db9。

同code_v2真实meta失败：torch._dynamo.trace_rules.get_torch_obj_rule_map在导入阶段报告Duplicate torch object `_meta_guards.<locals>.blocked` with different rules；guard将多个不同API赋同一函数，尚未触及实际模型构造/flash2检测。失败runtime为torch2.8.0+cu128/transformers4.57.1/safetensors0.7.0、defaultdtypeFP32、RSSpeak635305984B、原payload0、entity_loaded=false。完整result/traceback/meta.log保留evidence/jetson_300ms_20261009/packed_loader_meta_v2。执行者做独立wrapper identity红绿回归后v3；不提前放宽无CUDA门禁，不称meta不兼容或loader可用。旧失败不删除，实体load/eval继续硬拒绝。

code_v3实际复测：loaderSHA 5654934fa02c8b95b44c5948a2550b9827666e319f8c50ec4e7b59e29e1f58a3，testsSHA c7798d3fdf576af42a64a138225a70d6ca53289b1b30e181c1e422f1368b6aef，服务器18项回归exit0。packed_loader_meta_v3原资产审计通过同plan，但真实构造前import失败；完整cause为HF integrations.flash_attention→flash_attn_supports_top_left_mask→is_flash_attn_2_available→torch.cuda.is_available，命中精确CUDA guard，外层包装为Could not import module AutoProcessor。不是缺包也不是meta模型构造已失败；payload0/entityfalse/defaultdtypeFP32。result/完整traceback/cpu日志取回evidence/jetson_300ms_20261009/packed_loader_meta_v3。已升级独立高思考决策：ALP15-PACKED-LOADER-CPU-META-ADAPTATION-20261009.md，决定构造期可用性检查与无实体/无CUDA分配边界，不准执行者自行切SDPA或预导入绕过门禁。

数据口径再次明确：现有20clip与3seed是开发筛查，不等于60独立场景，更不能以300请求宣称独立泛化。按clip及更上层驾驶序列防泄漏；冻结候选后另用独立场景覆盖测试集，采集前锁定指标和样本量依据。大数据/模型/镜像仍不自动下载，先国内来源核实后交用户命令。本轮未因答疑修改数据名单或原结果。

混合v3实际执行：独立验收hybrid_v3_acceptance_20261009完整读冻结代码并39项CPU回归通过，允许新reference2 capture；GPU252/strict/Orin仍未验收。新服务器code快照experiments/jetson_300ms_20261009_hybrid_m1_code_v3，四新文件与执行报告冻结SHA一致。首次server unittest因SCP仍上传tests，0tests/exit5（cpu_tests.log），未据此宣称通过；确认SCP exit0与四源码/test SHA后，cpu_tests_v2.log记录39tests/12.987s/exit0。不覆盖首次失败。

新launcher scripts/run_alp15_hybrid_reference2_v3_20261009.sh独立创建，bash -n通过，不改旧v2。2026-10-09T04:40:46Z实际启动launcher2063807、Python2063812，命令--clips2 --steps10 --awq-forward reference --capture-decode-activations，输出experiments/jetson_300ms_20261009_hybrid_reference2_v3，日志code_v3/reference2.log；独占本项目GPU，原训练453201未停止。reference成功终态/原字节动态快照/receipt可跨进程消费后才新probe，旧v2资产和失败全部原样保留。

v3 reference终态：2026-10-09T04:44:01Z日志TERMINAL exit0、Python2063812结束，result completed_awq_hybrid_development_eval/exit0、6条64点轨迹，original_helper6/6通过，meanADE1.2327414155006409m，E2E均值5503.289ms/max5899.749ms，6/6>300ms。结果SHA256 02a163ddbadf9aec6c778d36a35e2e7b76515d17b8a384a35e55a9ace75d2356，服务器/本机一致；result/receipt/reference2.log/两次CPU日志保存evidence/jetson_300ms_20261009/hybrid_reference2_v3。dynamic库存明确_remote_module_non_scriptable、virtual库存torch.ops/torch.classes。新PID2068050 probe真实跨进程绑定后已进入CUDA实际矩阵，前3矩阵passed，尚非252通过；输出experiments/jetson_300ms_20261009_hybrid252_v3、日志code_v3/probe252.log，--devicecuda:0 --warmup3 --repeats10及paired reference/capture均v3；尚未运行hybrid2。后续仅沿该PID/终态，不因观察超时重启。

独立v3技术验收文档ALP15-HYBRID-M1-V3-ACCEPTANCE-20261009.md SHA d1a9304233867b5068a3e4b5c5fed2bfd59ea94e1f35917a95baef8c7d8d54d3；39CPU测试及额外来源负例通过，不是GPU/Orin证明。低峰值CPU-meta适配决策D-ALP15-PACKED-META-CAPABILITY-20261009已冻结，文档SHA741afd758b75c80fecf29b80e3c858218efa583a2b806335aa40afd86a52102f；主任务完整读取并派执行者做code_v4精确HF真实is_available查询例外/调用点日志/分配前guard/恢复负例，原flash2与Expert官方SDPA不改，查询可能cuInit须准确记录。其它CUDA模型分配、内核、payload和PhaseB仍禁止；真实meta结果未通过，不据决策文字宣称成功。

v3算子终态已验真：252/252实际矩阵及真实activation通过，反量化bitwise252/252、M1实际输出bitwise187/252、最大abs0.00390625、超容差0、disabled/M2/storage前后252/252一致；结果SHA3a26316c87508541b69b96a6571def779ed7cdb088b325a02fa25341c4efe4f3，服务器/本机一致。PID2068050结束/GPU释放后才新hybrid_smoke2_v3 PID2071254，已收到真实PID活跃输出，后续SSH状态读取延迟不作为失败或重启理由。完整结果见ALP15-HYBRID-252-RESULTS-20261009.md，矩阵加速比中位6.7893倍不是整模型或Orin加速；完整轨迹strict尚未验收、未扩20。

### 混合解码执行授权与低峰值来源审计

独立高思考决策D-ALP15-HYBRID-M1-20261008已冻结于ALP15-FLASHDRIVE-LIGHTVLA-NEXT-DECISION-20261008.md；批准仅M1融合，全部252矩阵桥接与同代码reference/hybrid完整轨迹strict smoke后才能扩20，不套用actualKV数值豁免。执行者hybrid_m1_execution_20261008独占新wrapper/probe/isolated_eval/test/执行报告五文件，禁止改正在运行code_v3、旧kernel、packed格式和原环境，GPU由主任务串行，代码生成不算CUDA验收。

低峰值决策者继续只读审计meta/分块装载；主任务取回实际父overlay小manifest，证据evidence/jetson_300ms_20261008/loader_provenance/parent-overlay-manifest.json，服务器与本机SHA256均d0133763c495781933cf9e5f6e32726b3c2577cb00319a27eb7c0c57bcc27d5c，原路径experiments/mlp_newdata8_recovery_20261002_165238/width_5376/distilled/manifest.json。未复制大权重、未改原manifest、未执行新模型加载。

本次继续检查确认full20实际PID1128980 live，201条completed、1条running、14个clip已开始，结果status=running。这是未终止的特定作业，不是失败或最终完成；没有重启、替换或删除。下一次沿同PID/JSON/日志核对，不据阶段输出提前选赢家。

### 新汇总口径smoke验收与full20实际启动

2026-10-08 21:50:16（北京时间）：launcher在smoke_v4退出0后实际进入full20_v1，子进程PID1128980，命令有--clips20、--steps10/6/4/2、--approved-smoke及显式new_candidate决策编号；launcherPID1124038仍live。这是实际启动，不是完成。新smoke30记录完整、保护模块和来源复验通过，4步meanADE1.22688889503479与旧smoke相同；同2clip教师mean1.3351951837539673已正确进入summary，相对变化-8.11164465%，不是用全20教师均值比较。L20请求max4311.8544ms，6/6请求超过300ms，Orin_gate_passed仍False。结果SHA256 5e18641c5a33973279d1e0a7e6e8ce96a9023c78e2bb084363b65b4dcf2600d9，本机/服务器一致，轻量JSON与memory记录在evidence/jetson_300ms_20261008/kv_candidate_smoke_v4/。下一轮核查full20实际PID/记录/终态，禁止观察超时就重启。

### 恢复执行：完整20clip（2026-10-08）

目标服务重新核查为active，上一轮因usageLimited停留属于无进展，本轮不重复状态说明代替执行。修正subset summary为同clip/seed教师均值，保留原全20教师来源校验；增加latency mean/p99/max及严格>300ms计数、clip ADE中位和改善/退化数。只改报告计算，不改模型/噪声/求解。旧smoke_v3原JSON与复核记录不变。

决策执行D-JETSON-KV-CANDIDATE-20261008扩评：新code_v3先固定2clip×3seed×5variant（30请求），同代码smoke通过来源/finite/配对与保护模块门禁后，再由脚本验真完整20clip×3seed×5variant（300请求）。四个actualKV步数10/6/4/2和fullprefill10全部保留，不据两clip提前淘汰或选择最终赢家。使用既有开发集，独立n=20而非60/300；不宣称泛化功效或Orin性能。共享GPU时间受干扰，保持既有variant顺序以复现数据，不能据此声称无偏的速度因果比较。最终赢家冻结前需独立数据与实机验证。

允许新增小型代码快照、日志和既有资产只读GPU评估；不下载大型资产、不改venv、不停止PID453201。新结果smoke_v4/full20_v1，若存在拒绝覆盖。launcher为scripts/run_alp15_kv_full20_20261008.sh，smoke退出非0自动停止，full20退出非0保留失败记录；禁止重启仅因观察超时。预计330请求，按此前5秒量级约30分钟（非保证），每行独立记录generation/Expert/端到端，不摊销多seed生成开销。

现场执行：服务器7项测试全通过无skip、退出0。本机/服务器入口SHA256一致为74886a867a8aae480dd848a6d068d2e3f817674d891774bcdabeb57f412292f8；tests SHA256 db22922d6dd27312124630b5e7b25a766628213b3983fdc636d5e5ff372587d7，launcher SHA256 83372f649023058057fb51edca22897538b4228761e3b14655f884a04e7933e3。launcher实际PID1124038，日志experiments/jetson_300ms_20261008_kv_code_v3/pipeline.log，2026-10-08T13:46:01Z开始smoke（北京时间21:46:01）。本轮结束不能据此宣称full20完成，后续须重新核查具体PID、进程退出与JSON终态。另启动GPT6.1Sol high独立适配决策，只写ALP15-FLASHDRIVE-LIGHTVLA-NEXT-DECISION-20261008.md，禁止并行GPU/大文件下载/改原环境。

### 2026-10-08追加：KV新候选决策与内核结果

决策D-JETSON-KV-CANDIDATE-20261008：独立路线决策智能体明确批准actualKV作为新精度候选，而非无损缓存替换。v1首clip/seed42严格门禁失败（退出2）原样保留；ADE差0.00039392709732055664m、trajectory NMSE 1.1041936546822378e-8、最大坐标差0.003307342529296875m，prefix/noise相同。执行2clip×3seed×10/6/4/2步探索；新候选需显式cache-policy和决策编号，默认strict不变。噪声/前缀/来源/有限值/缓存边界仍硬停。扩20前必须同代码/资产完整smoke，最终15%精度门限不变，不能据单条宣布通过。子智能体因服务额度退出后，主智能体接管KV脚本/tests唯一写入权。

v1结果SHA256为774a171c0477ff51c60b3227f15283f46d5bd596de3f29faa2cab12c5243697f，服务器/本机一致；轻量证据在evidence/jetson_300ms_20261008/kv_smoke_v1/。首条完整请求fullprefill6104.55ms、actualKV4550.05ms，只是共享L20诊断，非Orin或无干扰因果测速。

v2内核9项服务器CUDA/CPU测试全部通过无skip，探针退出0，BF16反量化权重逐元素完全相同。同次测量reference→fused同步wall中位ms：M1 1.26925→0.15014，M17 1.28552→0.45621，M256 1.36015→1.81336，M3086 2.54912→29.14265。输出通过rtol0.01/atol0.02算子门限，不代表全模型精度；因此不批准全量替换，仅支持后续小M解码专用路径评估。M1临时allocated从约354MB降至10752B。结果SHA256为3f23f82367f5d2d4fc45e2bbca60bd853de4ebd86055a941caec23cd0e9f2a66，服务器/本机一致。大型权重保持服务器原位，无下载。

用户新增最终适配要求：必须同时考虑FlashDrive与LightVLA，并计入最终端到端验收。先核实原作者仓库版本、模型结构/输入/位置编码/视觉token/KV/动作接口、量化内核及ARM64/SM87依赖，再锁定组合顺序与消融。不能直接相乘论文加速比，不能把机器人操作上的LightVLA结果当驾驶验证，不能把FlashDrive服务器成绩当Orin成绩。所有组合仍需同教师精度口径、完整请求延迟、冷启动和全系统统一内存门禁。

KV v2源码SHA256 b5a3966e309d1f446528f2dd452328a4bc7b055cba1870b7b87b0cc783abda42、tests SHA256 660ca30830db15db63b60ff981cf2374c9d61bd29bd5556246d1576c4f08b975，本机/服务器一致，服务器6项回归通过无skip。v2首次启动缺少隔离snapshot中两个packed helper，退出1且保留；复制冻结helper后启动新结果目录experiments/jetson_300ms_20261008_kv_candidate_smoke_v3。已观察新Python PID1098432在L20运行，原项目PID453201未停止。依赖与原模型源码未改；具体联合适配见ALP15-FLASHDRIVE-LIGHTVLA-ADAPTATION-20261008.md。

目标保持active，直到Jetson16GB、完整请求<=300ms、教师损失<=15%三项同时有相应范围的权威证据。缺实机不称完成，临时阻塞不妨碍继续L20可验证工作。下一次检查以真实进程/退出码/产物为准，不靠日志存在或旧意图判定在运行。
