# AWQ M1 混合路径独立技术验收

## v2 复核结论（当前有效，2026-10-09）

验收者与本文唯一写入者仍为 `hybrid_m1_acceptance_20261008`。本轮按原冻结决策与最新执行回报复核 v2，代码和其它文档只读；未运行服务器/GPU、未加载真实权重、未下载/安装。下文 v1 的 P1/P2 为历史发现，**其阻断已由 v2 代码修复及独立CPU负例验证解除**。这不是Git分支code-review，不假称固定比较点或分支审查。

**可以启动 code_v2 的 reference2 + capture。完成 exit0、完整6条轨迹和有效 final receipt 后，才可串行启动252真实activation CUDA probe；该探针252/252全部通过，再启动同代码hybrid2 strict smoke。** 主任务负责现场 root/config/import资产与GPU串行前提，不应复用v1 capture/probe，也不应上传到已运行流程的源码目录。当前没有发现阻止这三阶段按序执行的新增代码错误；这是实施准入，尚无CUDA/轨迹/性能/Orin验收结论。

### v2 冻结身份

| 文件 | 本次实测 SHA256 | 核对结果 |
| --- | --- | --- |
| docs/baselines/ALP15-HYBRID-M1-EXECUTION-20261008.md | 1ad1b648b954b7f0dede29ecce71af0bd48bd10b368ac462ce663c2668188f79 | 最新执行回报身份登记；不在文件内自引用 |
| scripts/alp15_awq_hybrid.py | 801c7f6cd1b314b73a882071480d617e444ef3ba7693e1c10a0ec53c52df270d | 等于回报v2冻结值 |
| scripts/probe_alp15_awq_hybrid.py | 0cd19e514c3b10988bc39061b2529424823c0c00ba0097123d7ae99f2d489075 | 等于回报v2冻结值 |
| scripts/run_alp15_awq_hybrid_eval.py | a2ccec42a954be1d8325c89afbc32337193808265121448ec770ec92a792e887 | 等于回报v2冻结值 |
| tests/test_alp15_awq_hybrid.py | e181dd82532bf06ec50cb2c1e8c25c0c8ea65652cba13e5542a1bc0652eaeb85 | 等于回报v2冻结值 |

原决策 SHA256 `5f567658ef404c2687d629d9051f6595b88651424f14a6c3c2e74a5cdbff3afb` 不变。旧kernel/full_packed/packed_w4a16/KV helper SHA仍等于下文冻结值。v2新增门禁没有改动原 kernel、packed格式、amax/pre_scale或上游生成/Expert流程。

### P1 关闭：成功终态、runtime与252真实capture绑定

`alp15_awq_hybrid.py:65-132` 的 `load_bound_capture` 现在同时检查：manifest v2及决策；reference/result/receipt的归属路径；receipt中的result与manifest SHA；reference completed、exit_code=0、reference身份；实际交付manifest/receipt/252 coverage；code/runtime/input/actual-import四个非空来源库存相等且逐文件SHA不漂移；保护摘要前后相同；capture clip与text seed；完整2clip×3seed且每条轨迹合法；252唯一完整投影与252唯一文件SHA。probe要求真实capture显式指定paired reference，读入Tensor时继续检查M1/BF16/连续/无梯度原内核约束。

终态顺序经完整runner核对：只捕获tensor及内存描述→最终保护/source/runtime/import/input审计→写manifest→保存completed result→写独立receipt→再次执行完整绑定复核。manifest只保存result/receipt路径；result保存manifest SHA与receipt路径；receipt保存result及manifest SHA，**没有哈希循环**。最后绑定失败会把result原子改failed/exit1，原receipt中的result SHA随之失效；缺失/孤立receipt同样不能背书成功。

`validate_probe_binding` 要求正式探针状态 `passed_all252_cuda_m1_probe_with_actual`、实际coverage252/252、每个真实矩阵都有actual_decode_bf16测试、capture/result/receipt SHA与paired reference绑定、公共来源和保护身份一致。随机输入诊断使用独立状态 `passed_all252_random_only_cuda_diagnostic`，不能晋升正式候选。扩20可复用2clip捕获的算子探针，但必须与20clip reference的同代码/完整资产/保护身份一致，并仍逐20clip做真实严格轨迹对照。

### P2 关闭：64×3、finite和重新计算strict门禁

`validate_trajectory_row` 要求双方恰好64×3，全部坐标/ADE（及存在的FDE）有限，冻结decision/10steps/clip/seed，合法noise/prefix SHA和token/future-start/prefix长度边界。`strict_pair_gate`先校验双方，再比较NMSE/ADE、noise、prefix/token及请求身份。`validate_smoke`重新读取绑定成功reference和probe，重算每一条实际轨迹的strict gate；不会只信任保存的passed=True。正式请求依旧由measured_trajectory锁定 `(1,64,3)` 和finite，不放宽原成功定义。

probe首测字段也已澄清为 `hybrid_first_for_this_activation`、`first_for_this_activation_includes_factory` 和JIT缓存说明，v1的实际输入第二次测试误称包含factory/JIT的问题关闭。

### 独立验证与剩余现场门禁

本轮完整读169行最新执行回报、312/244/529/473行四个v2文件；旧AGENTS/决策冻结身份不变。环境仍为本机 `/usr/bin/python3` 3.9.6、Torch2.8.0，CPU合成数据；没有真实safetensors/模型加载。

独立命令 `/usr/bin/python3 -m unittest discover -s tests -p 'test_alp15_awq_hybrid.py' -v`：exit0，**30项全部通过，3.094秒**，完整工具输出 `6ded21`。五文件SHA核查输出 `077804`，旧冻结SHA与时间输出 `16d384`（2026-10-09 12:17 CST）。负例包含：缺receipt、SHA仍一致的failed reference、runtime与actual-import漂移、声明runtime不配对、251层、重复文件、错paired reference/probe绑定、random-only/虚假实际coverage、双方1点或NaN、保存passed的短轨迹、finalization失败撤销completed。正例从实际生产finalization函数生成合成manifest/result/receipt，证明绑定可成功且不循环；不是手写CUDA通过数据的现场证明。

现场待验：新code_v2和旧helper的实际SHA/import路径；既有离线root/config/source manifest/overlay/20份cache；reference自主生成及6条轨迹/完整receipt；252真实权重random+actual数值和BF16逐元素dequant；每请求252M1实际分派、strict smoke；随后20clip/300ms统计。当前 `--preflight`仍不等于实际config/loader可用性，v1现场路径前提继续有效。主任务所述旧full20已完成/GPU释放不是本验收者独立核实的服务器证据，应沿用主任务已留痕终态与现场调度。

无新增代码阻断项。保留失败产物；strict轨迹失败退出2后不得套actualKV豁免扩20。后续任何源码变化重新冻结并遵守同代码配对；本轮不自动授权融合其它因素或改变10step/fullprefill。目标Orin16GB、完整请求max≤300ms、teacher meanADE损失≤15%及独立数据联合门禁仍未验收。

## v1 验收留痕（以下不代表当前v2状态）

日期：2026-10-08，Asia/Shanghai；验收者及本文唯一写入者：`hybrid_m1_acceptance_20261008`（GPT-6.1 Sol high）。对应冻结决策：`D-ALP15-HYBRID-M1-20261008`。本轮只读代码、运行合成 CPU 验证；无真实权重加载、无服务器连接/写入、无 GPU 操作、无安装或下载；未操作 full20 PID1128980。

## 结论

**允许主任务在现有 full20 安全终态、GPU 串行调度及服务器路径前提核实后，开始 reference2 + 真实 decode activation capture 的诊断运行。当前仅本地实现验收通过，不能自动宣称 hybrid CUDA 或完整轨迹通过，也不批准凭现有软件门禁自动扩20。**

包装器的共享 storage、M1 专用分派、保护模块摘要、同代码 reference/hybrid 配对、smoke→full20 资产集合和计时隔离未发现会阻止 reference2 的确定错误。下面两项门禁缺口须在 hybrid smoke 验收/扩20前由主任务补充实际证据或修正新文件：真实 activation 的完整来源与成功 reference 绑定不足；strict 配对门禁自身未锁定64点。若修改新源码，必须重新冻结 SHA，并按同代码约束重新生成受影响 reference/probe/smoke，不能将旧结果强行配对。

Orin NX16GB、完整请求 max≤300ms、同任务教师 meanADE 损失≤15% 三项联合目标保持不变。本轮16项 CPU测试、合成 storage/hash 证明均不能代替252真实矩阵、真实轨迹、L20终态或Orin实机。

## 验收基点与方法

本任务不是 Git 分支变更审查。已完整读取 `code-review/SKILL.md`；其固定比较点/三点 diff/双轴子代理流程不适用于主任务指定的五文件冻结交付，未假称执行该技能完整流程。本次按冻结决策、AGENTS 和文件 SHA 审查集成，不新增比较点，不派生子代理。

已完整读取 AGENTS、102行决策、147行执行回报和四个新代码文件。决策 SHA256：`5f567658ef404c2687d629d9051f6595b88651424f14a6c3c2e74a5cdbff3afb`。执行回报 SHA256：`262e6e4e6b5b2ca69d5c6372e9e66e31bd03f045d29b67e97a53bb902574762b`（主任务仅给四个源码冻结值；第五文件实测登记，不伪称已给定）。

| 文件 | 本次 SHA256 | 对照 |
| --- | --- | --- |
| scripts/alp15_awq_hybrid.py | 1941d88cf563261a9fcf2b6afc18a0e6df828cd1e5391bb77b6a19dacc7fec6a | 等于冻结值 |
| scripts/probe_alp15_awq_hybrid.py | 2bedc47681a7849ba0d54c53123ce8153d7aae50de090b9359b14503f36e12f9 | 等于冻结值 |
| scripts/run_alp15_awq_hybrid_eval.py | 34f4332b0e0995ea9d26f778e3d49d9e55b102f60097251c88c3a135a13c675e | 等于冻结值 |
| tests/test_alp15_awq_hybrid.py | 5bb8838531fd74ed6d2644d77c4a360bb248a93755a99a1cd14db1c1b102c483 | 等于冻结值 |

旧 kernel、full_packed、packed_w4a16、kv_steps_probe 的 SHA 均等于执行回报/决策冻结值。进一步只读完整 KV helper 相关函数、历史 full20 源码快照的 MemoryRecorder、sampled、loader 验证逻辑。历史快照只用于判断接口，不当成当前服务器 import 身份；当前 runner 会核验真实文件 SHA。

## 风险发现与进入边界

### P1：activation 来源没有完整绑定成功 reference 与当前 runtime

位置：`run_alp15_awq_hybrid_eval.py:98-105`，`probe_alp15_awq_hybrid.py:69-91,125-145`，runner `:234-247`。

capture 确实来自注册在252个真实投影上的额外自主生成 pre-hook，且生产者要求 captured 集合恰好 TARGET_NAMES；不存在用随机补齐的代码。但 manifest 只带 `code_sha256/input_sha256/protected_sha256`，未带 `runtime_source_sha256`、实际 import 来源、成功 reference result 标识/hash。manifest 在整个 reference 最终来源/保护模块复核前写出；若随后最终审核失败，留下的独立 manifest 仍能被 probe 接受。probe 只复核 capture 的 `code_sha256`、packed 文件身份及声明的 protected 摘要，没有验证其所属 reference 最终 completed 状态；runner 又不要求 probe 的 actual_activation_coverage 为252/252，也未绑定 probe 的 activation manifest 为 paired reference 实际交付的 manifest。

影响：不同官方/Transformers runtime、失败 capture 所属运行、部分实际 activation，均不能由当前自动门禁独立排除。random-only 的252矩阵探针按决策可做诊断，但不能描述成真实驾驶 activation 完整验收。

解除条件：主任务在 probe/hybrid smoke验收时确认 reference `exit=0/status=completed`，capture manifest/hash等于其 `decode_activation_capture`，实际252名称唯一/完整，各文件 SHA一致；比对 reference/current/probe 的官方及公共 runtime 来源、clip/text seed/packed/protected/Input身份。可在新文件加此校验；不应修改旧 runner。若采取人工补充验收，须把上述对照逐项留痕，不能仅信任 `status=passed_all252_cuda_m1_probe`。

### P2：strict 配对门禁允许非64点轨迹

位置：runner `:25-33` 调用冻结 KV helper `numerical_gate`（旧 helper `:69-84`）。该 helper只要求两条轨迹长度相同且非空，每点3维；未要求64点。CPU追加复现：两条 `pred_xyz=[[1.,2.,0.]]`，其余 noise/prefix/token/ADE 相同，`strict_pair_gate(...)[passed] == True`。现有16项测试未覆盖此情况。

影响范围有明确限制：正常 runner 的 `measured_trajectory` 已锁定 xyz `(1,64,3)`且finite，实际当次生成会拒绝非64点；这是可复用门禁与读入结果的完整性缺口，不是已证明真实生成返回了短轨迹。`validate_smoke`读取 `passed=True`而不重新检查记录完整shape，因此扩评前仍须独立复核每条 reference/hybrid 的64×3、finite、合法noise/prefix/token边界。noise和四个prefix字段的实际比较、NMSE≤1e-3与ADE差≤1e-4本身正确，旧 actualKV豁免未被套用。

解除条件：新 strict gate锁64×3并做负例测试，或主任务对实际全部结果64×3作独立验收留痕；不得改轨迹成功定义以通过小测试。

### P2：离线根目录与真实 import 尚待现场核实

runner `:153-218,261-300` 依赖既有 root/compression 与 root/scripts，模型/root路径、教师/overlay绝对路径、dataset/config目录和来源 manifest。本机只有历史源码快照，不能证明服务器部署布局。config路径与仓库既有离线交付文档一致，未发现拼接确定错误，但“旧 full20在运行”也不能证明新 snapshot 会 import 同一份helper。

reference2启动前由主任务核实：snapshot四个只读源码副本SHA、新文件SHA、所需真实helpers的绝对import路径及AWQ冻结hash；root/alpamayo1.5/venv、models、dataset、configs/dependencies/configs两组目录及configs/manifests；overlay和完整20份teacher cache。`--preflight`在实际模型/config资产验证前返回，所以不能把它的exit0当成离线config/loader可用证据。禁止为匹配路径下载资产、安装包或修改来源；前提失败保留日志升级。

## 已确认实现与证据

- `HybridAWQLinear` 使用相同 reference；仅enabled且真实M==1创建/调用原fused；M2/M17/M256/M3086或disabled不import候选/Triton、不隐式cast/contiguous或fallback。非法shape/device/dtype、空输入显式失败。installer先完整252预检再替换，实际buffer与逐文件资产比较；边界不含保护模块。
- late fused增加 `reference.*`/`fused.*` 的state_dict别名键，这是包装产生的真实变化；保护摘要以目标投影prefix排除全部别名，唯一storage统计按storage去重。CPU独立小模型前后 protected digest相同，persistent storage `484→484B`。不能把state_dict键数量当容量，也不能将这种state_dict直接当原checkpoint格式。
- 来源、packed hash、保护摘要、安装前后unique storage门禁均存在。模型安装后不会实体化新的persistent dense。runner在首M1前做的storage_after不能单独证明late fused后storage，本次CPU追加复核及未来逐矩阵probe的before/after提供补充；GPU现场仍需真实验证。
- 同provenance明确排除awq_forward字段，否则reference/hybrid无法配对；其code/runtime/input/steps/request/text/trajectory seeds完整比较是正确的。variant身份另由paired reference检查为reference，smoke检查为hybrid_m1。smoke/full20均冻结全部20个cache hash，再仅改变clip选择，因此资产集合不会随2→20意外变化。
- 逐252矩阵随机BF16和可选真实activation各自做M1数值桥接；dequant逐元素BF16检查；禁用/M2均调用原同一reference，要求逐元素相同。大M真实输出复现目前只有M2；M17/M256/M3086仅CPU合成覆盖分派，不能宣称真实大M CUDA测试已通过。逐层释放dense，不跨252层积累。实际252/252仍待GPU运行。
- probe first/warm的wall/event同步、reset peak按单项执行；runner只在MemoryRecorder外层phase重置peak，内部stage仅synchronize计时，没有互相重置污染。CPU回拷/数值oracle/保护hash/capture在正式E2E计时和请求memory phase之外。sampled验证在dispatch记录之后，未混入当次成功M1计数；它会影响后续请求热状态，两组同顺序执行并如实记录即可。
- reference/hybrid使用独立进程，variant_order已写为all reference then all hybrid，非同实例交叉区组。资产/保护/代码一致性可作为诊断对照，仍受时序与共享GPU负载影响，不作为无干扰因果测速。仅记录GPU前后进程，不能证明中途无其它负载；主任务负责串行和环境留痕。
- minor：probe对真实activation第二次测试仍使用字段名 `hybrid_first_including_factory_jit`，此时工厂/JIT往往已经由随机输入预热；解释时应视为该输入的首测，不冒称冷启动。首个随机测量才包含该模块fused工厂；相同shape内核跨矩阵的缓存也可能复用。

## 实际验证留痕

环境：macOS本机，`/usr/bin/python3` 3.9.6，Torch2.8.0，CUDA available=False；仅合成tensor，无safetensors资产加载。时间窗口约22:23–22:29 CST。

1. `shasum -a 256` 五个新交付文件及四个旧只读helper：退出0，值如本文；工具输出 `f5cfa1/7773bb`。
2. `/usr/bin/python3 -m unittest discover -s tests -p 'test_alp15_awq_hybrid.py' -v`：退出0，16项全通过，1.598秒；完整输出 `7773bb`。只证明CPU合成/mock行为。
3. 内存中构建1个真实PackedAWQLinear合成投影及Expert；调用 `protected_model_digest/storage_inventory_by_component`，mock fused工厂后前向：退出0，摘要不变，storage484→484，state_dict从4个reference键变成8个shared alias键；工具输出 `f44d7c`。同时复现1点轨迹strict错误通过，`truncated_1point_strict_passed=True`。
4. 只读historical MemoryRecorder/sample/loader源码；未在本机运行Linux RSS/CUDA路径。历史MemoryRecorder SHA `32ae3c3c0177b19f329ef2b736a61653243bbce6665cff42316d74f8cab7f1bd`；sample helper SHA `2d7049d548bce8ee0410e89a68c51750dcb6ca5a734ab34afa10b7d62afc3508`。这些值是本机快照身份，不替代现场import核验。

原任务代码只读；唯一新增/修改源文件为本文。后续顺序仍是：主任务安全串行条件核实→reference2完成及capture来源验收→252真实random+actual bridge→同代码strict hybrid smoke→主任务补齐门禁并独立验收→20clip配对。CUDA、质量、性能任一步失败保留产物停止新候选；不停止原full20、不套旧质量豁免、不宣布Orin完成。
