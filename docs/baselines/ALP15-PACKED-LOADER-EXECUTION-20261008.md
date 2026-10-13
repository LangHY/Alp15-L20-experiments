# Packed-only loader Phase A 执行回报

对应决策：`D-ALP15-PACKED-LOADER-20261008`。执行者/五文件唯一写入者：GPT-6.1 Sol high / `packed_loader_execution_20261008`。开始日期 2026-10-08，首版交付日期 2026-10-09（Asia/Shanghai）。本回报仅证明首版代码与合成回归；真实 L20 headers/meta 探针由主任务统一执行，尚未在本执行者任务中运行。未调用 superpowers、未连接服务器、未运行模型/GPU、未安装包。

## 交付与停止边界

- `scripts/alp15_packed_loader.py`：标准库资产审计与 LoaderPlan；实际原构造器 meta 路径、252 meta 占位、原/目标严格集合和参数/buffer种类、对象 alias、10 个非persistent buffer 及普通 Python trajectory ActionSpace 的显式审计与小官方类恢复；来源 hash/路径冻结；CUDA API/实体 checkpoint 入口/权重文件访问/ModelOpt/Triton 导入硬拒绝；finally 恢复 BF16 default dtype。
- `scripts/probe_alp15_packed_loader.py`：默认 headers，`headers|meta|load` 隔离进程入口；输出目录存在即拒绝；完整 traceback、失败前真实已导入路径/hash/version/default dtype/RSS 留痕，退出码 1。headers 成功为 `headers_only`，真实 meta 完成全部检查为 `meta_passed`，均 `entity_loaded=false`。
- `scripts/run_alp15_packed_loader_eval.py`：保留 `bf16_reference|packed_only` 默认 reference 的隔离参数入口，**评测未实施且硬拒绝**；不通过选择 reference 自动启动完整 CPU/GPU 模型。
- `tests/test_alp15_packed_loader.py`：15 项小合成回归，覆盖 bounded header read、重复/非法 header、路径逃逸、块上限、Plan/MetaAudit 绑定、shape/dtype/kind/key/alias、非注册 Python 子对象、default dtype 异常恢复、禁 CUDA/payload/依赖导入、252 meta 占位 forward 拒绝、即使合成 passed audit 也不得进入未授权实体加载。

PhaseB `load_packed_only_model` 与 `protected_hashes_chunked` **尚未实施**。实体函数先要求同来源 MetaAudit，随后明确报告“未实施/未批准”并终止。没有 907 参数 get_slice 实体复制、target readback hash、252 packed 实体验证/加载、桥接或质量评测，不能据合成 tests 宣称 loader 可部署。主任务明确要求优先交付 Phase A。

## 本机真实环境和验证证据

`/usr/bin/python3` 实际 executable `/Library/Developer/CommandLineTools/usr/bin/python3`；Python 3.9.6 / Clang 17.0.0；`platform.platform()`=`macOS-27.2-arm64-arm-64bit`；Torch 2.8.0 CPU。未导入本机 transformers/accelerate/hydra/safetensors。GPU/驱动/CUDA实机版本不适用本轮本机 CPU 合成验证。

命令：

```sh
/usr/bin/python3 -m unittest discover -s tests -p test_alp15_packed_loader.py -v
/usr/bin/python3 -m py_compile scripts/alp15_packed_loader.py scripts/probe_alp15_packed_loader.py scripts/run_alp15_packed_loader_eval.py tests/test_alp15_packed_loader.py
/usr/bin/python3 scripts/probe_alp15_packed_loader.py --help
```

实测测试输出 `Ran 15 tests in 1.880s / OK`，退出码 0；py_compile 退出码 0；CLI help 退出码 0。测试完整逐项输出已保留在本任务工具执行记录；不把这一合成测试记录当作 L20 官方大型 meta 结果。

另运行标准库 `read_safetensors_header` 对本机实际五个模型分片读取首 8 字节 + 原 JSON header（唯一允许的本机大模型读取）。模块导入时 `torch/transformers/modelopt/triton` 在 sys.modules 内的集合为空。结果：1159 original keys，252 dense 投影排除后 907 protected keys，protected bytes=`8,265,517,540`；总读取 prefix+header=`142,136B`（JSON 本身 `142,096B`），原 payload read=`0B`。五个 header SHA 完全等于决策表。此独立 header 检查没有完成资产审计：本机不具备决策所需36个 overlay和252个 packed 的全部实际文件。

| 分片 | JSON header bytes | 原 header SHA256 |
| --- | ---: | --- |
| 00001 | 51368 | `49df741ba611dbe3ca4dbba7cff0b35dd73de4cb0131178f374acd89d3d6058a` |
| 00002 | 19088 | `89c487d89c1de9be5f75f7adc5842acd6fb283a1f8c78e8db7c64bf2b655bb2b` |
| 00003 | 18864 | `422596c5eae572fa8a8c4e66f711622bc47eacf8d162838c84c3f9dc63b2c48d` |
| 00004 | 28096 | `1330aad43ef7dc0a6624ed9f345a2f535dbfad936a818896faa262d7e1a54031` |
| 00005 | 24680 | `d767e19b5997d6cc48aa1aed85a7f09bbaf862dedcd0e4609328345a74ae99c8` |

## 上传布局与主任务阶段命令

主任务可把三个新 scripts 上传至独立目录 `experiments/packed_loader_code_v1/scripts/`，tests 上传至该目录 sibling `tests/`。不覆盖 root/scripts 中旧 helper。通过 `--root` 指向真实服务器项目；`--helper` 可显式指定已冻结的原 helper，`--source-root` 指定真实运行源码树。两个参数均在 root 内 allowlist；实际源码 base hash 必须为现场 `1f2e49...`，不接受 Mac 旧 `94425eef...`。

下列模板由主任务用既有 L20 推理 Python 执行；本执行者未执行它们。输出目录必须是新路径；禁 CUDA 阶段不需要 `--device`，传入该参数也不使用它。

```sh
<L20已有推理Python> experiments/packed_loader_code_v1/scripts/probe_alp15_packed_loader.py --stage headers --root /home/USER/lang/Lighten_VLA --awq-artifact /home/USER/lang/Lighten_VLA/experiments/awq_full_20261008_best8_full20_v4 --overlay /home/USER/lang/Lighten_VLA/experiments/mlp_newdata8_recovery_20261002_165238/width_5376/distilled --output /home/USER/lang/Lighten_VLA/experiments/packed_loader_headers_v1
<L20已有推理Python> experiments/packed_loader_code_v1/scripts/probe_alp15_packed_loader.py --stage meta --root /home/USER/lang/Lighten_VLA --awq-artifact /home/USER/lang/Lighten_VLA/experiments/awq_full_20261008_best8_full20_v4 --overlay /home/USER/lang/Lighten_VLA/experiments/mlp_newdata8_recovery_20261002_165238/width_5376/distilled --output /home/USER/lang/Lighten_VLA/experiments/packed_loader_meta_v1
```

headers 的 `LoaderPlan.json` 绑定原 config/index/five header/源mtime-size-inode、36 overlay SHA/header、252 packed SHA/header、配置 bundle 来源 manifests、冻结 helper 与现场 snapshot；1663 目标完整 shape/dtype/kind。每个原 shard 明确 `payload_sha256_verified=false`，另留历史冻结 payload SHA。`plan_sha256` 为 schema 内容摘要，meta 重新验证来源未变化且 MetaAudit 带同一摘要。原 907 guard 的 `94b2256c...` 与父 overlay 1051 guard 的 `77b2770e...` 分别保存各自原模型 ordered keys，不混用；本轮均不读取原 payload 计算 guard。

meta若失败：`result.json` 包含 status/exit/error/runtime已导入现场，`traceback.txt` 是完整调用栈；不得生成 passed MetaAudit。meta若成功：`MetaAudit.json` 记录所有 persistent meta tensors、652B 非persistent 清单、Python子对象状态、token IDs/完整tokenizer序列化hash、generation config、RoPE字段与CPU alias、小buffer数值hash/default dtype恢复、RSS阶段预算。它仍明确 `entity_load_authorized=false` 和未完成的 reference/实体/系统pagecache/轨迹/Orin门禁。

## 前提、偏差与升级

静态源码发现：原 config 使用 `flash_attention_2`，HF 4.57.1 的 `modeling_utils.py` 中 `_check_and_adjust_attn_implementation` 与 `_flash_attn_2_can_dispatch` 会经过 `is_flash_attn_2_available`/`torch.cuda.is_available`。**这是静态风险，尚不是 L20 实测失败。** 主任务明确要求先保持原 flash2 与 CUDA硬拒绝，统一运行真实 CPU probe、保留栈后升级原决策者；本实现没有改 SDPA、伪造 CUDA返回值、绕过HF检测或回退实体CPU构造。

MetaAudit小状态的源码核验与恢复行为尚未在真实模型运行验证。构造前即检查现场源码hash，因此源码/hash不一致会在原构造前停止。未知nonpersistent buffer、Python tensor属性、持久alias、dtype/shape/key变化、禁用入口调用、RSS预算超限都停止。torch原生meta上下文覆盖构造；不调用完整模型 to_empty/.to(cuda)/from_pretrained。

仍未完成：完整 L20 headers/meta 与决策验收；真实 BF16 reference 的config/generation/tokenizer/small状态对拍；保护源payload与目标readback；实体阶段≤32MiB分块一次分配与预算采样；252原packed数值与全部bytes；严格两clip三seed后完整扩评；Orin NX16GB真实cold-start/热稳态、请求max≤300ms、teacher meanADE loss≤15%联合门禁；FlashDrive/LightVLA/后续hybrid适配。

## 首版代码冻结 SHA256

| 文件 | SHA256 |
| --- | --- |
| alp15_packed_loader.py | `e7b832c78fe86d78699faa2df002c99d68622e9f0b5eb21c8527d08726f8f4ab` |
| probe_alp15_packed_loader.py | `107c761a0974529bfad4971775177711e750af717ef8efba4fbd7cc0e15107e2` |
| run_alp15_packed_loader_eval.py | `68802e4c479bb0ed234d47528dec9f393cb1ff04d8a004ad11f7b220d404af76` |
| test_alp15_packed_loader.py | `cd4c56e3f47b064d2928a18a3c9166943f586310be53d2b423c2518b7c2b77fc` |

回滚只需退出新隔离进程；旧源码、权重、helper、AWQ内核、hybrid文件与既有环境均未修改。

## v2：真实 overlay 键契约修正（2026-10-09）

主任务 L20 独立 CPU 执行报告：v1 的15个合成 tests 退出0，但 `packed_loader_headers_v1` 退出1，`GateError: overlay 不得覆盖任何保护键`。原失败目录取回为 `docs/baselines/evidence/jetson_300ms_20261009/packed_loader_headers_v1/`，保持原样。主任务实际 `mlp_00` 连续两次复现失败；另实际 `mlp_29.safetensors` header 确認三键为 `gate_proj.weight/up_proj.weight/down_proj.weight`，BF16、gate/up `[5376,4096]` / down `[4096,5376]`。原 SHA 验证通过，错误是本执行者把 tensor key 错当为去后缀 module name。

修正仅将原 inline 验证提取为 `_validate_overlay_header`，精确要求上述三个 `*.weight` 键，down 分支精确匹配 `down_proj.weight`；不接受无后缀键、额外保护键、缺项、错误shape/dtype，不放宽任何来源/SHA/路径门禁。小 metadata fixture 先在原错误逻辑下运行：16项测试，1 error，退出1，栈精确落于 `_validate_overlay_header` 原集合判断，`GateError: overlay 不得覆盖任何保护键`。修正后同命令16项全通过，`Ran 16 tests in 0.859s / OK`、退出0；py_compile退出0。fixture 不分配真实大张量，也不读取原模型payload。

同时只读核实冻结 AWQ result 252/252 rows 的 file basename 完全等于 `name.replace('.', '_') + '.safetensors'`；现有代码的显式映射无需修改。AWQ result 的 code_sha256 与 `scripts/alp15_awq_full.py:180,206` 指向 **`/home/USER/lang/Lighten_VLA/compression/run_mlp_experiment.py`**，SHA=`15402eff7bf9f161758f7472e3868673a7264d68b263fdeb26ff5d30984a6833`；主任务已实际确认服务器该位置。它不是 root/scripts 文件。当前 PhaseA loader 不导入 run_mlp_experiment；仅来源冻结、无CUDA的 `test_alpamayo15_offline_compat.py` 位于 root/scripts 并固定 `1c769df0...`。未猜测 run_mlp_experiment 的路径或把它替换为此 helper。

新 v2 代码冻结：

| 文件 | SHA256 |
| --- | --- |
| alp15_packed_loader.py | `d00e4a6f586ea6d8f300601d2ffd2b8c93fcc5fd5941c08b8e9f3dab60066db9` |
| probe_alp15_packed_loader.py（未改） | `107c761a0974529bfad4971775177711e750af717ef8efba4fbd7cc0e15107e2` |
| run_alp15_packed_loader_eval.py（未改） | `68802e4c479bb0ed234d47528dec9f393cb1ff04d8a004ad11f7b220d404af76` |
| test_alp15_packed_loader.py | `a62d70e99e6aba9aab02fd367e6eb7e27b61d8429ccbf3d64350d2db4f9e6a18` |

主任务应冻结 `packed_loader_code_v2`、新 `packed_loader_headers_v2`，headers通过后再统一实际meta试验。v2没有运行服务器或GPU；真实meta仍未执行、实体阶段仍硬拒绝。

## v3：拦截器 callable identity 修复（2026-10-09）

主任务 L20 v2 headers 已通过：退出0，1159/907/252/1663，planSHA=`3dcc7b4a7c0368930de87a45879a0ebf7c1fc619edb5dffc6e92b7653aa68bfc`；服务器16项tests通过。真实 `packed_loader_meta_v2` 退出1，导入链 `transformers→torchvision→torch._dynamo` 在规则表构造时报告 `Duplicate torch object ...blocked... with different rules: TorchInGraphFunctionVariable/SkipFunctionVariable`。原证据在 `docs/baselines/evidence/jetson_300ms_20261009/packed_loader_meta_v2/`，完整栈已只读复核，失败前RSS peak=`635305984B`、default dtype=`torch.float32`、原payload=`0B`、禁用依赖import集合为空。此失败来自本执行者共享拦截函数对象，是拦截器自身错误，尚未进入真实模型构造；不能据此判定原constructor失败。

先加入Dynamo式rule map fixture及全部15个受保护入口identity/拒绝/异常后恢复的回归。旧代码下18项tests，2 failures、退出1：rule map复现不同规则绑定同一callable冲突，全部入口distinct检查为 `1 != 15`。修正为 `make_blocked(entry_name)` 为每个 torch.load/CUDA API/Tensor.cuda/Module.cuda 单独生成闭包对象和入口名称；每个闭包仍立即 GateError，既不调用原API也不返回伪CUDA检测结果。未预先import CUDA检查、Dynamo或Triton绕过问题；不改变payload文件guard/禁依赖import/网络/来源/实体阶段门禁。

修复后同命令18项全通过，`Ran 18 tests in 1.015s / OK`，退出0；py_compile退出0。新增fixture仅模拟object rule-map identity，不是宣称服务器Dynamo或真实meta构造已通过；由主任务冻结新v3后重新跑真实meta验证。

| v3 文件 | SHA256 |
| --- | --- |
| alp15_packed_loader.py | `5654934fa02c8b95b44c5948a2550b9827666e319f8c50ec4e7b59e29e1f58a3` |
| probe_alp15_packed_loader.py（未改） | `107c761a0974529bfad4971775177711e750af717ef8efba4fbd7cc0e15107e2` |
| run_alp15_packed_loader_eval.py（未改） | `68802e4c479bb0ed234d47528dec9f393cb1ff04d8a004ad11f7b220d404af76` |
| test_alp15_packed_loader.py | `c7798d3fdf576af42a64a138225a70d6ca53289b1b30e181c1e422f1368b6aef` |

v1/v2失败证据保留；本执行者本轮仍未连接服务器/运行GPU/安装依赖。PhaseB、完整meta验收及原flash2/CUDA检查的真实结果仍待主任务。

## v4：正式批准的真实能力查询与分配前门禁（2026-10-09）

完整读取 `ALP15-PACKED-LOADER-CPU-META-ADAPTATION-20261009.md`，对应批准决策 `D-ALP15-PACKED-META-CAPABILITY-20261009`；另使用 `/Users/USER/.agents/skills/diagnosing-bugs/SKILL.md` 的先失败回归流程。v3实测失败根因已由主任务证实为HF导入期能力查询被拒绝，原证据/新决策冻结的SHA未覆盖；不是AutoProcessor缺依赖。v4不修改原flash2、Expert官方SDPA、HF/NVIDIA源、config/helper、hybrid或环境。

修改范围：

- 在任何Transformers/flash-attn导入前，按实际package spec位置核验决策冻结的4个HF文件SHA；只接受3个函数调用点，实际caller的绝对路径、文件SHA、函数名必须全符。保存的原 `torch.cuda.is_available` 真实delegate一次，原bool/异常不被替换；unknown caller拒绝。每次query记录序号/phase/caller/result；环境CUDA_VISIBLE_DEVICES/NVML选项保持原值。
- 真实原query内部thread-local深度范围允许原device_count的直接NVML分支及原 `_C._cuda_getDeviceCount` Driver discovery，外部device_count/primitive仍拒绝。终于清零depth、恢复exact callable identity。其它CUDAAPI、Tensor/Module.cuda、PyTorch初始化、相关直接 `_C` 初始化/执行入口、torch.load/payload/import/network仍硬拒绝且各自独立callable identity。
- 生产probe预检Torch2.8.0与其真实cuda Python文件、原function名/路径，并冻结原两个函数源码SHA：is_available=`b91a896857c103611c58b84e1455f789393531efa0c56cf7971e931b99fdffce`，device_count=`16443ec7027f24db4f8c744133c12ee22d148ceeeb29b55d479f6b095b72e3b3`。这两项由新决策读过的本机Torch2.8.0函数原source得到；实际服务器源码若不相同即停止升级，不凭版本绕过。运行时留实际完整文件hash/function source与调用分支记录；未在本执行者任务中查询实际GPU。
- constructor和官方小状态恢复内使用TorchDispatchMode，执行前拒绝CUDA目标/输入，通过meta输出形状推演拒绝CPU单tensor>1MiB；不能meta推演的未知CPU算子停止升级。tensor/as_tensor/from_numpy额外前置check，防止CPU ingress先于lift_fresh分配；显式保持constructor的meta默认device及small restore的CPU默认device，退出还原factory。constructor运行中RSS增量>256MiB停止，原阶段最终预算继续核对。
- 记录outer/VLM/text/vision原flash2与Expert原SDPA，以及实际attention类。原907/252/1663、652B小状态、ordinary/alias/tokenizer/generation门禁保持。
- 删除旧 `cuda_api_called=False`，改为真实query/nested primitive日志、`cuda_driver_discovery_may_initialize=true`、cuda_execution_attempts、blocked_entry_attempts、Torch初始化前后、querycontext/guard/factory恢复表。`cuda_tensor_allocation_observed=null`明确表示native/PID证据尚未证明；不能用Python标志或dispatch结果冒称零GPU活动。probe在stdout启动即打印`PACKED_LOADER_PHASEA_PID`，并在result写pid，供主任务独立采集。
- 只读记录实际已导入flash-attn Python初始化AST的module级calls/imports、extension模块路径/文件SHA、distribution版本/安装位置；这不是native隐式分配/launch证明。若CPU/meta全部检查完成而主任务native/PID补验证据仍缺失，明确GateError停止升级，保留 `MetaAuditCandidate.json`（状态`cpu_meta_checks_passed_native_evidence_pending`），不生成passed MetaAudit、不启动实体阶段。

先红：在旧硬拒绝逻辑下新增“批准caller真实delegate bool一次”fixture，19项tests中1error、退出1，准确复现 `PhaseA 禁止 ... torch.cuda.is_available`。修复后最终29项tests全过，`Ran 29 tests in 1.394s / OK`，退出0；py_compile退出0，CLI help退出0；import loader仍stdlib-only（torch/transformers/triton/modelopt未导入）。负例覆盖unknown caller不delegate、False/True原样、query异常状态恢复、外部device_count/primitive拒绝、只在原query内delegate、初始/查询触发初始化拒绝、CUDAfactory/.to拒绝、直接native CUDA init拒绝、CPU>1MiB拒绝、meta大tensor允许、patched factory仍保持meta默认device、所有旧source/key/alias等回归。

fixture的mock bool/count/native primitive只验证机械契约，未用于实际硬件报告。没有服务器连接、GPU运行、包安装或额外联网。实际v4源码/HF/Torch查询/PID/native观察与完整constructor结果由主任务统一运行；native充分证据或未知调用点未解决前，不能宣称完整meta通过。

| v4 文件 | SHA256 |
| --- | --- |
| alp15_packed_loader.py | `0bfca8ba9fa511f639cca1215b16e31ad7cf176386a474dfe1292bbd6bcc9619` |
| probe_alp15_packed_loader.py | `747d24900778a6342e37a8d6cb7684b7c9e41a1f180a7ad0bbe977bb54e94b16` |
| run_alp15_packed_loader_eval.py（未改） | `68802e4c479bb0ed234d47528dec9f393cb1ff04d8a004ad11f7b220d404af76` |
| test_alp15_packed_loader.py | `1bc1c75bd443fd66452e32c0ba99215e867aae12c0049177242fe87ea0f55d36` |

主任务采用独立 `packed_loader_code_v4`/新 `packed_loader_meta_v4`，不覆写v1-v3。原PhaseB所有实体API及评测继续硬拒绝。

最终冻结前另强化：列表/ndarray即使以meta为目标，CPU ingress仍受1MiB上限；原已为meta的Tensor可保持大meta形状。最终29tests再次退出0（1.356s），compile退出0。上表SHA为本次最终文件，不是此前中间版本。
