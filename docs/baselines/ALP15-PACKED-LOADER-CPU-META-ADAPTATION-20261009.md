# packed-only PhaseA：真实能力查询与meta构造适配决策

决策编号：`D-ALP15-PACKED-META-CAPABILITY-20261009`。日期：2026-10-09，Asia/Shanghai。决策者及本文唯一写入者：GPT-6.1 Sol high / `hybrid_m1_acceptance_20261008`。风险：高。承接 `D-ALP15-PACKED-LOADER-20261008`；仅调整PhaseA的能力查询边界，不批准实体加载、性能评测或改变最终后端。本轮未运行服务器/GPU；执行者待主任务分配，GPU仍由主任务统一串行。

## 决定

**批准在独立PhaseA进程、全部原guard已经生效的范围内，向冻结HF调用点放行原 `torch.cuda.is_available()` 的真实只读查询，并记录它及其底层Driver/NVML查询语义；不修改其返回值、不改变模型flash2配置。** CUDA tensor分配、内核、PyTorch CUDA初始化、完整权重入口和原payload访问仍禁止。这里的“只读”是没有模型权重/张量分配或推理，不是“完全不调用CUDA Driver”：PyTorch默认查询可能经过 `cudaGetDeviceCount→cuInit`。此行为是本决策对原“任何CUDA API均拒绝”的唯一明确例外，不能再写 `cuda_api_called=False`。

首选保持原构造链与后端配置的方案，先解决已证实的import失败，再由真正meta构造捕捉后续阻塞。**拒绝本轮将构造临时改eager/SDPA、伪造available=True/False、跳过HF后端检查或修改HF/NVIDIA源文件。** 如果实际查询返回False或后续flash2导入/构造要求被禁止的GPU行为，保留失败升级；不为跑通meta自动更换后端。

本决策批准的是精确实现与一次新的真实CPU/meta探针，不是 `meta_passed` 结论。原907/252/1663、来源/config/tokenizer、652B小状态、alias/dtype、原payload0、RSS预算、hash和严格轨迹门禁全部保留。完整请求max≤300ms、teacher meanADE损失≤15%和Orin16GB联合门禁不变。

## 实际失败与证据身份

已完整读原packed-only决策、v3的63行traceback与70行result。下列为本轮实际SHA：

| 证据 | SHA256 |
| --- | --- |
| ALP15-PACKED-ONLY-LOADER-DECISION-20261008.md | 4b5d6e559e54a3793d121d6099898a3007c5d20a4c3e9281ce252401b9b97bb5 |
| evidence/jetson_300ms_20261009/packed_loader_meta_v3/traceback.txt | 53ba16588663a689989752a7391ae09d1fa0e24d2cb533140a6f9ea0aef66ad4 |
| 同目录result.json | 56bac1a33d0a77a6cb8bbbbc5a935256d4f5fc1274cdbb5a710b82d66f0a0cfc |
| scripts/alp15_packed_loader.py（v3） | 5654934fa02c8b95b44c5948a2550b9827666e319f8c50ec4e7b59e29e1f58a3 |
| scripts/probe_alp15_packed_loader.py（v3） | 107c761a0974529bfad4971775177711e750af717ef8efba4fbd7cc0e15107e2 |
| tests/test_alp15_packed_loader.py（v3） | c7798d3fdf576af42a64a138225a70d6ca53289b1b30e181c1e422f1368b6aef |

v3不是旧Dynamo callable身份冲突：实际HF导入已推进到 `processing_auto→processing_utils→modeling_utils→integrations.flash_attention`。后者顶层计算 `_use_top_left_mask`，调用 `flash_attn_supports_top_left_mask→is_flash_attn_2_available→torch.cuda.is_available`，被现有guard硬拒绝；最外层包装成AutoProcessor ModuleNotFoundError。根因不应误诊为缺AutoProcessor或要求安装依赖。

result明确status failed/exit1、原payload_read_bytes=0、entity_loaded=False；1159/907/252/1663是已完成header-plan计数，**不是meta构造成功**。torch2.8.0+cu128、Transformers4.57.1、accelerate1.12.0、safetensors0.7.0路径/hash已记录，defaultdtype在失败处FP32，RSS峰值640540672B，forbidden_modules_present为空。失败发生在 `Alpamayo1_5(config)` 之前；本记录没有新的实体CUDA张量或内核证据，不能把失败日志描述成模型曾完成构造。也不把已imported torch等同于独立证实零Driver活动。

## 一手源码依据与方案比较

本轮按web-access读取固定官方源码；CDP前置检查Node22.23.0通过、Chrome未连通，故使用技能允许的公开静态curl，无浏览器/安装/大资产下载。HF固定commit仍为 `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc`：

- [integrations/flash_attention.py:11](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/integrations/flash_attention.py#L11)在模块导入时查询top-left mask，不是forward。
- [modeling_flash_attention_utils.py:35-42](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/modeling_flash_attention_utils.py#L35)和[import_utils.py:1189-1215](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/utils/import_utils.py#L1189)使用真实package/version及CUDA可用性；伪造False还会改变导入期缓存的mask判断。
- [modeling_utils.py:_flash_attn_2_can_dispatch](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/modeling_utils.py#L2373)核查模型支持、package/version、dtype和实际设备；[后端调整与preload](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/modeling_utils.py#L2665)可导入flash-attn或尝试kernels-community路径，不能全方法mock为成功或允许网络kernel回退。
- [PyTorch v2.8.0 torch.cuda.is_available](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L163)默认使用CUDA Runtime查询并可能cuInit；NVML路径若失败也可能回落Runtime。设置 `PYTORCH_NVML_BASED_CUDA_CHECK=1` 不构成零Driver/零回退保证，本轮不靠它规避guard。已独立只读本机Torch2.8函数源码，与该分支一致；服务器实际源码仍由执行者核对。
- 实际NVIDIA现场快照 `base_model.py:227-235,382-392`将原config的flash_attention_2传入Qwen骨架；`alpamayo1_5.py:108-110`明确把Expert从flash2改SDPA。这是原有Expert语义，**不应在“保留flash2”检查中误要求Expert也flash2**。应保持外层/VLM/视觉与语言子config的原flash2以及Expert原SDPA，分别记录。

本次公开HF完整文件SHA（执行者必须在导入前以实际site-packages路径比对；公开值不是已核实的服务器文件身份）：import_utils `66b2ac40be2c538dbd9409370c1f6bf769934b8b232edac365e16299c5ec1108`；modeling_utils `bf1c6b2a43cf7c36fb79f37c981424dd6ae78eb863fcaa5d2a37e76c9828611d`；modeling_flash_attention_utils `293fe81c6bd38aac8a3bc6ae086ff21b9695b349e97c9650b1fe61e42a36d710`；integrations/flash_attention `850aa63f9473391188c357454cfb6a89339394d26f6024ccbe246f00f6344144`。Qwen modeling既有冻结SHA `dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da`继续核验。任一不同停止，不能仅凭版本号略过。

| 方案 | 优点与关键风险 | 本次决定 |
| --- | --- | --- |
| 精确放行原真实is_available查询 | 原flash2检查和mask语义保持；可能cuInit，需显式留痕及禁止分配/执行 | 批准下述窄上下文 |
| 构造期eager，之后重新设置flash2 | 可把骨架结构与目标设备可用性分开；仍不能解决HF模块导入期查询，改变NVIDIA Expert分支/多层config，恢复不能仅改一个字符串，实际HFset_attn_implementation还会重做能力校验 | 本轮不实施；若首选被实际证据阻塞，另行决策并单独审计与真实flash2复验 |
| 把available或can_dispatch固定返回True/False、跳过检查、禁用全部guard | 冒充硬件能力或改变导入缓存/后端，可能放出CUDA或权重加载 | 拒绝 |

## 精确实现单：guard、上下文和恢复

1. **独立进程和先后顺序。** 主任务分配新 `packed_loader_code_v4`/新输出目录；继承已验证来源、LoaderPlan和既有venv，不覆写v1-v3证据或旧流程文件。HF离线环境在首次HF import前设置；网络guard、payload guard、CUDA执行guard在首次Transformers/Alpamayo/flash-attn import前建立。禁止先无guard导入HF“预热”再恢复guard。仅标准库、原只读helper和Torch本身的既有导入可在设置guard前用于保存函数对象；保存Torch初始 `is_initialized()`、defaultdtype和已有相关module清单，初始Torch CUDA已初始化即停止。不得通过清空modules重新假装冷进程。
2. **一个允许的public入口。** 将cuda.is_available由硬拒绝改成独立wrapper：无参数，调用保存的原函数，真实bool原样返回，不catch异常后替换结果。入口仅在PhaseA上下文激活且以下已冻结HF调用点接受：`transformers.utils.import_utils.is_flash_attn_2_available`（本次失败点）；`is_flash_attn_3_available`（同一top-left helper先查FA3的官方路径）；`transformers.modeling_utils.PreTrainedModel._flash_attn_2_can_dispatch`（官方flash2检查的直接query路径）。用实际caller文件resolve路径+SHA+函数名匹配，记录每次调用序号/阶段/调用点/结果；未知调用点立即GateError并保留栈，不扩成整个HF或Torch都可查询。只有本次批准的这一个public查询，不给get_device_properties/capability/current_device开放权限。
3. **查询内部的实际分支。** 原环境的 `CUDA_VISIBLE_DEVICES`、`PYTORCH_NVML_BASED_CUDA_CHECK`、torch.version.cuda/hip、flash-attn安装元数据如实记录，不改它们凑结果。默认分支原 `_C._cuda_getDeviceCount`仅作为original_query内部的真实原语使用，不作为任意外部调用的绕过入口；若环境本来走NVML，允许保存的原device_count仅在查询wrapper的嵌套深度与PyTorch原is_available直接调用点同时匹配时执行，并记录它，外部 `torch.cuda.device_count()`仍硬拒绝。可用线程局部/contextvar标记“正在原查询内部”，finally清零；不得全局把device_count开放。源码/调用路径不同或需要更多CUDA原语则停止升级。不能以假device_count/假capability支持flash2。不能将query的Driver初始化解释成分配模型或通过GPU门禁。
4. **其它原guard不减弱。** 独立callable identity保留，避免v2曾发生的Dynamo规则对象冲突；torch.load、PreTrainedModel.from_pretrained实体权重、所有.safetensors/.bin/.pt/.pth payload入口、ModelOpt/Triton仍拒绝。CUDA init/_lazy_init、synchronize、set_device、get_device_properties/capability、Tensor/Module.cuda及任意.cuda/.to(cuda)/CUDA factory或forward/内核均禁止。不得因为预加载flash-attn就允许调用其函数；发现import-time GPU行为立即停机。网络kernel-community回退由禁网硬停，不准下载/安装补齐。
5. **分配前拦截与资源证明。** 保留 `_lazy_init` 硬拒绝，并使用Torch dispatch/factory检查在实际执行分配前拒绝任何CUDA目标或CUDA tensor运算；禁止仅构造结束后看state_dict再认为未短暂分配。constructor范围CPU tensor单次实体分配>1MiB也硬拒绝，meta不受此尺寸限制；小buffer恢复仅走原652B及已列普通属性官方小类，形状/dtype/path单独白名单。未知较大CPU表/初始化停止升级，不为配额放行权重。RSS增量≤256MiB保留。torch CUDA is_initialized前后/异常路径均应False；不能单靠此Python标志证明native extension没有GPU行为，需只读审计被实际导入flash-attn Python初始化与extension入口、记录版本/路径/文件SHA，并由主任务独立观测该探针PID的GPU进程/内存活动。发现GPU context/内存异常、native import是否分配或launch无法核实，就保留失败/未证明，不称零GPU分配或内核通过。观测不能替代guard，短暂峰值可能漏采也须报告。
6. **不改后端。** config/local memory只沿原来源路由到已核实Cosmos/Qwen；不写attn_implementation/eager/SDPA，也不patchHF的can_dispatch/get_correct_attn_implementation/_check_and_adjust_attn_implementation。真实metadata检查与flash-attn库的导入可发生，但不是forward或设备可用验收。构造采用原default_bf16→torch.device(meta)→no_init_weights，所有大参数meta；small restore/tokenizer/processor完整契约保持原样。输出outer/VLM/text/vision/Expert各自原后端config、实际attention类、source哈希和generation字典；Expert既有SDPA保留。任何未授权后端变化停止。
7. **恢复顺序和异常留痕。** ExitStack/finally恢复全部CUDA/load/import/open对象到exact原callable identity，查询wrapper的状态和defaultdtype无残留；禁止替换失败后保留wrapper或把整个guard提前退出以执行构造。若已导入HF模块缓存 `_use_top_left_mask`，必须真实查询导出而非假值；本进程仅PhaseA、绝不转实体/GPU，退出结束，不拿该进程代替部署runtime。成功和失败都带查询日志、guard恢复表、defaultdtype前后、is_initialized前后、来源、RSS与payload字节计数。将`cuda_api_called=False`改成准确字段：`cuda_capability_queries`、`cuda_driver_discovery_may_initialize=True`、`cuda_tensor_allocation_observed`、`cuda_execution_attempts`及各自证据/未证明项，不用新字段掩盖异常。

文件所有权：主任务将新实施版本 `scripts/alp15_packed_loader.py`、`scripts/probe_alp15_packed_loader.py`、`tests/test_alp15_packed_loader.py`及其执行回报交同一执行者独占；本决策者只写本文。不动hybrid文件、原kernel/helper、官方源码、原config/tokenizer/checkpoint/packed/overlay、venv。实际源码编辑本轮只授权上面最小queryguard/审计/测试变化，PhaseB实体API继续硬拒绝。远端新snapshot与输出由主任务统一安排，此决策不授权执行者自行并发GPU任务。

## 两组独立补验门禁

**CPU/meta门禁（本次可执行）：** 先CPU小fixture验证querywrapper仅真实delegate一次、不改bool、未知caller拒绝、嵌套device_count外部拒绝、异常状态清零/恢复exact identity、所有执行/weight/payload/importguard仍挡住、Dynamo规则callable不同；defaultdtype恢复；查询触发的GPU初始化或分配负例必须失败。fixture的mock值只证明wrapper机械契约，不能写进实际硬件能力报告。随后主任务在既有L20进程安排 `--stage meta` 真构造，新目录，stdout/result/traceback完整保存。验收必须同时满足原1159/907/252/1663集合、meta全部大参数、652B及ordinary小状态、完整tokenizer4000 IDs/序列化/特殊边界、processor/generation/config/aliases/dtype、payload0、CPU预算、guard/源码/环境恢复、真实query留痕、GPU分配和内核无发生的充分证据。真正运行返回meta_passed/exit0后再由Sol独立验收，不能以headers_only或18个旧CPU tests替代。

**目标flash2后端门禁（本次未执行、不得合并计分）：** 原L20可运行历史不等于新loader或Orin目标环境通过。PhaseA通过之后仍须另行批准PhaseB；目标部署全新进程解除PhaseA禁执行guard后，在实际设备真实核验flash-attn/PTX/torch/驱动/架构与BF16支持，无patch可用性、无临时backend，确认VLM原flash2与Expert原SDPA的实际分派。先目标小真实attention/合法head_dim/dtype/mask，再按原决策独立验证907源/target readback、packed252字节/桥接、无残留meta与小状态、全轨迹strict；2clip通过才能20clip。Orin还需实际ARM64/JetPack/SM87依赖、冷峰值、稳定完整请求max300ms、独立同任务meanADE≤1.15teacher、系统可用内存/无swap。CPU-meta成功不批准或声称该门禁通过，metadata query True不是BF16/flash2 kernel可执行证明。

## 停止、回滚与本轮留痕

未知查询caller、源码hash不同、metadata真实不可用、flash2导入触发GPU执行/张量分配/PyTorch `_lazy_init`（本决策明确允许的原query内Driver discovery/cuInit除外）、无法充分审计native import行为、任何原payload/大CPU权重、非meta持久参数、键/shape/dtype/alias/config/tokenizer不符、小状态不符、预算超限、guard恢复失败：立即停止，保留新result/traceback，不回退eager/SDPA、False/True mock或完整CPU模型。回滚只退出新PhaseA进程、撤销新源码版本；不覆盖/删除原资产或失败目录，不停止其它任务。

本轮实际工作仅本地读取既有决策/v3证据/源码/测试和固定官方公开小源码；只读本机Torch2.8 `is_available`函数，没有调用该函数或其它CUDA API。关键输出：v3trace/result完整读取 `4e14f2`；官方HF与本机Torch源码 `c09d88/8edc41/707540/b0e051`；证据/代码SHA与旧18CPU测试日志读取 `263d82`；公开HF文件SHA `c90b4f/eb2e61`。未运行真实meta、服务器命令、GPU、安装或下载大资产。唯一新增文件为本文；当前事实结论仍是v3失败，下一版通过与否待执行证据。
