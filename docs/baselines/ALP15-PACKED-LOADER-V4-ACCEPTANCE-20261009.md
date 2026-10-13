# packed-only loader v4 独立实施准入审查

日期：2026-10-09，Asia/Shanghai。验收者及本文唯一写入者：GPT-6.1 Sol high / `hybrid_m1_acceptance_20261008`。对应决策：`D-ALP15-PACKED-META-CAPABILITY-20261009`，承接 `D-ALP15-PACKED-LOADER-20261008`。审查对象是冻结实施物，不是 Git 分支比较，也不冒称运行了 code-review 技能流程。风险高。

## 结论

**当前冻结 v4 不准进入新的真实 `packed_loader_code_v4` / `packed_loader_meta_v4` CPU/meta诊断。** 独立复跑29项CPU合成测试全部通过，但分配前CPU factory门禁存在已复现的字节低估缺口；须按原决策修复、补失败fixture、重新冻结后再做准入复核。本文没有修改代码，不授权执行者自行改变后端或资源成功定义。

真实查询的窄放行、原flash2/Expert SDPA保持、异常恢复和native证据缺失时的pending状态总体符合决策。这些正确部分不抵消下面的分配前门禁缺口。当前没有真实v4 HF导入/原构造器结果，没有新的CUDA、Orin或实体加载通过结论。

## 冻结身份与读取范围

完整读取仓库 `AGENTS.md`、原packed-only决策、能力查询适配决策、执行回报全部v1-v4段落，以及下面四个完整文件；未调用superpowers。审查前后核验SHA一致：

| 文件 | 本次实际 SHA256 |
| --- | --- |
| `scripts/alp15_packed_loader.py` | `0bfca8ba9fa511f639cca1215b16e31ad7cf176386a474dfe1292bbd6bcc9619` |
| `scripts/probe_alp15_packed_loader.py` | `747d24900778a6342e37a8d6cb7684b7c9e41a1f180a7ad0bbe977bb54e94b16` |
| `tests/test_alp15_packed_loader.py` | `1bc1c75bd443fd66452e32c0ba99215e867aae12c0049177242fe87ea0f55d36` |
| `scripts/run_alp15_packed_loader_eval.py` | `68802e4c479bb0ed234d47528dec9f393cb1ff04d8a004ad11f7b220d404af76` |
| 能力查询适配决策 | `741afd758b75c80fecf29b80e3c858218efa583a2b806335aa40afd86a52102f` |
| 执行回报 | `f1980c95b53815c3d0b8ab285a3f20f2dbfce6413e6769c2b57aafb1e4780e06` |

v3的实际失败仍是HF导入期 `integrations.flash_attention → flash_attn_supports_top_left_mask → is_flash_attn_2_available → torch.cuda.is_available` 被拒绝；不应解释成AutoProcessor缺依赖。v3未进入原构造器；header-plan的1159/907/252/1663不等于meta成功。本轮沿用决策中已冻结的一手源码证据，未新增联网、模型或服务器执行。

## 阻断项与必须补齐的边界

### P1：factory入口未按真实输入字节在delegate前拒绝

`scripts/alp15_packed_loader.py:690-698` 对无显式dtype的输入使用固定 `item_bytes=8`。实际 `numpy.ndarray` 可为complex128（16B/元素），Torch Tensor也可有16B元素；嵌套容器还不能仅凭顶层flatten叶数完整描述内部数组字节。

本次仅用小合成ndarray和被替换的原factory spy，不调用原Torch factory，也不构造真实大模型或CUDA tensor：`np.empty(70000, dtype=np.complex128)` 的实际 `nbytes=1120000 > 1048576`；分别在进入 `_allocation_guard` 前以spy替换原 `torch.tensor/as_tensor/from_numpy`，然后调用guard后的同名入口。三者均到达spy，输出如下：

```text
tensor delegated [{'nbytes': 1120000, 'dtype': 'complex128', 'device': 'cpu'}]
as_tensor delegated [{'nbytes': 1120000, 'dtype': 'complex128', 'device': 'cpu'}]
from_numpy delegated [{'nbytes': 1120000, 'dtype': 'complex128', 'device': 'None'}]
```

这证明前置检查已放行超过预算的CPU实体输入。`tensor` 的真实delegate可能先建立CPU storage再触发lift/dispatch；后续dispatch拒绝不能补成“分配前拒绝”。`from_numpy/as_tensor` 的共享storage情形也不能使大实体输入白名单成立。该负例不声称实际发生了GPU活动或真实大权重分配。

最小修复应在delegate前对Tensor使用实际 `numel * element_size`，对支持的ndarray使用实际dtype/itemsize/shape或nbytes，分别核验CPU ingress与转换后的目标dtype尺寸；嵌套输入有保守、受限的完整核算。未知类型、无法确定字节或超预算立即GateError，不调用原factory。不能把固定8B改成另一个未经证明的通用常数，也不能靠调大1MiB门槛消除负例。保留已有大meta Tensor允许、meta默认device、exact factory identity恢复。加入上述无dtype/16B元素、嵌套数组及显式转换dtype的before-delegate负例，并证明原factory调用次数为0。

### P2：冷进程前提仅留痕，尚未硬拒绝预先导入

`scripts/alp15_packed_loader.py:498` 记录 `related_modules_before`，但未检查为空。合成向 `sys.modules['transformers']` 放入ModuleType后，`_meta_guards` 仍成功进入，证据清单为 `['transformers']`，退出恢复为True；未实际导入HF或执行模型。

独立新probe的正常入口目前是stdlib-only，故没有证据表明正常命令已经预导入HF。然而决定要求首次HF/flash-attn导入前guard生效，不能将带有无guard缓存/原API别名的进程当作冷诊断。新冻结版本应在生产PhaseA入口保存并检查初始相关模块清单，预先存在Transformers/Alpamayo/flash-attn/ModelOpt/Triton等相关链即停止并保留证据，不清空modules伪装冷进程；新增负例。此项是补齐已批准的启动前提，不批准预热或改变backend。

## 已符合的部分及其证明范围

- HF实际package spec路径与四个完整源码SHA先于HF import核验，三个批准callsite以resolve路径、文件SHA、函数名匹配；Torch2.8原查询函数source也核验，不凭版本号略过。保留原is_available真实返回值/异常，未知caller不delegate；nested device_count/Driver primitive限制于原query内部，thread-local finally清零。测试的mock bool/count仅机械验证，不是实际硬件事实。
- 网络、权重文件、torch.load、实体from_pretrained、ModelOpt/Triton、CUDA初始化/执行入口等没有全局放开；callable identity独立并exact恢复。默认/隐式CPU empty与zeros超过1MiB的额外负例实测在执行前GateError；大meta形状仍允许。它们不能替代上面的factory ingress负例。
- 原1159键、保护907、packed252、target1663、shape/dtype/param-or-buffer、持久对象alias、652B已列小buffer、普通trajectory ActionSpace属性、完整tokenizer4000 IDs/序列化、processor/generation和来源门禁仍存在。占位forward拒绝，原payload/hash尚未验证的字段不冒称完成。原outer/VLM/text/vision flash2与Expert原SDPA分别检查，不静默SDPA回退。
- 移除 `cuda_api_called=False`；真实query、可能Driver discovery/cuInit、Torch初始化前后、guard恢复表及 `cuda_tensor_allocation_observed=null` 准确区分已知与未知。flash-attn Python AST/extension路径/hash不是native无GPU活动证明。
- `build_meta_skeleton` 在完整CPU/meta检查以后仍将native证据置为未证明；触发GateError，并仅保存 `MetaAuditCandidate.json` 状态 `cpu_meta_checks_passed_native_evidence_pending`。probe为failed/exit1、entity_loaded=false，不生成passed MetaAudit；失败栈、PID和阶段证据保留。当前没有接收外部native证据并自动放行的实施口，所以不能期望本版获得完整meta_passed。
- PhaseB实体load、target readback及eval（包括默认bf16_reference）继续硬拒绝；没有把合成audit当实体授权。

## 本轮独立验证留痕

环境：Mac，`/usr/bin/python3` Python3.9.6，Torch2.8.0 CPU；GPU/驱动/CUDA实机不适用。未SSH、未运行真实模型/服务器/GPU、未安装或下载、未改环境。代码、执行回报、决策只读；本文是唯一新增交付文档。

```sh
/usr/bin/python3 -m unittest discover -s tests -p test_alp15_packed_loader.py -v
```

实际 `Ran 29 tests in 1.340s / OK`、退出0。完整输出在本验收任务工具记录 `9553a8`。另执行三段内联CPU合成诊断：隐式CPU allocation预算拒绝及factory恢复（`158f8a`，退出0）；complex128 factory spy负例（`9da767`，退出0）；预导入模块启动边界（`0790d9`，退出0）。SHA与对应行号证据 `a303b3/51364d/1b81ea`。退出0只是诊断脚本正常完成，**不是其被审查安全条件通过**。

## 重新准入与后续授权

修复以上门禁并重新冻结之后，Sol先复核新代码/失败fixtures；只有准入明确批准，主任务才可在既有L20环境分配新的独立CPU/meta诊断目录，保留旧v1-v4证据，不并发真实GPU主任务。不得通过先import、mock可用性、临时eager/SDPA、卸掉guard或更改依赖来绕过失败。

实际query可触发已批准的Driver discovery/cuInit，必须如实留痕，不再称“绝无CUDA调用”；模型CUDA张量分配/内核仍不允许。native/PID证据未充分证明时，预期结果应是上述pending GateError而不是完整通过。CPU/meta候选产物和native独立审计须另行复核，目标flash2真实执行更是独立补验；本报告不批准PhaseB、桥接/质量评测、Orin实机部署或任何服务器/GPU写入。

Orin16GB、完整请求max≤300ms及teacher meanADE损失≤15%的联合成功定义不变；不得用29CPU tests、header集合、meta候选或metadata query True替代这些门禁。回滚/停止仅不启动新探针或退出独立PhaseA进程，不操作既有hybrid/reference/GPU PID。
