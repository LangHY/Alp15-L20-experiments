# AWQ M1 混合路径执行回报

对应决策：`D-ALP15-HYBRID-M1-20261008`。执行者：GPT-6.1 Sol high（`hybrid_m1_execution_20261008`），本文件及下列四个新源码文件的唯一写入者。日期：2026-10-08–09，Asia/Shanghai。当前交付为 **v3动态来源生命周期修复**；v1验证时段22:16–22:22 CST，v2验证时段22:30–22:50 CST，v3最终本地回归2026-10-09 12:34–12:35 CST。状态：**本地实现和 CPU 回归通过；执行者未连接服务器、未启动 GPU，v3 GPU/轨迹/Orin 门禁未验收**。

用户后续授权及主任务决策承接历史门禁豁免，不把豁免记为原官方入口通过。v1实施时不修改当时在L20执行的 full20 PID1128980 所用源码或资产。v2开始前主任务通知该full20已exit0、300条、GPU释放；本执行者未连接服务器独立验证该终态，也未启动新capture。目标仍为 Orin NX16GB、完整请求 max≤300ms、教师 meanADE 损失≤15%，FlashDrive/LightVLA 后续联合适配；本次局部实现不能据此声称目标实现。

## v3根因、修复与当前冻结身份

主任务传回的真实v2终态已在本机只读核对：`docs/baselines/evidence/jetson_300ms_20261009/hybrid_reference2_v2/result.json` 为completed/exit0、6条请求、252/252 capture；同级 `hybrid252_v2/result.json` 为failed/exit1、matrices=0，`actual_import_sha256` 中 `/tmp/tmpu5d2gwvq/_remote_module_non_scriptable.py` 已消失。主任务服务器连续两次复现相同FileNotFoundError。v2结果和失败证据不变，不用本次补快照给旧结果背书。

按diagnosing-bugs先建失败fixture，再验证三种假设：进程退出自动清理、生成进程内替换/删除、路径误记。将临时module纳入旧receipt后删除源码，两次复现准确FileNotFoundError（工具输出`b42e2d`、`8dfd8d`，exit1）；初次fixture的macOS路径别名KeyError先用resolve纠正，未跳过来源检查。根因由实际Torch2.8源码而非推测证实：官方 [instantiator.py](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/distributed/nn/jit/instantiator.py) 创建TemporaryDirectory并注册atexit.cleanup，将 [remote_module_template.py](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/distributed/nn/jit/templates/remote_module_template.py) 格式化为原源码再import；生产者退出会清除原临时文件。主任务已只读核实服务器生成器SHA与本机一致。

v3仅接受精确的Torch2.8 non-scriptable remote module，生成进程仍存活时读取实际原字节到 `runtime_source_snapshots/<module>-<source SHA>.py`（排他创建、已有文件损坏拒绝覆盖）。保留原module/spec路径、生成器临时目录、实际生成器与模板路径/hash、版本/入口/生成模式，以及不含随机路径的稳定身份SHA。生产者终态仍要求原文件存在且hash正确；消费者可在原文件已清理后，核验完整映射、持久快照及实际生成器库存。未知/tmp来源仍按实际原路径逐一核验，丢失即失败；不重建丢失源码、不泛化/tmp豁免。跨进程reference/hybrid必须具有同一稳定身份。生成器/模板/动态源码固定如下：

```text
torch.distributed.nn.jit.instantiator                     567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615
torch.distributed.nn.jit.templates.remote_module_template 0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc
_remote_module_non_scriptable                            8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8
```

模板及动态字节hash在本机实际Torch2.8小源码实例化中核验；动态SHA与服务器v2实际库存完全相同。服务器模板字节需在v3运行时按此库存核验，不同则停止，不换模板放宽。严格全import检查还识别Torch的两个已核实虚拟namespace `torch.ops`/`torch.classes`：它们的 `__file__` 是 `_ops.py`/`_classes.py` 标记而非真实路径；精确module类身份绑定实际 `torch._ops`/`torch._classes` 实现文件及SHA，记录独立virtual映射，不忽略文件不存在。第一次严格化回归因此出现5 failures/1 error（37项，工具输出`6ceca1`），随后只修此明确身份映射。没有通配目录过滤。

v3 manifest 为 `alp15-m1-activations-v3`，完整runtime/import库存和动态/virtual映射贯穿reference成功终态、manifest、独立receipt、252probe及正式hybrid绑定。保留v2无循环终态链及其所有失败/strict门禁；旧schema2拒绝，必须新reference。最终39项回归含真实Torch生成器子进程退出、两个不同临时目录稳定身份等价、完整252合成capture/probe终态绑定、快照破坏/缺映射/生成器漂移/伪module拒绝、生产者原文件缺失拒绝、未知tmp仍失败、旧v2拒绝；未加载真实模型/packed权重，未初始化RPC或CUDA。

| 文件 | v3 SHA256 |
| --- | --- |
| scripts/alp15_awq_hybrid.py | `026eadf0bb1924e0356993e60172d732423e27264862aa751b792c3b6e791672` |
| scripts/probe_alp15_awq_hybrid.py | `7a7e8d8b2118a5487e17d68596e011da4d1d0f6b6b936fa5e7467d2a2f09b197` |
| scripts/run_alp15_awq_hybrid_eval.py | `fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd` |
| tests/test_alp15_awq_hybrid.py | `d22b76035b887f04024e405afea2ade585f5027b41d1e0be3d3c966f8ef56dd8` |

最终命令 `/usr/bin/python3 -m unittest discover -s tests -p test_alp15_awq_hybrid.py -v`，exit0，39项全通过，11.150s，完整结果工具输出`dde3d1`/`2b1c93`；四文件py_compile exit0（`bd0332`），两个CLI --help exit0（`221ee5`/`a8c082`），四新源码hash见`7ae2a4`，四旧只读helper仍与决策SHA一致（`5663a0`）。先前37项全绿输出`ed50fe`/`0ebdbb`保留，但最终以39项为准。本回报自身hash在写完后另报，不写自引用SHA。

CLI相对v2无新增/删除参数；主任务须上传全新code_v3目录，使用全新reference2_v3→hybrid252_v3→smoke2_v3。产物增加runtime_source_snapshots目录及结果/manifest的source_snapshot_directory、dynamic_import_sources、virtual_import_sources（provenance_version=3）。必须整体保留快照与receipt，不把原临时路径补造回v2。CPU源码生命周期回归不等于CUDA数值、完整请求质量/性能或Orin验收。

## v2验收修复与历史冻结身份（保留验收基点）

已完整读取 `ALP15-HYBRID-M1-ACCEPTANCE-20261008.md`，按P1来源绑定和P2短轨迹门禁修复；旧kernel、KV、packed只读值不变。以下v2 SHA仅是历史基点，不再代表当前源码：

| 文件 | v2 SHA256 |
| --- | --- |
| scripts/alp15_awq_hybrid.py | `801c7f6cd1b314b73a882071480d617e444ef3ba7693e1c10a0ec53c52df270d` |
| scripts/probe_alp15_awq_hybrid.py | `0cd19e514c3b10988bc39061b2529424823c0c00ba0097123d7ae99f2d489075` |
| scripts/run_alp15_awq_hybrid_eval.py | `a2ccec42a954be1d8325c89afbc32337193808265121448ec770ec92a792e887` |
| tests/test_alp15_awq_hybrid.py | `e181dd82532bf06ec50cb2c1e8c25c0c8ea65652cba13e5542a1bc0652eaeb85` |

终态绑定不循环：真实生成只保存252个tensor和内存描述；最终保护模块/代码/runtime/import/input审计通过后，写v2 capture manifest（完整来源库存、clip/text seed、保护身份及result/receipt路径），再保存完成result（manifest SHA、receipt路径），最后独立 `reference_final_receipt.json` 保存result SHA与manifest SHA。result不保存receipt SHA，manifest不保存result SHA。绑定器最后重新验证全来源和完整2clip×3seed终态。绑定失败会原子改result为failed/exit1，失败receipt保留但其result SHA失效，不用completed背书。

probe必须明确 `--paired-reference`，验证成功reference终态exit0、实际交付manifest、独立receipt、唯一完整252名字/文件、完整code/runtime/input/actual-import SHA、clip/text seed和保护身份；任何孤立/失败/部分capture拒绝。正式hybrid候选要求 `actual_activation_coverage=252/252` 且每个真实矩阵都有actual输入测试；random-only仍可诊断，终态单独为 `passed_all252_random_only_cuda_diagnostic`，不可通过正式候选门禁。带真实输入的终态为 `passed_all252_cuda_m1_probe_with_actual`。

strict门禁在新helper自身检查双方64×3、所有坐标/ADE/FDE有限、合法noise/prefix SHA、token/future-start边界、decision/10step/clip/seed身份，并严格比较双方。smoke扩评不只读旧passed标志，还重新读取绑定的reference和probe终态、逐条重新算strict门禁。旧actualKV豁免仍不可用。

v2最终CPU命令与结果：`/usr/bin/python3 -m unittest discover -s tests -p 'test_alp15_awq_hybrid.py' -v`，exit0，30项全通过，2.885s；同一工具输出`3e8862`包含完整30项结果、编译exit0和四文件v2 SHA。CLI帮助/编译exit0见`44f0fa`。独立新增失败fixture覆盖：孤立capture、failed reference（SHA仍一致）、runtime漂移、actual-import漂移、runtime声明不等、251覆盖、重复文件冒充另一层、错paired reference、错probe绑定、random-only准入、1point、NaN、伪passed短轨迹，以及绑定阶段来源失败后撤销completed状态。均为合成CPU fixture，不是GPU证据。

v1的实际失败和修复留痕保留如下；v1/v2旧SHA/测试数不再代表当前版本，不把不同源码capture/probe背书为v3。

## v1历史变更与冻结 hash（保留验收基点）

| 新增文件 | SHA256 | 行为 |
| --- | --- | --- |
| `scripts/alp15_awq_hybrid.py` | `1941d88cf563261a9fcf2b6afc18a0e6df828cd1e5391bb77b6a19dacc7fec6a` | 原 reference 共享包装，仅实际 M1 调 fused；首个 M1 才建 fused；统计成功分派；252 完整预检后安装 |
| `scripts/probe_alp15_awq_hybrid.py` | `2bedc47681a7849ba0d54c53123ce8153d7aae50de090b9359b14503f36e12f9` | 每个真实 packed 文件分别数值/反量化/禁用/M2桥接，记录 first/warm wall/event/peak/dispatch，每层释放 dense |
| `scripts/run_alp15_awq_hybrid_eval.py` | `34f4332b0e0995ea9d26f778e3d49d9e55b102f60097251c88c3a135a13c675e` | 默认 reference；独立完整 rawimages 请求；候选需252探针与同代码 reference；strict smoke 后才扩20 |
| `tests/test_alp15_awq_hybrid.py` | `5bb8838531fd74ed6d2644d77c4a360bb248a93755a99a1cd14db1c1b102c483` | 16项 CPU 合成回归，mock fused 与逐矩阵文件读取；不声称 CUDA 数值通过 |

本回报自身 hash 由主任务在交付后计算，避免写入自引用 hash。无其它任务文件变更；未安装包、未修改 venv、未下载大资产、未连接服务器写入。

以下只读旧文件的结束 SHA 均与决策冻结值一致：

```text
alp15_awq_kernel.py       3f09aea862990733b9bb1db536c450c79fd6e2ac93c7015eed6b1eef4f191025
alp15_awq_full_packed.py  150e1310ad3dbcb9f82966146f9627855e87bd12a29f45f410d0b94f1a96de0a
alp15_packed_w4a16.py     241edf75d838ad7cba1250374119237864780b95ec81cd8555a7d7c6d15fab12
alp15_kv_steps_probe.py   74886a867a8aae480dd848a6d068d2e3f817674d891774bcdabeb57f412292f8
```

新 runner 直接复用上述 SHA 冻结的 KV runner 中 raw prompt、生成参数、官方少步 sample、轨迹指标、保护模块 digest、storage 去重、来源配对工具；不调用其 `main()`，不复制或改造旧主入口。本轮固定 fullprefill10 请求，actualKV 与少步关闭。主任务已确认首轮先隔离检验计算收益；胜出 actualKV/steps 之后仍需独立决策与联合轨迹验收，不能停在此处作为 300ms 部署结论。

## v1环境、命令与结果（历史留痕）

本机为 macOS-27.2-arm64，Python 为 `/usr/bin/python3`（实际 CommandLineTools Python3.9.6），Torch2.8.0；`torch.cuda.is_available()` 为 False。系统 Homebrew Python、项目 `.venv` 及 Codex bundled Python 均无 Torch；主任务提供上述现有解释器。该解释器无 safetensors。本轮只合成 CPU tensor，没有加载真实模型/packed 资产。

初次回归 9项通过、5项错误，错误为 fixture 调用原 `PackedAWQLinear.save` 时 `ModuleNotFoundError: safetensors`。随后 fixture 明确改为合成文件 SHA + mocked 单矩阵读取；第一次改动后 12项通过、2项失败，原因是 macOS `/var` 与 `/private/var` 路径别名，统一 fixture 路径 `resolve()` 后修复。失败保留于本回报，不跳过检查、不安装依赖。生产 installer 与 GPU probe 仍调用原 safetensors loader，真实格式验证未被 mock。

最终命令（仓库 cwd）及终态：

```text
/usr/bin/python3 -m unittest discover -s tests -p 'test_alp15_awq_hybrid.py' -v
exit=0; Ran 16 tests in 1.508s; OK
工具输出 chunk_id=c020a3

/usr/bin/python3 -m py_compile scripts/alp15_awq_hybrid.py scripts/probe_alp15_awq_hybrid.py scripts/run_alp15_awq_hybrid_eval.py tests/test_alp15_awq_hybrid.py
exit=0
工具输出 chunk_id=c020a3（同时含最终runner hash）；其它三文件 hash见b8bb81

/usr/bin/python3 scripts/run_alp15_awq_hybrid_eval.py --help
/usr/bin/python3 scripts/probe_alp15_awq_hybrid.py --help
均 exit=0；工具输出 chunk_id=066166
```

16项最终测试逐项通过：

```text
test_buffer_objects_and_unique_storage_are_shared_after_factory ... ok
test_disabled_and_large_m_never_import_candidate_or_triton ... ok
test_disabled_is_original_bitwise_for_all_m_and_bias ... ok
test_enabled_m1_cpu_refuses_without_fallback ... ok
test_invalid_input_and_reference_are_explicit ... ok
test_m1_validation_does_not_cast_or_make_contiguous ... ok
test_only_actual_m1_uses_lazy_factory ... ok
test_duplicate_missing_extra_shape_and_dtype_are_rejected ... ok
test_installs_exact252_preserving_protected_modules_and_buffers ... ok
test_manifest_must_match_original_artifact ... ok
test_mismatched_last_buffer_leaves_entire_model_unmodified ... ok
test_wrong_asset_hash_and_format_rejected ... ok
test_expansion_requires_new_complete_strict_smoke ... ok
test_same_provenance_rejects_each_changed_source_and_step ... ok
test_source_receipt_rejects_changed_old_kernel ... ok
test_strict_gate_cannot_use_actual_kv_exemption ... ok
```

CPU 模拟 M1/M2/M17/M256/M3086，禁用 bias/无 bias 逐元素一致；验证原 buffer 对象身份、底层 storage 未增加，禁用/大M即使阻断 Triton 与候选模块 import 也可执行。installer 的合成 artifact 只在测试内 patch 原 result SHA，生产代码要求固定原 AWQ result SHA `16aa6b20915bac68597614361e3932c89b955e7c05c024f1f5025b435ddd76cc`。遗漏、重复、保护模块、多层、shape/group128、dtype、文件 SHA、实际 buffer 不符均拒绝；最后一层 buffer 不符时仍未替换任何模块。

## 服务器小文件布局与主任务串行命令

以下是**待执行**命令，执行者未运行。主任务先确认现有 full20 安全终态和无本任务并发 GPU，创建新的独占目录。上传新三个 scripts、一个 tests、一个文档；另外将旧四个冻结只读 helper 按原字节复制到新 snapshot（小源码文件，不复制权重），并核对上述 SHA。复用既有 root/compression 与 venv，其导入实际路径和 hash 由新 runner 记录。

```text
/home/USER/lang/Lighten_VLA/experiments/jetson_300ms_20261008_hybrid_m1_code_v3/
  scripts/alp15_awq_hybrid.py                    新
  scripts/probe_alp15_awq_hybrid.py              新
  scripts/run_alp15_awq_hybrid_eval.py           新
  scripts/alp15_awq_kernel.py                    只读原字节副本
  scripts/alp15_awq_full_packed.py               只读原字节副本
  scripts/alp15_packed_w4a16.py                  只读原字节副本
  scripts/alp15_kv_steps_probe.py                只读原字节副本
  tests/test_alp15_awq_hybrid.py                 新
  docs/ALP15-HYBRID-M1-EXECUTION-20261008.md       新
```

输出目录存在即拒绝。任何失败保留完整 result/traceback、memory_phases 与 stdout log；不得重用目录覆盖失败。首次录制真实 activation 是 reference 正式测量全部结束后额外的未计时自主生成，不影响 reference 的首推/稳态统计。每个 target 只保存首个真实 M1，完整252覆盖要求；v3 capture manifest 包含 packed/code/runtime/input/actual-import/动态源码快照/virtual/保护模块身份及独立成功终态绑定。若没有现成或捕获的真实 activation，probe 可省略 `--activation-manifest`，明确记录 `actual_activation_coverage=0/252`、random only；该结果只供随机输入隔离诊断，代码拒绝将其用于正式hybrid候选。

```bash
HYBRID_ROOT=/home/USER/lang/Lighten_VLA
HYBRID_CODE="$HYBRID_ROOT/experiments/jetson_300ms_20261008_hybrid_m1_code_v3"
HYBRID_PY="$HYBRID_ROOT/alpamayo1.5/.venv/bin/python"
HYBRID_ARTIFACT="$HYBRID_ROOT/experiments/awq_full_20261008_best8_full20_v4"
HYBRID_DECISION=D-ALP15-HYBRID-M1-20261008
HYBRID_REFS2="$HYBRID_ROOT/experiments/jetson_300ms_20261008_hybrid_reference2_v3"
HYBRID_PROBE="$HYBRID_ROOT/experiments/jetson_300ms_20261008_hybrid252_v3"
HYBRID_SMOKE="$HYBRID_ROOT/experiments/jetson_300ms_20261008_hybrid_smoke2_v3"
HYBRID_REFS20="$HYBRID_ROOT/experiments/jetson_300ms_20261008_hybrid_reference20_v3"
HYBRID_FULL20="$HYBRID_ROOT/experiments/jetson_300ms_20261008_hybrid_full20_v3"
export PYTHONPATH="$HYBRID_CODE/scripts:$HYBRID_ROOT/scripts:$HYBRID_ROOT/compression"

"$HYBRID_PY" -m unittest discover -s "$HYBRID_CODE/tests" -p test_alp15_awq_hybrid.py -v

"$HYBRID_PY" -u "$HYBRID_CODE/scripts/run_alp15_awq_hybrid_eval.py" \
  --root "$HYBRID_ROOT" --awq-artifact "$HYBRID_ARTIFACT" --output "$HYBRID_REFS2" \
  --candidate-decision "$HYBRID_DECISION" --awq-forward reference --clips 2 --steps 10 \
  --capture-decode-activations

"$HYBRID_PY" -u "$HYBRID_CODE/scripts/probe_alp15_awq_hybrid.py" \
  --awq-artifact "$HYBRID_ARTIFACT" --output "$HYBRID_PROBE" \
  --candidate-decision "$HYBRID_DECISION" --device cuda:0 --warmup 3 --repeats 10 \
  --paired-reference "$HYBRID_REFS2" \
  --activation-manifest "$HYBRID_REFS2/decode_activations/manifest.json"

"$HYBRID_PY" -u "$HYBRID_CODE/scripts/run_alp15_awq_hybrid_eval.py" \
  --root "$HYBRID_ROOT" --awq-artifact "$HYBRID_ARTIFACT" --output "$HYBRID_SMOKE" \
  --candidate-decision "$HYBRID_DECISION" --awq-forward hybrid_m1 --clips 2 --steps 10 \
  --matrix-probe "$HYBRID_PROBE" --paired-reference "$HYBRID_REFS2"

# 只有上一步 exit=0 且6条strict gate均通过，才执行两组20clip。
"$HYBRID_PY" -u "$HYBRID_CODE/scripts/run_alp15_awq_hybrid_eval.py" \
  --root "$HYBRID_ROOT" --awq-artifact "$HYBRID_ARTIFACT" --output "$HYBRID_REFS20" \
  --candidate-decision "$HYBRID_DECISION" --awq-forward reference --clips 20 --steps 10 \
  --approved-smoke "$HYBRID_SMOKE"

"$HYBRID_PY" -u "$HYBRID_CODE/scripts/run_alp15_awq_hybrid_eval.py" \
  --root "$HYBRID_ROOT" --awq-artifact "$HYBRID_ARTIFACT" --output "$HYBRID_FULL20" \
  --candidate-decision "$HYBRID_DECISION" --awq-forward hybrid_m1 --clips 20 --steps 10 \
  --matrix-probe "$HYBRID_PROBE" --paired-reference "$HYBRID_REFS20" --approved-smoke "$HYBRID_SMOKE"
```

同代码 smoke/expanded 比较 code、官方现场源码、全部原资产和完整20clip cache hash（只在请求选择上取前2或全20），不把旧 code_v1/v2 smoke 强行复用。每条 hybrid 需全部252投影有成功 M1 fused 计数；生成 token IDs、完整 prefix hash/长度、future-start ID、noise 必须一致，ADE差≤1e-4m、NMSE≤1e-3。strict 失败退出2，禁止套用 actualKV 的旧候选豁免。算子失败退出1；252层覆盖不足、NaN/OOM、来源变化或 unsupported activation 均停止。原入口未修改；回滚为新入口 reference 或退出新进程。

## 门禁对照与限制

- 本地 CPU 回归、编译、CLI：v3 39项通过；真实Torch小源码生命周期加合成manifest252边界、storage身份、终态/来源绑定，非真实模型/CUDA验收。
- 原源码/权重无改动：本轮只读旧源码 SHA 已核对；没有连接服务器，不能据此验证服务器正在跑的终态。
- 真实252矩阵及 activation CUDA 数值、BF16逐元素 dequant、first/warm/peak、真实分派：待主任务执行。
- 新2clip×3seed完整请求严格配对与20clip扩评：待执行；本轮没有模型、轨迹或性能数据。
- 基于同clip教师的 meanADE/FDE/clip尾部、全部请求及稳态 mean/p50/p95/p99/max/>300ms：实现统计，尚无实际结果。
- 原 full20 胜出求解步数、actualKV 与 hybrid 联合候选：等待当前 full20 质量结果和独立决策。
- 低峰值部署 loader、静态KV/Expert融合/动作缓存、LightVLA 筛选、Jetson ARM64/BF16依赖、Orin统一内存与300ms联合验收：仍未执行，不能外推L20或CPU结果。

本地相关完整 helper/模型来源文件位于服务器素材目录，未在本机加载；因此真实 loader、generation、官方 Expert 与 MemoryRecorder 的集成行为尚未在本轮运行。任何服务器 integration 前提不成立，保留失败并升级主任务，不自行改依赖、源文件或批准有损路径。
