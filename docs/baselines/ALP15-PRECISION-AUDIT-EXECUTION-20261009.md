# ALP15 已有开发证据精度审计执行回报

- 对应决策：`D-ALP15-PRECISION-VALIDATION-20261009-01`；决策文件 SHA256：`989b64ac1adfa3979556db82d035f71230d91132454b8eb35029eafacf0280db`。
- 执行者：GPT-6.1 Sol high；文件所有者仅此执行者。主任务独立验收尚待完成，本回报不自授最终验收。
- 正式审计开始/结束 UTC：`2026-10-09T15:06:53.312840+00:00` / `2026-10-09T15:06:53.552542+00:00`。
- 实际环境：本机 `macOS-27.2-arm64-arm-64bit-Mach-O`，CPU 架构 `arm`；Python `3.14.6`，Clang `21.0.0`。仅 Python 标准库；未使用GPU、torch、numpy、PT/pickle、SSH、网络、依赖安装或模型导入。驱动/CUDA/GPU资源观测不适用于本机JSON审计；原模型环境保留于输入结果，不宣称本次运行过。
- checkpoint/source/dataset 身份按输入核验：`7aba8293c09993f2e125c6819df05d7fa3e873ea` / `36aeb4c5938cbc2eb2aed33b22434773da4ab639` / `33f9bf447ed3bcb7d545ce13f4226f824214fafb`。

## 变更与冻结

新增 `scripts/audit_alp15_precision.py`（SHA `5fa573c996ffa03e6f4df67c8e511912a1f2d6d8e68de061bf4570501120b51f`）、`tests/test_audit_alp15_precision.py`（SHA `c1b25f4aef2dbd4722b884f56f81c012c6b9f1dcdf4939c132d906cf5fcf5d35`）及本执行回报。既有代码、证据、数据、缓存、配置未改写。正式输出目录 `docs/baselines/evidence/precision_20261009/development_audit_v1/` 此前不存在，由CLI独占创建；已有目录会拒绝覆盖，未提供force开关。

逐份输入解析前、解析后、正式审计后 SHA256 相等，未触发输入变化停止条件：

|输入|三次核验SHA256|
|---|---|
|`precision_20261009/teacher_development_result.json`|`67b328a1ecac9de414b47680ef6476c96f6cb30dab0c736a500a902970cebaa0`|
|`jetson_300ms_20261008/kv_candidate_full20_v1/result.json`|`db04c2ed3226e2c3c988c3f6ebd9218bd6150759d3334d25d7e193c7693ef84e`|
|`jetson_300ms_20261009/hybrid_reference20_v3/result.json`|`e83ef209381c6e166e862795e65fc21681528fe77d90c3bf48bd4d9e048e3b59`|
|`jetson_300ms_20261009/hybrid_full20_v3/result.json`|`a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6`|

输入路径均位于 `docs/baselines/evidence/`。SHA来自实际文件字节，不仅引用已有报告。

## 实际命令、退出码与日志

先运行CPU单元测试，再运行编译检查，再运行正式冻结审计：

```sh
python3 -m unittest discover -s tests -p test_audit_alp15_precision.py -v
python3 -m py_compile scripts/audit_alp15_precision.py tests/test_audit_alp15_precision.py
python3 scripts/audit_alp15_precision.py \
  --teacher docs/baselines/evidence/precision_20261009/teacher_development_result.json \
  --teacher-sha256 67b328a1ecac9de414b47680ef6476c96f6cb30dab0c736a500a902970cebaa0 \
  --kv docs/baselines/evidence/jetson_300ms_20261008/kv_candidate_full20_v1/result.json \
  --kv-sha256 db04c2ed3226e2c3c988c3f6ebd9218bd6150759d3334d25d7e193c7693ef84e \
  --reference docs/baselines/evidence/jetson_300ms_20261009/hybrid_reference20_v3/result.json \
  --reference-sha256 e83ef209381c6e166e862795e65fc21681528fe77d90c3bf48bd4d9e048e3b59 \
  --hybrid docs/baselines/evidence/jetson_300ms_20261009/hybrid_full20_v3/result.json \
  --hybrid-sha256 a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6 \
  --output docs/baselines/evidence/precision_20261009/development_audit_v1
```

三个最终命令退出码分别为 `0 / 0 / 0`。最终测试日志保存在本回报下列逐项记录，原工具会话为 session `6095`（输出chunk `226e6f`、`abdffa`）；编译检查chunk `cf2e05`无输出；正式审计stdout为chunk `670724`，正式审计完整argv/status/exit日志另保存于输出目录 `run.log`，环境、时序及输入摘要保存于 `input_manifest.json`。CLI实际stdout：

```json
{"status": "completed_development_audit", "output": "/Users/USER/Documents/ChatGPT/Lighten_VLA/docs/baselines/evidence/precision_20261009/development_audit_v1", "exit_code": 0, "diagnostics": []}
```

最终unittest完整日志（类名均为 `test_audit_alp15_precision.PrecisionAuditTests`）：

```text
test_changed_protection_code_input_dynamic_source_and_reference_binding ... ok
test_cli_wrong_sha_saves_failed_audit_and_nonzero_exit ... ok
test_coordinate_prefix_noise_tokens_and_boundary_mismatch ... ok
test_missing_dispatch_module_fake_totals_zero_fused_and_duplicate_targets ... ok
test_missing_duplicate_extra_seeds_and_wrong_teacher_group ... ok
test_parser_rejects_duplicate_keys_invalid_json_nonfinite_constants_and_overflow ... ok
test_quantile_linear_interpolation_has_independent_literal_expected_value ... ok
test_real_cli_writes_all_artifacts_and_refuses_existing_directory ... ok
test_real_frozen_results_reproduce_literal_means_and_60_exact_pairs ... ok
test_saved_pair_gate_and_KV_numerical_gate_are_crosschecked ... ok
test_short_trajectory_bad_coordinate_nonfinite_bool_and_negative_error ... ok
test_synchronously_falsified_252_target_names_are_rejected ... ok
test_synthetic_ratio_uses_ratio_of_clip_means_not_mean_of_ratios ... ok
test_teacher_ADE_swapped_bindings_fail_even_when_mean_is_unchanged ... ok
test_zero_teacher_clip_has_null_ratio_but_zero_overall_fails ... ok
----------------------------------------------------------------------
Ran 15 tests in 4.532s
OK
```

首次单元测试退出码1，共13测试出现2个测试夹具缺陷（工具session `27681`，chunk `63b031`、`7871df`）：p95浮点实际 `28.499999999999996` 对比数学字面量 `28.5` 应使用近似断言；教师错绑负例原先交换同clip/seed的同值导致未改变绑定，修正为交换不同ADE记录并保持整体和相同。修正后第二轮14测试退出码0（session `87930`），补充整体教师零值和逐clip零值测试后最后15测试退出码0。没有修改真实数据来通过测试。合成负例只在内存或临时目录处理，真实输入测试和合成比例案例名称明确区分。

## 门禁逐项对照与实际描述结果

|检查|结果|
|---|---|
|显式四路径及expectedSHA、决策SHA、输入不变|通过；原始命令及时间/平台/脚本SHA已保存|
|非法JSON/重复对象键/NaN/Inf/数字类型/负误差|负例全部拒绝；正式输入诊断0|
|教师四组与唯一dense/teacher组|每组60唯一请求；仅dense/teacher参与ADE教师比较，另180条明确排除|
|KV五variant覆盖与steps/status|300条，五组各20clip×3seed完整；没有把KV非exact候选丢弃|
|reference/hybrid与所有输入clip集合|各60条、completed、exit_code=0、fullprefill10；全部同20clip×seed42/43/44|
|64×3 finite轨迹、ADE/FDE/时间、逐键teacherADE与noise|通过；错绑但整体均值不变负例拒绝|
|60pair实际数组及全部prefix/token/noise/boundary字段|60/60 exact，max abs coordinate delta=0；不依赖保存的True|
|共同code/runtime/input、动态source stable身份、源码/checkpoint/dataset|通过；真实路径hash字典精确比较；动态临时路径按stable身份及源hash/generator比较|
|reference/hybrid四个保护摘要与paired_reference来源|通过；教师源SHA、reference gate SHA均绑定本次文件|
|252名称及每请求dispatch|36层×7真实投影精确集合；全部module计数与totals逐项核验；同步伪造名字/漏模块/假totals/zero fused均拒绝|
|保存strict/original/helper及KV numerical gates|数量/身份/布尔/数值检查；KV10 NMSE/ADE差/max坐标差与实际数组独立复算交叉核验，保留原数值门禁未通过的候选证据|
|聚合与异常输出|先clip内三seed均值，再20clip等权；整体ratio不是clipratio均值；整体teacher=0拒绝，单clip=0 ratio=null|
|全量表/尾部/失败率/限制|480请求、160clip全量输出，所有退化clip列出；无删除尾部；失败输入保存失败审计与非零退出码|

以下仅为旧开发集的描述性点估计。15%是整体mean ADE相对损失门限，并不是每clip必须≤15%的门禁：

|候选|mean ADE m|整体教师相对损失|退化clip/20|超过15%clip/20|ADE p95 m|ADE max m|
|---|---:|---:|---:|---:|---:|---:|
|教师dense/teacher|1.2323994789272548|0%|0|0|2.4410087347030647|3.6002918084462485|
|fullprefill10/reference/hybrid各自|1.282724219933152|4.08347632942061%|10|7|2.504727900028229|3.317171573638916|
|actualKV10|1.2825718538214763|4.07111295907836%|10|7|2.502070925633113|3.3122262557347617|
|actualKV6|1.2580879161755243|2.08442454638411%|10|7|2.454229362805685|3.125600258509318|
|actualKV4|1.2450251077612242|1.02447534665946%|9|7|2.412861675024033|3.027460217475891|
|actualKV2|1.2492179810379942|1.36469565253137%|11|7|2.6721907456715903|2.918438990910848|

每候选请求failed status为0/60、失败clip0/20；正式完整性诊断非finite/边界/缺失/重复/配对均0。仅通过全部完整性后才给出这些描述。学生FDE完整mean/SD/median/IQR/p95/max已保留在JSON/report中；教师FDE与相对教师FDE为null，未把两个预测之间的距离冒充对GT误差。

## 产物SHA256

以下均位于 `docs/baselines/evidence/precision_20261009/development_audit_v1/`：

|文件|SHA256|
|---|---|
|`audit.json`|`018f8f3792fa843cf0b4df9be569f48cbe0ca651f185806548c155ff4f28879f`|
|`input_manifest.json`|`59e1d3a8e6b71078860652246ea7d10da98982a2078e030ed478345bd4c0a892`|
|`per_clip.csv`|`4153b2486b976a3bf104d4e710a27313c2aa3653a66a895674e85ab4fc47d104`|
|`per_request.csv`|`7689b668cc4d9a2f75d62590f6228149a175af499a4171a9f9dc6fc014479f15`|
|`report.md`|`af795e438374d64dca0161b67debf38581e0ddac524458398b4e8ea8a4d358a2`|
|`run.log`|`64e983823edbc3a9230aa5ddf0f6c993ad874d11b1ce31466b35a077d751774e`|

`input_manifest.json`包含其他五产物的实际SHA；其自身SHA通过本回报/独立复核冻结，避免自引用。执行回报本身SHA由交接时外部字节核验提供。

## 限制、停止条件与移交

本轮没有新增独立数据或模型运行。数据反复用于开发选择，3seed属于重复测量而非60独立场景；没有提供GT、教师FDE、drive/session或真实场景metadata；没有计算p/CI、功效或制造采集前预注册。教师与KV的退出码未嵌入JSON，明确记录 `not_embedded_in_json`。教师当时使用冻结教师前缀，不能替代双方从新raw input自主生成的验证。输入来源一致不能证明本轮复核了远端所有权重字节。

当前仅得到“本地开发证据审计执行成功”及其描述结果；待独立Sol复核后才可称已有开发证据审计通过。不能称独立精度通过、完全精度验收、Orin/Jetson跑通、300ms达标、闭环质量、安全/泛化或NVIDIA原入口原样验收。KV4与hybrid仍为分开候选，没有联合实施或验收。

正式运行没有触发停止条件。以后发现输入/决策SHA漂移、教师键绑定失败、保护/来源冲突、dispatch不完整、真实GT/名单资格未解决，应保存失败输出并升级主任务/Sol；不得改输入凑数、删clip、补写FDE/退出码、换文件或绕过新GPU决策。原命令对已存在正式目录预期退出2；复核复跑请用临时目录，正式新增版本目录需主任务明确安排，不覆盖v1。

## v2：独立复核发现跨Python浮点求和问题后的修复与再验证

主任务独立验收在 `/usr/bin/python3`（3.9.6）发现 v1 原15测试1fail/2error，因此上文v1的本机3.14.6执行成功不能视为跨环境验收通过。v1目录与原日志原样保留；以下v2为主任务明确授权的修复记录。已完整阅读 `diagnosing-bugs` 技能，按真实失败回放→最小化→新增会失败的回归测试→修复→双环境全量回归执行，不增加模型机制。

### 实际失败与最小诊断

原命令 `/usr/bin/python3 -m unittest discover -s tests -p test_audit_alp15_precision.py -v` 退出1，session `57240`、chunks `591cf8`/`e6af79`：

```text
test_real_cli_writes_all_artifacts_and_refuses_existing_directory ... FAIL (returned 1 instead of 0)
test_real_frozen_results_reproduce_literal_means_and_60_exact_pairs ... ERROR
test_zero_teacher_clip_has_null_ratio_but_zero_overall_fails ... ERROR
Both errors: AuditError: kv numerical saved values disagree with arrays
Ran 15 tests in 9.359s
FAILED (failures=1, errors=2)
```

最小回放直接读取相同KV冻结JSON，SHA仍为 `db04c2ed3226e2c3c988c3f6ebd9218bd6150759d3334d25d7e193c7693ef84e`。旧Python内置float `sum`的逐项求和与Python3.12后改进求和产生舍入差；60个gate中旧sum 41个不exact，fsum 0个不exact。输入及公式一致。首个 `(06e20fbf-c4c7-4f47-8374-9a54f61f310b,42)`：旧sum NMSE=`1.1041936546822382e-08`，冻结/`math.fsum` NMSE=`1.1041936546822378e-08`。重算 `abs(old/saved-1)` 最大 `5.551115123125783e-16`（另一等价的 `abs(old-saved)/saved` 表达会有尾位差）；不是轨迹、模型或门禁差异。诊断chunk `20ab51`。

在测试中添加 `test_real_KV_gate_is_stable_when_runtime_sum_uses_legacy_left_fold`，对冻结真实key断言上述两个字面量，并强制将脚本局部的sum替换为旧逐项求和，进入真实 `audit()`完整门禁。修复前单条命令：

```sh
/usr/bin/python3 -m unittest discover -s tests -p test_audit_alp15_precision.py -k stable_when_runtime -v
```

退出1，chunk `5d69fc`，真实症状为 `AuditError: kv numerical saved values disagree with arrays`，1test/0.259s；因此新增测试在新Python也能捕获此回归，不依赖仅在旧环境偶然出现。新增测试首稿曾使用3.9不支持的starred subscript导致SyntaxError（chunk `78d67c`），立即改为tuple拼接后才记录上述真实red；未将语法错误冒称原数值问题。

唯一实现修复：NMSE分子与分母由内置 `sum` 改为 `math.fsum`，稳定计算相同数学式。没有改坐标逐元素exact、保存gate值精确比较、`1e-3` NMSE或`1e-4` ADE数值门禁、15%整体精度门限；没有放宽容差、改输入、重新生成模型结果。布尔/整数计数仍用原sum。v2源码SHA=`bb4ca945da96918ccf8604003e4973c5cf669b10b96c768db232b3100f5f6ea7`；v2测试SHA=`ff2e3e4338c40d7b28968b65940d20ffbc1bd8862c67f2eb233336c210fc5427`。

### 双环境回归与实际CLI

|本机解释器|全量unittest|py_compile|实际临时CLI|
|---|---|---|---|
|`/usr/bin/python3`：3.9.6|16tests/9.814s/exit0；session31111，8291c1/f684ab|exit0；0ea45e|exit0；54c6c8|
|`/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3`：3.12.10|16tests/5.838s/exit0；session46185，85fae0/384c7f|exit0；f34110|exit0；54c6c8|

两环境分别执行 `<executable> -m unittest discover -s tests -p test_audit_alp15_precision.py -v` 与 `<executable> -m py_compile scripts/audit_alp15_precision.py tests/test_audit_alp15_precision.py`。完整测试集是上文15名称加新增 `test_real_KV_gate_is_stable_when_runtime_sum_uses_legacy_left_fold`，两环境每项均ok。临时CLI是实际子进程命令，四路径与expectedSHA全部与原冻结命令一致；临时目录由标准库TemporaryDirectory创建并自动清理，未遗留调试原型。stdout均为 completed_development_audit/exit0/diagnostics=[]，stderr为空；完整argv在工具chunk `54c6c8`留存。

正式v2命令沿上文四路径与SHA，解释器换为 `/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3`，output换为 `docs/baselines/evidence/precision_20261009/development_audit_v2`，正式目录此前不存在。退出0，chunk `b64509`；完整命令已落 `development_audit_v2/run.log`。本机Python3.12.10/Clang20.1.0、平台 `macOS-27.2-arm64-arm-64bit`，不是远端服务器环境；本轮仍纯CPU/stdlib/no model/no SSH/network。正式UTC起止为 `2026-10-09T15:12:35.524002+00:00` / `2026-10-09T15:12:35.741109+00:00`。

v2完整性通过，60pair actual exact、480请求/160clip、四输入解析前后及审计后SHA不变；描述均值、比例、退化数量和尾部均未变化，两个CSV字节SHA与v1相同。教师mean1.2323994789272548，hybrid/ref/fullprefill10 mean1.282724219933152（+4.0834763294%），KV4 mean1.2450251077612242（+1.0244753467%）。教师FDE=null、独立精度/Orin pass=null，未新增p/CI或任何模型/独立验收结论。

|v2产物|SHA256|
|---|---|
|`audit.json`|`d8d3ba55842659fd7a986c26ba2d98ebc3a80183ff3f6336b72a3226cf35d6fc`|
|`input_manifest.json`|`9afc7be515a5903af974a32321a13caae1fe25f7d2a9c13f97627ca7bd76385c`|
|`per_clip.csv`|`4153b2486b976a3bf104d4e710a27313c2aa3653a66a895674e85ab4fc47d104`|
|`per_request.csv`|`7689b668cc4d9a2f75d62590f6228149a175af499a4171a9f9dc6fc014479f15`|
|`report.md`|`954866a0fb308aec427c16d2a38b4dfcf7345237edc2893d0bd8fcc659e098b8`|
|`run.log`|`0c2905d51e82b24357f0a9c46a478a550ca7d1c19aaf6fa41bc5762e16ff8b7d`|

v1的SHA与历史源码/测试SHA仍保留于原段落；v1不是被静默覆盖的“新成功”。本v2修复与证据交还独立验收者复核，执行者不自授验收。其它原门禁、限制与停止条件全部持续有效。

## v3：异常records容器的失败证据留存修复

主任务独立复核发现：同步坏输入expectedSHA后令 `teacher.records=null`，v2在 `len(None)` 后错误库存再次 `enumerate(None)`，退出1却只留下空输出目录。该问题违反已有决策的失败留痕条款；主任务明确授权最小修复。未见数据资格准备任务已暂停在只读决策/包身份检查阶段，三个准备文件尚未创建；不以新任务跳过本轮复验。

新增测试 `test_cli_malformed_records_save_first_failure_for_all_four_inputs`，使用真实本机CLI子进程，四个输入分别覆盖 missing/null/string/dict/bool/列表元素非dict，共24场景；每个坏输入的expectedSHA都由坏文件字节重算，不能因SHA不匹配提前退出代替结构验证。测试要求：退出1、stderr没有Traceback、保存audit/report/manifest/run.log、summaries为空、首诊断存在且不是SHA失败、坏输入身份SHA保留。未改变或覆盖任何真实冻结输入。

修复前红测命令 `/usr/bin/python3 -m unittest discover -s tests -p test_audit_alp15_precision.py -k malformed_records -v`：

```text
四输入各null/bool共8场景 ... FAIL
first error: TypeError: object of type 'NoneType' has no len()
handling error: TypeError: 'NoneType' object is not iterable
Ran 1 test in 6.082s
FAILED (failures=8)
```

原工具session `49723`，chunks `315fe2`/`0a5ba3` 保存各场景实际子进程Traceback。此前同一测试以直接CLI main接口复现8个二次异常（session64598，5bfa73/318c4c），随后改为真实子进程确认相同故障；没有将测试框架错误当作输入拒绝成功。

实现只增加读后结构预检与安全库存遍历：SHA和JSON解析完成后记录 `record_container_type`，仅list计算count，非list count=null并明确AuditError；list中的每项必须dict。失败库存仅处理dict输入内的list以及dict row，不对null/bool/字符串/对象容器调用enumerate，不再因坏容器覆盖最初诊断。没有添加宽泛Exception捕获或通过吞掉异常宣称pass。模型阈值、数组exact、NMSE稳定求和、四输入/决策SHA和指标公式都未更改。

v3代码 SHA `c3f34af61db9b201ff074c5be09c4df78da450302906b1c410ff28ac21a40faa`；测试 SHA `5ac86567ceef29d6708ef179fdf56a1f24621ec72f154bc81a48e253a378865a`。

|本机环境|完整回归|编译|实际CLI正例/records=null负例|
|---|---|---|---|
|`/usr/bin/python3` 3.9.6|17tests/14.206s/exit0；session48096，f643a1/db8cf6|exit0；4c7993|0 / 1，stderr均空；b20f66|
|`/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3` 3.12.10|17tests/9.089s/exit0；session35151，171f40/7c6f5b|exit0；ab9050|0 / 1，stderr均空；b20f66|

完整回归命令与v2相同，每项ok；新增结构测试的24实际CLI子场景也全部通过。实际独立CLI回放坏teacher文件SHA=`9b73db5e25f68c4aa884a86d2443b69789058e40c6113a970d9a41d5dba8c251`，expected/before/after_parse/after_audit一致，原教师文件未变化。两环境都保存全部六产物，首诊断 `teacher: records missing or not list`，input manifest明确 `record_container_type=NoneType`、`record_count=null`；summaries为空，不从失败输入计算达标。临时原型经TemporaryDirectory自动清理；该回放完整stdout在chunk `b20f66`。

正式v3继续使用全部冻结真实四路径/SHA，解释器为上述本机3.12.10，output为此前不存在的 `docs/baselines/evidence/precision_20261009/development_audit_v3`。完整argv保存 `run.log`，CLI exit0、diagnostics=[]（chunk283409）；UTC `2026-10-09T15:22:30.505777+00:00` 至 `2026-10-09T15:22:30.721843+00:00`。环境、CPU-only边界同v2，无GPU/模型/SSH/联网/安装。

正式输入各SHA解析前后/审计后相等，60pair exact、480请求/160clip；mean ADE、比例和尾部与v1/v2完全相同，两CSV字节SHA同原值，独立精度/Orin及教师FDE继续null。v1/v2各五产物与其manifest内原SHA逐项外部核验一致，manifest本身分别仍为 `59e1d3a8e6b71078860652246ea7d10da98982a2078e030ed478345bd4c0a892` / `9afc7be515a5903af974a32321a13caae1fe25f7d2a9c13f97627ca7bd76385c`，旧六产物未覆盖。

|v3产物|SHA256|
|---|---|
|`audit.json`|`072b98a033c01e4e8def97472804a0248d813c53d812ebea15f04e537b34dfa2`|
|`input_manifest.json`|`37ccb9c62b0cf4ccc8b3b8ed970ac7b8f4f4f83a12ab1ed34ef3f739c5e86979`|
|`per_clip.csv`|`4153b2486b976a3bf104d4e710a27313c2aa3653a66a895674e85ab4fc47d104`|
|`per_request.csv`|`7689b668cc4d9a2f75d62590f6228149a175af499a4171a9f9dc6fc014479f15`|
|`report.md`|`975875aa7046256c3da06ab1ab680ecb0e4ca2c014e3626b8350bb1cb8fd6e17`|
|`run.log`|`11e55eeb906442d4d114cfaedb2a1d32c88f2c3eebf04425a940b77451ce8a53`|

本v3源码/测试/回报与新证据冻结后交独立Sol复验；在明确恢复前，资格准备继续暂停，不自行SSH。原统计、GT、独立数据、Orin和官方原入口门禁持续未完成。

## v4收尾：2026-10-10主任务现态核验

昨日收尾时v4源码、测试和正式六产物已生成，原执行者在补写回报前被用户要求停止。本段由主任务独占追加，仅补齐当前可验证记录，不把中断阶段缺失的完整日志补造为已经归档。

主任务批准的v4边界是：针对number()中math.isfinite对10**400整数抛OverflowError的已复现问题，局部转为AuditError，保留正常数值、轨迹exact和既有精度门禁；不增加宽泛Exception捕获、不改输入。实际源码检查确认仅数值有限性检查局部捕获OverflowError。新增真实CLI测试覆盖四输入的ADE及坐标超大整数，共8子场景，坏输入expectedSHA按副本同步；所有副本只在临时目录，不改变真实输入。昨日失败回放状态由执行者消息报告为8fail，但本段没有其完整red日志，不冒称已补齐该历史日志。

冻结脚本SHA `85d978f4b9ebbadb55c168a9539af84316d95d2c1521305c6f5d9af3e34c6341`，测试SHA `dac4285a8f0f4874cde9b172deb03e65b119d78e2b96b0af58443e2ddea9e78a`，今天实际读取与昨日收尾一致。正式manifest记载本机Python3.12.10/macOS27.2-arm64、开始UTC2026-10-09T15:26:21.084937+00:00，完整四输入/expectedSHA及新输出路径保存在development_audit_v4/run.log，状态completed_development_audit、diagnostics=[]、exit0。不是服务器模型重跑，亦不是独立新数据测试。

2026-10-10主任务实际复跑命令：

```sh
/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B -m unittest discover -s tests -p test_audit_alp15_precision.py -v
```

18tests/12.398s/exit0，所有测试ok，session64507、chunksa21514/e8c84d。包含24个records结构CLI子场景、8个大整数CLI子场景、旧正常/负例及拒覆盖。主任务此前停止时的Python3.9.6复跑也已正常完成18tests/18.438s/exit0（session31352、chunk943738，见收尾记录）。本轮独立验收者另外对双环境正常CLI与records=null/大整数同步SHA CLI做复验，其完整环境/命令/输出/局限另见ALP15-PRECISION-AUDIT-V4-ACCEPTANCE-20261010.md；主任务不以测试数量代替独立验收结论。

正式六产物本轮字节SHA：

|v4产物|SHA256|
|---|---|
|audit.json|0a46b5a13c9d2679843bbeccc74b87707bc37a723bcaad5e5a3cb6edc2087d7c|
|input_manifest.json|8a33819e52c403ac436ab97ebefc1ffbbc0bba22897fba84f1ce72fd3bbc3a79|
|per_clip.csv|4153b2486b976a3bf104d4e710a27313c2aa3653a66a895674e85ab4fc47d104|
|per_request.csv|7689b668cc4d9a2f75d62590f6228149a175af499a4171a9f9dc6fc014479f15|
|report.md|68d62133601404b264128846891496dc1de2c2b3d6cb8d4eec9810867cc38ac9|
|run.log|c67dd73abf1eed4a187b20f9ddae97d8af5cae1572683fedc68137130a7db0a3|

四实际输入解析前/后/审计后SHA仍一致；CSV与原v1-v3同SHA，开发集均值/ratio/尾部未改变。独立精度、Orin及教师FDE仍null，不补造GT。各历史产物保留，不删除失败版本；跨Python描述SD末位差保留为限制，不追求全CSV跨版本字节exact。

本段仅证明主任务可复现v4和保存当前交付身份；新数据census、名单/QC/采集前设计、真实教师/候选自主生成、Orin16GB及300ms门禁仍未完成。数据准备文件所有权已另交holdout_prep_execution_20261010，原审计执行者不恢复写准备文件。
