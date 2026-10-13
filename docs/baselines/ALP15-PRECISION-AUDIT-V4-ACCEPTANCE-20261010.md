# ALP15 精度离线审计 v4 独立技术验收

- 对应决策：`D-ALP15-PRECISION-VALIDATION-20261009-01`；日期：2026-10-10（Asia/Shanghai）。
- 独立验收者：GPT-6.1 Sol high，依主任务分派执行。
- 结论：**v4 工具及冻结四输入的已有开发证据审计通过独立技术验收**。本结论限定为 `exploratory_reused_development`，不等于独立精度通过或“完全验证精度”。
- 范围：已完整读取仓库 AGENTS、精度决策、执行回报（初读至 v3，交付前补读主任务追加的完整 v4 段）、v2 拒绝验收、昨日收尾、当前 517 行源码与 293 行测试；读取 v4 六产物，解析全部 audit/CSV 内容并核对 manifest 与实际字节，核对 v1–v4 六产物留存。
- 写入所有权：验收者仅新增本文。未修改脚本、测试、执行回报、正式证据、四输入或 venv。临时 CLI/合成输入由标准库 TemporaryDirectory 创建并自动清理；解释器使用 `-B`。没有使用 superpowers。
- 执行环境：本机 `macOS-27.2-arm64-arm-64bit` / arm64，CPU 与 Python 标准库。没有网络、SSH、GPU、torch/numpy、模型加载、PT/pickle payload 或依赖安装；CUDA/驱动与模型资源观测不适用本轮。

## 1. 冻结身份与正式证据

下列为本轮实际字节 SHA256，不仅引用旧回报。脚本、测试在复验前后保持一致。

|文件|SHA256|
|---|---|
|`scripts/audit_alp15_precision.py`|`85d978f4b9ebbadb55c168a9539af84316d95d2c1521305c6f5d9af3e34c6341`|
|`tests/test_audit_alp15_precision.py`|`dac4285a8f0f4874cde9b172deb03e65b119d78e2b96b0af58443e2ddea9e78a`|
|`ALP15-PRECISION-VALIDATION-DECISION-20261009.md`|`989b64ac1adfa3979556db82d035f71230d91132454b8eb35029eafacf0280db`|
|执行回报读取时版本|`0dedf579f0540c66d8a754309e7d966389f4f421cb13054db18de4813816e3d0`|
|执行回报交付前追加版本|`f213d4f51f060a6b412f1a8ad81c4c72568f622fbcc3149e443180f8006d2890`|

执行回报路径为 `docs/baselines/ALP15-PRECISION-AUDIT-EXECUTION-20261009.md`，初读时尚无 v4 段，交付前已由主任务独占追加当前可验证的 v4 收尾，验收者完整补读并重算新 SHA。新增段的正式六产物 SHA 与本轮实读相符，并记录主任务今日 3.12 复跑 18tests/12.398s/exit0（session64507，chunksa21514/e8c84d）；它未修改被验收的脚本、测试或输入。昨日中断阶段的完整 red 日志没有归档，只有执行者消息曾报告八个失败；本文没有将该历史消息当作独立复核的完整日志，也没有宣称补齐。当前工具/真实输入验收以冻结源码、实际 v4 文件及本轮双环境独立复验为依据。被验收脚本、测试或四输入变化，需另行复核。

四输入位于 `docs/baselines/evidence/`，实际 SHA 与冻结 expected、正式 v4 manifest 的解析前/解析后/审计后三次 SHA 一致；双 CLI 及负例测试结束后再次核对未变。

|输入|SHA256|
|---|---|
|`precision_20261009/teacher_development_result.json`|`67b328a1ecac9de414b47680ef6476c96f6cb30dab0c736a500a902970cebaa0`|
|`jetson_300ms_20261008/kv_candidate_full20_v1/result.json`|`db04c2ed3226e2c3c988c3f6ebd9218bd6150759d3334d25d7e193c7693ef84e`|
|`jetson_300ms_20261009/hybrid_reference20_v3/result.json`|`e83ef209381c6e166e862795e65fc21681528fe77d90c3bf48bd4d9e048e3b59`|
|`jetson_300ms_20261009/hybrid_full20_v3/result.json`|`a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6`|

输入中的源码/checkpoint/dataset 身份为 `36aeb4c5938cbc2eb2aed33b22434773da4ab639` / `7aba8293c09993f2e125c6819df05d7fa3e873ea` / `33f9bf447ed3bcb7d545ce13f4226f824214fafb`。这是本地冻结来源字段核验，未联网或重核远端全部权重字节。

正式 v4 目录为 `docs/baselines/evidence/precision_20261009/development_audit_v4/`。正式运行解释器为 Python 3.12.10，UTC `2026-10-09T15:26:21.084937+00:00` 至 `2026-10-09T15:26:21.321232+00:00`，`run.log` 记录 `completed_development_audit`、exit 0、diagnostics=[]；manifest 保存完整 argv、Python/平台、脚本/决策 SHA 与四输入身份。实际产物 SHA：

|v4 产物|SHA256|
|---|---|
|`audit.json`|`0a46b5a13c9d2679843bbeccc74b87707bc37a723bcaad5e5a3cb6edc2087d7c`|
|`input_manifest.json`|`8a33819e52c403ac436ab97ebefc1ffbbc0bba22897fba84f1ce72fd3bbc3a79`|
|`per_clip.csv`|`4153b2486b976a3bf104d4e710a27313c2aa3653a66a895674e85ab4fc47d104`|
|`per_request.csv`|`7689b668cc4d9a2f75d62590f6228149a175af499a4171a9f9dc6fc014479f15`|
|`report.md`|`68d62133601404b264128846891496dc1de2c2b3d6cb8d4eec9810867cc38ac9`|
|`run.log`|`c67dd73abf1eed4a187b20f9ddae97d8af5cae1572683fedc68137130a7db0a3`|

对 v1–v4 每目录都实际计数六文件，五个 output hash 均与该版本 manifest 相符，manifest 自身 hash 外部重算。v1–v3 六产物未被本轮覆盖；两个正式 CSV 在四个版本字节一致，SHA 见 v4 表。其余历史 SHA 如下：

|版本|audit.json|input_manifest.json|report.md|run.log|
|---|---|---|---|---|
|v1|`018f8f3792fa843cf0b4df9be569f48cbe0ca651f185806548c155ff4f28879f`|`59e1d3a8e6b71078860652246ea7d10da98982a2078e030ed478345bd4c0a892`|`af795e438374d64dca0161b67debf38581e0ddac524458398b4e8ea8a4d358a2`|`64e983823edbc3a9230aa5ddf0f6c993ad874d11b1ce31466b35a077d751774e`|
|v2|`d8d3ba55842659fd7a986c26ba2d98ebc3a80183ff3f6336b72a3226cf35d6fc`|`9afc7be515a5903af974a32321a13caae1fe25f7d2a9c13f97627ca7bd76385c`|`954866a0fb308aec427c16d2a38b4dfcf7345237edc2893d0bd8fcc659e098b8`|`0c2905d51e82b24357f0a9c46a478a550ca7d1c19aaf6fa41bc5762e16ff8b7d`|
|v3|`072b98a033c01e4e8def97472804a0248d813c53d812ebea15f04e537b34dfa2`|`37ccb9c62b0cf4ccc8b3b8ed970ac7b8f4f4f83a12ab1ed34ef3f739c5e86979`|`975875aa7046256c3da06ab1ab680ecb0e4ca2c014e3626b8350bb1cb8fd6e17`|`11e55eeb906442d4d114cfaedb2a1d32c88f2c3eebf04425a940b77451ce8a53`|

## 2. 本轮命令、日志与退出码

工作目录为 `/Users/USER/Documents/ChatGPT/Lighten_VLA`。两个独立解释器实际执行：

```sh
/usr/bin/python3 -B -m unittest discover -s tests -p test_audit_alp15_precision.py -v
/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B -m unittest discover -s tests -p test_audit_alp15_precision.py -v
```

|解释器|18 项 unittest|真实临时 CLI|同步 SHA 三负例|已存在输出|
|---|---|---|---|---|
|`/usr/bin/python3`，3.9.6|18.654 s，全部 ok，exit 0|exit 0，diagnostics=[]，stderr 空|各 exit 1，stderr 空|exit 2，六产物字节不变|
|`/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3`，3.12.10|12.360 s，全部 ok，exit 0|exit 0，diagnostics=[]，stderr 空|各 exit 1，stderr 空|exit 2，六产物字节不变|

本轮工具会话完整日志位置：3.9 unittest session `3913`，chunks `935482`、`d16082`、`796162`；3.12 unittest session `84474`，chunks `014caf`、`01d472`。独立 CLI/数组/聚合/产物校验包装 session `44923`，提交 chunk `e1a9c2`，完整 stdout chunk `94e988`，包装 exit 0；运行 UTC `2026-10-10T04:41:30.395616+00:00` 至 `2026-10-10T04:41:32.722685+00:00`。补充全部八组公式、历史产物与 KV 门禁独立复算 chunk `4df39b`，exit 0。工具日志包括实际完整 argv、负例字节 SHA、诊断、CSV SHA、逐版本六产物 SHA 与断言结果，未保存或覆盖额外仓库日志文件。

真实 CLI 从正式 v4 manifest 逐一取四路径和 expected SHA，命令为下式；`<python>` 分别使用上表两个解释器，`<new-temporary-output>` 为 TemporaryDirectory 内此前不存在的 positive 子目录：

```sh
<python> -B /Users/USER/Documents/ChatGPT/Lighten_VLA/scripts/audit_alp15_precision.py \
  --teacher /Users/USER/Documents/ChatGPT/Lighten_VLA/docs/baselines/evidence/precision_20261009/teacher_development_result.json \
  --teacher-sha256 67b328a1ecac9de414b47680ef6476c96f6cb30dab0c736a500a902970cebaa0 \
  --kv /Users/USER/Documents/ChatGPT/Lighten_VLA/docs/baselines/evidence/jetson_300ms_20261008/kv_candidate_full20_v1/result.json \
  --kv-sha256 db04c2ed3226e2c3c988c3f6ebd9218bd6150759d3334d25d7e193c7693ef84e \
  --reference /Users/USER/Documents/ChatGPT/Lighten_VLA/docs/baselines/evidence/jetson_300ms_20261009/hybrid_reference20_v3/result.json \
  --reference-sha256 e83ef209381c6e166e862795e65fc21681528fe77d90c3bf48bd4d9e048e3b59 \
  --hybrid /Users/USER/Documents/ChatGPT/Lighten_VLA/docs/baselines/evidence/jetson_300ms_20261009/hybrid_full20_v3/result.json \
  --hybrid-sha256 a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6 \
  --output <new-temporary-output>
```

3.12 去除运行 provenance 后，全部 audit 内容与正式 v4 相同，两个 CSV 字节相同。3.9 门禁、轨迹、均值、比例、计数相同，只有第 5 节所列 SD 末位差；未将全版本全部字节一致作为未批准的新门禁。

18 tests 覆盖真实四输入/60 actual exact、真实 CLI/拒覆盖、SHA错误失败输出、跨版本 legacy sum 回归、四输入各六种 records 结构共 24 个实际 CLI 子场景、四输入 ADE/坐标超大整数共八个实际 CLI 子场景，以及原有 JSON/身份覆盖/错绑/轨迹数字/生成边界/来源保护/252 分派/saved gates/ratio/quantile/teacher-zero 负例。合成测试与真实结果明确分开，不以合成测试代替实际审计。

## 3. 局部修复与独立失败回放

确认当前源码只沿既有局部修复范围验证：NMSE 由 `math.fsum` 稳定计算；records 预检和安全失败库存保留首诊断；`number()` 将大整数 `math.isfinite` 的 OverflowError 转为 AuditError。实际数组 exact、saved gate 的精确比较、NMSE `1e-3`、ADE delta `1e-4` 和整体精度 15% 门限持续保留，没有放宽模型门禁。

除全量 unittest 的 32 个真实 CLI 子场景外，本验收包装独立构造以下三个教师临时副本，并按每个副本实际字节更新 `--teacher-sha256`；其余三个输入仍用冻结原件。两环境结果完全一致，均在 SHA 匹配后进入内容校验，非 SHA 提前拒绝。

|临时输入改动|临时文件 SHA256|首诊断|
|---|---|---|
|`records=null`|`9b73db5e25f68c4aa884a86d2443b69789058e40c6113a970d9a41d5dba8c251`|`teacher: records missing or not list`|
|首记录 `ADE_m=10**400`|`9e0e8faaaab5cdba3040d6c8e599dd71cf6ceff41ef0466ee34428bba3c14cc3`|`teacher records[0] ADE_m: numeric value cannot be represented as finite float`|
|首坐标 `pred_xyz[0][0]=10**400`|`a34f0e4361c6a46a467c4b259305198cedb63710ddac2ce9a2c87d5d70554586`|`teacher records[0] pred_xyz: numeric value cannot be represented as finite float`|

各负例均 exit 1、stderr 空、`integrity_passed=false`、`summaries={}`、`independent_precision_pass=null`，保存全部六产物；manifest 中坏输入 expected/before/after_parse/after_audit SHA 四者一致，五个 output SHA 与实际产物相符。records=null 的 `record_container_type=NoneType`、`record_count=null`；超大整数仍记录 list/240。v2 二次异常/空目录缺陷在冻结 v4 上未复现。原 v2 reject 结论及历史失败保留，未反向改写为通过。

## 4. 门禁逐项与开发集复算

独立复算包装未导入审计实现，直接读取冻结 JSON，选择教师键、比较实际数组、构造预定 252 名称、求和每个模块计数并计算聚合；补充包装独立重算 60 个 KV10 NMSE，逐项比对冻结 gate。

|门禁|本轮实际结果|
|---|---|
|四输入与 provenance|实际 SHA 与 expected/正式三阶段 SHA 一致，本轮前后不变；源码/checkpoint/dataset 与冻结身份一致|
|教师组与唯一比较组|四组各 60 唯一记录；仅 dense/teacher 的 60 键为教师，另 180 条排除组保留|
|候选覆盖与绑定|五 KV variant 各 60、reference/hybrid 各 60；统一 20 clip × seed 42/43/44；所有 teacher_ADE/noise 逐键绑定|
|实际坐标与生成字段|reference/hybrid 实际 64×3 数组 60/60 exact，最大坐标差 0；ADE/FDE、prefix/noise/token/boundary/decision/steps 配对一致|
|来源/保护摘要|共同 code/runtime/input 哈希字典、virtual 来源一致；动态 stable/source/hash 与生成字段由源码/测试/真实 CLI 核验；四保护 SHA 相等；reference provenance 绑定本次实际 JSON SHA|
|252 名称与分派|独立生成 36 层 × attention 四投影 + MLP 三投影；与 install 及每请求 modules 集合精确一致，hybrid fused>0、reference disabled 路径成立，totals 由实际 modules 求和相等|
|saved 与 KV gates|strict/original 的数目、键、布尔与数值由源码及实测交叉检查；独立 KV 60 gate 旧左折 sum 有 41 个不 exact，fsum 有 0 个；保留原 passed=false 58/60，不删除 KV 不利记录|
|正常/失败契约|18 tests 双环境全过；独立同步 SHA 三负例六产物/首诊断成立；已存在目录 exit 2 且产物未变|
|全量表与聚合|480 请求、160 clip 表项；两个 CSV 每行各字段与 audit 内容一致；八组均先 clip 内三 seed mean，再 20 clip 等权，主 ratio 均为整体均值之比减一|
|尾部/失败分母|完整退化表保留；正常每组模型记录请求失败 0/60、clip 失败 0/20；教师无行 status/teacher及KV无JSON退出码明确记 not_embedded_in_json|
|精度范围|教师 FDE/相对 FDE=null；independent_precision_pass/Orin_gate_passed=null；scope=exploratory_reused_development，无 p/CI 或独立验证通过结论|

实际点估计（米）：

|候选|mean ADE|整体相对教师损失|退化 clip|超过15% clip|ADE p95|ADE max|
|---|---:|---:|---:|---:|---:|---:|
|teacher dense/teacher|1.2323994789272548|0|0/20|0/20|2.4410087347030647|3.6002918084462485|
|hybrid/reference/fullprefill10|1.282724219933152|4.08347632942061%|10/20|7/20|2.504727900028229|3.317171573638916|
|actualKV4|1.2450251077612242|1.0244753466594592%|9/20|7/20|2.412861675024033|3.027460217475891|

hybrid 的平均 clip ratio 为 `0.11294743257279687`，KV4 为 `0.05909973018146081`，均与主指标不同。没有误用它们替代整体 ratio。两个候选的七个 >15% clip 相同，保留如下；逐 clip 15%仅为已批准诊断，不新增逐 clip 安全验收阈值。

|clip|hybrid 相对增量|KV4 相对增量|
|---|---:|---:|
|`44b4c663-5c9d-4c69-a5e5-15d966bd9823`|39.3773178461%|27.2045402628%|
|`60d13844-813d-4c35-a345-d59ba96aa681`|25.0483174445%|20.9111384680%|
|`7bd08ee7-3e9b-499e-b9af-c3b84aac728b`|34.4196819493%|26.5125152337%|
|`9147f7fb-a487-4be8-a038-59bdfaa295b7`|28.3636073406%|18.2361637471%|
|`914c6c62-6185-4648-b44a-51204244f1a1`|83.7804934373%|93.1313209205%|
|`b483042f-f4af-49df-93e7-59d848f34eec`|73.3613831887%|63.0386856737%|
|`fb80c64b-45e4-4253-a52d-79081fb67bad`|61.3531829910%|53.5971597355%|

## 5. 已知限制与结论边界

3.9 与正式 3.12 的 audit 去 provenance 后只有 13 处描述 SD 末位差：教师总体 SD 一处、seed SD 十二处，最大绝对差约 `2.22e-16`。例如教师 SD 为 `0.7790969513848408` / `0.7790969513848409`。3.9 per_request CSV SHA 与正式相同；per_clip CSV 为 `90a58f0d652e04f846e378aef594dcf8576d4b48f9e0e33c6bf20e16597aed75`，与正式不同。未改变任何门禁、均值、ratio、坐标或结论；不宣称跨版本全产物字节一致。

这 20 clip 反复用于开发选择，三个 seed 是重复测量。480/160 表项不是 480 请求或 160 clip 的独立样本来源。无 GT、教师 FDE、drive/session 或真实场景 metadata，不能重算 ground-truth ADE/FDE；学生 FDE 仅为原字段的描述，不能把教师—学生轨迹差冒充 FDE。教师当时使用冻结教师前缀，不替代双方对新 raw input 自主生成。

新数据资格/统计预案、独立精度、闭环质量、Orin/Jetson ARM64 与 16GB 统一内存、完整请求 max≤300ms、NVIDIA 官方原入口原样测试均未被本轮审计完成。KV4 与 hybrid 为分别评测的候选，未联合实现或验收。本轮未新增研究路线、模型机制、联网核验或 GPU 执行。

因此冻结 v4 在当前决策范围内通过独立技术验收，可以报告“已有开发证据审计通过”；当前可复现执行回报已追加，缺失的昨日完整 red 日志仍如实保留为历史留痕限制，独立精度与设备等后续门禁继续 pending。若后续脚本/测试/四输入 SHA 漂移，或发现教师绑定、保护/来源、派发覆盖冲突，保存证据并交执行者按新决策处理；验收者不自行修改实现。
