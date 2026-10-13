# ALP15 精度离线审计 v2 独立验收

- 对应决策：`D-ALP15-PRECISION-VALIDATION-20261009-01`。
- 日期：2026-10-09（Asia/Shanghai）；独立验收者：GPT-6.1 Sol high，按主任务当前安排执行。
- 结论：**v2 工具验收 reject / pending 修复**。固定真实输入的开发证据复现、跨 Python NMSE 门禁修复、现有 16 项测试通过；新增结构负例暴露失败审计留痕缺陷，不能据此宣布完整工具契约或“已有开发证据审计”已完成独立验收。
- 写入边界：验收者只新增本文件；未修改执行者脚本、测试、执行回报、v1/v2 证据或冻结输入。测试与 CLI 的临时目录由标准库 TemporaryDirectory 管理并清理，使用 `-B` 禁止新增字节码缓存。
- 执行范围：本机 CPU、Python 标准库；无网络、下载、SSH、GPU、模型导入/运行、PT/pickle 或依赖安装。已完整阅读决策、执行回报及脚本/测试，并核对 v2 六份产物。

## 1. 冻结身份与追溯

下列 SHA256 均为验收时实际文件字节重算，不仅引用执行回报：

|文件|SHA256|
|---|---|
|`scripts/audit_alp15_precision.py`|`bb4ca945da96918ccf8604003e4973c5cf669b10b96c768db232b3100f5f6ea7`|
|`tests/test_audit_alp15_precision.py`|`ff2e3e4338c40d7b28968b65940d20ffbc1bd8862c67f2eb233336c210fc5427`|
|`docs/baselines/ALP15-PRECISION-VALIDATION-DECISION-20261009.md`|`989b64ac1adfa3979556db82d035f71230d91132454b8eb35029eafacf0280db`|
|`docs/baselines/ALP15-PRECISION-AUDIT-EXECUTION-20261009.md`|`763e56f95563d22b358bda2468fdb45ccd281af15510c966c902e7d519a01341`|

版本边界：以上脚本/测试 SHA 是验收开始时实际核对的 v2；双环境 16 tests在该版本完成。文档完成时外部执行者已开始后续修复，最后检查测试 SHA变为 `5ac86567ceef29d6708ef179fdf56a1f24621ec72f154bc81a48e253a378865a`，脚本及正式 v2 audit仍为表中 SHA。本文不验收该新增测试版本，不将并发后续修复归于验收者，也不根据它改写 v2拒绝结论。

输入均位于 `docs/baselines/evidence/`，实际 SHA 与命令 expected、manifest 的解析前/后及审计后 SHA 四者一致。验收复跑后再次读取输入字节仍相同：

|输入|SHA256|
|---|---|
|`precision_20261009/teacher_development_result.json`|`67b328a1ecac9de414b47680ef6476c96f6cb30dab0c736a500a902970cebaa0`|
|`jetson_300ms_20261008/kv_candidate_full20_v1/result.json`|`db04c2ed3226e2c3c988c3f6ebd9218bd6150759d3334d25d7e193c7693ef84e`|
|`jetson_300ms_20261009/hybrid_reference20_v3/result.json`|`e83ef209381c6e166e862795e65fc21681528fe77d90c3bf48bd4d9e048e3b59`|
|`jetson_300ms_20261009/hybrid_full20_v3/result.json`|`a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6`|

以下正式 v2 产物均位于 `docs/baselines/evidence/precision_20261009/development_audit_v2/`。同时逐项重算 manifest 中的五个 output hashes；自身 manifest SHA 外部核验：

|产物|SHA256|
|---|---|
|`audit.json`|`d8d3ba55842659fd7a986c26ba2d98ebc3a80183ff3f6336b72a3226cf35d6fc`|
|`input_manifest.json`|`9afc7be515a5903af974a32321a13caae1fe25f7d2a9c13f97627ca7bd76385c`|
|`per_clip.csv`|`4153b2486b976a3bf104d4e710a27313c2aa3653a66a895674e85ab4fc47d104`|
|`per_request.csv`|`7689b668cc4d9a2f75d62590f6228149a175af499a4171a9f9dc6fc014479f15`|
|`report.md`|`954866a0fb308aec427c16d2a38b4dfcf7345237edc2893d0bd8fcc659e098b8`|
|`run.log`|`0c2905d51e82b24357f0a9c46a478a550ca7d1c19aaf6fa41bc5762e16ff8b7d`|

正式 manifest 保存 Python、平台、UTC 起止、完整 argv、脚本及决策 SHA，run.log 保存状态、退出码与诊断。v1 原目录及历史失败不被覆盖；此验收不把 v1 的 3.14 执行成功改写成跨环境成功。

## 2. 独立复跑与已通过部分

工作目录为 `/Users/USER/Documents/ChatGPT/Lighten_VLA`，本机 macOS 27.2 / arm64。命令如下，两个解释器的执行相互独立：

```sh
/usr/bin/python3 -B -m unittest discover -s tests -p test_audit_alp15_precision.py -v
/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B -m unittest discover -s tests -p test_audit_alp15_precision.py -v
```

|解释器|测试与退出码|独立真实 CLI|
|---|---|---|
|`/usr/bin/python3`，Python 3.9.6|16 tests / 9.814 s / exit 0|exit 0，completed_development_audit，diagnostics=[]，stderr 空|
|`/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3`，Python 3.12.10|16 tests / 5.823 s / exit 0|exit 0，completed_development_audit，diagnostics=[]，stderr 空|

两环境所有测试均 `ok`：输入 SHA/非法 JSON/重复键/NaN/Infinity/overflow 常量；缺失、重复、额外 seed 和错误教师组；教师 ADE 错绑且整体均值不变；64×3/数字类型/负值；prefix/noise/token/生成边界；保护/代码/输入/动态 source/reference provenance；漏 dispatch/伪 totals/零 fused/重复或同步伪造 252 名称；saved pair/KV gates；整体 ratio 公式；分位数插值；教师零值；真实 CLI 拒覆盖；真实冻结均值和 60 exact；强制 legacy sum 回归。

工具会话完整输出位置：3.9 测试 session `92620`，chunks `4c6cbf` / `fa6036`；3.12 测试 session `62361`，chunks `919650` / `850a3e`。双 CLI及独立复算输出 chunks `7f2ccc`、`5646ab`、`89c49f`。首个验收包装命令因过强的“跨版本全部 audit/CSV 字节一致”断言退出 1；其子 CLI 实际退出 0。后续明确定位为第 4 节的 SD 差异，并没有改模型门禁或输入绕过。

真实 CLI 以正式 manifest 的四路径及四个 expected SHA 为冻结参数，用上述各解释器执行 `scripts/audit_alp15_precision.py`，仅 output 替换为此前不存在的 TemporaryDirectory 子目录。两次均返回：

```text
status=completed_development_audit
exit_code=0
diagnostics=[]
stderr=""
```

3.12 复跑的 audit 去除运行 provenance 后与正式 v2 全部内容相同，两个 CSV 字节相同。3.9 的所有门禁、轨迹、均值、ratio、计数相同；描述 SD/CSV 限制另列，不宣称其全部字节 exact。

|独立检查|结果及范围|
|---|---|
|教师唯一组|仅 `model=dense AND prefix=teacher` 的 60 键为教师；四组各 60，另三组 180 条排除并保留身份说明|
|覆盖/同键绑定|五 KV variant 各 60、ref/hybrid 各 60；统一 20 clip × seed 42/43/44；逐键 teacher_ADE/noise 检查，不用整体均值代替绑定|
|hybrid/reference exact|直接实际数组比较 60/60 exact，最大坐标差 0；ADE/FDE、prefix、noise、token IDs、边界、decision 和 steps 全部配对|
|代码、输入与保护|代码/runtime/input hash 字典精确比较；动态源按 stable/source SHA与生成器字段检查；四保护 SHA相同；teacher 与 reference 文件 SHA provenance 正确绑定|
|252 实际模块|独立生成 36 层 × attention 4 + MLP 3 个投影的名称集合，与 install 及逐请求 modules 相同；totals 逐模块复算；脚本逐模块核验 hybrid fused>0 和 reference disabled 路径|
|KV数值门禁|3.9 旧 sum 与保存 NMSE 41/60 不 exact，math.fsum 0/60 不 exact；旧 gate false 58/60 原样保留，没有因 actualKV 非 exact 丢弃记录|
|聚合与全量输出|先每 clip 三 seed mean，再 20 clip 等权；480 请求、160 clip 表项来自 8 组 × 60/20，是重复记录，不是 480/160 个独立样本|
|拒覆盖/负例|现 16 tests覆盖的契约通过；新增结构负例失败留痕不通过，见第 3 节|

独立聚合未导入审计实现，直接从冻结 JSON 选择教师键并用标准库统计计算：

|候选|mean ADE m|主要相对教师损失|平均 clip ratio（非主指标）|退化 clip|超过15% clip|
|---|---:|---:|---:|---:|---:|
|teacher dense/teacher|1.2323994789272548|0|0|0/20|0/20|
|hybrid/reference/fullprefill10|1.282724219933152|0.0408347632942061|0.11294743257279687|10/20|7/20|
|actualKV4|1.2450251077612242|0.010244753466594592|0.05909973018146081|9/20|7/20|

report 的完整退化表保留全部尾部。hybrid 与 KV4 的 7 个超过 15% clip 相同，绝不能被整体约 +4.08% / +1.02% 掩盖：

|clip|hybrid 相对增量|KV4 相对增量|
|---|---:|---:|
|`44b4c663-5c9d-4c69-a5e5-15d966bd9823`|39.3773178461%|27.2045402628%|
|`60d13844-813d-4c35-a345-d59ba96aa681`|25.0483174445%|20.9111384680%|
|`7bd08ee7-3e9b-499e-b9af-c3b84aac728b`|34.4196819493%|26.5125152337%|
|`9147f7fb-a487-4be8-a038-59bdfaa295b7`|28.3636073406%|18.2361637471%|
|`914c6c62-6185-4648-b44a-51204244f1a1`|83.7804934373%|93.1313209205%|
|`b483042f-f4af-49df-93e7-59d848f34eec`|73.3613831887%|63.0386856737%|
|`fb80c64b-45e4-4253-a52d-79081fb67bad`|61.3531829910%|53.5971597355%|

教师 ADE p95/max 为 2.4410087347 / 3.6002918084 m；hybrid 为 2.5047279000 / 3.3171715736 m；KV4 为 2.4128616750 / 3.0274602175 m。按决策这些只是尾部描述；没有新增安全阈值，也不把逐 clip 15%诊断改为独立验证门禁。

## 3. 阻塞缺陷：结构无效输入无法保存失败审计

编号 `A-V2-01`；工具契约缺陷，验收 reject / pending 修复。位置 `scripts/audit_alp15_precision.py:458`、`:467`。决策 2.4 明确要求“完整性门禁失败应输出失败审计与非零退出码”，2.3 要求拒绝缺失/非法必填字段；当前实现仅保证本负例非零退出码，未保存失败审计。

合成负例仅在临时副本把教师 `records` 改为 `null`，按副本真实字节更新 teacher expected SHA，其余三输入及 expected SHA保持冻结值。该副本不是模型新结果，也没有改真实输入。SHA匹配使 CLI 实际进入结构校验，而不是仅测 SHA拒绝：

```python
import hashlib, json, pathlib, subprocess, tempfile
root = pathlib.Path.cwd()
manifest = json.loads((root / 'docs/baselines/evidence/precision_20261009/development_audit_v2/input_manifest.json').read_text())
args = manifest['argv'][2:-2]
with tempfile.TemporaryDirectory(prefix='alp15-v2-negative-') as td:
    directory = pathlib.Path(td)
    bad = json.loads(pathlib.Path(manifest['inputs']['teacher']['path']).read_text())
    bad['records'] = None
    teacher = directory / 'teacher.json'
    teacher.write_text(json.dumps(bad))
    args[args.index('--teacher') + 1] = str(teacher)
    args[args.index('--teacher-sha256') + 1] = hashlib.sha256(teacher.read_bytes()).hexdigest()
    output = directory / 'failed'
    result = subprocess.run(['/usr/bin/python3', '-B', str(root / 'scripts/audit_alp15_precision.py')] + args + ['--output', str(output)], text=True, capture_output=True)
    print(result.returncode, result.stdout, result.stderr)
    print(sorted(p.name for p in output.iterdir()))
```

实测 chunk `d49140`：CLI exit 1，stdout空，已创建 output 目录文件列表 `[]`。stderr完整关键链路：

```text
Traceback (most recent call last):
  File ".../scripts/audit_alp15_precision.py", line 458, in main
    info['record_count'] = len(data[name].get('records', []))
TypeError: object of type 'NoneType' has no len()

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File ".../scripts/audit_alp15_precision.py", line 505, in <module>
    raise SystemExit(main())
  File ".../scripts/audit_alp15_precision.py", line 467, in main
    for index, row in enumerate(obj.get('records', [])):
TypeError: 'NoneType' object is not iterable
```

根因：输入结构检查之前做 `len(records)`；已捕获的异常进入失败 inventory 时又假定 records 可迭代，从而第二次异常绕过 audit/report/manifest/run.log 写入。现 16 项测试没有覆盖这一失败留痕路径。已向主任务升级；验收者停在验收边界，没有修改执行者文件。

建议由执行者依据主任务决策以最小改动保护 record_count 与失败 inventory，新增真正的 CLI 结构负例，要求非零退出及失败产物、诊断、空 summaries，同时确认固定真实结果/模型门限不变。后续版本应使用新证据目录和新独立验收，不静默覆盖 v2 或本结论。

## 4. 非阻塞限制与未完成门禁

3.9 与正式 3.12 有 13 处 `statistics.stdev` 描述值末位差：1 处教师总体 SD，12 处 seed SD；观察到的最大绝对差约 `2.22e-16`。例如教师 SD为 3.9 的 `0.7790969513848408` 与 3.12 的 `0.7790969513848409`。因此 3.9 per_request CSV字节相同、per_clip CSV字节不同；门禁、均值、ratio和 exact轨迹仍一致。该限制不要求放宽模型门禁，也不构成更改冻结源码以追求全版本 CSV字节一致的理由。执行回报的 v1/v2 CSV同 SHA指正式 3.14 v1与正式 3.12 v2，未被此发现否定。

固定输入没有 GT或教师 FDE，教师 FDE及相对FDE均为 null；学生 FDE只是已记录字段的描述，不能独立重算 ground-truth ADE/FDE。教师和 KV退出码未嵌入 JSON，报告明确 `not_embedded_in_json`，没有补造。教师使用当时冻结教师前缀；双方新输入自主生成仍需后续验证。

20 clip反复用于开发选择，3 seed是重复测量，缺 drive/session与场景 metadata；没有独立新数据、p/CI、功效分析或采集前完整统计计划。`independent_precision_pass` 与 `Orin_gate_passed` 均 null，未称“完全精度验收”。本机来源摘要比较不能证明重新核验远端全部权重字节；本审计不能替代官方原入口、Jetson/Orin、300 ms、闭环质量、安全/泛化门禁。KV4与hybrid为分开候选，没有组合验收。

最终状态：v2固定真实输入复现部分保留；v2工具完整失败留痕契约拒绝验收，等待主任务安排执行者修复和新版本独立复验。不得从此文推进优化/模型机制或宣布完全精度通过。
