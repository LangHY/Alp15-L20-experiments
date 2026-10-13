# import 指纹引用角色独立窄域验收

验收编号：A-ALP15-PRECISION-IMPORT-REFERENCE-20261011。对应决策 D-ALP15-PRECISION-USAGE-GAP-20261010 §4.4。独立验收者：Sol high 子任务 `holdout_prep_execution_20261010`；查询、证据与执行回报由主任务编写/执行/独占。本验收仅写本文。日期2026-10-11。

**结论：通过6245条原external引用的“source指纹字段、不是该处数据输入引用”角色窄域验收。** 可在本 receipt 对这6245条出现位置签 `proven_non_usage`，但 scope 严格限于 `this_reference_is_a_code_fingerprint_not_a_data_input`。这不是对底层代码资产、其所有历史运行/输入或全usage的非曝光判定。原CSV/query/扫描框、模型与排除集合不修改。

## 1. 实际读取、环境与证据层级

本地 `/Users/USER/Documents/ChatGPT/Lighten_VLA`，`/usr/bin/python3 -B`、Python3.9.6/macOS27.2 arm64、stdlib。完整读取冻结执行回报（含最终日志取回补充）、v1失败查询、v2查询、v1完整失败stderr；完整解析3,922,806字节v2 JSON及全部6245 dispositions。producer全文 UTF-8 bytes 与前轮已完整阅读的本地 `producer_v3.py` exact相等，再独立 AST解析、读取实际 import函数/闭包/两个分支上下文。未执行producer或远端query。

本地直接核验：所有当前文本实际SHA、完整stdout字节数/JSON终态、失败traceback实际内容、两个真实空日志身份、实际producer源码身份与AST写入链、6245条原受限记录的字段/唯一出现位置/计数守恒。原CSV、scan-manifest、full20原result全文未传回；每条位置对应原result路径key的正则/hash相等、原CSV来源及前后SHA稳定是**已审完整源码的归档远端断言证据**，不是本地原文逐行重算。

没有SSH、网络、模型/GPU、安装、下载、生产脚本执行、payload读取；未打开、stat或重哈希任何 import引用目标。读取已有producer文本不等于跟随receipt里的引用目标。

## 2. 冻结身份与日志

证据简称U：`docs/baselines/evidence/usage_gap_20261011_v1/`。以下SHA为本地实际重算。

|文本|SHA256|
|---|---|
|usage-gap决策|ce674809d555c04adc7b2607fb74f6427ec624b2237ba875525f898861994a5c|
|ALP15-PRECISION-IMPORT-REFERENCE-EXECUTION-20261011.md（最终补充）|3e43f52878cdbbe3e401ddd346e0e16aa682697456d7a7df2e20ae6d7344a847|
|U/check_import_reference_roles.py（v2）|4095863dfb095bc56fa0b262aabaf4d698b4a88a4cb98bb4f3df00ef3e0723a1|
|U/check_import_reference_roles_v1.py|19d96d860dc5a3f7bc60372810e37c98948895fe9a86d56e88e0a8e7a30f379a|
|U/import-reference-roles-query-v2.json（3922806B）|0eb4d496584f735366b348938ca9b9f1fffdae31578c82fefbe8f0904c23133d|
|嵌producer全文 / 已完整阅读的producer_v3.py|fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd|
|U/import-reference-roles-stderr.txt（v1失败原文）|229126293a2ecac8de9c03d4be3a1b3ece9ce0e36bc655effa83110651880ede|

v2 stderr与v1失败stdout实际为空；分别为 `import-reference-roles-stderr-v2.txt`、`import-reference-roles-failed-stdout.txt`，两者SHA均 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。本轮曾观察传输中v1 stderr占位0字节，**没有采用它**；最终完整失败文本重算与主任务传回远端SHA一致后才采用。

v2 JSON.query_sha256 exact对应当前v2源码；归档 UTC02:34:43.575897–02:34:43.934501、exit0、input_texts_unchanged=True、coverage_complete=False、drive_disjoint_verified=False。远端执行命令/空CUDA/offline/拒覆盖/解释器及成功会话ca94f8，在冻结执行回报完整保留。没有 `-O`，源码assert门禁有效；本验收未亲自执行该命令。

## 3. 实际 source 指纹产生链：AST独立通过

`imported_source_receipt` 在冻结producer119–152行，AST只有一个同名函数：遍历 `list(sys.modules.items())`、取module.__file__，普通来源以canonical路径key写 `sha256_file(canonical)`；动态torch来源绑定持久source snapshot和generator来源；虚拟namespace绑定实现模块及来源SHA。特殊分支仍是代码来源指纹/实现来源记录，不能简化成所有key都是普通 `.py` 文件，更不能按扩展名豁免。

唯一嵌套 `source_receipt` 在276–277行，直接调用该函数并传result内snapshot/dynamic/virtual来源字典。独立AST核实际赋值，而不只采信查询的字符串包含检查：

- start字段在342行：`result['actual_import_sha256_start'] = source_receipt()`。
- final字段有两处实际赋值：378行是preflight分支，529行是正常运行分支，均为 `source_receipt()`。
- 267–271行save将同一个result以JSON保存；没有把这些字段的值改作load dataset的输入。

因此这些receipt字典key在该字段的用途是**为已import的源码/实现来源记录指纹**，不是该JSON位置声明数据加载参数。历史producer执行过hash目标文件是旧生产行为；本轮查询没有执行它或重新读取目标。底层被import的函数仍可能读取数据、训练、选择或校准，这个角色证明对这些行为不作豁免。

## 4. 原记录绑定与6245/6271差异

远端查询固定读取四个入口、逐项核实际文本SHA并在输出前再读复核：

|固定原输入|SHA256（归档查询断言）|
|---|---|
|unresolved-assets.csv|f398e3f054620d37544217e766fbafba2d6d052f084df75e22df9f41259a09c7|
|scan-manifest.jsonl|2177978ecf1ab7b3efe8150fb6fcfa7aabe1d60b32b4354584c24aa28b06014b|
|jetson_300ms_20261009_hybrid_full20_v3/result.json|a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6|
|hybrid_m1_code_v3实际producer|fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd|

查询核原result.code_sha256精确绑定producer实际路径；scan-manifest中该result路径唯一SHA等于实际result。按原result的一基行号识别两个顶层字段范围，逐行JSON解析该行唯一path→digest，校验其值等于完整result相应字典且digest为64位小写hex。字典行总计start2839/final3432，共6271。

随后只筛原CSV中该source的external记录；仅当location命中上述字段行、原扫描正则在该行恰一个路径、正则所得path exact等于该行JSON key、`sha256(path.encode())` exact等于原reference_sha256，才输出该原记录与role。保留原source/location/reference_sha256字段，不按目标路径、目录前缀、扩展名或exists去重/免检。

本地完整复算归档6245条：source均exact为上述full20 result、reason均external、location格式合法且6245个不同位置；reference_sha256/declared_import_digest格式合法，scope/disposition/两个False flag逐项正确；`(source, location, reference_sha256)` 恰为6245个不同组合，全部归档条目均已复核，不外推未归档记录。**归档中没有原path key原文，本地不能独立重新计算“path key→reference hash”；这项严格相等来自远端全量query，而非本地原result重算。**

|字段|字典条目（远端原文断言）|原CSV external exact命中（本地归档复算）|未命中原external|
|---|---:|---:|---:|
|actual_import_sha256_start|2839|2827|12|
|actual_import_sha256|3432|3418|14|
|合计|6271|6245|26|

26是字典记录未命中原external类别的差额，不补造CSV行，也不进一步判其原因或另给非曝光结论。6245条共有3418个不同reference hash，其中2827个hash在不同字段位置重复；两次出现均保留。6245是原引用**出现记录数**，不是6245个不同代码文件、数据资产或clip。

## 5. 守恒及可签范围

|原external范围|原记录|本receipt角色已分类|角色仍pending|
|---|---:|---:|---:|
|本full20 result|6280|6245|35|
|其它source|56897|0|56897|
|合计|63177|6245|56932|

`2827+3418=6245`、`6245+35=6280`、`6280+56897=63177`、`6245+56932=63177`全部本地断言通过。本源35和其它56897未处置；不得把6271替代6245导致错误余额56906。没有跟随引用目标，故**不是对所有这些目标代码资产本身的来源使用范围签发**，这些代码可能带来的其它历史数据输入仍需scope复核。

本 receipt 允许6245记录进入“引用角色已接受”余额，只签限定scope的 `proven_non_usage`；原query/CSV/ledger不回写。之前匿名剩650、其它binary来源剩2860不受此查询改变；全部4104条binary payload仍未验证。无新增/释放chunk，旧保守排除不撤销；coverage_complete=False、drive_disjoint_verified=False、最终独立名单未冻结、新数据精度未通过。

## 6. 失败留痕、限制和下一步

v1实际源码93行误断言 `len(dispositions) == 6271`；本地完整失败stderr明确这一行AssertionError，失败stdout真正为空。v1 exit1来自执行回报会话5feaca/4c6b07，不将本机文本阅读称为远端执行。完整v1→v2差异审阅确认仅复用result_lines、改为实际命中数与相应余额/按字段计数；没有放宽逐行JSON key/正则/hash精确匹配，没有改固定输入SHA。v2的非零上界与唯一性断言由本地归档exact6245复算补强，不依靠6271目标数凑齐。

取回期间主任务先遇半文件JSONDecodeError与SCP stderr停滞；本验收先报告未完整日志/0字节占位风险，未使用不完整文件。主任务仅停止本次三个本地传输进程，随后SSH文本读取848cb5退出0，按远端文本SHA用apply_patch归档小日志，最终本地SHA匹配。本文只记录已审冻结回报和本地终态，不冒称亲自操作这些进程，也不冒称SCP完整成功或远端JSON损坏。最终日志补齐，失败日志复核不再pending。

实际本地只读命令：`rg --files`、`sed -n`、`shasum -a 256`、`/usr/bin/python3 -B` heredoc（json/hashlib/ast/Counter/difflib）、`test ! -e`；全部exit0。工具会话日志：8afaa8/a980a9（初始完整执行回报/v2源码）、139882（完整v1源码）、e05d77（完整JSON解析/producer函数/输入身份）、8fdbdc（6245全subset和AST赋值、字段计数、守恒）、5647fe（producer闭包/两分支上下文；其中传输占位日志未采用）、93d594（v1/v2完整diff、3418hash/6245位置、AST补强）、161152（完整最终冻结回报）、4ed807（最终v1完整traceback）、9dd253（最终回报及三日志SHA）、1b10bf（新验收目标不存在）。它们是实际工具会话留痕，不冒称另有本地独立运行远端query的stdout。

可签：已冻结6245原出现记录的源码指纹字段角色；producer实际source SHA/AST生成链、归档逐行精确匹配与守恒证据；失败历史及最终完整日志身份。未验：原CSV/scan/result全文本地复算、引用目标内容或使用、底层代码从未加载数据、全部外部scope、全历史独立性、payload、精度或性能结论。

下一步按原usage-gap决策复核本源35及其它56897引用，并继续匿名/binary/revision/symlink/未扫描文本等scope缺口；每条仍按来源角色与同实验输入绑定，不用路径前缀批量豁免。全coverage与统计方案门禁未完成前不冻结独立名单，不启动模型实验或下载报价。
