# 精度留出集 usage 缺口分类复核决策

决策编号：D-ALP15-PRECISION-USAGE-GAP-20261010。任务与证据基点日期：2026-10-10；实际签发跨午夜，2026-10-11约00:03（Asia/Shanghai，UTC 2026-10-10 16:03）。按原授权保留20261010文件名。决策者：Sol high 独立决策子任务 `audit_v4_acceptance_20261010`。状态：**批准最小只读分类复核；coverage 未通过，最终独立名单不得冻结**。

## 1. 目标、范围与所有权

目标是核实既往训练、校准、监控、开发、选择、smoke、仿真等实际/引用数据的范围，使下一批独立精度数据不复用已见块。不是新模型实验，也不重新设计统计方法。承接 holdout prep v1/v2/v3 决策；不改冻结 v3，不重跑或覆盖其正式 census。本决策不替代官方原入口、Jetson 原始模型、设备资源/300ms 或闭环门禁。

风险：高，错误豁免会改变独立性结论。决策者只写本文；主任务是远端只读执行、补充证据目录和执行回报的唯一所有者，另一 Sol high 独立验收补充 receipt。先登记所有权再写；同文件不得并发。既有脚本、测试、决策、结果、manifest、日志、候选表、模型、数据均只读。

允许：使用已有 CPU Python/stdlib，读取明确的 UTF-8 源码、JSON/JSONL/CSV、日志及本节指定补充文本；对二进制仅 `lstat/stat`、文件名与已有文本 manifest 声明作核对。对源码/文本作 SHA256；不执行生产模型脚本、不导入模型包。新建拒覆盖的小证据目录和最小一次性只读查询，不新建通用审计框架。决策子任务没有 SSH/网络/安装/下载/模型权限；主任务只沿已授权远端 CPU 路径执行，不由本文新增网络或大资产传输权限。

禁止：`torch.load`、pickle、safetensors/ZIP/视频/NumPy payload 读取、二进制全文哈希、模型/GPU/训练/推理、安装下载、改动运行中的其它任务、按目录名或文件排序臆测 clip、为绕过门禁而补造历史 manifest 或修改 coverage 布尔值。已有 manifest 的二进制 SHA 只是历史声明，不冒称本轮重新核验 payload。

## 2. 已核事实与证据边界

本地完整解析四份小证据及远端回报终态段；本决策没有读取远端 46MB usage-ledger 或逐项清单。下面区分本地复核与主任务远端报告，不能把报告当本决策已经逐项验收。

本地目录 `docs/baselines/evidence/holdout_prep_remote_20261010_v3/`：

|文件|SHA256|
|---|---|
|run-manifest.json|022ae526670625ff44c1d69b7507bb1925e0a91cedd25442573cb9ccd29c3ff2|
|metadata-counts.json|64ac2b3035a3be42830c6b769a75f900a099a35dc03985443a22945041c65484|
|report.md|429b279db6ce66c01e5c3a50fe12ddab6b4ff2ebe37a31a71726121c0de71e30|
|census.log|beeee2467ee85f18833625d5b85e17aeda18b24dbf35bf79456e2c827259f7b4|

本地读取的 `ALP15-PRECISION-HOLDOUT-PREP-REMOTE-EXECUTION-20261010.md` SHA 为 `dfe1871697724b509eb5d0dede4c3b84dee9c6aab73f736d3f8c99ae881c24ea`。其实际命令、session53684/d4cfbc 终态为 exit0；manifest 独立确认 `completed_usage_census_pending_coverage_review`、diagnostics=[]、draft_generated=False，UTC 15:47:18.150641–15:49:47.818446。扫描稳定、声明目录清单完整、声明文本解析完整只针对冻结扫描契约，**不等于历史 usage 覆盖完整**。

冻结工具 SHA `ccebd4d237cf5229a422844579e5871405291e0f01f957f5600fda8148f04b9c`；tests SHA `c102a6f54abe950a7a58acc94aafe459d3b0126aa87b6e419429c890243ed5a5`；v3 决策 SHA `e53284d0ebcba64787b4f6ca4f5d60bbef8e779f62f29d7d1845b1910c1731f8`。dataset revision 固定 `33f9bf447ed3bcb7d545ce13f4226f824214fafb`，三 metadata SHA 与冻结决策/manifest 一致，before/after 字典完全一致。CSV 的 bytes7076/SHA `61105260827f5cf193c6c3ecbfed13b185432d04899d52bff3a7430ab4c0f304` 前后一致；before 多 `git_blob` 而 after 无该键，不用整个字典不等误报来源变更。manifest counts 与本地 metadata-counts 全结构一致，report/counts SHA 与 artifact_sha256 匹配。本子任务未独立读取另外七项远端产物，仅记录主任务已重算九项 SHA 的回报。

远端实际 Python3.12.14/Linux6.8.0-87-generic x86_64/glibc2.35，pandas2.3.3/pyarrow22.0.0/physical-ai-av0.2.0，offline 两 flag 与空 CUDA 控制均 True、official_api_called=False。当前映射219个引用 clip、保守排除21个 chunk；after_usage_exclusion=304075 clips/3125 chunks，test 候选池61301 clips/633 chunks，radar 缺失29771。全 metadata306152 clips/3146 chunks/radar 缺失145391。61599是未追加 usage 排除的旧 test 数，不能作当前余额。

主任务远端清点报告的待分类记录：匿名 cache1892、binary_payload_not_verified4104、external_path_reference63177、symlink_not_followed20，另有 dataset revision_conflict171、foreign_or_unknown_revision247。四类可能重叠（匿名 `.pt` 同时是 binary），外部引用按出现位置记录、不是63177个不同目录/资产/clip；不得相加作未知样本量。

冻结源码 `TEXT_EXTENSIONS` 仅 .json/.jsonl/.csv/.txt/.log/.md/.yaml/.yml/.toml/.sh/.py；`file_counts['text']` 是额外累加的**已扫描文本文件总数**。主任务实际 inventory 确认 `.text` 后缀0，而不是4542个 `.text` 被漏扫。ipynb4/patch2/rst1 确实在 inventory 中、没有按文本读取。其它未声明后缀/无后缀文件也须作范围分类，不能只补这七个便声称所有历史文本完整。

正式 `usage-ledger.json` 声明 SHA `0697cec28cd36ca841fe75ddeb95f1037269d69871bc2bda59ae831947e870d1`；`scan-manifest.jsonl` 为 `2177978ecf1ab7b3efe8150fb6fcfa7aabe1d60b32b4354584c24aa28b06014b`；`scan-final-inventory.json` 为 `fc608fb693fb047e6aa45675b2af1e5ea1ca0d0dc3140b115bc90680e41b5052`；`unresolved-assets.csv` 为 `f398e3f054620d37544217e766fbafba2d6d052f084df75e22df9f41259a09c7`。这些文本是远端补充查询的固定入口，不读取候选大表或 binary。

## 3. 归因门禁：必须是同一冻结实验的可证链

允许两种归因，均须绑定生产者实际 source SHA、同实验 result/provenance/manifest 的路径与 SHA、实际 argv/输入引用和输出目录：

1. 显式条目：文本直接给出 clip_id、对应 cache/file 路径、role/t0（存在时）与目录；逐一校验 clip 在固定 metadata 中及其 chunk。JSON 顶层 clip_id 对子 `activations` 的关系只有在**该版本**写入函数明确证实时才成立，不能给任意有顶层 UUID 的对象套这个规则。
2. 生产者确定性索引：冻结源码明确 `enumerate(records)` 如何形成 role_index 文件，且该次分支使用的**原始有序 records/list** 在同实验审计文件中完整保留，输入 SHA 与 provenance/argv 一致。核对 role、起点、连续索引、条目数、重复/遗漏、全部实际文件的 exact path 集合，无多余、无缺失、无 symlink。列表只证明整组数据范围时，可保守排除整组所有 chunk；没有逐项顺序证据时不输出伪造的 file→clip 映射。

禁止从另一实验的 split、sorted UUID、另一目录的 calibration/validation 数组、同名脚本当前版本或邻近时间推定该次顺序。计数相同不是来源证明；source SHA 只匹配而分支/输入未绑定也不足。失败/中断实验产生的缓存仍算曝光，不能因 result 非 completed 豁免；明确证明捕获/读取根本未发生才可降级。映射到任意旧训练/校准/选择/开发角色均保守 whole-chunk 排除，不因旧文件标为 holdout 允许再用。

## 4. 最小分类步骤与逐项 disposition

主任务新建 `experiments/precision_usage_gap_20261010_v1` 前先确认不存在，排他创建；若已有目录则停止并保留，不覆盖。只保存本次一次性查询源码/命令、原始 stdout/stderr、环境/退出码和小 receipt；不往旧 census 目录补内容。正文回报与小证据由主任务独占；本决策不要求新建常驻工具或重跑全 census。

### 4.1 固定入口和守恒盘点

先核旧四个入口文本 SHA 与 manifest，记录开始 UTC、解释器、offline/空CUDA、查询源码 SHA。完整读取 JSON/CSV 文本，按 `reason` 和规范化相对 source 路径分组；外部引用保留 `(source, location, reference_sha256, 原始出现序号)`，不要去重后丢弃重复记录。只输出计数/安全相对路径/哈希/必要 schema 字段，不打印任意日志原文、凭据、私钥或外部敏感路径。

为每项赋唯一 disposition：`attributed_used`（来源链闭合、保守排除块）、`bounded_used_group`（无法逐文件归因但同实验完整有限曝光集合可证、排除集合所有块）、`proven_non_usage`（可证不含/不新增本项目数据曝光）、`pending_evidence`、`blocked_scope`。原因类别可重复关联同一 source，但每条原始记录必须有唯一处置；分别验证1892/4104/63177/20条的分类数守恒、跨类交集和实际 unique source 数，保留未分类列表。不得先制定需要凑出的豁免数量。

### 4.2 匿名 cache：优先老师/prefix 和 decode activation

以生产实验目录逐组查 `provenance.json/result.json/split.json/optimization_split.json/new_data_audit.json/training_windows.json/teacher_calibration.json` 及其 argv 明确指向的文本。不是要求所有实验都有这些文件；缺少实际需要的源名单即 pending。读取实际固定 `code/` 源码（文本），比较该实验记录 SHA，再分析写入/复制分支。binary 只登记 path/bytes/mtime/is_symlink，不打开 payload、不把存量声明 SHA 记成实测 SHA。

本地已读 `docs/baselines/evidence/jetson_300ms_20261008/source_snapshot/run_interface_recovery.py`，SHA `1106c9488b8b05e11cd7212b4f722631ea803c0c80d05f3ba17b9f7d30da3850`，相关行164–195/275–322：教师 records 可来自 teacher[role]、training window audit 或 new_data 的 training/monitor_records；保存 `role_index.pt`，复用分支按当前 record.clip_id 从 donor 取再复制至当前 index。**该本地快照不自动代表1892个缓存的生产者版本。** 必须逐实验核实际 source SHA 与分支输入。donor 的 clip_id 在 PT 内检查过是生产源码行为，本轮禁止复现 torch.load；没有 donor 显式文本名单/范围链时保留限制，不能以旧成功日志替代逐文件归因。

两个 ref2 decode 目录可先做窄域归因：主任务 safe schema 查询 `9cb7ff` 报 v2/v3 schema、各252 manifest条目/252 `.pt` stat/252唯一basename、路径非symlink、sha格式合法、code/runtime与各自 result 一致；top clip_id=`1f62af5c-2991-47ba-a7c7-0a6f37d8bad3` 等于各 result.clips[-1]、已在used_clips。因此504个文件**有可核的候选来源链，不代表504个独立 clip，也未验 payload**。

本地分别完整解析两个 result 和 final receipt，SHA为 v2 `1b60bab73f88865e873f4bde777c3298f04a47286c6de5532a65df780b5770b8`、v3 `02a163ddbadf9aec6c778d36a35e2e7b76515d17b8a384a35e55a9ace75d2356`；声明 activation manifest SHA 分别 `7c269f8b0457cb6c637e7d842340ddddefda3f93cb04bc7a227aae44bc345ebc`、`3975408023cff735c3fcd9edb76e190be654359805fcbab7f6ff15d6a3c4bf04`，与各自 receipt 对齐。v3当前本地 producer `scripts/run_alp15_awq_hybrid_eval.py` SHA `fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd`，155–201行明确 capture保存 rows，clip为 provenance.clips[-1]，最终 manifest把该clip与activations绑定。v2 result声明 source SHA `a2ccec42a954be1d8325c89afbc32337193808265121448ec770ec92a792e887`，**须读取其实际冻结 snapshot 写入函数核验**；不能从v3推v2，也不能因v2后续probe失败把捕获曝光删除。正式 receipt 应补全该 safe query 的命令、日志及 SHA，再决定504的处置；本决策仍记待执行归因。

### 4.3 其余二进制：不因名字全部豁免

逐 producer/实验组区分数据缓存、激活/分数、训练adapter/checkpoint、packed权重、原始交付数据、静态包/构建产物。derived模型/分数也要查其校准/训练/选择数据；不是名字叫 model/weights 就与数据独立性无关。仅当固定源码与同实验 manifest证明某文件是确定输出、且其全部输入数据曝光集合已经闭合并排除，才可归入 attributed/bounded组；官方静态资产或依赖需明确来源/版本/用途证据才能归 proven_non_usage。

对 ZIP/视频数据保留交付文本中的 dataset revision/chunk/path/bytes/SHA声明；只证某chunk交付时保守排除该chunk，不声称内容/报价重新校验。未知二进制、目录来源混合、manifest无输入名单时 pending，不读payload寻捷径。不要求逐tensor数值正确；本任务验的是数据曝光范围。

### 4.4 外部 refs、symlink 和未扫描文本

外部refs先按源文件SHA/行号/引用hash归组，读取该行的受限上下文做语义分类：依赖/源码路径、日志历史路径、非文件的表达式/模板误命中、同一资产的明确挂载别名、实际外部数据/实验输入、未知。归类重复条目可以共享一份证据，但不能仅用 `/tmp`、site-packages、model等前缀全量删除。路径别名须由本实验argv/manifest/实际受限stat证明；不要展开未经确认环境变量或根据字符串替换视作同一资产。真实外部输入若需范围之外读取则 blocked_scope，列明所需最小新目录/文本和理由，等待明确范围授权。未知数据revision的171条与其它revision247条分别绑定 repository/资源身份核对，源码/模型revision不能无条件当dataset冲突，反之也不能全部当无关。

主任务随后传入只读查询 `601440`：171条dataset revision标记的当前源文本SHA与冻结扫描行一致，均行文本、无JSON结构字段，正则捕获token皆identifier_like。它只说明这171个标记的语法形态，**不能据此判全为误报，也不证明运行时revision正确**。`ced74c` 重算原正则且reference_sha256 Counter一致：hybrid full20 result（SHA `a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6`）6280条refs中6275为项目内但声明扫描根之外已存在路径，3为项目外不存在或非路径，2为项目外存在；LoaderPlan.json（SHA `17f8fa6d313faaeab1e75b01867297b5111a4a29e626ae8e2ac6c125657f6fb5`）3006条均项目内扫描根之外存在。故external不是“项目外”的同义词、exists不是“无数据曝光”的证明；这9286条仍须按来源角色/同实验输入归类，不能将两文件结果外推剩余refs。本决策未执行这些远端查询，原始命令、querySHA和stdout仍由主任务归档。

20个symlink不自动跟随、不读取target payload。记录link相对路径、原target字符串hash、内部/外部/断链/未知及是否与inventory已有target条目对应。若目标是已声明并已完整扫描的真实目录/文本，且link语义及实际target可证为重复入口，可引用既有覆盖证据；循环、动态/漂移、外部或未扫描target保留blocked/pending。确需新增外部target范围，先提交明确目录和允许文本类型的小范围补决策；不得把resolve成功当已覆盖。

额外读取 inventory中实际4个ipynb、2个patch、1个rst：ipynb只JSON解析 cells/source 与必要outputs的文本，不执行cell、不解码base64/附件；patch/rst只严格UTF-8文本引用扫描。记录前后文本SHA、路径和新增clip/ref/revision/parse失败。其它后缀/无后缀先按清单确认是否存在实际usage文本（如配置、执行回报、源文件），再针对明确文件补扫描，或证实静态非usage；不能依据未知后缀推定二进制无关。`.pem`等疑似密钥不读取，只有位置/风险，不输出内容。

### 4.5 最小 receipt 与排除集合增量

无需复制46MBledger或所有缓存。远端保存一份小 summary JSON和逐项 disposition CSV/JSONL（量大留远端），至少：原入口SHA、查询codeSHA/实际命令/UTC/环境/exit、每类计数守恒/交集、逐项证据source与SHA/location、producer/sourceSHA、同实验manifestSHA、名单原路径/SHA/count/role/index规则、output路径/stat、状态/原因、payload_verified=False、新增引用clip/wholechunk集合、未解决数与最小下一步。需要补审的原始文本只取明确小文件，取回SHA核对，不取回PT/权重/候选大表。

增量排除集合只能在旧21块基础上取并集；已被保守排除的块不因后来判为误命中自动放回。用固定metadata映射验证新增clip与chunk，列出旧/新增/并集及train/val/test余额的变化。若group只能给有限“可能已见”集合，排除整个集合；若未知集合无法限定到候选池之外，不能冻结任何受其潜在影响的名单。不要为了维持61301这个余额而调整范围。

## 5. 验收、停止与当前结论

逐项门禁：

- 入口身份一致、命令/环境/退出码/完整日志/前后文本SHA可追溯；输入并发变更、不完整读取或异常立即停止该组，保留失败证据。
- 四类原始记录分类守恒且每项可追溯，不以抽查代全组、不用计数/名字代source绑定；补扫描和revision疑点也有处置。新增发现不静默忽略。
- 全部来源链闭合，或有限曝光范围保守排除；无法限定的潜在数据输入维持pending/blocked。scope外目标、需payload、缺生产版本/同实验有序输入等触发升级，不自行扩大权限。
- 不修改旧census及工具coverage字段。即使分类复核通过，也只签发**明确roots、文本类型、历史实验时段和未读payload限制的独立scope receipt**；全局coverage需要独立范围验收，drive_disjoint仍另pending。scope之外未知历史不能靠声明消失。
- 统计草案仍 `draft_reviewed_not_ready`，固定N/抽样框、chunk统计目标/校准、GT时间坐标和失败分母等修订未解决；分类完成不是统计ready，也不是新精度pass。

回滚无需改原始资产：新证据出错就留存失败目录并拒绝使用其结论，另建拒覆盖版本；不删旧日志、不补造缺失生产者、不覆盖manifest。发现来源冲突、范围扩展或需要模型/payload读取即交Sol high新决策。

**当前结论：真实CPU census已经完成；批准先复核504个decode activation及老师/prefix cache的同实验来源链，再分类其它缺口。匿名1892/二进制4104/外部引用63177/symlink20尚未逐项闭合，coverage_complete=False、drive_disjoint_verified=False不变，最终独立名单未冻结，新数据独立精度未通过。** 本决策本身不是usage覆盖验收。

## 6. 本决策的本地核验留痕

cwd=`/Users/USER/Documents/ChatGPT/Lighten_VLA`，本机macOS27.2 arm64，`/usr/bin/python3 -B`3.9.6/stdlib；无SSH、网络、模型、安装、下载。实际只读命令：`sed -n`读取AGENTS/远端回报/report/相关源码行，`rg --files`与`rg -n`定位producer与历史冻结SHA；`shasum -a 256`核上述文本；`python -B`通过json.loads完整解析两份小JSON、两个ref2 result/receipt，并比较counts、metadata/CSV前后身份及本地两项artifactSHA；新文件不存在检查exit0。没有运行生产模型源码。

全部读取/查询退出0。关键工具日志：c4f388（84行远端回报）、91321d（完整小report）、ac98e0/559e24（冻结扫描契约）、4fb231（四证据及工具/回报SHA）、9207f4（完整JSON解析后的schema/终态/计数）、79a782（counts/metadata/CSV及artifact对照）、7886d9（v3生产者关联函数）、63bba3（老师cache索引/复用分支）、7b2755（两个本地producerSHA）、4379e8（完整AGENTS）、5a5e7a（独立ref2结果/receipt链）、08e729（本机环境）、48f327（目标新文件不存在）、db51bd（写入后冻结script/tests SHA仍一致）。它们是工具会话日志，未冒称另有已归档stdout文件。主任务的远端9cb7ff/601440/ced74c为传入报告，补充执行者须归档其原命令/日志与查询身份供独立验收。
