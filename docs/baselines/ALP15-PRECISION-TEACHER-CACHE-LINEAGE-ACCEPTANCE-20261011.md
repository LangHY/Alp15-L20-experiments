# 七组 teacher-cache 来源范围独立窄域验收

验收编号：A-ALP15-PRECISION-TEACHER-CACHE-LINEAGE-20261011。对应决策 D-ALP15-PRECISION-USAGE-GAP-20261010；验收者为独立 Sol 子任务 `holdout_prep_execution_20261010`。主任务编写、执行查询并归档 producer 与执行回报；本验收仅写本文件。日期2026-10-11，补证本地身份核对 UTC02:25:41（北京时间10:25:41）。

**结论：通过七组 teacher 文件的历史来源范围窄域验收，700个文件（698条原匿名记录、700条原binary记录）可在本 receipt 中签为 `bounded_used_group`。** 范围仅同一100clip有限曝光集合，不签逐PT内部clip身份。初次复核发现v2缺少历史 argv root/output绑定，已停止签发并要求另存补证；执行者未改冻结v2，新增七组绑定查询已完整审阅通过（第6节）。payload未验与完整历史coverage未通过仍不变；旧CSV/query/ledger不回写。

## 1. 实际权限、读取与证据层级

本机 `/usr/bin/python3 -B`、Python3.9.6、macOS27.2 arm64，stdlib CPU。完整读取当前 AGENTS、usage-gap决策、teacher执行回报（包括最新补证追加）、`check_newdata_teacher_lineage.py` 和 `check_teacher_argv_binding.py`；完整 JSON 解析两个查询结果，其中七份实际 source 全文分别 AST 解析及原 UTF-8 SHA 重算。完整阅读514行初始 newdata8 source，并阅读七版所有连续差异（共同文本不重复输出）；第一次未规范化行尾的差异输出被截断，已另用只读 `splitlines()` 比较完整补读，无截断。CRLF 仅用于阅读差异时归一比较，未修改原 source、未用归一后 SHA 代替原字节 SHA。

本地直接复核的内容：七份实际 producer 字节身份与全源语义、七个有序80/20名单、700个归档 stat 条目、六条 donor 边、1398条原 CSV 受限子集的所有条目、与已验收544路径交集、各 reason 计数运算。

原 provenance/audit/optimization_split/CSV/ledger 的原文仍在远端，仅有31项固定文本 SHA 与已审源码的成功查询证明。本验收**没有本地独立重算这些原文**；也未重新 lstat 服务器。已用 clip 包含关系、原 audit 的 records/角色记录、argv audit/donor 选择、同实验 provenance 字段等依赖归档远端完整断言。1398条 subset 在本地完整重算，不等于本地全 CSV 重算。

无 SSH/网络/安装/下载/GPU/模型包导入；没有执行生产脚本、torch.load、pickle 或 PT/其它二进制 payload 读取、内容哈希。本文不涉及新训练、优化、精度测试或 Jetson 运行门禁。

## 2. 冻结输入身份

证据目录简称 U：`docs/baselines/evidence/usage_gap_20261011_v1/`。

|本地实际文本|SHA256|
|---|---|
|usage-gap决策|ce674809d555c04adc7b2607fb74f6427ec624b2237ba875525f898861994a5c|
|ALP15-PRECISION-TEACHER-CACHE-LINEAGE-EXECUTION-20261011.md（含argv补证）|ef6682d62fd4437fbfc03236c819cab37457867bd969cb5dc31c5de8675cd1f0|
|U/check_newdata_teacher_lineage.py（实际v2）|d291a16e9dda34fbb1c56567746f04de8a0038d96c3a9f5277c6fef47d3e7b7c|
|U/newdata-teacher-lineage-query-v2.json|e062044eab401ab989c6183e505e69d00f663c0996e534ca7d9a8cdc730fbab9|
|U/check_teacher_argv_binding.py|0d49dd1be30fc7046dbc045ca1b7387c74780a3cdaa1163c5fba01622b52eb05|
|U/teacher-argv-binding-query.json|fdbe50b200e148508e65cd022cf934584db10c6d9468d245ed9b855157da5349|
|已冻结 activation/prefix 验收|774a8d400ab408629e6bc235c566289860b92f17ff1e58fef1eea832837bd7e2|
|U/activation-lineage-query-v3.json|0b25714cf65d718781cf9b7c1d42b39bd0e478847ca5aa4ecb85738f9c687e6c|
|U/prefix-lineage-query.json|106f600ea0722480ab11808fed773dcd499664c9078931fa8b8848f0e6e421ae|

v2 JSON.query_sha256 等于本地当前查询实际 SHA；stderr 实际0字节。归档 exit0、input_texts_unchanged=True、700末次 stat 大小/mtime 稳定断言，UTC2026-10-11T02:20:31.961982–02:20:32.456167，Python3.12.14。主任务执行回报保留实际 `python -B` 命令、offline两个flag、空CUDA、拒覆盖 stdout/stderr、退出0会话05a5e3；本机没有重跑该远端查询。v1也成功，但本文只以冻结v2及其新增 donor 字段/末次stat检查验收，不把当前v2源码冒称v1源码。

## 3. 七个版本与实际 audit→records→cache 分支

实验完整名称见下表；每组归档100项，角色为80 calibration/20 validation。全部 producer 的原 UTF-8 bytes SHA 与各组 `producer_sha256` 以及31项输入表对应实际 `code/run_interface_recovery.py` SHA exact 相等。

|实验（均 experiments/ 下）|原 producer SHA256|原 audit读取 / records覆盖 / enumerate / copy或prepare行|
|---|---|---|
|mlp_measured6p8_recovery_20261002_190306|bc5e3999b6416895b83c91aaef46178d3a596a2316daeffd24e4c3be86c045cd|164 / 286 / 287 / copy290|
|mlp_measured7_recovery_20261002_181113|69801df440a77311b355ff8cf4a42365ed06c9aab04eecff186e33a20c14a785|164 / 286 / 287 / copy290|
|mlp_measured7_recovery_20261002_182132|bc5e3999b6416895b83c91aaef46178d3a596a2316daeffd24e4c3be86c045cd|164 / 286 / 287 / copy290|
|mlp_newdata8_recovery_20261002_165238|f5f95930b99c3b31547d16acc01be992c2e6fd67d81fd6d81d5c56023585d457|158 / 259 / 260 / prepare261→save272|
|mlp_precision8_balanced_20261002_195311|60b6f7d78cdb762cb5cb0852e2783974d7e7485d248fe45d47fe3924abe0639f|167 / 292 / 293 / copy296|
|mlp_precision8_dense_20261002_195955|232589eb5a3d6357e865af278e83ffbbca3fbc786932d12a17a8614b2b579771|169 / 296 / 297 / copy300|
|mlp_rebalanced7_recovery_20261002_210458|1106c9488b8b05e11cd7212b4f722631ea803c0c80d05f3ba17b9f7d30da3850|173 / 301 / 302 / copy305|

六种不同 SHA 不是七种；两 measured 版本相同。rebalanced source 原有606个 CRLF，原 SHA 完整保留。七个 prepare 函数静态语义相同，调用 `load_physical_aiavdataset(record['clip_id'], t0_us=record.get('t0_us',5100000), ..., maybe_stream=False)`；仅为源码阅读归一行尾的函数文本 SHA 为 `798cd7880d063d58065e60628132c42d4cad03dc856fbc1e3d570a4c64d48af9`，不作生产 source 身份。

查询审阅通过的绑定断言：

- 从各 provenance.argv 唯一 `--new-training-audit` 取原 result，实际 SHA与复制 `new_data_audit.json`、provenance.new_data_audit_sha256、optimization_split.data_audit_sha256均绑定 `472a5d107e3876ce0e81727519c0c67b7c1e6f44fdb6aae90814ce32ecbda349`；原 audit 完整对象相等、status completed、dataset revision 对 ledger 相等。
- 无 training-window-audit/resume；audit.training_records80、monitor_records20各唯一互斥，records的 clip集合等于这100集合；optimization split角色集合相等、gradient_updates_on_monitor=False；100clip在 ledger.used_clip_ids。
- 本地七份 `ordered_role_clip_ids` **完整有序列表 exact 相同**，而不是从 split 排序数组重建顺序；查询直接提取原 audit 角色记录的先后顺序。原 optimization split 的 sorted role IDs 仅用于集合检查，不用于 index 映射。
- 本地31 input paths exact为七实验的 provenance/source/audit/split共28项，加原 `newdata_chunk0000_audit_20261002_165238/result.json`、固定 CSV和ledger。不存在未声明 donor 外部文本；七份 audit SHA相同，七份 split实际文本SHA也相同。

实际 source 先读取原 new audit并复制，再在 role循环里覆盖早先 teacher/window的 `records`，之后 `enumerate(records)`。newdata8 是 direct prepare→save；其余六组先读 donor缓存，以 `(role, e['clip_id'])` 建表，身份/重复/t0/完整性检查后按当前 `(role, record['clip_id'])`选取，并复制到当前 `role_index.pt`，append 后 continue，不进入直接 prepare。没有用 sorted donor文件名去推 UUID顺序；源码的 sorted glob只控制加载遍历。

这些检查证明候选**100clip有限来源范围**。复制分支内部身份检查依赖历史PT内容，未在本轮重做；本 receipt 不输出“已重新验过每个PT内部clip”的精确身份保证。相同数据范围也不能证明七实验都训练成功，更不能删除失败/中断曝光。

## 4. donor 六边和700文件集合

|recipient（简写末段）|donor|
|---|---|
|measured6p8_190306|measured7_182132|
|measured7_181113|newdata8_165238|
|measured7_182132|newdata8_165238|
|balanced_195311|newdata8_165238|
|dense_195955|newdata8_165238|
|rebalanced7_210458|measured7_182132|

本地遍历六边图无环、所有边终止在本轮 direct newdata8组；没有图外 donor。查询核 donor argv路径、reused_teacher_cache.directory、同audit SHA、角色集合、source_commit/checkpoint_revision/dataset_revision/logit_weight/expert_weight相等、donor rollout_prefix=True；每份 declared source文件集合 exact等于该 donor的80 calibration/20 validation路径。这里的 PT source_sha256 仅历史文本声明且格式合法，不是本轮计算 payload SHA。

本地完整核700项 stat：每组100不同 source，role/index集合 exact为calibration0–79、validation0–19，source exact属于该实验 teacher_cache 下对应basename；跨组不重复，bytes/mtime_ns正整数。原查询初次lstat核regular/non-symlink，目录 `.pt` 集合无多余缺失，末次检查这700项stat未漂移。**它没有末次重新枚举整个目录以检查期间新增文件，故只能签“已列700文件末次stat稳定”，不能称全目录反向新增/删除稳定复验。**

## 5. CSV子集与差异700/698

原 CSV SHA固定 `f398e3f054620d37544217e766fbafba2d6d052f084df75e22df9f41259a09c7`；ledger固定 `0697cec28cd36ca841fe75ddeb95f1037269d69871bc2bda59ae831947e870d1`。查询读取全CSV后筛选700 source的两reason，输出所有1398条原受限记录。本地对全部子集按reason/source Counter重算：binary700 source exact为700文件集合、每path恰一次；anonymous698 source为其子集、每path恰一次。原查询的总数+唯一性断言配合本地exact集合复算满足该子集守恒；不是只比较698/700总数。

唯一两个原无anonymous记录的路径：

- `experiments/mlp_newdata8_recovery_20261002_165238/teacher_cache/calibration_0.pt`
- `experiments/mlp_newdata8_recovery_20261002_165238/teacher_cache/calibration_1.pt`

这两个有binary记录，不补造anonymous记录，不把700硬改成698文件。旧544 source本地重算后与新700无交集，联合1244不同文件。

|维度|原记录数|原已签范围|本次bounded_used_group|累计后的来源待分类数|
|---|---:|---:|---:|---:|
|anonymous|1892|544|698|650|
|binary来源|4104|544|700|2860|

第6节补证通过后，本 receipt 正式签发本次698/700的有限来源范围；累计匿名1242、binary来源1244，分别与650/2860余额守恒。两个类别共享文件，不相加为样本量。**全部4104条payload仍未验，700个文件也不是700个clip。** 外部63177、revision171/247、symlink20本次均未处置，不能借同目录或同checkpoint豁免。

## 6. 发现缺口→另存补证→同实验绑定通过

usage-gap决策第3节要求同一冻结实验的实际 argv/输入引用和输出目录绑定。原v2从 provenance.argv解析audit/donor，但初次复核发现未校验：

1. 唯一 `--output`经受限规范化后 exact等于当前实验 `out`；
2. `--root`绑定真实项目root，以及 argv[0]的实际生产入口归属与冻结 code SHA之间的明确证据（入口可位于原source目录，不要求臆造其等于 snapshot路径）；
3. 对应受限 argv 字段的归档值及其 provenance实际SHA，以供独立审阅。

这些字段不在v2 JSON，不能由本验收补猜；代码同名SHA本身不足以证明输出归属。执行者接收缺口后另存最小拒覆盖只读 `check_teacher_argv_binding.py`，未改冻结v2/query/provenance。新增query源码/完整JSON均已读；实际SHA见第2节，归档4bf62f exit0、UTC02:23:54.709944–02:23:54.716756、stderr实际0字节。

本地逐组完整复算新增七条binding：父v2 JSON实际SHA exact相同；新增读取的各 provenance SHA exact等于父31项身份表；七实验集合exact、无重复遗漏；各 argv 的唯一 `--root`、`--output`是绝对路径，分别exact为项目根与该实验out；argv[0]记录 `compression/run_interface_recovery.py`；实际冻结snapshot SHA exact等于父producer及provenance.code_sha256。source初始版本明确先从 `Path(__file__).parent.glob('*.py')` 复制code到当前绝对out，再记录这些snapshot SHA，与这条归档关联相符。父query与provenance前后SHA稳定。

因此本窄域所需历史输入/输出关联缺口解除；**只签归档argv脚本身份、绝对root/output及同实验冻结snapshot关联，不从相对argv[0]猜历史cwd或绝对执行入口，更不宣称当前compression同名源码还等于七次历史版本。** 历史绝对cwd、当前源码身份或更强运行时证明未核，但本有限来源 receipt 不依赖它们替代已冻结snapshot。

本次未复算固定 metadata clip→chunk；已used和新增排除为空来自归档ledger查询，不能叫本地metadata逐条验收。无新增clip释放，旧21块保守排除不撤销。若新证据出现额外输入clip，只在旧排除基础取并集并按固定metadata复验；未闭合来源不得冻结独立名单。

## 7. 留痕、可签范围及下一步

全部本地只读命令 exit0：`rg --files`定位、`sed -n`完整读取、`/usr/bin/python3 -B` heredoc以json/hashlib/ast/difflib/Counter做文本与集合断言、`test ! -e`确认新验收目标不存在。工具会话日志：baf03c（完整AGENTS）、4d0e83（完整当前决策）、b96699（完整teacher回报）、643d1f（完整query源码）、ab4183（完整JSON解析schema/七source身份）、bcb42c（完整514行base）、56de93（首次差异被截断，不作完整阅读证据）、ded01c（全部版本完整差异补读）、e0db63（700stat/1398subset/六边DAG/旧544无交集/100有序名单及31身份断言）、968674（各版行号/prepare身份/本地输入SHA）、9e6636（31输入路径exact及缺口定位）、1424e7（目标文件不存在）、66453c/55f144（补证源码/完整JSON）、7fb1e7（完整最新执行回报）、672b03（补证七条binding全部SHA/字段/父子关联断言）。它们是实际工具会话留痕，不冒称另有独立归档stdout。远端执行05a5e3/4bf62f及取回由冻结执行回报记录，不冒称本验收执行SSH。

可签：本地文本身份和完整静态源码验收；归档700不同文件、同100clip有限范围及组内donor无外溢；同实验历史argv root/output与snapshot关联；1398条归档原CSV子集exact计数；本次698匿名/700binary来源维度的 `bounded_used_group`。**不可签：逐PT内部身份、payload正确性、七实验训练成功、adapter/checkpoint全部祖先、实时远端目录状态、全CSV/ledger本地独立复算、完整scope、drive disjoint或新精度/性能通过。**

下一步继续核剩余650匿名、其它binary来源2860，以及其它scope缺口；不修改旧receipt、query、CSV或ledger。coverage_complete=False、drive_disjoint_verified=False、最终独立名单未冻结仍不变；全部payload4104仍未验，剩余scope与统计方案必须另行完成既定门禁。
