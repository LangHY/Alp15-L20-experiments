# activation / expanded20 prefix 历史曝光来源链窄域验收

验收编号：A-ALP15-PRECISION-ACTIVATION-LINEAGE-20261011。对应决策：D-ALP15-PRECISION-USAGE-GAP-20261010。独立验收者：Sol 子任务 `holdout_prep_execution_20261010`；本来源查询、producer 快照与执行回报由主任务独立编写、执行和归档。验收日期：2026-10-11；本地末次身份核对 UTC 02:11:42（北京时间10:11:42）。本文件为唯一写入产物。

**结论：通过明确的历史曝光来源链窄域验收。** 两组共504个 decode activation 和 expanded20 的40个 teacher/student prefix 文件，共544个不同路径，可在本窄域 receipt 中由候选归因签为 `attributed_used`。这只说明其生产输入范围属于已见数据，不说明文件 payload 被重新核验，也不修改原 ledger、CSV 或 query 的候选状态。完整 usage coverage、drive-disjoint、最终独立名单冻结和新精度通过均未签发。

## 1. 范围、证据层级和执行边界

完整读取上述决策、当前执行回报、两版 producer、两个查询源码；使用本机 `/usr/bin/python3 -B`（Python3.9.6、macOS27.2 arm64、stdlib）完整 JSON 解析两个查询结果、两个原 activation manifest，以及既有 ref2 v2/v3 result/final receipt。只执行文本 SHA、JSON 断言、AST/source 片段对照、路径/列表/集合与计数运算。未执行任何生产源码；没有 SSH、网络、安装、下载、模型/GPU、PT/pickle 或其它二进制 payload 读取及内容哈希。

证据层级明确分开：

- **本地直接复核**：归档文本身份；activation manifest 全部条目与归档 stat basename 集合；两个本地 result/receipt 的实际文本 SHA 与交叉引用；完整冻结 producer 的数据加载、捕获、写入语义；归档列表唯一性、列表哈希和40条 prefix 映射结构。
- **审阅后采信的归档远端查询证据**：原 CSV 全文的 Counter；原 ledger 的已用 clip 包含关系；prefix 原 reference split、复制 split、原 result 的输入 SHA/argv/记录集合；远端文件当时的 regular/non-symlink stat 和输入文本前后稳定检查。查询源码完整可审，命令未使用 `-O`，成功原始 JSON 与空 stderr 已归档；本验收没有在远端重新执行这些检查。
- **仍未直接本地重算**：11MB 原 CSV、46MB ledger、prefix 原 result 与两份 split、远端当前目录/文件状态、metadata 的逐 clip→chunk 映射。不能把查询中的摘要/哈希当作本地已读取这些原文。此限制不影响本 receipt 对归档查询所证历史曝光范围的有限签发，不授权任何全局覆盖结论。

## 2. 冻结文本身份

路径基点为 `docs/baselines/`；证据简称 U 指 `evidence/usage_gap_20261011_v1/`。全部 SHA256 由本地实际文本重算，末次核对仍一致。

|文本|SHA256|
|---|---|
|ALP15-PRECISION-USAGE-GAP-DECISION-20261010.md|ce674809d555c04adc7b2607fb74f6427ec624b2237ba875525f898861994a5c|
|ALP15-PRECISION-ACTIVATION-LINEAGE-EXECUTION-20261011.md（含 expanded20 追加）|752763f13a35225e0cdf13e5856ff508a757dbdcb952025a78cbe1f7bd859ec6|
|U/check_activation_lineage.py|67dbb977404eaf3cb9325d0839f3d1660b96b9074e61864a422e778bccf58670|
|U/activation-lineage-query-v3.json|0b25714cf65d718781cf9b7c1d42b39bd0e478847ca5aa4ecb85738f9c687e6c|
|U/producer_v2.py|a2ccec42a954be1d8325c89afbc32337193808265121448ec770ec92a792e887|
|U/producer_v3.py|fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd|
|U/activation-manifest-v2.json|7c269f8b0457cb6c637e7d842340ddddefda3f93cb04bc7a227aae44bc345ebc|
|U/activation-manifest-v3.json|3975408023cff735c3fcd9edb76e190be654359805fcbab7f6ff15d6a3c4bf04|
|U/check_prefix_lineage.py|1f47739a39510dda165634691c768f52743bffafdde26f5f7dda81d7d1100fc6|
|U/prefix-lineage-query.json|106f600ea0722480ab11808fed773dcd499664c9078931fa8b8848f0e6e421ae|

prefix 查询内完整 `producer_source` 按 UTF-8 重算 SHA 为 `92e858ca556deab9edb26b8f98231d332bdae1376c4907b52853baba27623313`，与查询绑定的实际实验 source SHA 相同。两个 JSON 的 `query_sha256` 均与各自本地源码一致。activation v3 和 prefix stderr 均实际为0字节。

## 3. 两组 activation：逐项集合与 receipt 绑定通过

实验分别为 `jetson_300ms_20261008_hybrid_reference2_v2`、`jetson_300ms_20261009_hybrid_reference2_v3`。实际 producer 分别位于对应 `hybrid_m1_code_v2`（20261008）和 `hybrid_m1_code_v3`（20261009）冻结目录；不是用当前同名脚本替代两版历史 source。

|检查项|v2|v3|
|---|---|---|
|manifest / stat 数量及唯一 basename|252 / 252 / 252|252 / 252 / 252|
|归档 stat bytes 合计|2660558|2660558|
|完整 stat 列表 SHA256|5074e78adfa417e7e6fc5fe4f71268bf494f76e767e29ce3fc3102fa52528ffa|fe85263230a293b267b75c4b4fbb252db9e4beeae88643e2343eb8489f9a3dd0|
|本地 result 文本 SHA256|1b60bab73f88865e873f4bde777c3298f04a47286c6de5532a65df780b5770b8|02a163ddbadf9aec6c778d36a35e2e7b76515d17b8a384a35e55a9ace75d2356|
|本地 final receipt 文本 SHA256|85b770f7ce45e4704022b600117535b1840231b3785c4df8c24495031c5eef66|583c5bbcdaecef095f5fefcceed592effff558db49aebbf3b797f08568a7d6df|

列表哈希复算严格采用查询中的 `sha256(json.dumps(file_stats, sort_keys=True).encode())`，保留列表顺序，不换序列化方式。两组各252条 stat 全路径唯一，分组间无交集，合计504。条目路径均属于各自实验的 `decode_activations/`；basename 安全、为 `.pt`，无 `..`。stat 保存的 bytes/mtime_ns 为正整数；regular/non-symlink 检查来自原查询，未声称本机重做服务器 lstat。

两个原 manifest 的全部252条 file 名集合与各自归档 stat basename 集合 exact 相等，无重复、无遗漏；条目 SHA 字符串格式合法。**这些 PT SHA 只作为 manifest 历史声明，未在本轮重新读取 payload 计算。**

本地实际 result/receipt 来自 `evidence/jetson_300ms_20261009/hybrid_reference2_v2/` 和 `hybrid_reference2_v3/`。实际文本 SHA 与查询及 receipt/result 声明相等；manifest 的 `reference_result`、`final_receipt` 绑定同一实验路径，receipt/result 对 manifest 的引用 SHA 与本地原 manifest 实际 SHA 相等。manifest 的 code/runtime/input/actual-import 字典均与对应 result 相等；v3 snapshot/dynamic/virtual-import 字段同样匹配。producer 实际 SHA 与该 result 所记冻结 source 路径及 SHA 完全对应。两 result 记载完成状态、exit0、capture argv、相同固定 dataset revision；并非只因顶层 UUID 相同就作归因。

## 4. 当前 data → 最后 clip 捕获 → manifest 的源码语义通过

两版分别独立完整阅读，并用 AST 对照归档的实际函数片段。

- v2 421–423、v3 452–454 行在 `for clip in clips` 中以当前 `clip` 调用 `load_physical_aiavdataset(... t0_us=5100000, maybe_stream=False)`，赋值当前 `data`。
- v2 492–493、v3 523–524 行仅在 capture flag 且 `clip == clips[-1]` 时，将该次循环的同一 `data` 传入 capture；未发现 capture 前覆盖为另一 clip 的赋值。
- capture hook 将当前生成过程中投影输入保存为文件，积累同一 `rows`；函数返回的文件条目与 `provenance['clips'][-1]` 属于同一 capture 对象；最终 writer 将该对象的 clip 与 activations 一起写入 manifest，并由 final receipt/result 绑定 manifest 文本。

两个 manifest 的 clip 都是 `1f62af5c-2991-47ba-a7c7-0a6f37d8bad3`，等于各自本地 result.clips[-1]；原查询确认在冻结 ledger.used_clip_ids 中。故504个文件是**同一个已用 clip 的两次派生捕获**，不是504个独立 clip，也不是本轮证明生成张量/数值正确。不能因后续 probe 失败或目录曾标 holdout 消除历史曝光。

## 5. expanded20 的40个 prefix：确定性索引来源链通过

完整审阅 `check_prefix_lineage.py` 与 JSON 内实际 `controlled_8b_diagnostic.py` 全文。归档查询固定实验 `controlled8_expanded20_20261002_125000`，实际 result SHA 为 `67b328a1ecac9de414b47680ef6476c96f6cb30dab0c736a500a902970cebaa0`，split SHA 为 `e25cd24fa473a21cf14d9d46306bdd5b051ba1776d47bccd30be333ccaccd20c`。

关键链为：该次 result.argv 的 `--expanded-validation-dir` → 原 `mlp_stage8_joint_20261001_182631/expanded_checks/split.json` → 其实际 SHA 等于 result.inputs 声明 → 实验复制 split 同 SHA、同完整有序20 clip → 实际 expanded 分支的 `enumerate(clips)`。归档查询明确没有 `--resume`。不能用另一实验的 sorted UUID 或附近 split 推定顺序。

源码134–145行逐项准备当前 clip，写入 teacher `validation_{index:02d}.pt` 并按原顺序保留 `paths`；156–163行沿这些 teacher path 取同一 example/clip，调用 `generate_prefix`，将 student 存到 `path.name`；`generate_prefix` 明确保留 example.clip_id。这是静态核实历史 producer 行为，不是本轮执行 `torch.load`。

**`teacher_cache_provenance.json` 是源码89行从父实验复制的文本，不能作为本次 expanded20 名单或顺序依据。** 本 receipt 只使用上面的 argv→原 reference SHA→同 SHA split→enumerate 链。

本地完整解析40条映射：40个 source 不同，每个 role 恰20项，indices exact 为0–19；basename 与 index 一致；两 role 的 index→clip 字典 exact 相等、20个 clip 唯一，stat 正整数、payload flag False。原查询核两目录 `.pt` 文件集合 exact 相等且 regular/non-symlink，核原 result.records 的 clip 集合等于20名单，核20 clip 均已在 ledger.used_clip_ids、revision 相同、所有读入文本前后 SHA 不变。原 result 记录内容和240条数量未在本地原文独立重算；执行回报的240条不扩大本验收范围。

## 6. CSV 守恒与可签发 disposition

activation 查询完整读取固定 SHA `f398e3f054620d37544217e766fbafba2d6d052f084df75e22df9f41259a09c7` 的原 CSV，对两个 reason 分别检查 `Counter(selected) == Counter({path: 1 for path in attributed})`，不是只比计数504。prefix 查询对同 SHA CSV 的40路径分别执行同一 exact-once Counter。两查询都绑定 ledger SHA `0697cec28cd36ca841fe75ddeb95f1037269d69871bc2bda59ae831947e870d1`；activation 另绑定原 scan manifest/final inventory SHA。全部 input_text 前后稳定检查在归档查询中通过。

本地独立检查504与40路径集合无交集，联合544；归档 JSON 计数与原决策基数的算术守恒通过。**原 CSV 未传回，下面是已审源码和成功远端全量 Counter 证据支持的窄域归类，不是本地原 CSV 逐行重算。** 一条原记录只获一个 disposition；跨 reason 共享 source 不计作新样本。

|reason / 证据维度|原记录|本 receipt 来源归因|来源仍待分类|
|---|---:|---:|---:|
|anonymous_cache_without_explicit_local_mapping|1892|544 attributed_used|1348 pending_evidence|
|binary_payload_not_verified（只分来源）|4104|544 attributed_used|3560 pending_evidence|
|external_path_reference|63177|0|63177 pending_evidence|
|foreign_or_unknown_revision|247|0|247 pending_evidence|
|revision_conflict|171|0|171 pending_evidence|
|symlink_not_followed|20|0|20 pending_evidence|

**payload 未验证仍为全部4104条**，不能把来源已归因544条扣掉便宣称只剩3560个 payload 未验。activation clip 已包含在 expanded20 的20个 clip 中；本范围不是544个 clip。归档查询均无新增排除块，依据为所有范围内 clip 原已 used；本轮未新增 metadata/chunk 映射复验、未释放任何旧21个保守排除块。原 query 保留 candidate 状态，本文签发归因不回写旧 census。

## 7. 实际验证留痕、失败历史与门禁

本地所有只读检查 exit0。实际命令族：`sed -n` 完整读取决策/回报/producer/query，`shasum -a 256` 重算身份，`/usr/bin/python3 -B` 的临时 heredoc 仅用 pathlib/hashlib/json/ast/collections 解析与断言；`test ! -e` 确认本验收文件不存在。工具会话留痕：1bd925（完整 activation JSON、列表哈希与 AST 片段）、391053（本地 result/receipt 和当前 data→capture 绑定）、450eb7（原 manifest 全 entries↔stat 与40项 prefix 映射）、4d6811（prefix 源语义及544联合集合）、af0cbb（当前10份文本 SHA 再核）、aa7fca（本机版本、两个空 stderr、embedded source SHA）、41ed47（归档计数/schema）、056a43（完整决策重读）、1db3f3（完整当前执行回报）、22052c（完整 prefix 查询重读）。这些是工具会话日志，不冒称另有本地独立归档 stdout 文件。

远端原 activation 成功执行0a2564（UTC 2026-10-10 16:06:39.461565–16:06:40.025864），prefix 成功执行ebb0ce（UTC 2026-10-11 02:08:28.412896–02:08:28.884638）；完整 stdout 是对应 query JSON，stderr 实际为空；命令/解释器/offline/空 CUDA 在冻结执行回报中可复核。此处是审阅已有执行证据，不声称本验收亲自执行远端命令。

保留执行回报记载的失败历史：activation 最初错误 v3 日期目录导致 FileNotFoundError/exit1，随后依据实际 result 路径另存修正版本；取回时 SSH master 过期 exit255 后仅重新取回已完成文本；prefix 定位不存在日期目录 exit2 后依据实际 inventory 定位。未因失败豁免曝光，也不把当前成功源码冒充失败版本。失败原始日志未在本轮逐项完整核验，失败上下文采信冻结执行回报，不据此新增结论。

|门禁|结论|
|---|---|
|两版实际生产 source / result / receipt / 原 manifest SHA 与引用|通过，本地直接核验|
|两组252 entries exact 对归档 stat，列表哈希、联合504唯一|通过，本地直接核验；远端 stat 是历史证据|
|当前数据加载→末 clip 捕获→同 manifest entries|通过，两版分别完整静态阅读|
|expanded20 实际 producer / argv / 原名单 SHA / 索引链|通过静态语义及归档查询证明；原 split/result 本地重读 pending|
|两 reason 每路径 exact-once、守恒|通过归档远端完整 Counter 证据；本地原 CSV 重算 pending|
|payload 正确性、实时远端状态|未验，不签发|
|全历史 scope / coverage_complete / drive_disjoint_verified|未通过，继续 False / pending|
|最终独立名单或新数据独立精度|未冻结 / 未通过|

## 8. 下一步

本窄域来源链可签发，不能推进最终独立抽样、下载报价、模型实验或优化门禁。继续按原 usage-gap 决策逐实验核剩余匿名1348、其它 binary 来源3560，以及外部引用/revision/symlink/未扫描文本；未知来源未被有限范围保守排除前保持 coverage False。若要将本 receipt 升级为本地原文重算，最小待补是原 CSV、ledger 相关原文与 prefix 原 result/两份 split 的明确范围授权及身份核验，不得假造条目或读 PT 寻捷径。任何新增排除块仅能在旧21块基础上取并集；完整 scope 验收与统计方案门禁另行签发。
