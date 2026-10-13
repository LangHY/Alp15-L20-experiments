# 完整候选池报价与用户下载工具独立验收

验收编号 A-ALP15-PRECISION-FULL-POOL-DOWNLOAD-20261011；对应决策 D-ALP15-PRECISION-FULL-POOL-DOWNLOAD-20261011。独立验收者 Sol high 子任务 `holdout_prep_execution_20261010`。执行者主任务独占脚本/测试/清单/执行回报；本验收只写本文。日期2026-10-11，本地验收核对 UTC03:06:27（北京时间11:06:27）。

**结论：通过本表冻结版本的完整metadata test候选池清单、逐文件报价、默认plan-only和用户显式下载门禁验收。** 初版输出目录/cache/receipt安全缺陷先报告并阻断；执行者修复后才验当前代码。可交给用户决定是否承担全池成本并自行运行命令；不授权代理启动下载。本轮未取得任何ZIP，不签“下载完成”“独立最终测试集”或新精度/设备性能通过。

## 1. 边界、身份及证据层级

本机 cwd `/Users/USER/Documents/ChatGPT/Lighten_VLA`，`/usr/bin/python3 -B`，Python3.9.6、macOS27.2 arm64、stdlib CPU。完整阅读决策、两脚本、最终tests、最终执行回报、用户命令文档；完整解析candidate snapshot/plan/官方features CSV和61,301行test CSV。独立运行9项合成CPU测试和无参数plan-only命令；没有运行export、quote、真实下载、SSH、网络、模型、解压或ZIP内容读取。测试中的 `--download` 负例在缺少确认参数门禁停止，不导入HF或进入下载循环。

按code-review skill的规范/决策两个维度审查。该skill默认Git固定点/并行agent流程不适用于本轮冻结SHA验收，且任务明确禁止新增agent；已告知执行者后直接有界审查指定文件，不扩到未授权来源或无限安全框架。

本地直接复核的内容：以下实际文本SHA、CSV全名单/块集合/排除交集、所有3165文件路径与五个实际官方CSV模板的exact绑定、全部报价size/hash格式与总和、代码门禁/安全修复、实际CPU测试和默认CLI。原45MB eligible全表、原file-plan/ledger未传回；从原表筛split=test的来源与输入前后稳定是已审export源码及成功远端执行回报层级。官方API报价与本日国内小查询来自冻结执行回报/工具会话，不称本验收自行联网复验。ZIP SHA是官方LFS声明，不是本轮payload实测。

|冻结文本|SHA256|
|---|---|
|FULL-POOL-DOWNLOAD-DECISION-20261011|9a969761e29b27762b353913924919760494214d95df5826e2ffc2512be8f836|
|FULL-POOL-DOWNLOAD-EXECUTION-20261011（含第6节修复）|c7625afc13b140a1ae67870bf0c1665bff9e79463f82cdab88a40bb43a83b422|
|scripts/download_alp15_full_test_candidates.py|93911fd3542bfbac7f9a14a79b4062873aacc446d6cd5510ecb191b2e3d9c8aa|
|scripts/export_alp15_test_candidate_snapshot.py|805593baaf559f54b90a7dc6c7bb6cf8f640a6ce0c246683a9e941c82de9b1ef|
|tests/test_download_alp15_full_test_candidates.py|bfacf13ebdd4580166abad56bcce4920d026a9d3700d93585db5d1a80e1f13a0|
|evidence/full_pool_download_20261011/candidate-snapshot.json|e75bb6813da2f51767e701ea1f7fc9f920da8e943abb89bcb18f06a27b1b2096|
|同目录test-candidates.csv（3119214B）|ad03761a64e92503b6a1bd6954b04898b7d57cb7badb45d194e9f9f7611d248a|
|同目录download-plan.json|2a2352b5e8d690763c7d66c40d3baed2154095987b64bd33c8206614ac93e8f0|
|同目录features.csv（7076B）|61105260827f5cf193c6c3ecbfed13b185432d04899d52bff3a7430ab4c0f304|

验收只适用于这些冻结清单/代码，不泛化为任意用户自造 `--snapshot` 或报价的来源合格。运行时以CSV SHA、snapshot SHA和用户确认的plan SHA绑定这批文件；本次验收独立重算了CLI无需每次重读计数的完整CSV。

## 2. 完整名单、五feature模板与来源闭环

完整test CSV header exact为clip_id/chunk/t0_us，61,301个clip UUID唯一且canonical；观察到633个不同chunk，exact等于snapshot.chunks，不按连续编号ranges补齐。snapshot的21排除块唯一，与CSV观察块集合无交集；本CSV t0_us均5100000只是名单字段核对，**不是输入窗口QC通过**。

完整snapshot有3165个不同path，`(chunk, feature)`集合exact为633观察块×五项：egomotion、camera_cross_left_120fov、camera_cross_right_120fov、camera_front_tele_30fov、camera_front_wide_120fov。实际features.csv解析36个唯一feature，五个原chunk_path如下；所有3165项都由对应实际模板 `.format(chunk_id=chunk)` exact复算相等，没有只凭目录习惯猜路径。

|feature|冻结官方CSV原chunk_path|
|---|---|
|egomotion|labels/egomotion/egomotion.chunk_{chunk_id:04d}.zip|
|camera_cross_left_120fov|camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_{chunk_id:04d}.zip|
|camera_cross_right_120fov|camera/camera_cross_right_120fov/camera_cross_right_120fov.chunk_{chunk_id:04d}.zip|
|camera_front_tele_30fov|camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_{chunk_id:04d}.zip|
|camera_front_wide_120fov|camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{chunk_id:04d}.zip|

本地正式v3 run-manifest实际SHA `022ae526670625ff44c1d69b7507bb1925e0a91cedd25442573cb9ccd29c3ff2` 的artifact_sha256明确绑定原 eligible CSV `cf620cf9b278bacd149378591d356e30a8efa3679e2696850a5bc743c717d8e7`、原file-plan `b76b687b43f3d9923c1a213e2b34ad3a3e64936d2c6c90a31470f7cf83a1b0ac`、ledger `0697cec28cd36ca841fe75ddeb95f1037269d69871bc2bda59ae831947e870d1`；其features_csv_before与本地原件bytes/SHA相同，Git blob声明 `65de094a4037902a40bd42de7b17082029b78fe2`。本轮未重新验远端Git对象。

export源码固定原三SHA、只筛split=test、直接取实际chunk/relative_path和ledger排除块，核名单/文件数/每块五项，排他新建目录、open-x输出、结束再核三输入SHA。归档远端export5f2b19 exit0、取回666388 exit0和产物SHA对照，通过这个已冻结输入链采信来源；不声称本机重新筛过未传回的原45MB全CSV。

## 3. 全量报价与成本

plan的3165路径唯一且exact等于snapshot；每条size为正整数、SHA为64位小写hex，逐项完整检查。plan.snapshot_sha256 exact等于本地snapshot实际SHA，dataset/revision固定为NVIDIA官方 `nvidia/PhysicalAI-Autonomous-Vehicles` / `33f9bf447ed3bcb7d545ce13f4226f824214fafb`。authority_source和transport_source都是 `https://huggingface.co`。

逐项求和：**3500560168089 B = 3500.560168089 GB = 3260.15070834104 GiB = 3.500560168089 TB = 3.1837409261142966 TiB**。不是用单chunk估价乘633。初次工具门禁要求剩余ZIP字节另留5%，约3.68TB空闲；缓存/解压/工作空间额外开销未知，不保证这些用途都被5%覆盖，也不自动复制或解压第二份全池。

quote源码只使用显式官方endpoint的dataset_info/get_paths_info小API、先核revision、分100路径batch核返回path集合、读取LFS size/SHA，quote以open-x保存，未调用ZIP下载函数。官方API真正执行exit0和全部路径报价来自执行回报b2a5f5；本验收未重跑网络或校验LFS payload。

国内传输证据只采信本日有界小查询：执行回报记魔搭同名仓只有四个顶层项、无camera/labels；HF-Mirror所核入口308到官方。同名仓/提交不等于完整镜像。当前只可交官方备用传输，不签“国内大文件镜像可用”或实际大文件吞吐、gated许可/网络已满足。用户须自行具备官方许可/认证；脚本无第三方endpoint/token转送入口，也不代接受许可。

## 4. 两维代码审查与修复验收

### Standards

当前冻结版本保留小脚本结构，没有引入下载框架、安装、模型或解压。输入路径只接受受限ZIP相对路径；用户输出要求绝对无`..`、规范化、非root/home、祖先无symlink；ZIP冲突不覆盖。**初版硬阻断已解除**：此前遗漏HF local_dir缓存与两个receipt路径，可能越过已检查ZIP路径向链接目标写入。最终prepare_output在任何HF导入前检查out祖先、整个已有.cache树不跟随symlink、cache/HF目录类型冲突、receipt symlink/类型/异plan/条目身份；合法partial缓存保留、不清空任何用户内容。receipt写入带同plan SHA。

缓存内部非目录冲突由HF正常创建目录时失败也不会覆盖文件；本验收不把prepare_output称作所有未来并发写入的安全隔离框架。用户应使用独立输出目录，不在运行期间自行修改缓存/目标/receipt。未做真实网络续传、SDK大下载或恶意并发压力试验。

当前规范维度无阻断发现。首版7tests/SHA仅是执行回报第4节历史；本receipt签第6节最终9tests与本表代码。

### Spec

- **默认plan-only**：只读取本地snapshot/CSV/plan和打印规模；没有HfApi/hf_hub_download导入、GET、安装或文件输出。help在读取前由argparse处理；quote/download互斥。
- **用户显式门禁**：只有 `--download` 且plan SHA/confirm-bytes/独立output三项匹配才继续；plan与snapshot/CSV冻结身份已绑定。任何非官方transport被拒，调用HF函数仍显式固定官方endpoint/revision，不把token输出或交第三方镜像。
- **保留已有文件**：下载启动前全部已有ZIP须bytes+SHA同时匹配才计为已有；冲突/链接停止，不覆盖或删除。磁盘不足在HF导入/下载前停。partial留在HF自己的缓存用于原命令重跑，未承诺本轮已测试续传完成。
- **逐文件完成校验**：只有实际下载/已有文件bytes+SHA通过才追加verified receipt；全3165通过才追加completion，independence/input_qc标False。校验失败保留文件/缓存并停止；不改SHA或删条目凑齐。
- **成本与范围披露**：用户命令文档明确3.50TB、空间、官方备用、候选非已验独立、用户手动三确认、无代理代下载；不缩成便利小集。ZIP可能附带非test clip，不授权把所有成员跑模型。

当前决策符合性维度无阻断发现。用户尚未确认实价后全池成本和目标，不触发实际下载；完整usage/drive/QC/planning/最终名单门禁仍未通过。

## 5. 实际独立验证与完整测试日志

本地真实命令：

```sh
/usr/bin/python3 -B -m unittest discover -s tests -p test_download_alp15_full_test_candidates.py -v
/usr/bin/python3 -B scripts/download_alp15_full_test_candidates.py
```

第一条exit0，9tests/0.188s，完整测试stdout/stderr如下（工具会话34214a，非主任务014bbc复述）：

```text
test_complete_snapshot_is_accepted (test_download_alp15_full_test_candidates.CandidateDownloadTests) ... ok
test_default_cli_has_no_network_or_download (test_download_alp15_full_test_candidates.CandidateDownloadTests) ... ok
test_download_requires_explicit_hash_and_bytes_before_creating_output (test_download_alp15_full_test_candidates.CandidateDownloadTests) ... ok
test_duplicate_or_missing_paths_are_rejected (test_download_alp15_full_test_candidates.CandidateDownloadTests) ... ok
test_escape_paths_are_rejected (test_download_alp15_full_test_candidates.CandidateDownloadTests) ... ok
test_masked_sha_and_bad_total_are_rejected (test_download_alp15_full_test_candidates.CandidateDownloadTests) ... ok
test_output_aliases_and_cache_symlinks_are_rejected_without_touching_target (test_download_alp15_full_test_candidates.CandidateDownloadTests) ... ok
test_receipt_symlink_and_foreign_receipt_are_preserved (test_download_alp15_full_test_candidates.CandidateDownloadTests) ... ok
test_used_chunk_is_rejected (test_download_alp15_full_test_candidates.CandidateDownloadTests) ... ok

----------------------------------------------------------------------
Ran 9 tests in 0.188s

OK
```

第二条独立exit0（6e2e73），打印3500560168089B、上述TB/TiB、plan SHA及候选scope，末行 `PLAN ONLY: no ZIP download; not a certified independent test set.` 没有运行quote/download。synthetic tests并不替代完整实际CSV/官方模板验收；这两类验证本轮分别做过。

其余实际只读命令为sed/rg、stdlib json/csv/uuid/hashlib完整解析与断言、test目标不存在。工具留痕：79fcaf（完整决策）、962613/e04bb8（首版download/export完整阅读）、4839d0（全CSV/plan名单、排除、正size/hash/总和）、a47d71/836a7b/c87f19（最终代码/tests/执行回报全文）、591459（实际features7076B/36rows/五template全3165复算及冻结SHA）、8d4ce1（正式manifest原三artifact身份/flags）、d1fdbd（用户命令全文）、6e2e73（真实默认CLI）。除测试本文保留的原始日志外，其余为可追溯工具会话，不冒称另有未创建的独立日志文件。

## 6. 签发范围与下一步

可签：冻结完整候选名单/五feature路径/真实API报价清单；本机默认只读计划；用户手动启动前的确认、冲突与路径/cache/receipt保护；真实CPU回归。当前无未解除阻断代码缺陷。

未签：真实ZIP下载、传输/缓存续传成功、payload/解压/窗口QC、引用scope/drive完全独立、最终61,301clip均可评测或独立测试集、模型/设备精度与性能。官方LFS hash/ZIP bytes是报价身份声明，不据此叫文件已获取。最终独立框可能因额外usage/QC/planning排除进一步收紧，旧21块不释放。

下一步先向用户呈现3.50TB实价及独立目录/空间选择，只有用户自行确认并运行冻结命令才下载；执行者不代启动。完成获取后仍需单独验实际bytes/SHA/receipt，再按既定usage/QC/planning及统计门禁冻结最终名单。若用户主动改较小方案另作概率设计，不称完整池，不以100clips或便宜chunk冒充充分。
