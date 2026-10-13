# ALP15 未见数据资格准备决策单

- 决策编号：D-ALP15-PRECISION-HOLDOUT-PREP-20261009-01。
- 日期：2026-10-09（Asia/Shanghai）；决策者：GPT-6.1 Sol high；主任务协调远端CPU执行与独立验收。
- 上游冻结决策：`ALP15-PRECISION-VALIDATION-DECISION-20261009.md`，SHA256 `989b64ac1adfa3979556db82d035f71230d91132454b8eb35029eafacf0280db`，保持原文件及其所有权不变。
- 用户当前目标：先完全验证精度，暂停FlashDrive/LightVLA新实施。本决策只完成历史使用清点与候选资格实证，为后续独立精度方案提供输入。
- 风险：数据资格与研究结论为高风险；执行范围仅为既有服务器CPU只读盘点。禁止GPU、模型推理/训练、读取PT payload、依赖安装、网络与数据下载。
- 当前批准：实现并验证准备脚本，在主任务既有L20环境运行一次CPU census，独立验收台账与候选输出。此批准不等同于冻结最终测试名单、下载数据或开始模型评测。

## 1. 已核对的事实与固定来源

已完整读取 `AGENTS.md`、上游精度决策、最新 `ALP15-PRECISION-PRIORITY-STATUS-20261009.md` 与 `ALP15-PRECISION-DATA-DOWNLOAD-SOURCES-20261009.md`。本会话已完整读取 experiment-design、statistical-analysis 技能及其计划/报告模板；继续遵循采集前锁定设计、正确独立单元、明确探索性范围与失败记录，不把已有数据或准备结果写成验证性统计结论。

数据权威来源为 NVIDIA PhysicalAI-Autonomous-Vehicles，revision `33f9bf447ed3bcb7d545ce13f4226f824214fafb`。以下三个全局元数据在服务器 `datasets/physicalai_chunk0000_20261002_165238/files/` 已有且已由主任务只读核验，各306152行，大小和实际SHA与官方固定revision相同：

| 相对metadata根路径 | bytes | SHA256 |
| --- | ---: | --- |
| `clip_index.parquet` | 11132300 | `9630a3de5ee057ded194deaa94d3b9b42af6f18c301785706e50141314a6935c` |
| `metadata/feature_presence.parquet` | 11209157 | `a9946feb269124e3a7cbfe22d97669585735134f634c2ca97332e7abc4314790` |
| `metadata/data_collection.parquet` | 11339400 | `9624750cadabf8028264cc1404e75e81777654be9def06b00e35c9f834b173d5` |

`features.csv` 官方固定Git blob为 `65de094a4037902a40bd42de7b17082029b78fe2`、7076 bytes，现存文件按Git blob算法校验后登记实际SHA256；如缺失或不符则停止文件计划阶段，不能调用构造器重新下载。

官方API固定 `physical_ai_av==0.2.0`，源码提交 `59d578b53ef186ffc3f1de1d1b2598717a982d64`；官方 `dataset.py` SHA256 `b35f5f1bbb42ba92c83c835f7525c33e8ee915eb5ef071ac3dd972d91b0d9560`。决策者已完整读取本地同SHA源码：构造 `PhysicalAIAVDatasetInterface` 会走HF接口并下载元数据，不允许在本轮执行；使用既有pandas/pyarrow读取本地parquet、CSV，再使用官方 `physical_ai_av.dataset.Features(features_df).get_chunk_feature_filename(chunk_id, feature)` 解析五路文件名即可，禁止手猜目录模板或批量下载API。运行时记录实际包版本、模块文件SHA、pandas/pyarrow/Python版本。

资格要求为 clip_is_valid=True 且五真实列均为True：egomotion、camera_cross_left_120fov、camera_cross_right_120fov、camera_front_tele_30fov、camera_front_wide_120fov。主任务现清点306152clip均合格；仅排除chunk0和3119后305952clip/3144chunks，官方split真实计数train153425、val90928、test61599。上述是基线计数，不是完整usage排重后的余额。实际 `split=test` 可优先作可行性候选，不能将其称为Alpamayo专门benchmark协议。

`data_collection` 有country/month/hour_of_day/platform_class/radar_config/clip_id，无drive/session ID。新clip和chunk不重叠不证明行程独立；必须保持 `drive_disjoint_verified=False`。t0_us固定5100000，但四帧历史/64步未来窗口实际完整性需将来下载后QC，本轮不能验收。

## 2. 文件所有权与执行命令契约

主任务分派一位GPT-6.1 Sol high执行者，唯一写入：

- `scripts/prepare_alp15_precision_holdout.py`：本地元数据与历史资产census，候选/文件计划，不含模型或下载功能。
- `tests/test_prepare_alp15_precision_holdout.py`：CPU unittest，含合成资产树与parquet夹具；禁止torch/pickle/PT读取。
- `docs/baselines/ALP15-PRECISION-HOLDOUT-PREP-EXECUTION-20261009.md`：新的执行回报；若路径已有内容，先由主任务选择新路径，不能覆盖。
- 新输出目录由CLI `--output` 指定，存在即拒绝（不允许force/续写/清空）。推荐远端 `experiments/precision_holdout_prep_20261009_v1/`。输出目录、冻结脚本目录不纳入usage扫描，并在清单列明排除原因，避免自引用。

CLI至少包括 `--root`、`--metadata-root`、可重复的 `--usage-root`、`--output`；可选 `--draft-feasibility-clips 100 --sampling-seed 20261009`。默认只生成完整资格表、usage台账与文件名计划，不进行抽样。生产路径在主任务核验后显式传入；不能隐式扫描个人HOME或服务器其它项目。脚本内冻结三parquet预期bytes/SHA、数据revision、API版本/SHA和features.csv blob，不使用不固定的main。

```sh
alpamayo1.5/.venv/bin/python -m unittest discover -s tests -p test_prepare_alp15_precision_holdout.py -v
alpamayo1.5/.venv/bin/python -m py_compile scripts/prepare_alp15_precision_holdout.py tests/test_prepare_alp15_precision_holdout.py

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
alpamayo1.5/.venv/bin/python -B scripts/prepare_alp15_precision_holdout.py \
  --root /home/tuqiang/lang/Lighten_VLA \
  --metadata-root /home/tuqiang/lang/Lighten_VLA/datasets/physicalai_chunk0000_20261002_165238/files \
  --usage-root /home/tuqiang/lang/Lighten_VLA/experiments \
  --usage-root /home/tuqiang/lang/Lighten_VLA/datasets \
  --usage-root /home/tuqiang/lang/Lighten_VLA/docs/baselines \
  --usage-root /home/tuqiang/lang/Lighten_VLA/scripts \
  --usage-root /home/tuqiang/lang/Lighten_VLA/compression \
  --output /home/tuqiang/lang/Lighten_VLA/experiments/precision_holdout_prep_20261009_v1
```

命令是实现后生产契约，不表示已经执行。主任务先核验各root存在；发现另一个历史数据delivery、恢复路径或资产引用落在范围外时，将其显式追加只读usage-root或记覆盖缺口，不能默认已扫描。现有约1152份experiments JSON作为盘点参考，实际数量随并发记录可能变化；冻结开跑清单和SHA、结束时复核，不将不同数量直接当不存在的文件可忽略。源码快照可由主任务按现工作流上传，冻结目录须新建、拒覆盖；不改远端原venv、不触碰其它项目GPU/PID。

## 3. 使用历史census的可执行规格

1. 建立显式扫描manifest：每root、路径类型、相对路径、bytes、mtime_ns、文本文件SHA256、读取/解析结果、排除/未处理原因；重叠roots去重，拒绝逃逸root的路径，symlink只记录目标及范围内状态，不跟随到未经声明目录。无权限、扫描中变化、损坏/解析失败均列明。
2. 在experiments中完整枚举JSON，解析所有结果、split、manifest、receipt、partition、cache索引；也读取相关JSONL/CSV/TXT/LOG/MD/YAML/TOML/SH/PY文本以查训练/评测/缓存引用。用途分类用于审计，不以不确定用途作为保留依据。扫描文本时查找UUID形式并与metadata clip_id集合精确匹配，JSON字符串值与key均查；保守记录所有metadata-mapped clip引用及来源位置，包含仅作为日志/决策提及者，统一排除。不得读取或复制完整日志内容/凭据到输出；输出仅clipID、结构key/行号和文件SHA，不输出任意敏感片段。
3. 每clip引用映射至官方metadata chunk，记录roles（training/calibration/monitor/development/smoke/holdout/simulation/unknown等）、first/all source refs及所有关联chunk。只要曾引用，即列 `used_or_referenced_clip_ids`；其整chunk加入保守 `excluded_chunk_ids`，连同已知0/3119，不以“只是holdout字段”豁免。
4. PT/pickle/cache文件仅做文件名/目录索引，记录路径、size、mtime与可识别clipID，不打开payload、不反序列化、不扫描权重tensor。`validation_00.pt`等匿名文件名只能依同一冻结split/manifest/receipt的显式映射绑定clip；找不到映射则列 `unresolved_cache_indices`，不能猜按另一个目录顺序。PT payload未核验是明确限制，不能为了完整性读取它。
5. ZIP/权重/视频等大二进制仅枚举文件路径、类型、size/mtime；模型权重无需算大文件SHA。dataset ZIP可从既有receipt/manifest读取clip名单，filename含chunk则保守排除chunk。若二进制或缓存可能含未记录的历史输入、却没有可追溯索引，列coverage gap。相关已存在JSONL/日志不得因文件大就静默跳过；如实现设读取上限，超过上限显式列未审计并阻止完整coverage声明。
6. JSON解析失败仍可从可读文本提取匹配UUID用于保守排除，但文件仍是失败，不算完整解析。UUID未出现在本revision metadata的引用另列 `unmapped_uuid_refs`，与已映射clip区别；不能把模型manifest UUID全猜成clip，也不能把未知引用全部删除后声称完整。数据revision冲突和外部路径引用列疑点，等待复核。
7. 由metadata得到`referenced_clip→chunk`映射后进行保守候选排除；不读取ADE/FDE或预测误差用于筛选/排序，不把解析时见到的指标值写入候选采样逻辑或输出。角色不明也排除，已知用过的chunk整体排除。
8. `coverage_complete` 默认False且**脚本不接受手工开关设True**。同时给出机器层级 `declared_roots_inventory_complete`、`all_declared_text_parsed`、`unresolved_cache_count`、`unresolved_external_path_count`、`scan_stable`等证据。只有独立Sol复核扫描边界、未处理文件/日志、外部引用、历史资产范围与匿名缓存映射，并确认无相关未覆盖项后，才可在新独立验收receipt声明针对明确范围的coverage；仍不能推论所有未知历史或行程独立。本轮脚本可成功产出 `completed_usage_census_pending_coverage_review`，False不是运行失败或阻塞实现。

不得用`coverage_complete=False`就跳过实证台账，也不得为了source报告下载命令要求True而补填假值。源报告命令B只在后续账本审定且冻结名单成立后可用。

## 4. 元数据资格、候选表与文件计划

三个parquet索引/clip_id规范化后必须唯一、键集一致，clip_id字符串、chunk非负整型（bool不能冒充整数）、clip_is_valid及五feature严格布尔、split保持真实值；缺值或集合错位时停止而不是fill后吞掉。先逐项复现总306152、仅排除0/3119基线305952/3144及真实split计数，再报告追加usage排除后的新计数。参考计数不符则保存差异、非零退出，升级，不自动换metadata。

输出 `eligible-candidates.csv` 含所有尚未被保守排除的合格clip及官方split/chunk/country/month/hour_of_day/platform_class/radar_config/t0_us，标记 `candidate_only_pending_usage_coverage_and_input_window_qc`。输出按所有split、每chunk及真实粗粒度字段分组的完整counts；官方test作为优先可行性池另计数，保留train/val背景计数而不改标签。

每剩余chunk用官方五feature常量及Features接口生成5唯一zip路径，输出 `candidate-file-plan.csv`：chunk、feature、官方相对路径、当地是否已有对应receipt/文件、size与SHA来源。如没有既存固定revision报价证据，`official_bytes`/`official_sha256`为null、`quote_status=pending_fixed_revision_query`；不能将chunk0001容量例8,649,375,436bytes按块数乘成准确报价。本輪不实例化联网API或查询新报价，不下载zip。

产物至少含 `scan-manifest.jsonl`、`usage-ledger.json`、`referenced-clips.csv`、`unresolved-assets.csv`、`eligible-candidates.csv`、`metadata-counts.json`、`candidate-file-plan.csv`、`report.md`、`run-manifest.json`。manifest登记完整argv、UTC起止、环境/API/脚本/本决策SHA、元数据前后SHA、扫描范围/排除输出目录清单与产物SHA。ledger记录revision、used_clip_ids（保守引用集合）、used_chunk_ids、audit_evidence、False coverage和drive flag、未解决事项；保留兼容source报告字段但不得以格式兼容为由伪造验收。

## 5. 可选100clip可行性草案：不等于最终样本

默认本轮不抽样。若主任务显式传`--draft-feasibility-clips 100`，仅从当前剩余官方split=test池按chunk分组构造可复现草案：对整数chunk列表升序排序后，用固定seed20261009的独立`random.Random`打乱；chunk内clip_id排序后同一固定随机流程打乱，按chunk顺序依次取clip直到100，无放回。算法/seed/Python版本和完整候补顺序记录；不得按误差、速度、模型输出或事后便利换chunk。这是减少整块下载数量的可行性取样取舍，可能集中于少数chunk，须报告chunk数及country/month/hour/platform覆盖，不声称总体代表性或行程独立。

草案文件名 `draft-feasibility-selection.json`，state=`draft_pending_coverage_review_quote_and_input_qc`；不能命名为已审定frozen-selection。若候选不足100则明确不足，不自动转train/val。草案可以提前用于文件报价规划但不能越过usage coverage门禁进入下载/最终评测。

后续Sol审定资格/草案后，利用来源报告既有官方`HfApi.get_paths_info(...,revision=33f9...)`报价接口逐路径求size/SHA；那是**后续联网只读报价**，本轮不执行。按去重chunk的五ZIP真实bytes给用户规模与命令；过多chunk导致大下载时先给报价/方案，用户自行通过命令下载。不能将用户授权准备精度验证解释为代下载大资产。当前尚无经核验的完整国内传输源，沿来源报告证据边界处理。

100clip没有功效依据，仅可行性规模；若其结果用于修改候选/统计计划，它立即成为开发/规划数据。最终采集前统计计划仍pending：需锁定mean ADE15%主门禁、合理单侧CI、独立单位/关联敏感性、样本量依据、两候选多重比较和失败处理。A=KV4、B=fullprefill10 hybrid仍分别冻结，未合并；本轮不实现模型输入/评测/推理框架。

## 6. 验证、停止与独立验收

CPU测试至少覆盖：metadata SHA/blob不符、非法类型/非唯一clip/键错位/feature缺失；引用藏在JSON key/list/嵌套字段与日志行；同clip多source去重映射；引用整chunk保守排除；unknown引用/解析失败/匿名PT索引未解决使coverage保持False；不打开PTpayload；外部symlink/读取失败/扫描中变化；存在output拒覆盖；官方test优先且确定seed草案可复现；Features解析文件名而非手猜；从旧baseline计数到usage排除计数不混淆。测试可用合成小元数据时显式test fixture契约，真实生产不能放宽306152和来源门禁。

正式执行须CPU测试全部通过、无跳过，py_compile退出0；服务器运行实际元数据/资产树并检查输入前后SHA不变。只描述环境GPU型号，禁止主动GPU初始化/资源测试。执行回报给实际命令、退出码、完整日志、变更文件/产物SHA、总JSON/text/PT索引扫描数、解析/覆盖缺口、候选counts、是否生成草案、所有门禁结果与偏差。主任务取回小证据后由独立Sol复核，才能称“本轮usage census与资格准备通过”；coverage待复核仍保留。

源码/API/元数据身份不符、扫描无法安全限定、输入变化、metadata键错位、试图读取PT/GPU/下载时立即停在安全边界并留失败日志，升级主任务；不修改旧证据、不删除目录、不换revision/依赖/样本绕过。来源hash和coverage实证不足时可以产出失败/待复核报告，不能宣布“独立测试就绪”或完成精度验收。
