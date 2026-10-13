# ALP15 留出集准备 v3：本地工具独立技术验收

- 对应决策：D-ALP15-PRECISION-HOLDOUT-PREP-20261009-01/-02 与 D-ALP15-PRECISION-HOLDOUT-PREP-20261010-03；冲突按 v3/v2 的限定替代范围处理。
- 日期：2026-10-10（Asia/Shanghai）；独立验收者：GPT-6.1 Sol high，按主任务分派执行。
- 结论：**冻结 v3 通过本地工具契约独立技术验收**，状态 `local_tooling_accepted_remote_census_pending`。本文不验收真实服务器 census、usage 范围完整性、最终名单或精度。
- 写入边界：仅新增本文。未修改准备脚本、测试、三个决策、执行回报、原日志、模型或 metadata。没有网络/SSH、下载、安装、模型/GPU或真实 PT/ZIP payload 操作；不使用 superpowers。
- 已完整读取 v1/v2/v3 决策、冻结 661 行脚本和 457 行测试、执行回报原段及 v3 新追加段、执行者 v3 原始回归日志。执行者冻结前只读取决策与整理清单，未运行动态版本。

## 1. 冻结身份与环境

下表均为实际文件字节重算，脚本/测试在本轮复验前后未变。

|文件|SHA256|
|---|---|
|`scripts/prepare_alp15_precision_holdout.py`|`ccebd4d237cf5229a422844579e5871405291e0f01f957f5600fda8148f04b9c`|
|`tests/test_prepare_alp15_precision_holdout.py`|`c102a6f54abe950a7a58acc94aafe459d3b0126aa87b6e419429c890243ed5a5`|
|准备决策 v1|`3efbfea76b5b336d5199c93337863bff49c120f2b3bd92c9919d6f26e979297c`|
|准备决策 v2|`32b534ea574677928fca9c6e9277498807f00839fe8de852f14f01d7a9e7a23b`|
|准备决策 v3|`e53284d0ebcba64787b4f6ca4f5d60bbef8e779f62f29d7d1845b1910c1731f8`|
|执行回报初读至 v1/v2 时版本|`e545f8dabbfe8c4ecf95bf64df69e9ee4addafda80500a3a11cca749b74cc3d9`|
|执行回报交付前含 v3 追加段版本|`756008795134ce424dd6ce761c477390f6171d147ba7bdfaab06610807d6026c`|

决策与执行回报均位于 `docs/baselines/`，执行回报为 `ALP15-PRECISION-HOLDOUT-PREP-EXECUTION-20261010.md`。追加由执行者独占完成，未修改被验收脚本和测试；验收者补读后区分旧 23-test 版本与新 28-test 版本，不将旧测试用于替代 v3。

本机解释器 `/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3`，Python 3.12.10，`macOS-27.2-arm64-arm-64bit`，pandas 3.0.5、pyarrow 25.0.1、physical-ai-av 0.2.0。CUDA/驱动/设备资源不适用本机 CPU 工具验收。环境设置 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''`。

实际调用 `environment_identity()` 只通过 importlib.metadata 定位发行版并读取源码/AST：dataset.py SHA 为 `b35f5f1bbb42ba92c83c835f7525c33e8ee915eb5ef071ac3dd972d91b0d9560`，__init__.py SHA 为 `40f7c62c601c74f9b44460745a137db8d4fafdff7d9719b402e962e6d97c22f0`，公开入口仅 HfRepoInterface/PhysicalAIAVDatasetInterface，`official_api_called=False`。独立包装确认没有导入执行 physical_ai_av 或 torch；pandas/pyarrow 仅处理新合成 metadata。

固定生产 metadata 的三份 bytes/SHA、CSV 7076 bytes/Git blob、数据 revision `33f9bf447ed3bcb7d545ce13f4226f824214fafb`、API/源码身份保持原决策值。脚本 DECISIONS 三项与实际文件 SHA 逐一匹配；沒有 CLI 夹具/coverage/force/skip 来源门禁开关。

## 2. 实际验证、命令与日志

工作目录 `/Users/USER/Documents/ChatGPT/Lighten_VLA`。实际全量回归：

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  /Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B \
  -m unittest discover -s tests -p test_prepare_alp15_precision_holdout.py -v
```

28 tests / 0.787 s / OK / exit 0，无跳过；完整 stdout/stderr 工具 session `82653`、chunk `b2c6ef`，终态补取 chunk `ad5841`。原有 23 项与 v3 五项均在本轮冻结源码实际执行。

独立验证包装使用同一解释器 `-B` 执行 Python 标准库及已有 pandas/pyarrow；完整提交 chunk `f83d35`、session `5076`、完整 stdout `969652`、exit 0，UTC `2026-10-10T15:47:50.501776+00:00` 至 `2026-10-10T15:47:51.340253+00:00`。包装包含独立 14-clip 数据、真正 parquet roundtrip、逐层 Counter 复算、CSV 文本检查、来源失败真实 CLI、拒覆盖、禁止打开二进制的 guard、其他必需 null 负例及生产计数函数正负例。所有临时数据由 TemporaryDirectory 自动清理，没有新增正式 census 目录。

编译实际使用 `py_compile.compile(source, cfile=<TemporaryDirectory>/<stem>.pyc, doraise=True)` 分别编译脚本与测试，exit 0；.pyc 仅位于临时目录，没有写源码目录。追加独立全缺失草案与空池校验 chunk `64bc6a`，exit 0。

执行者新 v3 归档日志也实际全文读取/核验，不覆盖它们：

|归档原始日志，位于 evidence/holdout_prep_local_20261010/|bytes|SHA256|
|---|---:|---|
|`unittest-v3-1545.log.txt`|4942|`92cb68d6dd1a61bdd4200e30f00ed0ca6bf287c9a703de1ec877c600a068f18d`|
|`py_compile-v3-1545.log.txt`|0|`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`|

前者记录执行者 28 tests/0.869 s/OK，后者为空编译诊断。它们与本轮独立日志分开；本轮不重造先前失败日志。执行回报 v3 追加段及日志 SHA 核验 chunk `5bc74c`。

## 3. radar 缺失契约与独立复算

独立 14-clip 合成集合包含 chunk0/3119 与其它块，三种 split，七个原始 radar null、真实字符串 `nan`/`None`/`unknown`/`real`；另有一个 clip_is_valid=False、一个必需 feature=False。只有原资格 bool/feature 及整块引用影响资格，radar 缺失不作为资格筛除条件。

直接用原始 fixture 行集合和 Counter 独立计算预期，再与脚本各层输出比较：

|层级|clips|radar missing|非缺失类别总数|验证范围|
|---|---:|---:|---:|---|
|all_metadata|14|7|7|全部原始行|
|qualified|12|7|5|仅排除资格 false 的两行|
|known_chunk_only_baseline|10|6|4|仅追加排除0/3119|
|after_usage_exclusion|9|5|4|再保守排除引用所在 chunk8|
|official_test_priority_pool|5|5|0|保留真实 test 标签，全 radar 缺失仍合格|

各层每个 split/chunk/country/month/hour_of_day/platform_class/radar_config 的非缺失分组之和 + missing 数等于 clips；missing 数为数值 int。normalize 前后的原 radar Series 用 `pd.testing.assert_series_equal` 比较相同，不填充，不造类别，不改原输入。

eligible DataFrame 的 `radar_config_is_missing` 为 bool dtype。独立 CSV 文本解析确认五个 null 对应空字符串/`True`，四个非缺失类别保留原字面值/`False`，不是将字面 `nan`、`None`、`unknown` 误当缺失。真实 parquet 写入/读取后七个 null 与非缺失类别仍分开；夹具自算 SHA 只显式用于内部纯数据函数，不能从生产 CLI 解除固定来源。

另以全部 radar 缺失的三行夹具复核可选 draft：两条 test clip 全被保留，draft counts 为 clips2/missing2/非缺失类别0，各列合计完整，state 仍待审查；全 usage 排除形成空池时，两候选层 clips0/missing0/空分组均可核对。不是生产抽样或最终名单冻结。

独立逐列将原必需字段设 null：chunk、clip_is_valid、split、五输入 feature、country/month/hour_of_day/platform_class 共12项，全部 PreparationError。radar 列缺失、资格 bool 非bool、chunk bool/float/string/负数、索引重复/非canonical/键错位等原测试亦通过；本例外没有变成一般 metadata 缺失豁免。

生产计数函数的独立合成契约正例只隔离检查：all_metadata/qualified 各306152且各 radar missing145391，原305952/3144与真实 split 基线。两个 radar 缺失层分别传0、145390、145392、306152共八负例全部拒绝；生产 CLI 无传入该合成 summary 的入口。此正例说明固定门禁逻辑，**不证明实际服务器 count 已复现**。

## 4. 真实 CLI、安全边界与失败产物

独立真实子进程命令为：

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  /Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B \
  /Users/USER/Documents/ChatGPT/Lighten_VLA/scripts/prepare_alp15_precision_holdout.py \
  --root <new-temporary-root> --metadata-root <new-temporary-root>/metadata \
  --usage-root <new-temporary-root>/assets --output <new-temporary-root>/failed
```

实际完整 argv 及临时绝对路径保存于 chunk `969652`。输入为上述14行合成 parquet，未 patch 生产 CLI、固定来源、发行版或预期 SHA：exit 1，stdout `failed`，stderr 空；新目录仅 `report.md` 与 `run-manifest.json`，首诊断 `PreparationError: 固定来源身份不符: clip_index.parquet`。manifest 保存三决策、expected metadata、实际来源前后身份、环境、阶段、exit及 False flags，report 的实际 SHA 与 manifest artifact_sha256 匹配。没有产生草案、eligible 或伪造生产成功。

同命令再次运行 exit 2，stderr `output已存在，拒绝覆盖`，两个已有产物字节 SHA 未变。--help 未提供 fixture/coverage/force/skip；实际追加四个未知绕过参数均 argparse exit 2。原测试另确认 root 外 output 不创建。读取来源身份失败不调用联网构造器或替换版本，也不把失败数据当正式 candidates。

独立 scanner 夹具放入六种二进制哨兵 `.pt/.pkl/.zip/.safetensors/.mp4/.npy`，对 Path.open 设置一旦打开上述文件即抛错的 guard；census_usage 与 file_plan 均通过。只枚举 filename/stat，保留 binary_payload_not_verified、匿名缓存 unresolved2；引用 clip 所在整 chunk8 被排除，scan_stable=True，但 coverage_complete=False。文件计划五路 quote 均 null/pending、official_api_called=False。原测试还覆盖 receipt 仅为未核 payload 报告、路径碰撞/symlink/外部引用/匿名映射/JSON key/损坏日志/读取失败/新增删除及同size+mtime SHA变化，不以 False coverage 跳过台账。

源码静态边界未导入 torch/pickle/physical_ai_av/HF/network 模块；官方 CSV 模板解析依 v2，不调用内部 Features 或构造数据接口。元数据 parquet 本身按准许读取，二进制哨兵仅为新合成文件，不触碰真实模型资产。

## 5. 验收门禁与尚未完成项

|门禁|本轮结果|
|---|---|
|冻结 script/test/三决策与来源代码|实际 SHA 匹配，复验前后不变；包公开入口/版本/源码身份通过|
|radar 原始缺失与资格|Series/真实 parquet/CSV 独立核验通过，null 不减少资格，不推测缺失原因|
|分组+missing与CSV bool|五层、draft与空池合计成立；真 nan/None/unknown 与 null 区分；strict bool标记对应|
|其余必需字段/生产145391|12项 null 独立负例拒绝；生产固定缺失量负例拒绝，其它原门禁保留|
|CPU tests/compile|本机28tests全部通过；两文件临时 cfile 编译成功|
|来源失败真实CLI/拒覆盖|exit1留两失败产物，拒覆盖exit2且字节不变，无生产夹具绕过|
|usage/文件计划边界|二进制 open guard通过，引用整chunk排除，quote null，coverage/drive False|
|服务器真实census|本文未执行、未验收；独立远端终态回报 pending|
|资格范围/最终名单/输入QC/精度|全部仍待后续明确证据；本工具成功不替代这些门禁|

主任务另外报告：五文件新远端快照 SHA 匹配，既有服务器 CPU 28tests/0.886s/exit0、py_compile0，真实 census session53684 尚在运行。该信息是主任务通知，非本验收者联网实测；本文不从 live 进程推出真实 census 成功，也不将本机 pandas3.0.5/pyarrow25.0.1 等同远端版本。远端产物、实际306152/145391/基线计数、前后来源 SHA、usage覆盖缺口与终态日志由主任务单独归档/复核。

因此可以报告“留出集准备 v3 本地工具技术验收通过”。coverage_complete 与 drive_disjoint_verified 保持 False；未知历史完整排除、最终测试独立性、raw窗口/GT、下载报价/receipt、抽样与CI匹配、固定样本量/规划、模型采集、独立精度、Orin16GB/300ms、闭环和官方原入口均未由本文完成。脚本/tests SHA 变化须另行复核；缺陷由执行者按决策修复，验收者不自行改实现。
