# ALP15 未见数据资格准备：本地 CPU 执行回报

当前版本为 **v3 本地冻结、待独立验收及远端 CPU census**：28 tests/0.869s/exit 0，py_compile exit 0。script SHA `ccebd4d237cf5229a422844579e5871405291e0f01f957f5600fda8148f04b9c`，test SHA `c102a6f54abe950a7a58acc94aafe459d3b0126aa87b6e419429c890243ed5a5`。v3 实施与完整日志身份见文末追加；以下 v1/v2 记录保留为历史，不能将旧 23-test 结果当 v3 回归。

- 对应决策：D-ALP15-PRECISION-HOLDOUT-PREP-20261009-01 与 -02（v1 + v2，冲突处以 v2 为准）。
- 执行者：`holdout_prep_execution_20261010`，唯一拥有本准备脚本、对应测试和本新回报；主任务继续拥有旧决策、旧审计与状态文件。
- 日期：2026-10-10；本次暂停后验证窗口 UTC `2026-10-10T15:37:16Z` 至 `2026-10-10T15:37:42Z`（Asia/Shanghai 23:37:16 至 23:37:42）。早期实施开始未单独捕获 UTC，不补造时间。
- 第一阶段历史状态：`local_cpu_verified_blocked_by_radar_config_null_policy`；后续由文末 v3 决策实施记录替代。
- 第一阶段历史结论：本地冻结代码及 23 项 CPU 回归通过；当时未冻结为可部署验收版本。当前仍未运行真实 metadata/历史资产 census，未取得未见数据资格或精度验收结论。

## 决策与代码身份

完整读取 AGENTS.md、冻结 v1/v2、DAY-END-SUMMARY 和 PRIORITY-STATUS 后续接；不使用 superpowers。v1 的内部 Features 方案在实施前已由 v2 纠正，本代码仅按已核验 CSV 原始 `chunk_path` 做标准库格式解析，没有调用官方 API、内部类、私有 API 或伪造接口对象。

| 文件 | SHA256 |
| --- | --- |
| `ALP15-PRECISION-HOLDOUT-PREP-DECISION-20261009.md` | `3efbfea76b5b336d5199c93337863bff49c120f2b3bd92c9919d6f26e979297c` |
| `ALP15-PRECISION-HOLDOUT-PREP-DECISION-V2-20261009.md` | `32b534ea574677928fca9c6e9277498807f00839fe8de852f14f01d7a9e7a23b` |
| `scripts/prepare_alp15_precision_holdout.py` | `73c6d06b57b73f56825804cf554793d5127de3ba59857a084627028802634eb7` |
| `tests/test_prepare_alp15_precision_holdout.py` | `50f6518f8185d571f4ad36ed942da2e471322d3a6ba0d9477e3c28e5659f6ca9` |

暂停后实际重新读取代码 SHA，与暂停版本逐一相同（工具 chunk `e82289`）；最终环境核验后重读仍相同（`6feadf`）。上述是本地待复验的冻结身份，不代表生产流程已通过。

## 实际环境与执行边界

既有解释器 `/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3`；Python 3.12.10，macOS-27.2-arm64-arm-64bit，pandas 3.0.5、pyarrow 25.0.1、physical-ai-av 0.2.0。本轮 CPU 文件处理未初始化 GPU，未加载模型或 checkpoint，未读取 PT/pickle/ZIP payload，未安装依赖，未执行网络、SSH、数据报价或下载。CUDA/驱动和目标 L20/Jetson 环境没有由本执行者实测，不能以本机环境替代远端环境。

以 `importlib.metadata` 定位发行版，读取文件字节与 AST 而不导入执行 physical_ai_av，复核结果 exit 0（`6feadf`）：

- `dataset.py` SHA256 `b35f5f1bbb42ba92c83c835f7525c33e8ee915eb5ef071ac3dd972d91b0d9560`。
- `__init__.py` SHA256 `40f7c62c601c74f9b44460745a137db8d4fafdff7d9719b402e962e6d97c22f0`。
- `__all__` 为 `HfRepoInterface`、`PhysicalAIAVDatasetInterface`；`official_api_called=False`。

本次恢复初始，主任务通报 SSH `REDACTED_IP:2222` 为 Connection refused、exit 255；本执行者没有重试连接。随后主任务重新认证并完成 metadata 只读查询（`8a5242`、exit 0），发现 `radar_config` 有 145391 个 null，当前严格无空门禁真实不成立。连接阻塞已由主任务解除，但部署仍停止在 metadata 口径决策边界；本执行者不修改现有门禁，等待新 v3 决策及来源 SHA 复核。

主任务 `8a5242` 的实际字段摘要：三个 parquet 均 306152 rows；clip_index 确含 `clip_is_valid`（bool、无空）、`chunk`（int64、无空）、`split`（object、无空）；presence 五 feature 均 bool、无空；collection 的 country/object、month/int64、hour_of_day/int64、platform_class/object 无空，radar_config/object 有 145391 个 null。这里是父任务实际只读观测的转述；本执行者未读取服务器 parquet，该次来源 SHA 复核仍由主任务负责。

## 变更范围与关键行为

实际变更为准备脚本、对应测试、本新回报；此次收尾经主任务新增授权，将实际 `/tmp` 日志用 `apply_patch` 原样归档为两个小型 evidence 文本文件。既有其它工作树变更不改、不覆盖。

脚本固定三项 metadata 的 bytes/SHA、306152 行与仅排除 0/3119 的 305952 clip、3144 chunk 及真实 split 基线，以及 CSV 7076 bytes/Git blob、发行版与源码身份。生产 CLI 不提供夹具门禁绕过、coverage True、force 覆盖或自动下载选项。小型测试夹具只用于直接调用内部纯数据函数的明确身份契约；真实 CLI 用小夹具必定来源失败，不当成生产成功。

显式 usage roots 去重后清点全部指定文本类型；JSON key/value/list 与日志行里的 UUID 均记录为结构位置/行号，引用按 lowercase canonical 匹配 metadata，整 chunk 保守排除。metadata clip_id 本身必须为小写 canonical UUID。未映射 UUID、损坏文本、匿名缓存、二进制未核验、外部路径、symlink 与扫描变化保留台账；不输出任意日志片段、JSON key 内容或异常正文。通用模型/source revision 记 `foreign_or_unknown_revision`；只有明确 `dataset_revision` 或明确 dataset repository 的 revision 冲突才记 `revision_conflict`。

扫描冻结开始清单与末端反向清单，复核新增/删除及 bytes/mtime，并重算文本末端 SHA。PT/pickle/ZIP 等只取文件名和 stat；匿名缓存只在同一明确对象含 clip_id + 文件引用时绑定，不猜目录顺序。输出和冻结代码目录显式排除；冻结决策从 `Path(__file__).parents[1]/docs/baselines` 校验，与生产 `--root` 资产目录分离。源码直接位于实际项目 root 时，仅排除准备脚本和测试本身，而不把整个历史 scripts root 排除。

文件计划仅使用经过安全校验的 CSV 模板；五路径须唯一、相对 POSIX ZIP 路径且不经 symlink 逃逸。已有 receipt 只按固定 dataset repository/revision 和真实安全目标路径关联；receipt 报告的 SHA 明确标为 payload 未核验，不能当官方报价或实际 payload 验证。无固定 revision 报价时 `official_bytes`、`official_sha256` 为空、quote pending；没有规模外推。默认不生成草案；可选草案仍为待覆盖审查、报价与输入 QC 的官方 test 可行性池，不是最终样本。

`coverage_complete=False`、`drive_disjoint_verified=False` 保持；机器扫描完成只可能得到 `completed_usage_census_pending_coverage_review`。本地测试输出不证明历史资产范围完整、行程独立、输入窗口 QC、官方原始模型跑通或精度门禁通过。

## 本次真实命令、日志与验证结果

实际回归命令（工具 chunk `c555cc`）：

```sh
set -o pipefail
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  /Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B \
  -m unittest discover -s tests -p test_prepare_alp15_precision_holdout.py -v \
  2>&1 | tee /tmp/alp15-holdout-prep-20261010.YzSHz6/unittest-resume-local-1537.log
```

实际结果 `Ran 23 tests in 0.863s`、`OK`、exit 0，无跳过。完整 stdout/stderr 已从真实 `/tmp` 文件读取，用 `apply_patch` 原样归档为 `docs/baselines/evidence/holdout_prep_local_20261010/unittest-1537.log.txt`；4036 bytes，SHA256 `c5dfad893fa59a1ce9fc91cacb33f4794672deb8b6e069e556a5f01092a2f3e9`。源日志与归档 SHA 相同（`b83b1a`），不是根据回报重造日志。

实际编译命令（`9dbf51`）：

```sh
set -o pipefail
PYTHONPYCACHEPREFIX=/tmp/alp15-holdout-prep-20261010.YzSHz6/pycache-resume-1537 \
  /Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -m py_compile \
  scripts/prepare_alp15_precision_holdout.py \
  tests/test_prepare_alp15_precision_holdout.py \
  2>&1 | tee /tmp/alp15-holdout-prep-20261010.YzSHz6/py_compile-resume-local-1537.log
```

exit 0，无诊断。实际空日志归档为 `docs/baselines/evidence/holdout_prep_local_20261010/py_compile-1537.log.txt`，0 bytes，SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`，与原文件一致。pycache 仅位于 `/tmp`，未写入项目源码目录。

| 门禁 | 本地实证 | 限制 |
| --- | --- | --- |
| 固定来源身份 | 小夹具 bytes/SHA 负例、CSV blob/bytes 负例、包源码/公开入口核验通过 | 真实全局 parquet/CSV 未在本次本机读取 |
| metadata 键集与 dtype | 真正写入/读取合成 parquet；唯一/规范字符串 UUID、集合对齐、严格整数/布尔、空值/缺列负例通过 | 父任务实测 radar_config 145391 个 null；现严格门禁不允许部署，须新决策 |
| 模板路径 | 字段、conversion、format spec、穿越/URI/控制字符、碰撞、symlink 负例及合法填零通过 | 官方 CSV 实际模板远端端到端解析待执行 |
| 使用引用 | JSON key/list/嵌套、损坏 JSON、日志、多 source、大小写 UUID、整 chunk、unknown 引用通过 | 真实历史内容台账尚未产生 |
| 资产/覆盖证据 | 显式匿名 cache 映射、未解决资产、外部路径、重叠 roots、symlink、读失败、二进制 payload 不读通过 | scope/匿名索引/receipt 独立验收 pending |
| 输入稳定与拒覆盖 | 新增/删除/改写、同 size+mtime 但 SHA 变动、metadata/CSV SHA 变动、已有 output 拒覆盖、root 外 output 不创建通过 | 生产并发资产变化须实际 census 后再审 |
| 候选与可选草案 | 已知块基线与使用排除分离、官方 test 优先、固定 seed 可复现、不足时不转 train/val 通过 | 本轮未抽样、未冻结最终测试名单 |
| 生命周期与失败留痕 | 真实 subprocess CLI 来源失败留报告/manifest，exit 1；再次同 output exit 2 且 hash 不变 | 没有真实生产 CLI 成功结论 |

## 失败、修复与升级历史

昨日代码仅含模板解析，测试引用尚不存在的 census；本轮完整实现后初次 2-test 回归因 macOS `/var` 与 `/private/var` canonical root 不一致而失败，exit 1（`e53450`）。修复为显式声明 root 的安全 canonical 比较，并拒绝 leaf symlink root，未放宽扫描到范围外。

添加报告拼接时有一次 SyntaxError 导致测试导入失败，exit 1（`ca7301`）；补齐字符串连接后 20-test 回归通过（`4eb0fa`）。早期这些失败日志留在工具原始记录，未另存独立 stderr 文件，不补造所谓完整失败日志。

随后 21-test 中间版本通过（`d6581a`），script SHA `9797b24dc2b1f2ecaca92f9e1ae2193c4fafc0f83a0a674a9cb418762d145826`、test SHA `52fe49496f3d12820b519252a8f4b6b29a684f4ae93b4e14eebaa68a307e2707`；主任务审查指出 revision 分类与 UUID 大小写漏排风险，已撤回该中间冻结并修复，23-test 版本通过（`2319f3`）。本次恢复后再次跑当前 23-test 版本及 py_compile，结果如上；没有将旧测试结果冒充新源码测试。

主任务还指出真实 collection 粗字段可能合法缺失、clip_is_valid 的实际文件归属未在 v1 中唯一指定。本执行者保持严格门禁并升级；主任务后续 `8a5242` 已核实字段归属与 null 计数，确有 radar_config 145391 个 null。该事实与当前 strict collection 无空门禁冲突；代码不修改、不部署，不删除行、不补造值。主任务准备新 v3 决策，仅拟允许 radar_config 的官方 null 显式保留为 unknown 层级；尚未授权本执行者实施，本回报不提前把拟议口径写成已实现。

主任务另独立复跑相同本地 SHA 的 23 tests，0.767s、exit 0（session 88055、chunks `c043c5`/`6842ce`）。此为主任务实际回归的转述，不冒充本执行者日志；本回报归档的原始日志来自本执行者 `c555cc` 运行。

## 待决项与后续安全步骤

1. 主任务已核实字段归属与 radar_config null 计数；继续复核实际来源 SHA、canonical clip_id，并发布明确的新 v3 决策后，执行者再据其范围实施。当前不放宽、不换来源、不删除行、不补造观测；本 v1/v2 严格版本不得当可部署验收版本。
2. 对本地冻结 SHA 完整复审；在既有 L20 pandas 2.3.3/pyarrow 22.0.0/physical-ai-av 0.2.0 环境独立回归。上述远端版本仅来自先前主任务记录，本次未刷新，不可代替实际回归。
3. 在全部来源/schema 门禁明确后，主任务新建拒覆盖的代码 snapshot（含固定决策副本），显式纳入原五 root 和 logs、incoming、旧 clip delivery、旧 L20 audit、Self-realization。JSON 数为 0 的目录仍扫描文本；默认无草案，运行一次 CPU census 并保存真实输入前后 SHA、环境、argv、日志和产物。
4. 独立审定 census 资产范围与未解决项后，才考虑名单、输入窗口 QC、报价和采集前统计方案。本执行者不自行连接远端，不下载，不开始模型评测；当前 coverage 仍为 False。

## v3 追加执行：官方 radar_config 缺失的局部处理

- 新决策：D-ALP15-PRECISION-HOLDOUT-PREP-20261010-03。
- 完整读取并现场重算 `ALP15-PRECISION-HOLDOUT-PREP-DECISION-V3-20261010.md` SHA256 为 `e53284d0ebcba64787b4f6ca4f5d60bbef8e779f62f29d7d1845b1910c1731f8`（`62aa9a`）；绑定原 v1/v2，不改三个旧决策文件。
- 本阶段实际窗口 UTC `2026-10-10T15:43:04Z` 至 `2026-10-10T15:45:30Z`；本地同一既有 Python/pandas/pyarrow 环境，执行者未访问服务器。
- 当前状态：`local_v3_cpu_verified_pending_independent_acceptance_and_remote_census`。
- 当前冻结：script `ccebd4d237cf5229a422844579e5871405291e0f01f957f5600fda8148f04b9c`；test `c102a6f54abe950a7a58acc94aafe459d3b0126aa87b6e419429c890243ed5a5`。旧 SHA 和旧原始日志保留，不覆盖。

v3 事实依据由主任务 `8a5242` 与 `dace90` 对真实行数、dtype/null 和固定 bytes/SHA 逐项复核，均 exit 0；本执行者依新冻结决策实施，不推测缺失原因。主任务又以 `1c2480` 实测服务器 pandas 2.3.3、pyarrow 22.0.0、physical-ai-av 0.2.0，十个显式 roots 均存在、非 symlink；这些是主任务观测的转述，不是本执行者远端运行。

本阶段修改限以下内容：

1. 仅允许 collection.radar_config 的 null。其列仍必须存在，原始值不填充、不转成 unknown/nan/None 字符串、不删除 clip；其它原必需列无空、canonical 索引、键集、bool/chunk/split 门禁保持。
2. 各层级 counts 增加 `missing_by_column` 数值计数，分组只列非缺失真实类别，并校验每列类别总数 + missing 等于 clips。真实字符串 `nan`、`unknown` 与 null 独立计数，空池同样输出可核对的零计数。
3. eligible CSV 的 radar_config 缺失保存为空字段，增加严格 bool `radar_config_is_missing`；资格仍仅由 clip_is_valid 和五输入 feature 决定。生成 report 明示雷达粗字段缺失不影响输入资格、不证明覆盖充分，并列各资格层 clips/missing 摘要。
4. 生产门禁新增全局 all_metadata 和 qualified 的 radar_config 缺失数均须为 145391；306152 合格 clip、305952/3144 已知块基线和真实 split 计数不变。DECISIONS 增加 v3 固定 SHA，没有 CLI 夹具或缺失数量绕过。

新增五项测试验证原始 null 保留且资格不缩减、真实类别 nan/unknown 不撞缺失、每层类别+missing合计、CSV 空字段/布尔标记对应、其它粗字段或资格 bool null 仍失败、radar_config 缺列仍失败、生产缺失数不符失败。另真正写入/读取合成 parquet，证实 null 与真实字符串类别经 pyarrow/pandas roundtrip 后仍可区分。生产缺失计数测试只直接调用计数门禁函数，不替换 CLI 来源读取/hash；原真实 CLI 小夹具仍来源失败。

首轮 27 tests/0.828s/exit 0（`4863f6`）；补齐真实 parquet null roundtrip 和 report 计数摘要后，冻结源码的正式本地回归为 **28 tests/0.869s/OK/exit 0，无跳过**（`21e89a`）。本阶段没有失败尝试，不以先前 23/27 tests 冒充最终 28-test 版本。

```sh
set -o pipefail
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  /Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B \
  -m unittest discover -s tests -p test_prepare_alp15_precision_holdout.py -v \
  2>&1 | tee /tmp/alp15-holdout-prep-20261010.YzSHz6/unittest-v3-1545.log

PYTHONPYCACHEPREFIX=/tmp/alp15-holdout-prep-20261010.YzSHz6/pycache-v3-1545 \
  /Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -m py_compile \
  scripts/prepare_alp15_precision_holdout.py \
  tests/test_prepare_alp15_precision_holdout.py \
  2>&1 | tee /tmp/alp15-holdout-prep-20261010.YzSHz6/py_compile-v3-1545.log
```

编译实际 exit 0、无诊断（`061f9e`）。实际 `/tmp` 全日志经读取后由 `apply_patch` 原样归档，源/归档 bytes/SHA 相同（`01f36d`）：

| v3 归档原始日志 | bytes | SHA256 |
| --- | ---: | --- |
| `evidence/holdout_prep_local_20261010/unittest-v3-1545.log.txt` | 4942 | `92cb68d6dd1a61bdd4200e30f00ed0ca6bf287c9a703de1ec877c600a068f18d` |
| `evidence/holdout_prep_local_20261010/py_compile-v3-1545.log.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

本地 v3 已按批准口径完成，可交独立复验；尚未完成服务器回归、真实 census 或范围验收。后续由主任务按冻结 SHA 新建拒覆盖快照，必须同时携带 v1/v2/v3 决策副本；默认无抽样/报价，在既有 L20 环境复验后再运行已授权 CPU census。coverage_complete 和 drive_disjoint_verified 仍为 False；最终名单、窗口 QC、采集前统计口径及精度验收均未完成。未执行 SSH、GPU、模型、PT/ZIP payload、安装、下载，也未改原始 metadata 或旧资产。
