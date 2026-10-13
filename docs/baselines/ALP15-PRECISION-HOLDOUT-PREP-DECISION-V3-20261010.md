# 留出集准备 v3：保留官方 radar_config 缺失观测

- 决策编号：D-ALP15-PRECISION-HOLDOUT-PREP-20261010-03；日期2026-10-10（Asia/Shanghai）；决策者主任务 Sol；执行及验收使用 GPT-6.1 Sol high。
- 上游 v1/v2 文件保留，SHA 分别为 3efbfea76b5b336d5199c93337863bff49c120f2b3bd92c9919d6f26e979297c、32b534ea574677928fca9c6e9277498807f00839fe8de852f14f01d7a9e7a23b。本 v3 仅覆盖雷达粗元数据空值、对应计数/CSV 表示和来源门禁绑定，不改变模型输入资格、排除、下载及评测授权。
- 目标：使已授权 CPU 历史使用清点准确消费官方元数据，不因合法雷达配置缺失而删除样本。风险高（评测资格），操作限本地代码/合成测试和既有服务器 CPU 只读文件。

## 事实依据

主任务实际 SSH 只读 pandas 查询退出0（8a5242）：三个 parquet 各306152行、索引 clip_id 无空；clip_index 真实包含 clip_is_valid/bool、chunk/int64、split/object且均无空；五必需 feature 均 bool 无空。collection 的 country、month、hour_of_day、platform_class 均无空，radar_config/object 有145391个空值。

紧接着实际字节/SHA复核退出0（dace90），与固定 NVIDIA PhysicalAI 数据 revision 33f9bf447ed3bcb7d545ce13f4226f824214fafb 的冻结来源一致：

| 相对文件 | bytes | SHA256 |
| --- | ---: | --- |
| clip_index.parquet | 11132300 | 9630a3de5ee057ded194deaa94d3b9b42af6f18c301785706e50141314a6935c |
| metadata/feature_presence.parquet | 11209157 | a9946feb269124e3a7cbfe22d97669585735134f634c2ca97332e7abc4314790 |
| metadata/data_collection.parquet | 11339400 | 9624750cadabf8028264cc1404e75e81777654be9def06b00e35c9f834b173d5 |

因此缺失不是本地文件损坏证据，也不表示摄像头或 egomotion 不完整。没有已核验的缺失原因，不推断无雷达、天气、车辆状态或传感器故障。

## 执行规格与所有权

原执行者继续唯一写 scripts/prepare_alp15_precision_holdout.py、对应 tests 和当前执行回报；主任务唯一写本决策。独立验收者另写验收文档，不改代码。

1. 所有原必需列仍必须存在、键唯一/匹配、索引 canonical、类型严格。只允许 collection.radar_config 空值；其余字段的无空门禁不变，资格 bool 和 chunk/split 不填补。
2. 原始 radar_config 保持缺失，不改成虚构配置，也不丢弃对应 clip。资格只由 clip_is_valid 和五必需输入 feature 决定，不读误差。
3. 各计数层级增加 missing_by_column 数值计数；by_radar_config 只列非缺失真实类别，缺失数单独记录，避免把缺失转成字符串 nan/None/unknown 与合法类别撞名。其它分组也保持类别数与缺失数可核对；每列非缺失组计数之和加缺失数须等于该层级 clips。
4. eligible CSV 保留 radar_config 原始空字段，并增加严格 bool radar_config_is_missing 列；该标记是缺失事实，不是场景标签。摘要/report 明示 missing radar_config 不影响输入资格，不证明覆盖充足。
5. 正式来源固定SHA不变；全局 metadata-counts 的 radar_config 缺失数必须复现145391，qualified 仍306152，已知块排除及 split 基线计数仍沿 v1。合成夹具不受生产数量约束，不能从CLI开放夹具绕过。
6. DECISIONS 绑定新 v3 SHA，同时保留 v1/v2。来源不符或异常类型不得以本例外绕过。CSV模板解析/top-level API/no网络构造器边界沿v2。

## 验证与停止条件

新增合成测试：radar null 允许并保留、eligible不因缺失减少、CSV标记与空字段对应、每层类别+missing合计完整；真实字符串 nan/unknown 类别不与缺失合并；country等其它字段null仍失败；资格布尔null仍失败；生产缺失数不符失败。所有既有23tests及新增tests通过、py_compile0，归档真实日志与SHA。

执行者冻结代码后，主任务核对并上传小型新快照（拒覆盖），在既有 L20 venv 用 CPU 测试及新输出目录运行真实 census。禁止安装、GPU、模型、PT/ZIP payload读取、大下载、停止其它进程或重写旧资产。输入变化、固定来源计数不符、安全范围异常即留失败产物并停止升级，不换版本/依赖绕过。

验收按决策→执行→独立复核进行。coverage_complete 仍False，真实 census 成功也不自动代表未知历史全部排除、最终名单独立、输入窗口QC或精度通过。本决策不批准最终抽样/模型评估，抽样与CI匹配问题另行解决。

回滚：保留旧SHA及旧测试日志，若新实现失败只停止使用新快照；不删除或恢复覆盖用户文件，不改官方元数据。允许修复局部实现并以新日志复验，不能伪造缺失来源。
