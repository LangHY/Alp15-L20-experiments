# ALP15 未见数据资格准备决策 v2：纯离线模板解析

- 决策编号：D-ALP15-PRECISION-HOLDOUT-PREP-20261009-02。
- 日期：2026-10-09（Asia/Shanghai）；决策者：GPT-6.1 Sol high。
- 替代范围：修正 v1 中非顶层 `Features` 调用方案；本修订与 v1 非冲突条款共同构成当前执行规格，冲突时以本 v2 为准。
- 冻结 v1：`ALP15-PRECISION-HOLDOUT-PREP-DECISION-20261009.md`，SHA256 `3efbfea76b5b336d5199c93337863bff49c120f2b3bd92c9919d6f26e979297c`。原文不修改，保留为实施前发现约束冲突的证据；不能继续按其 Features 调用条款实施。
- 上游精度决策：`ALP15-PRECISION-VALIDATION-DECISION-20261009.md`，SHA256 `989b64ac1adfa3979556db82d035f71230d91132454b8eb35029eafacf0280db`，不修改。
- 本轮动作：仅写本修订文档，不实施/运行准备脚本，不连接服务器，不联网、不下载、不加载模型。

## 1. 冲突证据与修订依据

用户“仅使用 physical_ai_av==0.2.0 顶层官方 API”的直接限制优先。v1 错把官方源码中的内部 `physical_ai_av.dataset.Features` 当成可调用顶层接口，实施前应纠正。

本次已完整重读 v1，并核对相关来源报告与本地固定版本源码。`physical_ai_av/__init__.py` 的真实 `__all__` 仅为 `HfRepoInterface`、`PhysicalAIAVDatasetInterface`；`Features` 不在顶层公开列表。官方固定 `dataset.py` 已现场重算SHA256为 `b35f5f1bbb42ba92c83c835f7525c33e8ee915eb5ef071ac3dd972d91b0d9560`，与来源报告固定提交 `59d578b53ef186ffc3f1de1d1b2598717a982d64` 相符。

该源码215–217行的 `get_chunk_feature_filename` 规则为读取当前feature行的 `chunk_path`，执行 `.format(chunk_id=chunk_id)`。这是本轮**纯数据格式解析的依据**，不是批准调用内部类或宣称执行了官方 API。构造顶层 `PhysicalAIAVDatasetInterface` 会下载元数据并调用 `api.file_exists`，与本轮脱网边界冲突，不能为绕过冲突而mock接口、用`__new__`伪造对象、猴子补丁/注入下载器或换其它私有API。

## 2. 替换 v1 的具体实现条款

以下替代 v1 第1节 Features 调用、第4节“官方五feature常量及Features接口生成路径”及第6节对应测试要求：

1. 用既有pandas/pyarrow读取本地三个已冻结SHA的parquet与 `features.csv`，不导入/调用 `physical_ai_av.dataset.Features` 或其它内部模块对象，不实例化任何 PhysicalAIAV/HF 接口。纯census实现不需要 `import physical_ai_av`；使用标准库 `importlib.metadata` 核验发行版 `physical_ai_av==0.2.0`，读取安装位置的 `__init__.py`/`dataset.py` 字节或AST做公开入口与源码SHA核验即可，不执行源码。
2. `features.csv` 固定7076bytes、官方Git blob `65de094a4037902a40bd42de7b17082029b78fe2`。生产预检按 `sha1(b'blob '+ASCII(len(content))+b'\0'+content)` 精确核验Git blob，计算实际文件SHA256并写入run manifest；解析前后SHA必须相同。三parquet冻结bytes/SHA及306152行门禁仍沿v1不变。文件缺失/身份不符即停止，不自动获取CSV。不能只校验CSV结构而跳过字节身份。
3. 只读取明确五项feature行：`egomotion`、`camera_cross_left_120fov`、`camera_cross_right_120fov`、`camera_front_tele_30fov`、`camera_front_wide_120fov`。这些名称来自已核验feature_presence真实列及同revision CSV，不是私有API常量。CSV `feature` 行键必须唯一、五项完整；其 `chunk_path` 非空字符串，不能以NaN、重复行或默认模板补齐。
4. 对经安全校验的CSV原始 `chunk_path` 调用 Python 标准字符串 `.format(chunk_id=chunk_id)`；模板内容必须来源于CSV，不手写猜测 `camera/...` 或 `labels/...` 路径，不截取再重新拼模板，不拼任意root绝对地址，不执行模板为代码。输出记录 `resolution_method=verified_features_csv_chunk_path_format`、CSV SHA、原始模板、feature、chunk_id、解析路径，明确 `official_api_called=False`。
5. chunk_id 来自官方clip_index真实chunk列。先保证整型dtype非bool、无缺失，规范化为Python整数时验证转换不改变值；拒绝bool/float/字符串/负数及不在当前metadata chunk集合的chunk。单元测试可传小型明确metadata集合，生产不得绕过来源门禁。

## 3. 模板与相对路径安全门禁

模板解析先用标准库 `string.Formatter().parse` 检查，不对未审查模板直接format。允许字段仅为恰好一次的 `chunk_id`，禁止位置字段、其它变量、属性/索引访问（如 `chunk_id.real`/`chunk_id[0]`）、conversion（`!r/!s/!a`）、嵌套格式字段。format_spec只允许空串、`d`或有限宽度的十进制整数格式（例如官方模板的`04d`）；有限宽度限定1–12，禁止对齐/填充表达式及异常巨大宽度。若已核验官方CSV实际模板不符合该明确集合，则记录冲突并升级，不能悄悄放宽或改CSV。format结果应再校验，下游只消费通过的相对路径。

模板字面量及解析结果均拒绝：空串、空路径组件、重复斜杠、`.`/`..`组件、绝对路径、反斜杠、盘符/冒号、URI scheme、查询/fragment、NUL与控制字符、首尾空白。使用POSIX相对路径语义，不以OS自动normalize消除穿越痕迹。仅允许ASCII字母数字、下划线、连字符、点及分隔符 `/`，且后缀为`.zip`。不得把URL或root外路径当数据文件名。每个chunk解析五项必须得到五个唯一相对路径，不得两个feature碰撞；不符合时文件计划失败并保留diagnostic。

若检查本地是否已有计划文件，路径必须在显式metadata/数据root内，解析真实目标后确认不通过symlink逃逸；不打开ZIPpayload。文件名解析本身不能触发网络、download或原接口构造，也不表示实际ZIP已存在、下载或窗口QC完成。

## 4. 新增负测与验收条款

原 v1 所有census、coverage、metadata、输入稳定性、拒覆盖、sampling草案与失败用例继续适用；移除调用内部Features类的测试，新增：

- 固定CSV blob/bytes或SHA前后不符、重复feature、缺任意五feature、空/非字符串chunk_path。
- 无chunk字段、多次chunk字段、未知字段、位置字段、属性/索引访问、conversion、嵌套格式、超大format宽度和不允许format spec。
- chunk_id为True/浮点/字符串/负数/不在metadata集合，以及合法整数填零格式化。
- 模板或结果为外部绝对路径、`../`、反斜杠、URI/盘符、控制字符、空组件、重复斜杠、不是ZIP、两feature路径碰撞。
- 本地候选路径symlink逃逸已声明root，不读取PT/ZIPpayload，不实例化联网接口；实现源码/依赖边界检查不得出现内部Features或私有API调用。
- 合法CSV模板结果逐字对应固定源码所示格式化规则，仅用合成模板和经校验真实CSV验证，不导入内部类作“对拍”。报告区分纯模板解析与官方顶层API执行。

执行报告绑定本 v2 SHA与原v1 SHA；说明v1问题在实施前发现、本轮按v2完成，无伪造官方API对象。包版本、公开入口、源码SHA、数据文件hash、模板安全、CPU测试及实际census由独立Sol复核；任何内部接口/网络使用或安全门禁绕过即拒绝验收。

## 5. 保持的授权与执行路径

文件所有权继续限执行者的 `scripts/prepare_alp15_precision_holdout.py`、`tests/test_prepare_alp15_precision_holdout.py` 与新执行回报；本决策者仅拥有此v2文档。v1的usage-root/CLI、完整历史引用台账、保守排除整chunk、coverage默认False、official split=test候选、可选100clip草案、quote null与统计计划pending均保持。下载上限等待用户选择不阻止census，但不能据草案自行下载；100clip/单chunk不能作为充分独立性或功效证据。

推荐生产新输出改为 `experiments/precision_holdout_prep_20261009_v2/`，代码快照目录也显式使用v2；存在即拒覆盖。执行者读完v1与v2后，可按主任务授权实施CPU准备脚本；本次修订任务本身不执行脚本。无GPU/模型/训练/PTpayload/安装/联网/下载边界不变，最终名单、实际输入窗口QC、采集前统计计划与精度推理仍需后续独立决策。
