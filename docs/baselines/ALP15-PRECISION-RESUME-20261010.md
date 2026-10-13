# 精度优先阶段续接记录：2026-10-10

主任务唯一写入。上一轮为有实质进展：实际开发证据复算、失败留痕缺陷发现/修复和停止时产物保存；随后用户要求当日停止，goal设paused。今日产品goal查询返回active（updatedAt1791607116），据此续接，不重启旧实验。长期目标仍为Orin NX16GB、完整请求≤300ms、教师精度损失≤15%，未完成。

## 当前范围与所有权

先完全验证精度，FlashDrive/LightVLA新机制、训练、组合量化/KV、实体loader、ONNX/TensorRT继续不实施。审计和数据准备仅CPU；无新大文件下载。

- audit_v4_acceptance_20261010：独立审计冻结v4，只写新验收文档。
- holdout_prep_execution_20261010：唯一写prepare_alp15_precision_holdout.py、对应测试、今天的新准备执行回报；昨天实际代码仅43行模板解析，尚未完成census。
- precision_eval_design_20261010：只写采集前精度方案草案，不实施runner或采集数据。
- 主任务：本记录、原审计执行回报的v4收尾段；验证冻结SHA、远端CPU环境及以后新census，不改其它执行者拥有的文件。

当前文件未提交且用户未提供提交差异固定点，故本轮不宣称code-review skill提交差异流程已完成；按冻结SHA、决策和技术试验验收。已使用web-access核对联网边界，仅系统SSH访问既有服务器；experiment-design用于采集前草案，样本量、GT、失败处理和重复测量不凭旧均值自授通过。

## 已完成的本轮实际核验

主任务Python3.12.10实际命令`python -B -m unittest discover -s tests -p test_audit_alp15_precision.py -v`：18tests、12.398s、exit0（session64507，chunksa21514/e8c84d）。冻结脚本SHA85d978f4b9ebbadb55c168a9539af84316d95d2c1521305c6f5d9af3e34c6341、测试SHAdac4285a8f0f4874cde9b172deb03e65b119d78e2b96b0af58443e2ddea9e78a；正式v4 audit SHA0a46b5a13c9d2679843bbeccc74b87707bc37a723bcaad5e5a3cb6edc2087d7c与昨日代码身份一致。原执行者昨天被中断导致未补齐v4回报，主任务已追加当前可验证收尾，明确历史red日志缺失而非补造。

旧SSH复用会话失效，BatchMode认证失败exit255（ca8738）；ssh-agent无身份。随后按既有用户授权、既有隔离known_hosts严格检查恢复SSH，pwd=/home/USER/tmp/alp15-precision-20261010-control。不记录认证秘密，不改系统known_hosts或服务器依赖，不启动GPU/停止其它PID。

远端既有venv只读metadata、文件SHA和pyarrow行数查询exit0（fb0df0）：pandas2.3.3、pyarrow22.0.0、physical-ai-av0.2.0。三个全局parquet各306152行，SHA仍为9630a3de5ee057ded194deaa94d3b9b42af6f18c301785706e50141314a6935c、a9946feb269124e3a7cbfe22d97669585735134f634c2ca97332e7abc4314790、9624750cadabf8028264cc1404e75e81777654be9def06b00e35c9f834b173d5；CSV7076bytes、SHA61105260827f5cf193c6c3ecbfed13b185432d04899d52bff3a7430ab4c0f304。CSV实际列feature/directory/chunk_path/clip_files_in_zip、36行，五项chunk_path采用{chunk_id:04d}（7260de）。纯CPU读取，无模型/PT/ZIPpayload或下载。

## 尚不能宣布完成的部分

独立技术验收文档待归档；完整历史usage内容清点、未决资产范围复核、新名单/下载规模/窗口QC、采集前样本量和统计计划、教师与候选新raw自主生成及GT保存都未完成。旧开发集Hybrid+4.08%/KV4+1.02%与每者7/20clip退化>15%保持原结论，不能用其替代独立精度验收。Orin内存/300ms、闭环仿真亦未完成，goal保持active。

## 后续终态：v4独立技术验收通过

ALP15-PRECISION-AUDIT-V4-ACCEPTANCE-20261010.md已由唯一验收者新增，SHA e10ff555dfebed82757c462fe0da9d35adfe76f8194515f44a83a4b0889396ca；主任务完整读取全文。3.9.6/3.12.10各18tests、正常实际CLI、同步SHA非法结构/大整数失败六产物、拒覆盖、60坐标exact、252名称/计数、四输入及v1-v4产物、meanratio与全部尾部均通过。原执行回报的主任务追加段SHA f213d4f51f060a6b412f1a8ad81c4c72568f622fbcc3149e443180f8006d2890已交付前补读，昨日完整red日志缺失如实保留。

该终态只支持“已有开发证据审计通过”。不支持独立新精度/GT重算/Orin/300ms/原官方入口通过；上节其它未完成门禁不变。审计源码与测试不再改写，数据准备继续按独占文件边界完善。

## 用户明确继续后的本轮复核

用户在收工暂停后明确说“继续”，goal 查询返回 active，恢复精度优先工作。历史收工记录不改写。主任务读取当前 AGENTS、web-access、experiment-design 及预注册模板，全文读取独立评估草案和准备测试；无 superpowers。

SSH 只读预检使用既有隔离 known_hosts、StrictHostKeyChecking=yes、BatchMode=yes、ConnectTimeout=10，目的为查询三个 parquet 的字段/dtype/null。实际返回 `ssh: connect to host REDACTED_IP port 2222: Connection refused`、exit255（19387e），连接前即失败，远端 Python 未执行，不能声称取得空值事实。预检草拟路径中的 clip_index 已根据固定决策和脚本纠正为 metadata-root/clip_index.parquet（不是 metadata/clip_index.parquet）；下一次连接必须使用正确路径。未绕过信任、重新下载、尝试其它未授权主机或启动 GPU。

主任务用既有 `/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python -B -m unittest discover -s tests -p test_prepare_alp15_precision_holdout.py -v` 实际复跑：23tests、0.767s、exit0（session88055，c043c5/6842ce）。准备脚本SHA73c6d06b57b73f56825804cf554793d5127de3ba59857a084627028802634eb7、测试SHA50f6518f8185d571f4ad36ed942da2e471322d3a6ba0d9477e3c28e5659f6ca9，与暂停快照一致（b71d68）。此测试仅合成CPU夹具及失败CLI，不证明真实parquet schema符合脚本。

真实待决项仍是 clip_is_valid 字段归属和 collection 五粗字段是否存在合法缺失值；保持严格门禁，不靠猜测放宽。执行者仅做本地日志归档和回报收尾。独立验收者另审采集前草案（唯一写新review文件），重点抽样/整群CI估计目标与独立性前提。没有启动真实census、模型采集或大文件下载；目标未完成，待用户恢复端口转发后继续服务器事实核验。

## 端口恢复后的实际证据与 v3 修订

用户报告端口已开启。旧 socket BatchMode 认证失败255（9d31f1）；主任务按原授权和隔离 known_hosts 严格核验重新认证，pwd=/home/USER/6363ab）。无凭据落盘或输出，不改系统信任。

新 SSH 复用会话实际执行既有 venv 的 pandas.read_parquet + columns/dtypes/isna().sum 查询（8a5242，exit0）：三文件各306152行、clip_id 索引无空。clip_index 的 clip_is_valid/bool、chunk/int64、split/object均无空；feature_presence 五资格特征均 bool 无空。data_collection 的 country/object、month/int64、hour_of_day/int64、platform_class/object均无空；radar_config/object缺失145391。随后三文件 bytes/SHA 查询退出0（dace90），全部与 v1 固定来源一致。源文件未写入，未执行模型或下载。

因此真实 schema 字段归属疑点已解除，粗元数据全无空假设被反证。新增 v3 决策 D-ALP15-PRECISION-HOLDOUT-PREP-20261010-03，文件SHA e53284d0ebcba64787b4f6ca4f5d60bbef8e779f62f29d7d1845b1910c1731f8，仅允许原始 radar_config null；独立 missing_by_column 计数、候选CSV缺失flag、不删样本/不造类别、保留其余严格门禁，生产复现145391。执行者依据新决策实现；尚未实际census。

独立评估草案审查文件 ALP15-PRECISION-INDEPENDENT-EVAL-REVIEW-20261010.md，SHA b8652a5b21cd83eaf97f3ee2d5e9246709ac7638351ffdf7a9829103dbb13133 已全文读取。结论 draft_reviewed_not_ready：clip SRS/有限框与观察chunk bootstrap需明确匹配，CI覆盖率需校准，QC后框与预测时间轴需绑定，family功效/失败分母待冻结。原草案未改，不以算法分位数或重采样次数称精度上界有保证。

用户另要求周额度剩余2%时免费重置一次。本轮额度工具返回周窗口已用92%（剩8%），可用免费重置3次；未达阈值，未使用重置。当前工具只有只读查询，没有重置执行入口，不能承诺自动重置；不使用购买额度替代。

## 本轮收尾事实（目标继续 active，不再暂停）

用户进一步确认已开启明确请求下自动免费重置功能；授权范围为周额度剩余≤2%时一次免费重置。最新只读查询周已用96%、剩4%、免费次数仍3，未达阈值，未执行/未声称重置成功。不以重复查询代替实际执行入口或购买额度。

v3代码冻结、主任务本机28tests/0.911s/0、L20同SHA28tests/0.886s/0及编译0；独立本地工具验收通过。真实L20 CPU census session53684终态exit0，主任务复核产物SHA与来源内容稳定；219引用clip/21排除chunk，剩test61301clip/633chunk。1892匿名cache、63177外部引用等仍待分类，coverageFalse。远端详细命令、日志、大小/哈希、最初CSV字典比较误报的核验纠正均见 ALP15-PRECISION-HOLDOUT-PREP-REMOTE-EXECUTION-20261010.md。正式来源与输出没有篡改；四小证据已取回归档。

下一步按冻结实验源码与显式索引分类匿名cache/activation/checkpoint/score，以及外部引用和未按文本契约读取的文件，形成独立范围receipt；不得反序列化payload或猜索引顺序。采集前草案not_ready的抽样/CI/QC/time/power问题继续解决。没有模型新测量、GPU任务或大下载；长期Orin16GB/300ms/≤15%目标与独立精度均未完成。此次属于实际进展，不是等待阻塞或仅重复状态。
