# ALP15-WHITENED-ALL36-MEASURE-20261007：全36层FFN敏感性测量决策

采前冻结日期：2026-10-07（Asia/Shanghai）。决策：GPT-6.1 Sol high；执行代码：GPT-6.1 Sol high；独立复核：GPT-6.1 Sol high；现场协调：当前主智能体。该文件在采集前保存副本与SHA256，执行回报单独追加文件，不事后改写设计。

## 目标与授权

直接用户要求“寻找更强压缩方案并执行”，随后要求全面记录并继续。本轮承接三轮局部pilot，限定为L20探索性FFN敏感性测量；沿用此前对旧阶段门禁的用户限定授权例外，不代表NVIDIA官方原始入口、Jetson或完整6B验收通过。

目的：在同一父模型、真实参数预算、输入和算法下识别层间异质性，为后续训练侧预算分配提供证据。预设的rank896/1024是测量档位，不强迫所有层采用低rank。局部NMSE改善不能证明完整模型轨迹精度。

## 冻结事实与一手来源

- NVIDIA Alpamayo源码commit `36aeb4c5938cbc2eb2aed33b22434773da4ab639`；checkpoint revision `7aba8293c09993f2e125c6819df05d7fa3e873ea`。来源与结构验证见 [先前决策](ALP15-6B-LR-PILOT-20261007.md)。
- 父模型 `experiments/mlp_newdata8_recovery_20261002_165238/width_5376/distilled`，36层、hidden4096、FFN5376；冻结profile `experiments/recovered8_importance_20261002_175834`。
- 冻结引擎SHA256 `fa2e18bf86a768e724ec08536db1a0e6a9ed88125fe199fcebf04317ef86b007`；基础模块SHA256 `c2975113b04a0a5e295ab14f0ba09db4152568fdb751ec957b45c87b3693ad69`。本轮不修改二者。
- 数学来源为 [SVD-LLM作者论文](https://arxiv.org/abs/2403.07378) 的协方差加权分解思想，实际算法以冻结脚本为准；不是官方完整复现。论文与作者源码固定版本已在先前决策核对，不安装其旧环境。
- 现场只读预检：同一NVIDIA L20，空闲显存41MiB、GPU利用率0%；36份`train_00.pt`至`train_35.pt`缓存齐全；服务器工作路径剩余1.6T。来源、张量形状和内容仍须每个child严格preflight，文件存在不等于完整性通过。

## 设计与选型约束

探索性、clip内配对的逐层测量，不作功效/显著性或泛化声明。真正样本单位为clip；36层、多个rank/方法和token都不是额外独立样本。

- 固定64 fit clips、16 calibration-eval clips、split seed20261007；两者来自历史80训练clips，不是独立测试集。此前失败clips不排除。
- 每层9185 fit tokens、2319校准评估tokens（现场preflight必须确认）；统计/闭式拟合固定抽取8192 fit行，同seed。不得随机拆token代替clip分离。
- rank896/1024、gesvdj、FP64 Gram/闭式拟合、正则比1e-4、真实BF16因子执行均不变。
- 每层9配置：父FFN，两个rank各自的通道对照、仅白化、父特征down重建、压缩特征down重建。单层原参数66,060,288；rank896/1024分别25,460,736/29,097,984参数，通道对照预算一致。
- fit-only规则选择三种低秩方法中train NMSE最低者，固定方法顺序打破平局；通道只作对照。校准评估不选方法、层或rank。暂不分配完整模型预算。
- 固定随机执行顺序：Python标准库`random.Random(20261007).shuffle(list(range(36)))`，保存全部顺序；不是按既有误差安排层次。
- 6/12/24/35层也重新测量作为重复一致性检查，记录与旧结果差异，不选择两次中更好者。全层汇总统一采用本阶段结果。
- 缺失/失败不插值、不丢弃、不另换正则；无离群clip剔除，无事后显著性检验或置信区间。

## 执行边界、文件所有权与资源

执行者独占新入口、测试和独立汇总器；主智能体独占本决策与执行回报。旧源码、checkpoint、8B/7B父overlay、缓存、历史实验全部只读，不覆盖或删除。

每层串行独立child进程，使用原引擎、仅在内存设置`LAYERS=(layer,)`；独立新子目录、argv、完整stdout/stderr、退出码、manifest和源码哈希。root保存采前决策/代码快照与SHA256、原子进度、一把fcntl排他锁。child结束释放GPU，输入来源仍由冻结preflight验证。

`--resume`仅跳过已完成且退出码0、9条唯一配置、全部artifact/perclip哈希、父模型/split/seed/rank/正则和fit选择全部一致的层。running/failed/不完整层保持原样并停止，升级决策者；不自动创建新attempt。只在36/36独立复核后标为completed。

预计校准峰值约2GiB是历史测量参考，不是资源保证。持续记录GPU使用、RSS、耗时；若出现异常增长、其他GPU任务竞争或连接/来源问题，保留进度，只停止本实验进程，不触碰其他进程。36个child各自完整preflight会重复读取全部36份overlay及80份teacher缓存，禁止为提速绕过验证；15–25分钟仅为粗略估算，不作ETA承诺。

明确不做：训练/蒸馏、权重导出、完整模型组装、真实ADE测试、Q/O新输入采集、量化、ONNX、TensorRT、Jetson操作。Q/O仅可读取冻结官方调用约束，另立采集决策。

## 采前锁定的记录与验收

必须36个唯一层、324条配置、72组**fit所选低秩方法**与等参数通道配对，每组16clips。全量保存候选，不跨层混合SSE掩盖敏感层。

每层/每rank记录：参数数/保留比、fit能量NMSE（原记）、eval能量NMSE（独立重算）、clip等权mean/median/p90/p95/max、SSE/目标能量/token、feature RMSE、逐clip改善/恶化与最坏病例、leave-one-clip-out两种口径、父缓存重现NMSE、正则实际量、BF16有限性、校准分解/方法/总耗时、CUDA allocated/reserved峰值与GPU/RSS观测。分位数用排序后`(n-1)*p`线性插值。

fit逐clip/SSE/目标能量若没有保存，明确标缺失，不伪造；新增观察性sidecar不可改变原引擎选型调用。资源指标是校准过程，不是完整VLA推理延迟或部署内存。

配置误差高或比通道差是有效结果；来源/哈希/形状不符、非有限、Cholesky/SVD失败、父缓存重现NMSE>1e-3、退出异常才触发停止。缺失层禁止宣称全36测量完成。重复层数值差异必须报告，不能按优势忽略。

本轮成功仅允许形成逐层敏感性结论与下一阶段决策。未测完整模型ADE/FDE、重载一致性、自主生成、稳态延迟、模型大小、Jetson，不宣称6B精度成功。

## 归档与复现

计划文件作为`--decision`输入，启动前复制并校验SHA256；代码最终hash、原始命令、实际环境与时间、各层退出码、指标与验收结论写入独立执行回报 `ALP15-FFN-SENSITIVITY-RESULTS-20261007.md`。独立汇总器应验证root/child状态、配置与来源、artifact与逐clip哈希及fit选择后再生成报告，输出目录必须尚不存在。
