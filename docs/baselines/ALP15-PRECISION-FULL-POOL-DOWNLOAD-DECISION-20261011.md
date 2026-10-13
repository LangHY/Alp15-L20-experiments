# 完整候选池用户下载准备：最小决策

决策编号 D-ALP15-PRECISION-FULL-POOL-DOWNLOAD-20261011。日期2026-10-11（Asia/Shanghai）。决策者 GPT-6.1 Sol high，子任务 `audit_v4_acceptance_20261010`。本文唯一写入者为本子任务；清单/脚本/报价/执行回报由主任务明确所有权后另建，不覆盖任何旧证据。

**批准准备“完整候选池下载清单”和默认 plan-only 的用户手动下载脚本。未批准代下载、模型执行或把它命名为已验收独立最终测试集。** 用户要完整集合，不授权改成随意100clips或少量便宜chunk便宣布充分。

## 事实与“完整”的范围

正式v3 census已exit0，固定 dataset revision `33f9bf447ed3bcb7d545ce13f4226f824214fafb`；保守排除21个chunk后的metadata test候选池61301 clips/633 chunks。coverage_complete=False、drive_disjoint_verified=False，后续来源验收仍剩650匿名等scope缺口；输入窗口QC、规划数据转dev、最终名单和新模型运行均未ready。下载齐全不消除这些限制。

本次“完整”仅指**该冻结metadata候选快照的所有test clip所需、按chunk去重的egomotion+四camera原始文件**。它不是全仓库所有数据、不是全306152clip、不是所有道路/drive，也不是QC已合格框。chunk ZIP可能附带不在test名单内的clip，下载范围与评测ID范围必须分别保存；不对这些额外clip自动运行模型。未来新增已见块/QC/planning排除只能使最终独立框收紧，原21块不释放。

必须从正式v3 manifest绑定的 `eligible-candidates.csv` 文本筛 `split=test`并读取明确chunk_id：原产物SHA `cf620cf9b278bacd149378591d356e30a8efa3679e2696850a5bc743c717d8e7`。若服务器实际另名 `candidates.csv`，先证明字节SHA及字段来源与该正式产物绑定，不能用旧下载文档中只排0/3119的候选表代替。核61301唯一clip、633唯一块、与21块无交集；不从clip UUID猜chunk或从编号连续区间补块。

## 最小执行步骤和权限

1. 主任务沿既有授权路径只读正式候选CSV/manifest及冻结features.csv文本，导出test clip→chunk的完整名单和633块×五特征的实际路径；以固定模板解析，不初始化数据下载接口。预期3165个chunk-feature条目，必须核条目唯一、每块五项齐全、无未声明额外feature，实际路径碰撞/缺失即停止，不凭预期数补造文件。
2. 由主任务查询官方固定revision小型文件信息，逐文件取得exact path/bytes/LFS SHA256，存完整报价manifest及SHA；若SHA遮蔽/未知或来源revision不符则pending，不交可执行大下载。求去重总bytes、GB/GiB/TB/TiB，分开列ZIP、缓存复制/解压/工作空间；未知开销单列，不将一个chunk容量乘633冒充真实总价。不要重下现有metadata、模型、其它传感器或整个仓库。
3. 优先复查国内源**小API/HEAD**，分别保存 authority_source/revision 与 transport_source/revision、状态、重定向目标、size/SHA可对照程度。国内仓同名或索引同revision不等于文件传输/字节已验；若重定向官方，明确是官方备用而非“国内大文件镜像已可用”。旧20261009国内源观察仅历史依据，不冒称本日复查。禁止将既有token发送给未获用户明确授权的第三方镜像；不代接受gated许可、不打印凭据。
4. 交给用户完整清单、真实总量/存储需求、当前可用传输路径及明确下载命令。脚本无参数默认只读取本地冻结清单、检查/展示计划，**不连接文件GET、不下载metadata/ZIP、不自动安装依赖**。下载分支必须由用户显式启用，绑定报价manifest SHA、目标目录和确认的总bytes；不在生成、help、测试或plan阶段触发下载。用户后续自己运行；本轮执行者不跑下载分支。
5. 下载分支只针对所列固定路径，拒覆盖已有未知/冲突文件；完整已有文件只有bytes和SHA均匹配才可跳过，partial留作明确续传状态，结束逐文件校验并出receipt。校验失败停并保留失败，不以改SHA或删项凑齐。不要自动解压、读取payload或启动模型。脚本/路径/参数验收仅CPU合成fixture，不用真实大文件“试一下”。

本决策子任务禁止SSH/网络/GPU/payload/下载；仅本地读证据并签本文。上面第1–3步由主任务依其已有授权实施小文本查询，本文不新增未授权端点、登录、安装、目标目录或大资产传输权限。所有查询保留命令、UTC、环境、输入/输出SHA、退出码/失败日志；新产物拒覆盖，原census/tool/CSV/usage receipt不改。没有来源证据时标pending并报告最小缺项，不无限扩展审查。

## 报价后的用户选择与测试门禁

用户当前明确请求完整集合，所以**全633块候选池清单是主交付，不主动缩成抽样**。由于真实总量/存储尚未确定，准备清单不等于用户已承诺未知TB成本：先展示实价，再让用户确认是否仍下载全池、可用存储及目标位置；只有用户主动改选较小方案，才能另按v2审查认可的概率设计/固定样本量规划准备“足量抽样”清单，不能叫完整池，不预设100clips充分。用户未作选择或预算不足时，保留完整plan，不代运行、不悄悄删大块。

即使用户完整下载并校验，也只能签“该候选清单获取齐全/字节校验通过”。称为“独立最终测试集”仍须已有usage范围审定、确定且误差不可见的QC/去重、planning转dev排除和最终名单冻结；新评测还需教师/A/B资产、真实时间/GT、完整planned请求及审定分析。若对最终完整QC框普查，有限框抽样误差为0，不再凭抽样n/观察chunk bootstrap制造CI；这不取消曝光/测量/失败/道路泛化及设备门禁，也不授权本轮执行模型。

因此最小下一步是**完整633块的逐文件官方报价+国内传输小证据+plan-only交付**；在实价后确认用户全池选择。coverage/drive标志不改，新独立精度、Orin16GB/300ms、闭环及官方原入口不据此通过。

## 本轮本地核验留痕

cwd `/Users/USER/Documents/ChatGPT/Lighten_VLA`。只读 `sed -n`读取正式小report、旧下载来源文档相关部分及当前v2审查；`shasum -a 256`核冻结身份；`/usr/bin/python3 -B`仅json/pathlib读取本地manifest确认status/coverage/drive、61301/633与正式CSV声明SHA；`test ! -e`确认本文新目标不存在，均exit0。无SSH/网络/模型/安装/下载/payload。

输入SHA：run-manifest `022ae526670625ff44c1d69b7507bb1925e0a91cedd25442573cb9ccd29c3ff2`；report `429b279db6ce66c01e5c3a50fe12ddab6b4ff2ebe37a31a71726121c0de71e30`；usage-gap决策 `ce674809d555c04adc7b2607fb74f6427ec624b2237ba875525f898861994a5c`；teacher窄域验收 `3ef45a7222ced1343bd7381666e3af587a2907a7a408017bb5a45257826f968e`；v2审查 `7fd9d8bc0e2c188c21bd953e100cc0c8c1bbd0ef2513fa8374966e246f510799`。实际工具留痕5ec91e（report/旧来源上下文）、667a4f（当前v2审查）、56caee（身份/新目标检查），补充manifest查询命令与输出留在本轮工具会话。没有伪造新的全池bytes、镜像可用性或逐chunk名单。
