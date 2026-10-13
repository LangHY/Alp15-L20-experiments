# Mac 下载与 L20 同步流水线决策

决策编号：D-20261013-MAC-L20-PIPELINE-01。日期：2026-10-13（Asia/Shanghai）。
决策者：6.1 Sol 决策子智能体；执行者：协调者指定的 6.1 Sol 执行子智能体；最终验收：6.1 Sol。
用户先要求本机 Mac 下载固定 949 clip 数据，并同时传输 L20、支持断点续传和自动重试；最新限定为“直接给我一个脚本让我下载”。因此本轮只准备脚本、离线验收和小型只读 API 预检，真实大文件执行留给用户手动启动，不由智能体启动。按本轮直接人类指令使用 6.1 Sol；仓库旧版 AGENTS.md 的模型名称由该直接指令覆盖，决策、执行、独立验收及文件所有权要求继续适用。

## 决策与任务边界

批准实现以下有界流程，用户手动启动时仍须全部预检通过：一个下载工作线程、一个上传工作线程、容量为 2 的已验证文件队列。Mac 每下载完一个 ZIP，先验证冻结清单中的大小和 SHA256，再入队上传；下一文件下载可与已完成文件上传重叠。此处“同时传输”是文件级流水线，不是将尚未完成的下载文件交给 rsync。

同时批准 `--run --download-only --confirm-bytes 54382445102`：用户可在 L20 隧道未恢复时手动先完成本机下载。该模式仍须固定清单、官方 HF认证、全部50文件metadata及本地105%余量检查；只省略远端预检、远端写入、上传工作线程。默认 `--run` 保持双端流水线及严格 SSH启动门禁；不因远端失败自动退化为 download-only。模式须绑定到运行日志，后续切回流水线时重新验证本地完整文件并入队上传。

风险等级：较高，原因是约 54.38 GB 网络传输、双端目录写入及认证状态；不涉及模型、评测口径或环境核心依赖变更。该工作只准备当前官方原始模型基线的数据资产，不解压、不运行模型、不改变数据样本、不使用剪枝、蒸馏、量化、视觉 Token 剪枝、ONNX 或 TensorRT，不替换数据源或 checkpoint。下载完成不等于官方入口、Jetson、精度或样本独立性通过验收。

## 冻结事实与本次核对

- 官方来源：`https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles`。
- 固定 revision：`33f9bf447ed3bcb7d545ce13f4226f824214fafb`。
- 唯一清单：`deliverables/alp15_windows_949/small-download-plan.json`，SHA256 `7a2356ea86921d535a96ef84e49f1018f8209dd31f7011423c6574372703471f`。
- 唯一候选名单：`deliverables/alp15_windows_949/test-candidates.csv`，SHA256 `9fbf80b944f8b810557838da52e7318d9cc7d3855746bc7f39b183dd539e76af`。该 LF 版本沿用 Windows 包的冻结名单，不混用历史 CRLF 字节哈希。
- 固定数量：949 clip、10 chunk、50 ZIP；总大小 `54,382,445,102` bytes。首次双端各需至少 `57,101,567,358` bytes 可用空间（总量的 105%，向上取整）；不包括解压所需空间。
- 固定 Mac 目标：`/Users/USER/Downloads/alp15-small-test-949-20261013`。
- 固定 L20 目标：`/home/USER/lang/datasets/alp15-small-test-949-20261013`。
- 本次只读重算两个文件 SHA256，均与上述值一致；用 jq 核对 50 文件及大小和等于 `total_bytes`。未联网重新核查官方 metadata、未下载 ZIP、未连接远端。
- 来源与抽样记录：`docs/baselines/ALP15-SMALL-TEST-CANDIDATES-20261011.md` 和 `docs/baselines/ALP15-WINDOWS-DOWNLOAD-DECISION-20261011.md`。冻结计划本身标记 `coverage_complete=false`、`drive_disjoint_verified=false`；本决策不提升这些证据等级。

本次只读核对命令：

```bash
shasum -a 256 deliverables/alp15_windows_949/small-download-plan.json deliverables/alp15_windows_949/test-candidates.csv
jq '{dataset,revision,clip_count,chunk_count,total_bytes,files_count:(.files|length),files_bytes:(.files|map(.size)|add)}' deliverables/alp15_windows_949/small-download-plan.json
```

## 文件所有权与允许变更

- 本决策子智能体独占本文件；不修改其他文件。
- 执行子智能体独占 `scripts/download_alp15_mac_l20_pipeline.py`、对应独立测试文件及新建远端传输 helper；具体测试/helper 文件名在执行回报记录。不得重写已有 Windows/完整池下载器或冻结清单。
- 协调者独占本轮执行回报、运行日志、状态与验收记录；远端 helper 只能写本任务独立目标目录中的绑定状态、锁和 `.incoming`，不得修改现有 L20 数据或其他任务配置。
- 不调用 superpowers；不提交、展示或在日志保存 token、密码、私钥、认证 cookie、账号个人身份或带认证的 URL。

## 实施与启动门禁

1. 默认入口仅输出冻结计划，不访问网络、不导入或调用下载 API、不创建数据目录。必须同时有 `--run` 与 `--confirm-bytes 54382445102` 才能进入实际运行；缺确认应在写入前失败。使用固定官方 `hf_hub_download`，`repo_type=dataset`，固定 revision，`HF_HUB_DISABLE_XET=1`。复用 Mac 独立 venv 中既有 `huggingface_hub==1.31.0`，先核验版本与 Python 环境，不改全局环境、不静默升级依赖。
2. 下载开始前，官方 HF `whoami` 及固定 revision 的目标文件 metadata 必须成功，路径、大小、LFS SHA 与冻结清单一致。通过官方认证读取本机现有凭据；禁止硬编码、输出或将令牌发送给第三方。401/403 立即停止，不做认证绕过。
3. 双端模式下载开始前核验 SSH 配置与 host `nvidiaL20`（连接项可通过明确 host/port 或已有 ControlSocket 配置覆盖），实际远端身份及目标路径。使用 BatchMode、连接超时和严格主机密钥检查；不禁用 host-key 验证，不盲信新 host-key，不自动写认证信息。远端身份可在内存核对，日志只记录核对结论与环境类别。显式 download-only模式不调用 SSH或创建远端目录。
4. 核对两个目标都是独立绝对目录；逐级拒绝 symlink、路径穿越、非普通文件、身份不明的已有状态、根目录或主目录作为数据目标。远端 helper 须检查目标固定路径及全部路径组件，并把状态绑定到清单/CSV/revision/目标。为整个本地任务持有系统 `flock` 语义的独占锁（Python `fcntl.flock` 可用于 Mac）；远端发布也须独占锁，禁止双实例同时下载或上传同一目标。
5. 双端模式本地与远端各自核对可用容量至少为该端未完成、未验证文件总字节数的 105%；download-only只检查本地。只从经真实 size/SHA 验证的完成文件扣减剩余量；不凭 receipt、文件名、partial 长度扣减。运行前和逐文件推进时重复检查空间；不根据未验证的 `.part` 推定可回收容量。
6. 预检通过后才启动双工作线程。下载在指定本地根中使用官方 local-dir/cache 续传机制；完整 size/SHA 验证后入队。采用阻塞容量 2 队列提供背压，任何工作线程终止须通知另一线程并释放队列等待，防止永久挂起。SIGINT/SIGTERM 安全停止，保留完整文件和部分文件，不删除本地缓存/partial。
7. rsync 使用 Mac 已有兼容能力：`rsync -a --partial -P`。系统 rsync 2.6.9 不支持 `--append-verify`，不得假设该选项可用。上传到 `.incoming/<冻结相对路径>.part`，远端最终路径不会暴露未完成文件。每次重试在记录后继续同一任务 staging；不使用 `--delete` 或其他清理选项。
8. 上传前，远端最终文件若存在则先检查真实 size/SHA：完全匹配才跳过；不匹配立即停止并保留。任务绑定的 staging 可由 rsync 恢复；若上传前 staging 内容不完整或不匹配，保留错误记录，允许在同一 staging 上重传修复。上传结束后必须远端核验完整 size/SHA；失败立即停止，不发布。完成发布须采用同文件系统排他 hard-link 或等价不覆盖的原子操作；不能先 `exists` 后普通覆盖 rename。发布时出现最终文件冲突，保留全部文件并停止。最终路径发布后再核对 size/SHA 并写绑定 receipt。
9. 每个下载/上传操作允许初次尝试后最多 6 次有界网络重试，指数退避且单次等待不超过 60 秒。明确区分瞬时网络断连与认证、校验、目录、空间故障。rsync 网络断线可重试；SSH 认证失败、host-key 失败或明确权限拒绝立即停止升级。不得无限重试，所有尝试保存事件与退出码。
10. 日志与 JSONL 状态记录清单哈希、环境版本、阶段、相对路径、字节数、校验结论、次数、退出码与时间；屏蔽敏感认证内容。每个 run 单独日志，完成记录须绑定 50 文件及实际双端核验结果；状态本身不能替代重启时的真实验证。

## 当前前提与停止标准

协调者报告当前 UU `REDACTED_IP:2222` 连接拒绝，旧 `/tmp` ControlSocket 已失效；小型 Mac HF预检得到 `AUTH_VALID`、首个目标metadata可访问及HF库1.31.0。本决策子智能体未独立联网确认，这些只作为协调者预检报告，不能替代用户启动时全部50文件metadata核验。允许先完成实现、离线测试与小型HF预检；最新人类限定下不由智能体启动大文件。用户手动双端模式SSH预检未通过时必须停在“大文件未启动”的边界，用户可明确改用download-only。不可声称流水线已启动，不因“自动重试”而启动无限等待或新的外部协调操作。

以下情况立即停止并保留数据与日志：清单/CSV/revision不一致；HF 401/403；已有最终文件大小/SHA 冲突；下载完整文件或上传完成后 SHA 不符；身份、绑定状态或锁冲突；symlink/危险路径；不足容量；认证/host-key/权限失败；瞬时网络重试耗尽；用户停止。预检或执行发现冻结前提不成立，执行者须升级决策者，不替换源、样本、版本、目录或研究口径。回滚为安全停止进程、保留既有文件与 partial；本决策不授权自动删除、覆盖冲突最终文件或清理其他任务数据。

## 验收与留痕

实现验收必须覆盖：默认入口无联网/无目录写入，字节确认拒绝，冻结绑定，目录/链接/锁/已有文件冲突拒绝，双端空间检查，限次退避与认证 fail-fast，下载验证后入队，队列/线程失败无死锁，rsync partial重启恢复，远端排他发布及 SHA错误不发布；download-only不调用SSH且仍执行全部本地/HF门禁，后续双端模式复验并上传本地完成文件。采用小型本地文件、模拟 HF/SSH/rsync，禁止测试意外触发54 GB真实下载。交付须给出默认计划、download-only与双端流水线各自的实际命令。

实际执行回报必须记录本机操作系统、Python、HF库、rsync、SSH版本，远端操作系统/身份核对结果/存储容量，实际命令、开始结束时间、初始预检、输入哈希、50 ZIP逐文件日志与状态、退出码、双端输出位置、断线尝试与限制。该任务不运行 GPU：驱动/CUDA/GPU仅在有关时记录为未使用，不伪造模型环境验收。

阶段判断：

- 仅代码和模拟通过：只能称“流水线实现通过离线验收”。
- 真实预检通过并有首个实际下载进度：才可称“下载已启动”。
- 有真实 rsync字节进度：才可称“L20上传已启动”。
- 全部50 ZIP 在Mac与L20最终路径真实size/SHA一致、绑定完成记录存在、流程退出0并由独立Sol复核：才可称“固定949资产双端传输完成”。

最终复核不可将网络预检、mock测试、目录创建或 completion文本当作真实 ZIP传输完成；不把该资产传输验收延伸为官方原始模型基线成功。
