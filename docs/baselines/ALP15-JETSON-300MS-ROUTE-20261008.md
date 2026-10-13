# ALP15-JETSON-300MS-ROUTE-20261008：16GB / 300ms / 15% 路线决策

日期：2026-10-08（Asia/Shanghai）；决策者：GPT-6.1 Sol high；本文件独占所有者：jetson_300ms_route。状态：路线决策与公开来源核查完成，首轮实验批准；本文件作者未运行 GPU、训练或部署。用户已直接豁免补齐旧官方原入口验收，授权持续优化；不将该豁免改写成历史官方测试成功。

## 1. 结论和首轮授权

现有最佳 8B AWQ 是质量良好的继续研究起点，但保持约 3k prompt tokens、串行文字生成、10 步高精度 Expert 的现有链路，达到 Orin NX 16GB 端到端 300ms 的希望很低。量化真正内核、减少重复 prefill 和低峰值加载必须做；这些变化本身不能消除算法所需的大量权重读取和计算。

批准第一实验：**在 L20 对同一冻结 8B AWQ，比较生成 actual KV 与当前重跑 full-prefill KV；固定 20 开发 clip、seeds 42/43/44、10 步及噪声，记录质量、阶段资源和缓存形状。**先两个 clip 对拍，再扩 20。随后同一候选、同一 actual KV 上配对筛查 diffusion 10/6/4/2 步。第二项改变求解器口径，须单独标明，不混入“无损 KV 复用”结论。

最有希望的完整路线是：**压缩视觉/策略输入 → 单次前向的小策略学生（短或无显式 CoT）→ 蒸馏少步或直接动作输出 → Orin 可用的真实低位内核和低峰值加载 → 实机联合验收。**原 8B AWQ 保留为高质量教师/候选，禁止覆盖其权重。小学生没有参数量硬指标，首先看完整硬件/时延/质量约束；不直接重试已失败的全层统一低秩 6B。

## 2. 本轮核查的权威证据

| 事实 | 来源与状态 |
| --- | --- |
| 源码 | NVIDIA Alpamayo 1.5 `36aeb4c5938cbc2eb2aed33b22434773da4ab639` |
| checkpoint | `nvidia/Alpamayo-1.5-10B` revision `7aba8293c09993f2e125c6819df05d7fa3e873ea` |
| 数据 | PhysicalAI revision `33f9bf447ed3bcb7d545ce13f4226f824214fafb` |
| 8B AWQ 成功 | `evidence/awq_full_20261008/completed_full20_v4/result.json`，status completed，20 clip × 3 seed；SHA256 `16aa6b20915bac68597614361e3932c89b955e7c05c024f1f5025b435ddd76cc` |
| 本轮记录环境 | torch 2.8.0+cu128 / CUDA 12.8 / NVIDIA L20 / ModelOpt 0.39.0；不是 Jetson |
| Orin NX 16GB | NVIDIA 官方产品表：16GB、128-bit LPDDR5、102.4GB/s；该表将 NX Super 总体能力标为 157 sparse INT8 TOPS，不能换算为本 VLA 的 FP16 吞吐 |
| 官方生成与 Expert 接口 | 固定源码 `models/alpamayo1_5.py:261–373`：生成保留 past_key_values，StopAfterEOS 多生成一步使 future-start 进入 KV，Expert 使用该 KV，逐步裁剪新增未来 token |
| 官方 diffusion | 固定 `diffusion/flow_matching.py:29–46,48–85,160–179`：Euler、默认 10 步、通过 `inference_step` 覆盖；官方 config 未另设步数 |

公开来源于本轮逐页核查：[Orin 官方规格](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)、[固定模型推理源码](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/models/alpamayo1_5.py)、[固定 FlowMatching](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/diffusion/flow_matching.py)、[固定 BaseDiffusion](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/diffusion/base.py)、[固定 checkpoint config](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/7aba8293c09993f2e125c6819df05d7fa3e873ea/config.json)。

按 web-access 读取公开官方页面。依赖检查 `node /Users/USER/.agents/skills/web-access/scripts/check-deps.mjs` 返回 Node 22.23.0 可用、Chrome 未连接；本任务不需要登录或浏览器交互，继续使用 skill 允许的静态公开读取，未操作用户浏览器或下载大文件。

## 3. 现场指标与瓶颈，不将推导写成实测

AWQ mean ADE = 1.282724219933152m；冻结教师 mean ADE = 1.232399m，因此相教师增加 4.0835168%。按最新用户 15% 约束，同口径上限为 1.41725885m。20 clips 是反复选型的开发集，60 seeds 不是 60 个独立场景；此结果只是开发筛查通过，不证明最终泛化或安全性。

AWQ 持久模型 storage = 10,272,486,512B（9.566999GiB）。报告中的首次 AWQ eval allocated 峰值约 18.18GiB；当前仍 BF16-first loader，不能称 16GB 已可运行。`scripts/alp15_awq_full_packed.py:66–82` 每次先完整反量化再 F.linear；这是格式/质量参考，不是 W4A16 加速实现。

本轮直接以 result 的 `packed_layers[].shape/persistent_bytes` 复算：252 个语言投影共 3,888,119,808 参数；纯 INT4 qweight = 1,944,059,904B，全部 AWQ buffer = 2,006,968,320B。AWQ 20 个 clip 的 generated_tokens 为 10–17、均值 13.5。**没有几百 token 的“长 CoT”现场证据。**数量短，也仍包含多轮大模型串行读取。

在普通串行 decode、语言矩阵不能常驻片上缓存、每步读一次完整 qweight 的假设下，仅 qweight 读流量除以官方理论峰值 102.4GB/s：

| 生成轮数 | 理想读 qweight 时间 |
| ---: | ---: |
| 10 | 189.8496ms |
| 13.5（均值） | 256.29696ms |
| 17 | 322.74432ms |

这是带条件的乐观预算推导，**不是 Orin 实测或任意算法的硬下界**；未包括 scale、embedding/head、KV、视觉、初始 prefill、Expert、动作转换、数据处理或有效带宽损耗。生成 helper 多生成的末 token 不进入 prefix，但对应 forward 更新 future-start KV；精确 forward 次数须实验计数，不只读 token 数猜测。投机/并行解码或更小学生会改变上述假设；不能以这张表排除所有优化。

`evidence/quant_route_20261008/full20/result.json` 记录 RTN prefix_length 3096–3104、prompt_length（prefix减该条生成量）恒为 3086；RTN 的 token 分布不能替代 AWQ 的 10–17。两者沿用相同输入缓存，AWQ 未单独输出 prompt 长度，第一实验须重新记录。约 3k token 对 prefill 也很重。旧 full20 的秒级阶段时间来自共享 L20 与完整反量化参考，不能当 Jetson 时延或优化内核收益。

另一个必须检查的瓶颈是 Expert。固定 checkpoint expert_cfg 设 hidden=2048、intermediate=8256、16 个 Q head；当前 Cosmos 配置为 36 层、8 KV head、head_dim128，其余由官方构造器继承。按 7 个 dense projection 推导，Expert 投影约 2,279,079,936 参数、BF16 约 4,558,159,872B；10 步每步至少读一次、除以峰值带宽约 445.13ms。**这是配置推导，须主任务实际模块 inventory 核验，不作已测参数或时延结论。**它解释了为何必须同时考虑 Expert 少步/蒸馏/量化。

## 4. 第一实验执行决策单

编号 ALP15-300MS-20261008-01；风险中高；执行者与验收者 GPT-6.1 Sol high；GPU 调度和执行归主任务；所有输出创建独立目录且存在即拒绝。当前子任务只拥有本 Markdown；主任务须明确新诊断 runner/测试的文件所有者。

事实依据：`scripts/alp15_awq_full.py:276–284` 中 generate_prefix 返回 `generated, actual` 后删除 actual，再调用整个 `model.vlm.model` 重做视觉/语言 prefill。已冻结 helper `evidence/quant_route_20261008/full20/code/controlled_8b_diagnostic.py:32–62` 保存 actual KV；`expert_interface_distill.py:29–52` 可接受含 past_key_values/rope_deltas 的输出。既有 helper SHA256 分别为 `2d7049d548bce8ee0410e89a68c51750dcb6ca5a734ab34afa10b7d62afc3508`、`1d64290ad58e55f67d58f55e04accc8f15aff541f0fc93852e91f9ebfa3efb8d`。

执行边界和步骤：

1. 来源/数据/父 overlay/AWQ manifest 哈希检查，直接装载已冻结 AWQ252层，拒绝重新搜索 calibration scale 或修改权重。记录保护模块 hash。既有 BF16-first loader 允许本轮 L20 诊断，但明确该限制。
2. 两个开发 clip，一次自主生成后同时保留 actual KV 和生成 prefix。当前 full-prefill 作为诊断对照，两者共享同一完整 future-start prefix，不调用 cached 教师文字替代自主生成。记录 prompt/generated length、全部 token ID、prefix hash、rope_delta、逐层 KV shape/dtype/length、generation forward 次数。
3. 对三 seeds 分别从一致的 RNG 状态生成完全相同的初始动作噪声；按旧 sampled 10 步固定 Euler 对拍 actual-cache 与 full-prefill。采样必须保护 cache 不被 seed1 修改后污染 seed2；记录采样前后 KV 长度/哈希或复建轻量 cache wrapper。
4. 记录 first-step vector field、64 步 pred_xyz、ADE/FDE、最大坐标差、相对轨迹 NMSE。原 full-prefill 结果须对冻结 AWQ 逐条 ADE ≤1e-4m，noise/prefix 对应完全一致；失败先归因来源/数值/设备状态，不能以新标准替换旧证据。
5. 实测实际缓存复用是否减少 vision/model full-prefill 调用。CUDA 同步的 compute timing 单独记录；含 image decode、processor、传输、动作后处理的 wall E2E 另记录。单纯重用离线 GPU input 缓存不是端到端验收。保留不做资源重置嵌套的 MemoryRecorder 方式。
6. 若 exact-comparison 门禁通过（cache 路径逐条 ADE 差≤1e-4m且有限、边界/噪声一致），扩到20 clips。若仅 NMSE≤1e-3，而 ADE 超此阈值，不称无损；可作为新部署候选独立扩20，重新比较教师阈值和尾部，明确基线口径改变。NMSE 小不能代替 ADE 门禁。
7. actual KV 候选稳定后，固定每个 clip 的同一 prefix/cache及三 seeds，筛查 `inference_step=10,6,4,2`。仅改步数，保持 temperature=1、同一初始噪声与动作空间；记录每一步 Expert forward 调用数。不能缩短 future64/6.4s、换 minADE、减少 sample 数或改变时间点凑阈值。

允许此实验新增 runner、CPU cache/RNG边界回归、小证据报告；未经独立决策，不改原 helper、旧 runner、原 venv、源仓库、任何旧模型或数据。预期新 CLI 由执行者实现后再给真实命令，不在这里提供不存在的脚本“可执行命令”。首轮无需新增大模型/场景下载，已有权重与数据足够。

验收：退出0；全部 provenance/hash匹配；0边界缺失/非有限；actual/cache形状与生成后future-start长度吻合；full-prefill复现；所有noise逐条配对；所有60轨迹有完整指标；内存/分阶段耗时有首调用和稳态；共享GPU限制清楚。首轮可接受实际收益未知，不能以 L20 300ms 或某几条样本作为最终通过。

停止：来源不符、边界缺失、噪声错配、缓存污染、保护权重变化、OOM、非有限、原标准口径复现失败。保存失败产物/日志，不覆盖、删除或默默回退。若旧数据/代码前提不成立，升级主任务决策后继续。

## 5. 完整路线及各阶段的硬门禁

| 阶段 | 具体变化 | 必须证明 |
| --- | --- | --- |
| A：清除重复工作 | actual KV 复用；明确 source StopAfterEOS/位置与 cache；无用 logits 不保存 | 相同 prefix/噪声精度对拍，视/语言重计算次数下降；分阶段预算实测 |
| B：缩短视觉 prefill | 首先训练侧按四相机四帧覆盖保留的空间缩减/合并预算；候选总 prompt约256/512/1024作为实验预算 | 官方 processor pixel bounds、image token数量、image_grid_thw、mRoPE、Qwen DeepStack路径一致；训练不含20开发/未来测试clip；真实轨迹比较 |
| C：减少串行文字前向 | 短 CoT/动作边界 token 教师蒸馏，或不在实时链路生成自然语言、改用单次 latent policy state | 自主输入、无教师未来/文字泄露；实际decode次数下降；总 ADE ≤教师×1.15；prompt/边界改动明确记录 |
| D：减少 Expert 成本 | 先10/6/4/2步质量筛查；若2/4仍不够，少步 flow distillation 或直接64步标准化加速度/曲率学生 | 同初始噪声/真实动作输出、多步稳定性、有限/动态可行、ADE/FDE/尾部及闭环；不能只看局部vector field NMSE |
| E：硬件实现 | 真W4A16内核、FP16全链路复测、流式packed loader、静态KV/workspace、可验证CUDA Graph | 实际SM87/ARM64/JetPack支持、prescale/amax/dtype/shape桥接、252层完整轨迹对拍；启动+首推+稳态统一内存和E2E |
| F：冻结与独立验证 | 冻结一次部署artifact；新独立测试与Orin实机同输入口径 | 最终目标全部联合成立，不用开发集或L20替代实机 |

视觉合并放在 ViT 之后只减少语言 prefill，不自动降低 ViT 运算；若视觉本身超预算，还需早期 vision token reduction、输入分辨率减小或小视觉学生。降低 min_pixels/max_pixels 不能只改输出 token 序列而保留旧 pixel/grid/mRoPE；需要正式独立代码/数据决策。目标预算不代表已验证尺寸或准确率。

小学生最有希望采用更少层/更窄的视觉策略模块，以教师轨迹和 causal state 监督，实时单次输出 action。若继续接原 Expert，则必须有显式 learned KV adapter：原 Expert 需要36层 cache，Qwen3-VL-2B config只有28层，不能直接更名/加载或少8层KV凑兼容。当前目录的2B文件是config/tokenizer，不证明2B权重存在。没有小权重时可以从已有模型训练/裁剪研究，但不能暗中下载GB权重。直接动作学生须继续使用官方 UnicycleAccelCurvatureActionSpace/64waypoint/dt0.1与历史条件，保持相同任务而不是换成简化任务。

已有80训练clip支持pilot，不足以宣称覆盖驾驶长尾；20clip多轮调参已是开发集。最终候选需要训练未见的独立clip评测和AlpaSim闭环补充。教师CoT/轨迹只在训练侧作为监督，验收时学生必须从当前图像/历史自主运行。可解释文本可作为另外研究输出，但实时预算中不得漏计仍依赖的文字生成。

真实低位内核研究不得假设 NVIDIA FP8/NVFP4配方支持Orin。TinyChat作者仓库说明Orin W4A16，但支持清单没有Alpamayo1.5，也无本checkpoint的Qwen3-VL/mRoPE/36层Expertcache桥接证据；因此这里只接受“底层矩阵内核可研究”，不接受“安装后全VLA就兼容”。[作者TinyChat README](https://github.com/mit-han-lab/llm-awq/blob/main/tinychat/README.md)。任何所选backend先冻结实际commit，不能凭main链接实施。

## 6. 最终验收定义和剩余门禁

最终验收需要目标 Orin NX16GB 的独立现场证据。记录实际型号、JetPack/L4T/CUDA、功耗模式、频率/温度、存储、可用内存、共享系统负载与所有artifact hash；禁止用 sparse TOPS 或 L20 的时间作比例换算验收。

端到端计时起点为同一时刻四相机当前输入及历史数据可用，终点为最终64步轨迹可供控制使用；包括图像decode/预处理/搬运、视觉、策略、全部仍依赖的reasoning/Expert、动作后处理、GPU同步。相机历史窗口等待不与compute time混淆；若预解码输入在部署中真实存在，可另列该部署输入契约，但不能拿模型内核时间冒充完整E2E。冷启动、首推、稳态分开；权重不需要每个周期重载，但**启动与首推不能OOM**。每个clip/seeds以一个完整部署请求测量，三seed重复评估不能只摊销一次generation到三个请求得低数字。

300ms按用户“不得大于”保留为上限：报告每条请求/稳态重复的mean、p50、p95、p99、最大值和超时数；目前未获用户允许以平均/p95替代，故任意验收请求超300ms不能宣称该门禁通过。连续稳定负载和热平衡也要测，不能只取最快样本。

16GB约束以真实全系统统一内存为准：包括OS、图像/processor、CPU、GPU、KV、workspace、加载/首推与长期运行。记录 tegrastats、进程RSS、系统available、swap/majorfault和CUDA指标；共享统一页不可直接将RSS与CUDA峰值相加。无swap/offload掩盖、无OOM、留有系统/相机运行空间，才算满足目标。CPU offload或换页若存在必须计时，且不能将“权重9.57GiB”写成“全系统够用”。

精度主门禁沿同一clip、时间点、64步和三个配对seed，以clip为独立层级统计mean ADE：候选/教师≤1.15；同时报告FDE、clip p95/max、失败率，不用minADE或更短未来替换。候选冻结后独立clip上的教师与学生同口径复测；尽可能以clip bootstrap量化不确定性，置信区间不能只靠60seed假装独立。若最终选择actualcache/FP16等新部署口径，教师与学生都补齐该口径，旧full-prefill教师1.232399仅保留历史开发参考。

截至本决策，精度仅开发集初步通过；统一内存、最终内核、端到端300ms、独立泛化和Orin实机证据均未通过。目标保持完整，不因当前只有L20而降低验收标准。
