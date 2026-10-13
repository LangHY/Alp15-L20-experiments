# Alpamayo 实验交付审阅与下一步方向

日期：2026-10-07。性质：只读证据审阅及方向建议，不是训练执行批准单。

## 1. 结论

保留最佳 8B 为精度优先主候选，最佳 7B 为资源约束备选；暂不继续向更小参数规模剪枝，也不继续原样延长现有蒸馏训练。优先诊断“代理损失改善为何未转化为轨迹改善”，然后再决定是否开展单因素恢复实验。

这些结论来自交付包快照，不代表服务器实时状态。SSH 服务可达，但截至本次审阅，跳板机密码认证未成功；未运行服务器命令、训练或模型评测，未修改 SSH 配置或生成密钥。

本仓库现行 AGENTS.md 仍要求官方开发机基线及 Jetson 原始模型实机结果通过验收后才进入优化阶段。交付材料记录了后续压缩实验，但不能自动撤销该门禁，也不能替代 Jetson 证据。本次仅评价已有结果；任何新增优化实验须先解决门禁与实验记录之间的状态冲突。

## 2. 交付与复核证据

输入：`/Users/USER/Downloads/Lighten_VLA_MacBook_Handoff_20261004.zip`、同目录 `SHA256SUMS.txt`、`MACBOOK_HANDOFF.md`。

- ZIP SHA-256：`a15650fcbe58d71aeb8823aef5ab7f3f838dd0d53c2a1f60beaebbcf55cb8b56`，与清单一致。
- `unzip -tq` 检查退出码 0，未报告归档错误。
- 隔离解压至 `/tmp/alp15-handoff-a8uKHc/Lighten_VLA`，没有覆盖当前工作区，也没有执行包内脚本。
- 阅读核心交付、精度恢复、部署预算与失败分析报告，并检查训练探针/受控推理源码及三份逐轨迹原始 JSON。未声称审阅所有归档文件。
- 独立重算最佳 8B、最佳 7B、8B 后续训练终点的 ADE：三份各 20 clips × 3 seeds，先 clip 内平均，再跨 clip 平均；全部与汇总字段一致。
- 三份 JSON 的 60 个对应 clip/seed 初始轨迹噪声哈希完全一致。此检查确认这三份结果之间的噪声配对，不能推广为全部历史实验均已配对。

原始 JSON 路径均相对于归档内 `Lighten_VLA/`：

| 文件 | SHA-256 |
| --- | --- |
| `compression/results/newdata8_recovery_final_20261002/experiments/mlp_newdata8_recovery_20261002_165238/task_validation/step_000400/result.json` | `68190c17deaa675d406ba519cc2dd209d8806991ff64b601fdb6065110d4d59c` |
| `compression/results/measured7_recovery_final_20261002/experiments/mlp_measured7_recovery_20261002_182132/task_validation/step_000400/result.json` | `1dfb6debc20ca5bec6fd5acb31024a023270000cc57f75f738c87263dda92353` |
| `compression/results/precision8_dense_final_20261002/experiments/mlp_precision8_dense_20261002_195955/task_validation/step_000320/result.json` | `083c4271f9b4724a463f8d6de7cab9a5d2b1c467c5d9234b149719d9563c5a60` |

其他主要来源：`PRECISION8_FINAL_REPORT.md`、`NEW_DATA_8B_RECOVERY_REPORT.md`、`PRECISION_RECOVERY_PLAN.md`、`DEPLOYMENT_TARGET.md`、`compression/results/rebalanced7_local_analysis_20261002/REPORT.md` 与 `analysis_summary.json`。

## 3. 当前模型取舍

以下是修正噪声配对后的固定 20 clip 验证口径，不与旧单 clip 基线或旧 RNG 口径混用。ADE 越低越好，3 seeds 是取均值而非 best-of-3；历史字段名 `mean_minADE_m` 不改变实际聚合语义。

| 候选 | 总参数量 | ADE（m） | 相对教师退化 | 建议 |
| --- | ---: | ---: | ---: | --- |
| 原始教师 | 11,078,526,194 | 1.232399 | — | 保留参照，不因裸 BF16 体积提前淘汰 |
| 最佳 8B，step 400 | 8,020,878,578 | 1.299637 | 5.46% | 主候选 |
| 最佳 7B，step 400 | 6,998,516,978 | 1.329894 | 7.91% | 资源约束备选 |
| 6.797B | 6,797,190,386 | 1.428902 | 15.95% | 不作为当前主线 |

7B 比最佳 8B 的 ADE 高约 2.33%。模型名称是总参数近似，不是独立 VLM 参数量；目录 `width_5376` 等也不能当作所有层均匀宽度。

这些验证 clips 被反复用于选模型/选训练步，结果只能说明当前验证集表现，不能声称独立测试集泛化。69 个历史 holdout 需审计使用记录并按新口径重新规划；被旧协议评估过的数据不能称为从未触碰。

## 4. 为什么暂停原样训练

- 新数据 8B 恢复确实有收益：同结构训练 step 400 达到 1.299637 m，但继续到 step 800 回升至 1.341245 m，而监测损失继续下降。
- 从最佳 8B 开始的 dense 恢复训练 320 steps，终点 ADE 1.319240 m，较父模型退化约 1.51%；按已记录的 20-step 检查点，最佳仍为 step 0。监测损失约下降 0.41%。这不能排除未评测中间步偶有改善。
- 7B rebalanced 训练 80 steps，ADE 从 1.329894 升至 1.352809 m（退化 1.72%）；监测损失下降约 0.56%，Expert velocity NMSE 下降约 10.96%。9 clips 改善、11 clips 变差。
- 7B 最大记录梯度范数约 0.917，小于裁剪阈值 1；现有记录不支持“梯度爆炸”为解释。降低 Hidden 权重同时还改变了其他训练设置，因此也不能认定 Hidden 权重单独导致结果。

最可靠结论是：现有局部接口/速度监督指标不足以代理最终轨迹质量。过拟合、条件分布偏差、采样路径误差等都仍是候选解释，尚未确证唯一根因。

源码依据：`compression/expert_interface_distill.py:1-6,12-28` 明确探针为固定噪声与真值 action 的线性插值，t 为 0/0.25/0.5/0.75，不是完整轨迹采样，也不是官方原始训练流程。`compression/controlled_8b_diagnostic.py` 则使用模型自主生成前缀与多步 diffusion 采样。二者存在可检验的分布差异，但差异存在不等于已证明它导致退化。

## 5. 建议下一步顺序（待门禁和连接问题解决）

1. **先审计，不训练。** 核实 L20 主机/GPU、当前进程、依赖版本、官方源码提交、模型/overlay 哈希、split、最终产物及保护模块哈希；确认交付快照之后是否出现新实验。不能依据文件夹名称选模型。
2. **补齐比较可信度。** 对最佳 8B/7B 做独立重载复现；先检查自主生成 KV 与标准化 full-prefill KV 的一致性。保留独立文本 RNG 与配对轨迹 RNG、噪声哈希、逐 clip/seed 指标、退出码和日志。已有验证集用于诊断，不冒充独立测试。
3. **不更新权重的条件/采样诊断。** 从训练/监测 split 取样，比较教师/学生网络分别 replay 教师/学生 token 前缀的四种条件；每个网络须对同一所选前缀重新计算自己的 KV，不能把不同长度的缓存直接拼接。最佳 8B、最佳 7B 应分别测试，旧 8B 的诊断不能自动推广到恢复后的模型。再比较固定真值插值探针和真实采样路径上的速度误差随时间变化，以及终点 ADE，判断代理指标是否与任务指标脱钩。
4. **有诊断依据后才设计短期单因素试验。** 以最佳 8B 为主候选，保持父权重、数据、预算、LR、loss 权重等一致，只改变前缀条件或采样状态来源中的一项。预先锁定主指标、监测频率、停止条件和留出规则；不直接宣称必须端到端 ADE 反传，也不同时改多个因素。
5. **最后选择与独立评估。** 冻结方案后再使用审计后的留出数据；区间估计按 clip 处理，不能把同 clip 的 3 个 seeds 当作 3 个独立场景。记录轨迹 ADE、尾部误差、生成边界失败、延迟、加载及运行峰值；新阶段是否增加其他指标须先定义口径。

本次没有启动上述任何 GPU 操作。没有因接收附件而授权剪枝、量化或框架改造。

## 6. Jetson 方向的取舍

交付中的目标是 Orin NX 16GB 上完整推理栈与系统稳定运行、精度优先，而非硬性压到 6B。资源预算是估算，不是实机测试：按其“语言 Transformer W4、其他部分 16bit”假设，8B 约 9.56 GiB，7B 约 9.07 GiB，差距仅约 0.49 GiB；不能用 BF16 参数削减量直接估计最终部署收益。

因此先保持 8B/7B 双候选，做框架兼容性与资源核算；只有完整栈确实超预算才决定继续剪枝。必须计入视觉/Expert、KV、草稿网络、量化辅助缓冲、编译/CUDA Graph、加载峰值、输入解码及系统共享内存。LightVLA 与 FlashDrive 的节省比例不能直接相乘，L20 的 GPU 内存也不能作为 Jetson 16GB 通过证据。

若 Jetson 实机证据缺失，应明确记录为未验收，而不是用 L20 成功替代。附件里的框架源码预算只能支持后续验证问题，不能证明 ARM64、目标 JetPack 或 GPU 内核兼容。

## 7. 本轮执行边界与待补信息

- 本轮文件所有者：主智能体；仅新增本报告，保留工作区既有修改。
- 本轮没有删除文件、覆盖模型、变更环境、启动训练或改写服务器 SSH 授权。
- 连接：`REDACTED_IP:2222` SSH 服务可达；无交互认证及密码认证均未通过。已停止重试，待用户确认跳板机用户名和密码归属。密码不写入本报告或脚本。
- 当前唯一实时状态结论：SSH 认证未成功。包内模型、运行结果与 L20 环境均为交付快照信息；服务器核查后应追加独立执行记录，不能静默改写历史结果。

## 8. 同日服务器只读复核（后续追加，替代上述连接阻塞状态）

用户确认本机 2222 直接转发到 L20 的 22 端口，并确认新 ED25519 指纹 `SHA256:Ayyswj6/LciwUQMReHioDam7zCWTsspgbSposPm2o2Q` 后，密码认证成功。使用独立临时 `UserKnownHostsFile`，保留原有信任记录；检查完毕已退出 SSH。没有生成公私钥，没有改动服务器授权。

实际环境：主机 `zwl-acloud`，Ubuntu 22.04.5 LTS、x86_64、内核 6.8.0-87-generic；NVIDIA L20，46068 MiB，驱动 580.95.05。检查时显存占用 41 MiB，`nvidia-smi --query-compute-apps` 没有计算进程；不能据此推断整个服务器无后台任务。

实际项目路径 `/home/USER/lang/Lighten_VLA`。用 `alpamayo1.5/.venv/bin/python` 查询安装元数据：torch 2.8.0、transformers 4.57.1、safetensors 0.7.0、physical-ai-av 0.2.0。官方仓库 HEAD：`36aeb4c5938cbc2eb2aed33b22434773da4ab639`。本次没有导入/加载模型，未检验仓库工作树是否有修改，也未重新执行 GPU 推理。

主要只读命令与检查：

```sh
hostname
id -un
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
find experiments -maxdepth 2 -name result.json -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -20
find experiments -maxdepth 1 -type d -newermt '2026-10-03' -printf '%TY-%Tm-%Td %TH:%TM %f\n'
git -C alpamayo1.5 rev-parse HEAD
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

另以只读 Python 解析结果/manifest、统计 `mlp_*.safetensors` 文件数、计算第 2 节三份验证 JSON 的 SHA-256、查询依赖安装元数据。完整交互命令与返回输出保存在本次 Codex 对话工具记录；未另写服务器日志文件。

核查结果：

- 三份验证 JSON 的服务器哈希与交付包完全一致，ADE 分别为 1.2996366459876298、1.3298937315742174、1.3192404765635728。
- 最佳 8B 和 7B 均选择 step 400，各自 overlay 有 manifest 和 36 个 MLP safetensors 文件。这里只确认存在性与结果一致性，没有逐文件验证权重哈希或重新加载，不能称为模型完整性验收。
- 7B rebalanced 最终选择 step 0，确认其后续训练没有替代最佳父模型。
- 三个实验的 `controlled_reload_check.json` 均存在且标记 completed，范围注明为固定两个验证 clips × 三个配对 seeds，不是完整 holdout。本次只检查该历史状态与范围，没有重算其全部比较字段，不能把它当成本轮重新推理。
- 顶层 `result.json` 仍标记 `selected_pending_holdout_and_reload`，与另存的 completed reload 记录存在状态汇总未同步问题；不可据顶层状态认定重载从未发生，也不可据小规模重载认定全部验收完成。
- 当前检查范围内，最新顶层实验结果为 10 月 2 日的 `mlp_rebalanced7_recovery_20261002_210458`；没有发现 10 月 3 日以后新增的顶层实验目录。这不排除其他目录中的操作或旧目录内容更新。

结论：实时服务器证据支持第 1/4/5 节方向判断，连接阻塞已解除。下一步优先补齐状态/模型审计，并针对最佳恢复模型做无更新的条件与真实采样路径诊断；本轮没有启动新实验。
