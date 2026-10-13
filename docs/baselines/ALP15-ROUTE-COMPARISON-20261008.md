# ALP15-ROUTE-COMPARISON-20261008：精度优先部署路线比较

日期：2026-10-08。决策复核：GPT-6.1 Sol high；量化模块执行：GPT-6.1 Sol high；协调及现场执行：当前主智能体。

## 授权、目标与边界

用户直接授权：不补齐官方原始入口基线，以现有离线脚本数据为准，比较6B剪枝、8B量化及更优候选并执行。这覆盖旧AGENTS阶段门禁，仅允许本次探索；不改变历史验收、不宣称官方原样运行或Jetson已通过。最终目标Orin NX16GB，精度第一，参数6B不是硬约束。

保护原始checkpoint、最佳8B/7B、所有旧缓存及实验。不停止其它项目训练，不改变现有推理环境。所有新输出独立目录，拒绝覆盖。主智能体独占本报告与路线runner；执行子智能体独占packed模块及其单测。

## 事实与比较决策

- 冻结离线20 clips×3 seeds参考：教师11,078,526,194参数、ADE 1.232399m；最佳8B step400为8,020,878,578参数、1.299637m（+5.46%）；7B为1.329894m；6.797B为1.428902m（+15.95%）。这是重复选型使用的开发集，不是独立测试。
- 全36层低秩测量：52/72组能量NMSE改善，但34/72组p95或最大误差恶化。完整6B未组装，Q/O未测。不能用局部NMSE推断轨迹ADE或延迟收益。
- 原始结构选择性量化可避免先承担剪枝损失，应作为精度竞争者；8B选择性量化是更小资源候选。两者必须同格式、同clip/seed、同自主前缀与真实diffusion采样口径比较。
- NVIDIA官方1.5量化配方只列FP8/NVFP4及混合，验证5090/B300，不能当Orin INT4兼容证据：固定[提交3f6504e](https://github.com/NVlabs/alpamayo-recipes/blob/3f6504e215da86eb501860c7eea925d1e5b75b0d/recipes/alpamayo1_5_quant/README.md)。[MIT TinyChat](https://github.com/mit-han-lab/llm-awq/blob/main/tinychat/README.md)提供Orin W4A16，但未列Alpamayo1.5。格式存在不等于本模型全栈兼容。

主线顺序：最佳8B W4A16与原始结构W4A16筛查 → 配对20clip开发评估 → 训练侧驾驶校准的AWQ/混合敏感层精度增强及优化packed内核 → 候选冻结后的Orin资源和延迟验收。6B仅保留独立研究路线，不原样执行已失败统一rank方案。没有用户允许精度下降百分比，不擅设3%/5%为验收门限。

## 首轮执行单

风险中高；L20执行，Mac仅编辑/静态检查。第一阶段使用group128、对称RTN INT4 [-7,7]、FP16存储scale、高精度激活；量化语言Transformer线性权重，不动视觉、Expert、embedding/head、action投影、KV。实际高精度计算沿现有BF16离线参考；Orin最终FP16变化必须另测。

RTN是透明筛查基线，不冒称AWQ、ModelOpt或最优量化器。真实uint8两nibble打包；save/reload后替换线性模块，不保留dense副本。参考forward按需反量化，可能慢，不能据其延迟淘汰量化路线或承诺加速。记录真实模型storage和CUDA allocator峰值；加载仍先加载原始BF16，不是低峰值部署loader。

KV仍保存16bit，不量化缓存；Q/K/V权重量化会改变KV数值，不能宣称KV值不变。保护哈希仅排除明确的7类projection模块，注意力q_norm/k_norm等其余参数仍受保护。

先跑模块CPU/GPU单测；再固定开发集排序前2 clips×seeds42/43/44、最佳8B原精度复现及8B/教师packed推理。无异常后同格式扩20clips。不因eval误差改层/阈值。停止条件：来源/哈希/噪声配对不符、原8B复现误差>1e-4m、非有限、生成边界缺失、重载不一致、OOM。失败留痕并停止相关配置，不删缓存。

指标：ADE/FDE及逐clip/seed配对差、clip均值/中位/p95/max、生成边界与有限性、token数量、初始噪声哈希、模型权重storage/packed文件哈希、源配置/代码哈希、环境、加载/打包/推理时间、CUDA allocated/reserved峰值、采样RSS峰值、数据解码/生成/prefill/采样分阶段耗时。共享GPU上时间只作诊断，未测功耗/碰撞率/Jetson全系统统一内存需明确缺失。

执行前已实时确认主机zwl-acloud、NVIDIA L20 46068MiB；另一个项目训练占15428MiB，不停止。torch2.8.0、Triton3.4.0；现有环境无nvidia-modelopt，不在离线服务器暗中pip安装。

## 执行状态

服务器模块10项CPU/CUDA测试全部通过、无跳过、退出码0；本地runner指标/输出保护4项回归通过。独立6.1Sol high实施前验收通过。日志：`experiments/quant_route_20261008_code/cpu-tests.log`。

首轮`experiments/quant_route_20261008_smoke`在preflight停止，退出码1：旧`selected_task_validation.json`及step400原结果没有`status`字段。现场检查二者records完全一致，完成态位于`controlled_recovery_result.json`。兼容修复改为完成manifest=completed，并核验selected step400、missing_markers=0、seeds42/43/44及60条配对记录；新增完成manifest哈希。该轮未加载模型，失败result/log保留，不删除任何历史下载/权重。

修复后新目录`experiments/quant_route_20261008_smoke_v2`，包装器PID592615，日志`experiments/quant_route_20261008_code/smoke_v2.log`。执行命令：

```sh
PYTHONPATH=scripts:compression alpamayo1.5/.venv/bin/python -u \
  scripts/alp15_quant_route_pilot.py \
  --root /home/USER/lang/Lighten_VLA \
  --output /home/USER/lang/Lighten_VLA/experiments/quant_route_20261008_smoke_v2 \
  --clips 2
```

### 两clip筛查验收

v2 completed，包装器退出码0，209.887秒，18条轨迹（8B原精度、8B量化、原始结构量化各6）。原8B重载逐记录ADE差为0；两量化配置均252矩阵真实save/reload一致，无边界失败、噪声逐条配对、保护哈希一致。CSV、PNG/SVG、资源日志、既有记录导出器均成功。

| 两clip候选 | ADE(m) | FDE(m) | 本候选原精度参考ADE | 真实模型持久storage |
| --- | ---: | ---: | ---: | ---: |
| 最佳8B BF16重载 | 1.298883 | 3.279880 | 1.298883 | 原精度，不是可部署结论 |
| 最佳8B W4A16 RTN | 1.494978 | 3.804118 | 1.298883 | 9.564990GiB |
| 原始结构 W4A16 RTN | 1.510415 | 3.772303 | 1.335195 | 11.033313GiB |

注意原始结构在这两个clip的原精度参考ADE不是全20clip的1.232399。RTN两clip出现退化，只说明需要完整开发评估/更强量化器，不能据此证明量化都失败或推荐6B。

L20 sampled运行CUDA allocated峰值：8B量化10.815GiB、原始结构量化12.318GiB；加载阶段最高约21.217GiB。不是Orin统一内存，也不是最终packed-only loader；RSS/CUDA独立峰值不能直接相加作目标设备结果。

不含权重的证据包已取回`docs/baselines/evidence/quant_route_20261008/smoke_v2_evidence.tar.gz`，SHA256 `5a9d02a246d5e0bd283671a73997a8aa57bbcd7b3d68115d1a7d43185327d212`，本地与服务器一致。解包后独立核对completed/18轨迹/0失败、两份252矩阵manifest、重载差0及资源记录通过。实际packed权重留在服务器，不复制到Mac。

### 20clip开发比较

同一冻结代码、格式、范围扩展到20clip×3seeds；输出`experiments/quant_route_20261008_full20`，包装器PID597420，日志`experiments/quant_route_20261008_code/full20.log`。命令同上，仅改`--clips 20`与新output；每个量化候选60轨迹，8B原精度仅前2clip作重载门禁，不额外重跑官方入口。

完整运行completed，包装器退出码0；126条轨迹（8B原精度重载6条、两个量化候选各60条），无失败。所有初始噪声逐条配对、保护模块哈希及packed重载一致。

实际elapsed574.800秒。6.1Sol high独立复算聚合、代码/manifest哈希、字节预算、clip/seed噪声及资源记录通过。补充服务器全量复核32项源/输入哈希与504个实际safetensors哈希全部通过、退出0。首次哈希审计仅因导入辅助模块缺PYTHONPATH失败，没有计算模型；改为标准库独立审计后成功，两个日志均保留。

全程最高CUDA allocated21.217GiB、reserved21.736GiB；进程采样RSS最高10.453GiB。仅量化推理阶段8B allocated10.816/reserved14.029GiB，原始结构12.319/12.920GiB。47项资源phase均completed、没有采样异常。独立峰值不能相加，仍非Orin结果。

| 全20clip候选 | ADE(m) | 相对自身原精度ADE | FDE(m) | clip均ADE p95(m) | clip均ADE max(m) | 持久storage(GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 最佳8B原精度（冻结参考） | 1.299637 | — | 本轮没有重跑全部20clip | — | — | 约14.94裸权重 |
| 最佳8B W4A16 RTN | 1.442645 | +11.00% | 4.101184 | 2.867057 | 3.552225 | 9.564990 |
| 原始结构原精度（冻结参考） | 1.232399 | — | 本轮没有重跑原始基线 | — | — | 约20.64裸权重 |
| 原始结构 W4A16 RTN | 1.403494 | +13.88% | 3.989972 | 2.587843 | 4.386285 | 11.033313 |

原始结构RTN比8B RTN平均ADE低约2.71%，但多1.47GiB持久权重，最差clip均ADE更差。两个候选均相对自身原精度退化，不能把普通RTN定为精度优先的最终解，也不能由此反推6B更好。下一阶段优先ModelOpt AWQ/混合精度校准，原始结构与8B均保留，不删除旧模型。

尾部补充（clip内3seed均值后统计）：8B FDE p95/max=8.907679/11.188625m，原始结构7.737179/14.443861m。两量化候选相对自身原精度分别仅7/20、4/20 clips改善；原始结构RTN相对8B RTN仅9/20 clips更好。数据集被重复选型使用，胜率不能当独立泛化证明。

证据包`docs/baselines/evidence/quant_route_20261008/full20_evidence.tar.gz`（不含大权重），SHA256 `04d2d66969e184e69e00a357cf8d3383f8a1915c8a364ec6853303411d42f538`，已与服务器一致。独立解包到同目录`full20/`；原始代码、逐轨迹JSON、CSV、PNG/SVG、资源日志、两份矩阵manifest齐全。

完整执行/失败/依赖日志另包`execution_logs.tar.gz`已取回并解包同目录`execution_logs/`，SHA256 `5a5ff3f534a06ba68e4efe18abac7c40329e17a07c9f839afaa43299def88ac3`一致；包含主过程PILOT_EXIT=0和全量权重实际哈希审计。无需为此复制大权重到Mac。

所有时间是共享L20上的控制评测路径（自主生成＋另一次full-prefill＋diffusion），不是部署入口端到端延迟，也不是Orin吞吐。尚缺可接受延迟上限、Orin连接/JetPack及统一全系统资源结果，已向用户非阻塞询问。

## 下一阶段：ModelOpt AWQ依赖与桥接准备

决策6.1Sol high核查[ModelOpt0.39.0固定源码](https://github.com/NVIDIA/Model-Optimizer/tree/f329b19d4d0512b4a0774838a94f684e20347cb7)：torch>=2.6、transformers>=4.48且<4.58，覆盖当前2.8.0/4.57.1。INT4_AWQ_CFG为group128，awq_lite/alpha_step0.1，激活量化关闭。只是通用Linear API可适配，不是Alpamayo1.5认证。

官方wheel SHA256 `32f05317c81be1ff2ffeab749e5258b7bea8e4c6e60a09c760584f25ad03f648`。已在联网Mac准备缺失依赖并交付，整包SHA256 `e580f16aaf50832ce2bf73e739c1a86a7879ee12aa016547fd4839ee34cb247f`。只在`experiments/quant_route_20261008_code/modelopt039_vendor_v2`离线`--no-deps --target`安置，没有修改原venv；现场`modelopt.torch.quantization`导入退出0，版本及配置确认。

首次依赖解包错误地早于SCP完成，部分tar报EOF、安装退出1；保留该失败目录和日志。等待传输退出0并核对整包SHA后，在独立`_v2`目录重做，安装/导入均退出0。没有删除已有文件、没有使用不完整包继续校准。

依赖固定清单来自本轮实际resolver，非官方全部推荐版本：annotated-types0.8.0、cpsat-logutils1.0.0、markdown-it-py4.2.0、mdurl0.1.2、ninja1.13.2、nvidia-ml-py13.615.71、nvidia-modelopt0.39.0、orloge1.0.0、pip25.3、pulp4.0.0、pydantic2.13.5、pydantic-core2.46.5、pygments2.21.0、rich15.0.0、torchprofile0.0.4、typing-extensions4.16.0、typing-inspection0.4.4；numpy/scipy/torch等复用原环境。隔离导入成功不等于完整AWQ成功。

允许的下一步先是tiny BF16 CUDA Linear的官方AWQ API与数值桥接检查，非完整VLA实验。AWQ必须保留权重amax和input_quantizer.pre_quant_scale；后者即使激活量化关闭仍执行。不能丢掉输入逆缩放、不能重新RTN absmax覆盖AWQ校准amax。完整校准必须用训练侧真实驾驶图像、自主生成/full-prefill/diffusion，不使用20clip开发集选scale；所有目标层记录搜索与回退，之后才能进入完整AWQ任务比较。

### AWQ API数值桥接结果

脚本`scripts/alp15_modelopt_awq_smoke.py`，SHA256 `14a53f12ccbbbe28f91b72d0267a3efc898af4d7b7eecaab04a68e7df1e19398`，由执行者6.1Sol high实现、独立6.1Sol high实施前验收。独立随机Linear256→64，每group128内部高低方差交错；没有使用VLA checkpoint或任何clip。

```sh
PYTHONPATH=experiments/quant_route_20261008_code/modelopt039_vendor_v2 \
TORCH_EXTENSIONS_DIR=/home/USER/lang/Lighten_VLA/experiments/quant_route_20261008_code/awq_api_extension_cache \
alpamayo1.5/.venv/bin/python -u scripts/alp15_modelopt_awq_smoke.py \
  --output /home/USER/lang/Lighten_VLA/experiments/modelopt_awq_api_20261008_smoke
```

结果`passed_api_smoke_only`，退出0，59.156秒；扩展在独立实验cache成功加载54.5秒，callback两遍，无警告。原位对象identity、input量化关闭/weight量化启用、INT4/group128/static状态通过；全值finite，原权重/输入、fake-quant状态与diagnostics已safetensors保存。

- input pre_scale为BF16 [256]，范围0.150390625–6.625。
- 实际weight amax为BF16 [128,1]（64输出×2个128输入group展平），保存实际amax与step_size，step_size=amax/7检查通过。
- 正确保留输入缩放的显式前向与ModelOpt输出最大绝对差为0。
- 丢掉输入缩放则最大绝对差61.9375，证实不能直接套现有RTN格式。

这只是官方AWQ API/输入缩放桥接通过，不是完整VLA AWQ、不是实际packed AWQ导出、没有新ADE或部署性能结论。原ModelOpt环境未全局安装、原venv保持不变；本轮所有GPU测试正常结束，服务器剩余训练仍是原其它项目PID453201。

桥接证据`awq_api_smoke_evidence.tar.gz`已取回同一本地evidence目录，SHA256 `8c60a5f8afd0d97932fed296458e22fe1c6f784d6b8f0972816e970e6859c936`匹配；解包为`awq_api_smoke/`。主智能体独立读取JSON确认pass/2callback/0warning/代码hash，并解析safetensors头及payload确认保存的quant_output与manual_output形状、dtype、原始tensor字节完全一致。

下一次实施从真实驾驶训练侧校准和含pre_scale/amax的新packed格式开始；先两clip门禁再完整20clip。当前决策保留最佳8B为资源优先主候选、原始结构为精度竞品；不构建失败的统一低秩6B，不把RTN或tiny smoke称作最终最优。
