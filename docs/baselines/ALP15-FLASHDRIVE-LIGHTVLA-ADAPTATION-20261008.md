# FlashDrive / LightVLA 联合适配决策

日期：2026-10-08；文件所有者：主智能体。状态：一手源码审计与实验规划完成，未安装、未转换权重、未宣布适配通过。

## 一手来源与版本

- [FlashDrive原作者仓库](https://github.com/z-lab/flashdrive)：本次GitHub树API核实提交 `26265174f0413db77c8a16d99993dbeecacf458f`。
- [LightVLA原作者仓库](https://github.com/LiAutoAD/LightVLA)：本次commit API核实提交 `a4680fda5ffe73029190ac97328aa34b0e87a45a`。仓库网页显示已归档；不能期待持续维护替代本项目验收。
- 当前教师与学生仍为既有离线开发口径；不能采用论文minADE替换本项目配对mean ADE，不能把原作者服务器速度当Orin速度。

## 适配判断

FlashDrive应优先作为系统侧主线参考，LightVLA作为需训练验证的视觉侧适配，而不是最后才粘接两个插件。两者均要纳入最终交付，但是否启用某项机制取决于联合实测，不强行启用退化机制。

| 范围 | 必须适配的边界 | 验收与风险 |
| --- | --- | --- |
| FlashDrive加载 | 本地base/PARO/draft路径、权重格式、显式机制开关 | 当前AWQ并非PARO；不能直接换名字装载。不得自动下载或覆盖原venv |
| 连续帧缓存 | camera/frame身份、时间顺序、vision边界、stream reset | cold首窗与steady分记；首窗返回None不是成功轨迹；跨clip/乱序/重置不得污染 |
| DFlash | 当前学生的hidden维度、目标层、token与量化变化 | 不假定原10B草稿适配恢复8B；新候选需检查接受率、退回路径、质量和额外内存 |
| LightVLA剪枝 | Qwen3-VL位置编码、DeepStack、掩码与专家KV接口 | 保留非视觉/历史/动作边界；batch1先验；需训练而非零样本直接宣称精度保持 |
| 联合缓存 | 原帧patch身份→保留token→position/cache slot映射 | 动态保留集合改变时失效/重建；不能沿用未剪枝定长shift索引 |
| 最终部署 | ARM64/SM87、实际JetPack依赖、量化与graph workspace | 16GB为全系统统一内存；完整请求max<=300ms与教师ADE<=15%同时达标 |

### FlashDrive实际源码限制

[`__init__.py:72-108`](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/__init__.py#L72)：优化路径由`z-lab`前缀判断，并派生`-PARO`/`-DFlash`来源。直接传本地绝对路径不能按默认代码假定进入优化分支；offline适配必须显式区分路径和机制，禁止伪装远端名称触发下载。默认先BF16加载再搬GPU，仍需专门解决冷启动峰值。

[`pyproject.toml`](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/pyproject.toml)：Python3.12、torch2.9.1、transformers4.57.1、vLLM0.16.0、paroquant0.1.15，与现有torch2.8.0不一致。SM满足最低计算能力不等于ARM64 wheel、CUDA工具链和Orin实机内核已验证。先独立环境/编译可用性审计，不直接uv sync原环境。

[`quantization.py`](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/quantization.py)：使用旋转参数和Marlin W4A8状态，现有W4A16 AWQ张量格式不兼容。两路线做独立候选，不叠加量化同一矩阵。

[`streaming.py:116-164,241-291`](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/streaming.py#L116)：默认4视角×4帧，保留每视角最新帧，再以已建立帧范围移位KV。当前单时间点开发数据不足以证明跨窗口收益；需连续窗口与帧身份审计，不以重复同一帧冒充流式输入。

[`diffusion.py`](https://github.com/z-lab/flashdrive/blob/26265174f0413db77c8a16d99993dbeecacf458f/flashdrive/diffusion.py)：缓存速度跳过指定Expert调用。它不是本轮官方Euler减少inference_step的同一个因素；需另设完整10步/少步/缓存速度消融，记录实际Expert调用次数。

### LightVLA实际源码限制

[`modeling_prismatic.py:45-125,127-189`](https://github.com/LiAutoAD/LightVLA/blob/a4680fda5ffe73029190ac97328aa34b0e87a45a/prismatic/extern/hf/modeling_prismatic.py#L45)：实现按CLS、固定patch区段和task拆分token，以task条件评分筛选并同步掩码/位置索引；下游是PrunedLlamaModel，非Qwen3-VL即插即用补丁。

推论与待证：Qwen3-VL适配需同步三轴位置编码、所有DeepStack分支、视觉标记、attention和cache_position；原作者推理中按batch整体boolean选择再view，对各样本动态保留数不同的情况需另核实，不将batch1可用外推batch>1。首阶段保留原序并记录camera/frame/patch身份，再验证动态选择与流式缓存能否共存；若选择改变，必须正确重建相关缓存，收益也须计入重建成本。

## 锁定执行顺序

1. 完成本轮actualKV与官方10/6/4/2步配对，保留严格一致性失败，不改最终15%要求。
2. 现有AWQ仅先评估小M融合解码；大M探针变慢，禁止全模型无条件替换。
3. FlashDrive逐机制移植/评估：Expert融合与缓存速度→static cache/graph→真实连续帧缓存；PARO/DFlash先审计所需本地权重和Orin依赖，再决定是否下载。大文件仍只提供已核实国内来源的下载命令。
4. LightVLA在独立训练split适配Qwen3-VL，先用“不剪枝”路径证明接口等价，再记录实际保留数、开销、ADE/FDE与尾部风险。冻结策略前不使用独立测试集选参数。
5. 联合消融：固定可运行base、FlashDrive侧、LightVLA侧、联合侧四组；不得直接相乘两篇论文加速比例。连续窗口的教师同样使用相同观测与采样口径。
6. 最终Orin16GB联合验收：冷启动、首窗、稳态、重置与恢复；完整请求mean/p50/p95/p99/max、超300ms数、统一内存/功耗/温度、ADE/FDE与失败率。模型服务API和仿真接口使用一致版本、时间戳、坐标、64点dt0.1轨迹协议。仿真闭环通过后仍不能替代Orin硬件验收。

## 本轮执行留痕

2026-10-08：主智能体接管KV文件；显式new_candidate模式回归6项服务器测试全通过无skip。第一次candidate smoke_v2缺少隔离snapshot中的两个既有packed辅助模块，退出1；未加载模型，保留结果及日志。将冻结code_v1辅助模块原样复制到kv_code_v2后，启动candidate smoke_v3（2clip×3seed、10/6/4/2步），目录与旧失败隔离。源码、输入哈希及请求分阶段指标由脚本记录，不覆盖venv、权重或历史源码。

后续同日复核：smoke_v3退出0、30条完整记录，保护模块hash未变。结果与复核在evidence/jetson_300ms_20261008/kv_candidate_smoke_v3/；其中acceptance.md单独修正两clip配对教师的汇总口径，不改原JSON。暂保留4/6步进一步筛选，未扩20、未宣布Orin或两框架适配通过。目标服务当前usageLimited，不是complete。
