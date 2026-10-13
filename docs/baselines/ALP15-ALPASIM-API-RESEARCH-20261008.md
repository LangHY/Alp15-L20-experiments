# Alpamayo 1.5 接入 AlpaSim 的 API 仿真评估研究

日期：2026-10-08。所有者：主智能体。状态：官方接口与文档调研完成，未部署、未运行闭环仿真。

## 结论与授权边界

用户当前要求研究：暂时不能在 Orin 验证，先把模型服务封装成 API，在 Alpamayo 官方仿真平台测试。

方案可行且值得采用：官方平台为 AlpaSim，已支持 Alpamayo 1.5；其驾驶策略本来就是 gRPC 微服务。建议复用官方驾驶服务和预处理，通过独立模型插件加载本项目原始模型与压缩候选，不先自建 REST/JSON 协议。

这里的“可行”指官方存在相应集成点，不表示本项目服务、量化权重或服务器容量已经通过运行验收。本轮不安装服务器依赖、不下载场景或镜像、不修改模型、不运行仿真。保持用户“不补齐官方原始入口基线”的要求；后续相同仿真场景的未压缩参考是公平比较所需，不是补跑旧官方门禁。

## 核查版本及来源

`git ls-remote https://github.com/NVlabs/alpasim.git HEAD` 返回：

`affc2eab209fa43bdfa2f26c0f8d437922d78a68`

下列源码使用这一提交读取；官网硬件页为当日公开文档，页面标注最后更新 2026-08-25。正式实施仍应冻结镜像 digest、uv.lock、场景 artifact UUID、数据集实际 commit 与 SHA256，而不仅记录可变标签。

- [官方教程](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/TUTORIAL.md)：Alpamayo 1.5、外部 driver、NuRec 场景。
- [EgodriverService 协议](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/grpc/alpasim_grpc/v0/egodriver.proto)：会话、观测、导航与轨迹响应。
- [1.5 官方适配器](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/driver/src/alpasim_driver/models/alpamayo1_5_model.py)：模型加载、processor、导航与完整推理调用。
- [共享 Alpamayo 适配器](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/driver/src/alpasim_driver/models/alpamayo_base.py)：历史插值、坐标系、采样与输出时间戳。
- [设计](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/DESIGN.md)：分布式服务与非实时设计目标。
- [配置](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/configs/base_config.yaml)、[1.5 driver](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/configs/driver/alpamayo1_5.yaml)、[外部 driver](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/configs/driver_source/external_static.yaml)。
- [指标和运行说明](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/OPERATIONS.md)、[场景版本](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/data/scenes/README.md)。
- [依赖与权限](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/ONBOARDING.md)、[NuRec 硬件](https://docs.nvidia.com/nurec/basics/hardware.html)。

## 建议的服务结构

`AlpaSim runtime → EgodriverService → 官方观测/导航/坐标适配 → 本项目模型 → 轨迹 → controller/physics → 新状态 → renderer → 下一步观测`

服务可与渲染器同机或异机运行，只要 runtime 可访问服务地址。外部服务方式为 `driver_source=external_static`，地址池配置为 `wizard.external_services.driver`；不要从 Docker 容器内盲目把 localhost 当作宿主机。

模型插件保持官方接口及输出转换，仅替换 checkpoint/overlay/packed 加载；使用独立环境，不把 AlpaSim 最新工作区依赖直接同步进现有推理 venv。官方目前提供 `alpasim.models` entry point，1.5 对应 `Alpamayo15Model`。现有 RTN 推理脚本是控制诊断 runner，不能直接作为闭环 driver，也不能把教师缓存前缀喂给实时仿真模型。

### 协议不是一个无状态 predict 请求

官方 `EgodriverService` 定义八个 RPC：`start_session`、`close_session`、`submit_image_observation`、`submit_egomotion_observation`、`submit_route`、`submit_recording_ground_truth`、`drive`、`get_version`。

每个 rollout 的相机、运动历史和随机状态需要按 session 隔离。`drive` 返回带姿态和时间戳的 `common.Trajectory`，可附调试信息与候选轨迹。遵循官方 rig/local 变换和四元数约定，不能直接返回裸 XY 数组。

1.5 官方默认：四路相机、每路四帧；运动历史为 16 点、间隔 0.1 秒、覆盖 1.5 秒；从 route 构建导航文字；BF16/SDPA 完整自主 VLM reasoning 与 diffusion 推理。配置为 batch size 1、单条轨迹、选择第一条、轨迹优化关闭。输出时间戳从 t0 后一个 0.1 秒步长开始。

后续先保持这些口径，使用相同场景/种子/控制器/采样参数比较。不能使用真实未来轨迹挑选候选，否则会造成评测泄漏；官方基础配置 `send_recording_ground_truth=false` 应保持。导航路线是任务输入，与用未来 GT 选择最优预测不是一回事。

## 硬件与资产：当前真正的待办

- NuRec 官方当前支持列表包含 L20；未列 V100。不能承诺 V100 可以运行当前渲染器。官网列 Linux x86_64、CUDA 12.8+、GPU 内存大于 24GB，并建议系统内存大于 48GB；该要求不是一个场景的实测渲染峰值。
- AlpaSim onboarding 要求 NRE 容器对应的驱动支持 CUDA 12.8（570.x+），以及 Docker/Compose/Buildx、NVIDIA Container Toolkit、uv 0.9.17+，本地相关扩展还涉及 Rust。
- 1gpu 配置让模型、渲染器、physics 等共享 GPU，且其默认并发为 4，不是最低显存配置。建议首次单场景、所有相关并发降为 1，缓存/worker 联动减少；具体命令须生成配置复核后再运行。
- 1.5 教程写约 40GB 普通推理/60GB CFG，而 driver 默认单样本；模型 README 又按采样数区分约 24/40/60GB。不能把任一数字当本项目实际峰值。当前 L20 是否能同时容纳原精度模型和 NRE 需要运行验证；不要停止其它项目训练来制造空闲。
- 如果单 L20 不够，首选另一个官方支持 GPU 承载渲染器、L20 承载驾驶服务。CPU runtime/eval 可分离，但不会消除 GPU 渲染的占用。当前不能拿两张 V100 的显存简单相加当作支持的渲染设备。
- 已下载的 PhysicalAI 视频与 egomotion ZIP 不是 AlpaSim 直接所需的场景。还需要 `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` 的 USDZ 重建资产；场景映射、重建版本和渲染镜像需一致。原测试 clip 是否有匹配重建资产，尚未检查。
- 当前官方推荐 `public_2601` 套件有 913 场景，也有 26.04 套件。首次不要下载或跑全部；选一小批 UUID 固定资产，后续独立保留测试集。对场景缓存变更采用新目录，不照搬文档中的删除旧缓存建议。
- 若 L20 仍不能联网，应在可联网机器准备源码、依赖/镜像、NuRec 场景、必要配套资源，再上传本地运行；Ubuntu22.04 基础镜像本身不等于完整仿真镜像。资产授权、容器可下载性以及服务器 Docker 权限尚未实时检查，不在代码或报告保存 token。

## 测试口径与允许的结论

第一层：离线开环回归继续使用现有参考，不补齐官方原样入口。接入服务后，对捕获的同一组合法观测检查预处理、导航、轨迹坐标和随机种子的一致性，区分压缩差异与服务适配错误。

第二层：AlpaSim 闭环评估。官方已有责任碰撞、后方碰撞、驶离道路、最大真实路径偏离、20秒无失败完成比例、事故间行驶距离等指标；输出原始逐 rollout parquet、聚合报告、ASL 日志和失败视频。报告同时列场景数量、失败/超时/无效 episode，不把无效项静默排除。ADE/FDE 保留为开环指标，不用它们取代碰撞与越界。

第三层：模型时延敏感性。默认 `planner_delay_us=0`；非实时仿真可以等待慢模型，因而“控制步长 100ms”不表示模型达到实时 10Hz。记录 wall-clock 的预处理/模型/RPC p50/p95/p99 和负载；另设固定 planner delay 实验前先核验 runtime 的缓冲语义。同延迟比较精度，不同延迟研究鲁棒性，不能把 L20 RPC 耗时自动当作 Orin 延迟。

第四层（当前挂起）：Orin 上相同候选的实际 dtype/kernel 数值回归、统一内存峰值、冷启动和端到端实时性。服务器仿真支持“该模型在指定仿真条件下驾驶质量如何”的结论，不支持“Orin16GB 已可部署/已实时运行”的结论，也不是现实道路安全认证。

建议先做一场景接入烟测，再做冻结小规模闭环比较：现有未压缩参考、最佳8B原精度，以及完成真实驾驶校准后的AWQ/混合精度候选；现有RTN可保留为诊断对照。先建立闭环测量能力，不凭开环均值继续扩大剪枝。完整AWQ尚未执行，不能将其列为已经存在的已验收模型。

## 本轮执行留痕与限制

开发机：macOS；工作区 `/Users/USER/Documents/ChatGPT/Lighten_VLA`。读取现有量化路线报告、runner、git 状态；未执行模型。

公开网页搜索用于定位来源，技术结论使用官方源码/官网。使用 `urllib.request` 读取上述固定提交源码；没有执行远程脚本。源码具体读取包括 `egodriver.proto`、两个 Alpamayo adapter、driver/main.py、配置、设计、运行和依赖文档。

网络失败留痕：Chrome CDP 前置检查未连接，使用公开静态文档继续；GitHub tree API 返回403；浅克隆临时路径 `/tmp/alp15-alpasim-research.bxWNc2/repo` 因 SSL 失败未生成 checkout，之后改用固定提交 raw 源码读取。没有将克隆成功当成验收依据。

本轮仅新增此文档；不改既有脏文件。仍待服务器资源/容器权限检查、NuRec 访问授权确认、资产下载、API 实现与真闭环运行。所有建议的实施步骤均未启动。
