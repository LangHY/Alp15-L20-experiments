# ALP15-ALPASIM-PREFLIGHT-20261008：首轮仿真测试前置检查

日期：2026-10-08。执行与文件所有者：主智能体。状态：前置检查完成，渲染与闭环测试尚未启动。

## 决策单

- 用户授权：开始仿真测试；同时明确要求并行执行模型优化。
- 目标：先完成单场景、单并发 NuRec 渲染烟测，再接入模型 API 做闭环驾驶质量测试；模型优化另一路进行，不替代仿真结果。
- 来源：`ALP15-ALPASIM-API-RESEARCH-20261008.md` 中固定的官方 AlpaSim 提交与接口。
- 范围：检查指定 L20 的身份、资源、容器运行时、场景和镜像可用性、外网连接；不修改系统/现有 venv、不停止其它项目、不绕过 TLS 校验、不覆盖缓存。
- 停止条件：缺少容器运行时、资产或管理员授权；GPU 不足；来源/版本不一致。
- 验收：前置满足后才能启动真实渲染；必须保存原始退出码、GPU/RAM峰值、图像输出及后续完整 rollout。没有这些不能声称烟测成功。
- 文件分工：主智能体负责仿真记录与服务器调度；GPT-6.1 Sol high 子智能体负责独立 AWQ full 脚本、测试与文档，不自行启动远程 GPU。

## 环境及实测资源

通过已授权的本地2222端口转发连接，沿用专用可信 known_hosts、`StrictHostKeyChecking=yes`，未修改默认信任记录。

服务器 hostname：`zwl-acloud`；Ubuntu22.04.5、Linux6.8.0-87、x86_64；用户tuqiang。

- NVIDIA L20：46068MiB总显存，检查时使用15477MiB，驱动580.95.05。
- 其它项目进程PID453201使用15428MiB；没有停止或修改。
- RAM：125GiB总量、约116GiB available；Swap2GiB、约1.8GiB已使用。
- `/home/USER

以上是检查瞬时值，不能作为后续并行任务峰值或当前空闲资源的长期保证。

## 执行命令与结果

首条 SSH 非交互检查包含 `hostname`、`nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv`、`free -h`、`df -h /home/USER -v docker`、`command -v uv`。前三类检查成功，末尾命令未找到使整体退出码为1；不是GPU查询失败。

交互环境继续检查：

```sh
command -v docker; command -v podman; command -v apptainer; command -v enroot; command -v uv
ls -l /usr/bin/docker /usr/local/bin/docker /var/run/docker.sock /home/USER/.local/bin/uv 2>/dev/null
dpkg-query -W -f='${Package} ${Status}\n' docker.io docker-ce docker-compose-plugin nvidia-container-toolkit 2>/dev/null
sudo -n true
find /home/USER/lang -maxdepth 3 -type d \( -iname '*alpasim*' -o -iname '*nurec*' -o -iname '*nre-artifacts*' \) -print
find /home/USER/lang/Lighten_VLA -maxdepth 5 -type f \( -name '*.usdz' -o -iname '*nre*.tar' -o -iname '*alpasim*.tar' \) -print
curl -I --connect-timeout 5 --max-time 10 -sS https://nvcr.io/v2/
curl -I --connect-timeout 5 --max-time 10 -sS https://huggingface.co/
```

结果：

- PATH未找到Docker/Podman/Apptainer/Enroot/uv；所查常见Docker二进制、socket和uv路径无输出。
- 所查容器软件包未返回已安装记录；没有据此声称整个系统绝无其它自定义路径安装。
- `sudo -n true`退出1：目前没有确认可用的免密管理员能力；未尝试用SSH密码执行sudo安装。
- 指定查找范围内未找到AlpaSim/NuRec目录、USDZ或离线镜像包。
- 两个HTTPS请求都返回curl60：`SSL certificate problem: self signed certificate`。保留TLS校验，没有使用`-k`、修改CA或输出代理配置。

原始终端输出保留于本聊天的SSH工具记录；此文档保存可复现命令与摘要，并不是渲染日志。

## 并行优化前置检查

CPU只读检查现有训练缓存：`experiments/mlp_newdata8_recovery_20261002_165238/optimization_split.json`包含80训练clip和20监控clip；训练clip与`controlled8_expanded20_20261002_125000/result.json`的评测clip交集为空。

`teacher_cache`包含80个`calibration_*.pt`与20个`validation_*.pt`；用现有推理Python通过`torch.load(weights_only=True,map_location='cpu')`读取单例，没有加载模型或使用CUDA。

单例包含真实图像tokenized输入、位置及logit_mask、四个expert探针。已将schema传给优化执行者。完整teacher前缀不能冒称自主生成校准；可以沿既有`generate_prefix`从原prompt裁剪并重新自主生成。真实全模型AWQ与新的任务指标仍待执行。

## 用户约束与无特权备选路线

用户明确回复管理员安装与换服务器“两种方法都不行”。因此不执行特权安装、不更换服务器，转向现有 bubblewrap 加离线官方镜像 rootfs 的分阶段验证；方案详见 `ALP15-ROOTLESS-SIM-20261008.md`。

用户随后明确大文件获取边界：先核实国内镜像及对应版本/校验信息，只提供下载指令，不自行下载；小文件可直接准备。该限制覆盖渲染镜像、场景、权重和大型依赖包；不存在可靠国内镜像时说明未找到，不臆造下载地址。

实测 `unshare -Ur true`、bubblewrap 用户命名空间、绑定设备后的 `nvidia-smi` 均退出0。只读绑定宿主根目录时，小型 PyTorch CUDA 运算初次因 Error304 失败；补充 `--proc /proc --tmpfs /tmp --tmpfs /dev/shm` 后同一现有 venv 通过，输出 `BWRAP_CUDA_COMPUTE_PASS NVIDIA L20 1024`，退出0。该组合成功不能单独归因于其中某一项，更不能证明独立 OCI rootfs 或 NuRec 渲染已成功。

Mac 侧保持 TLS 校验，已登录 HF 凭据的只读检查结果：NuRec `26.01` 元数据可读，但默认场景 `sample_set/26.01_release/01d503d4-449b-46fc-8d78-9085e70d3554/01d503d4-449b-46fc-8d78-9085e70d3554.usdz` 下载 HEAD 返回403（大小2004837747字节）。NGC 标准匿名 pull token 流成功，`nvidia/nre/nre-ga:26.04` manifest 返回200，压缩层合计14287662578字节；未输出任何 token，未下载场景或大镜像。镜像元数据成功不替代场景授权、镜像架构/digest 与实际渲染验收。

当前仿真停止在资产下载权限门禁；不绕过访问限制、不代替用户接受许可。模型优化独立继续：v3 的6条 BF16 参考轨迹复现通过，但 AWQ 因嵌套资源 phase 记录错误退出1；失败产物保留于 `evidence/awq_full_20261008/failed_v3/`。v4 修复只改变日志记录边界，采用新代码快照与新输出目录，不覆盖v3。

### 后续实测更新

国内魔搭存在26.04同clip资产，其大小/SHA256与官方HF26.04固定提交完全一致；官方AlpaSim支持26.04，批准将其作为不同版本的新烟测候选，不冒称26.01复现。下载命令与校验步骤见 `ALP15-SIM-DOWNLOAD-COMMANDS-20261008.md`。没有自行下载场景或镜像，尚未进行独立rootfs渲染测试。NRE26.04 manifest digest和linux/amd64/config/入口已实际核验；未找到可靠国内镜像，因此备用命令明确标注官方NVCR来源。

AWQ v4结果为 `completed_awq_development_pilot`：2个训练clip校准、2个开发评测clip×3seed，252层全部打包/重载；6条BF16参考ADE精确复现，量化后meanADE=1.2327414155m，参考=1.2988827825m。持久storage16041757808→10272486512字节，减35.96%。这是小样本开发结果，不是Orin/闭环/最终精度结论。Ninja缺失触发ModelOpt C++扩展CPU fallback的警告已保留，须独立核实不等于AWQ算法退回普通max校准。首次量化评测allocated峰值19517032960字节，后次11617141248字节，不隐藏首次较高峰值，不能把持久storage当全系统峰值。

本地取回的三份v4证据与远端SHA256均一致：result=`f847aeddcedec56f15d07c5e2831b2cd9f613368cb1b60feccb136f65e82f824`，calibration_manifest=`8f882cfd813a95a89efa8dada07c37aeb5753e9cb4b199db4dd07a5eecc53413`，memory_phases=`ed9c2ea98f0a8dcd21fa6076c3504aed51f81f15d3bc4555259396740340f6a8`。证据位于 `evidence/awq_full_20261008/completed_v4/`，详细优化验收见独立AWQ报告。GPU任务结束后只余原其它项目PID453201，未停止该进程。

### 完整开发集评测完成

新目录 `experiments/awq_full_20261008_best8_full20_v4`，冻结code_v4、同2个训练clip校准、同ModelOpt legacy torch后端，参数 `--evaluation-clips 20`；原命令仅改变输出路径和评测数量，完整stdout/stderr保存 `experiments/awq_full_20261008_code_v4/full20_v4.log`。SSH前台执行进程退出0，result状态 `completed_awq_development_pilot`。

20clip×3seed：meanADE1.2827242199m、meanFDE3.6538041871m；clip均值ADE median1.1807787816m、p95=2.5047279000m、max=3.3171715736m。相冻结8B BF16参考meanADE1.2996366460m的配对均差-0.0169124261m。不把重复选型开发集当独立泛化或闭环证明。

远端独立标准库审计重新读取所有252个实际packed文件并计算SHA256，与本轮manifest均一致。小样本/完整评测的整文件SHA仅17/252一致，进一步逐文件解析safetensors header，按tensor名称核对dtype、shape、data_offsets指向的原始payload SHA256及metadata：252层全部exact match（退出0）。区别来自文件序列化表示，不能仅凭整文件哈希差异宣称张量变化；每轮各自文件哈希仍需独立保留。前6条轨迹的唯一字段差异为diagnostic_elapsed_from_clip_start_s，其ADE/FDE/pred/noise等一致。

本地完整评测证据位于 `evidence/awq_full_20261008/completed_full20_v4/`，4个文件与远端SHA一致：result=`16aa6b20915bac68597614361e3932c89b955e7c05c024f1f5025b435ddd76cc`；calibration_manifest=`8f882cfd813a95a89efa8dada07c37aeb5753e9cb4b199db4dd07a5eecc53413`；memory_phases=`f1eaf2b675deba5acba88d3764402c4a48f90f96aa375d2d0bc2b20134166789`；full20_v4.log=`fb9d6f4fb66e528e7527dbcca5d98ebd7d6ac1efa4b977cd09ce622d532a76e4`。完整过程的资源口径、局部退化、限制及下一步由独立AWQ报告复核。完成后只余其它项目PID453201；没有下载大文件或更改原推理环境。
