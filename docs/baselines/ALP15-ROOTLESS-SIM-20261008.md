# ALP15-ROOTLESS-SIM-20261008：同一 L20 的无管理员仿真路径核查

日期：2026-10-08。文件所有者：rootless_sim_20261008 子智能体。范围：公开一手源码研究、决策草案和主智能体现有只读检查的归档；本执行者未连接服务器、未安装软件、未下载模型/镜像/场景、未启动渲染。

用户最新限制：非常大的文件先核查国内镜像，并向用户提供下载命令；智能体禁止自行下载。小文件可准备。因此NRE约14.3GB的压缩layers、26.01约2GB/26.04约1.72GB的USDZ均仅做元数据研究和命令准备，不自动获取。国内来源仍需核验发布主体、版本、digest/hash和资产使用权限，不能猜镜像地址或绕过授权。

## 结论

存在值得继续验证的第三条路径：在用户目录准备官方镜像的 Linux amd64 合并根文件系统，以服务器已有 bubblewrap 建立挂载命名空间，绑定现有 NVIDIA 设备与驱动库，再运行官方 NRE 服务。当前已验证命名空间、宿主根文件系统内 NVML 查询及轻量CUDA kernel；尚未验证独立镜像根文件系统、NuRec 渲染或 AlpaSim 闭环。不能从宿主rootfs路径推出独立镜像已通过。

这是一项容器启动与服务编排适配，不改变模型结构。NRE 的公开标准软件要求仍是 Docker 与 NVIDIA Container Toolkit；bubblewrap 的可运行性需要本项目单独验收，不能写成 NVIDIA 官方保证。

Docker rootless 不是当前最短路径：官方要求 newuidmap/newgidmap，以及至少 65,536 个 subordinate UID/GID；主智能体现有检查显示均缺少。Enroot 的运行部分使用单 UID/GID 映射，不受这两个缺项直接阻塞，但其官方 GPU hook 需要尚未安装的 nvidia-container-cli。Enroot v4.2.0 的 `import` 与 `load` 也有不同权限路径，不能混为一谈。

## 一手来源和冻结点

下列原文在本次通过公开 HTTPS 实际读取。没有关闭 TLS 校验。

- AlpaSim：提交 `affc2eab209fa43bdfa2f26c0f8d437922d78a68`。继承已冻结提交并重新读取 [ONBOARDING.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/ONBOARDING.md)、[base_config.yaml](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/configs/base_config.yaml)、[wizard.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/wizard.py)。`base_config.yaml:137` 固定默认 renderer 标签为 `nvcr.io/nvidia/nre/nre-ga:26.04`；标签 digest 尚未读取。
- Enroot v4.2.0：本次 `git ls-remote` 返回 tag commit `9367f9d1b1479c51480f97258295723ae300d32a`。原文：[requirements.md](https://github.com/NVIDIA/enroot/blob/9367f9d1b1479c51480f97258295723ae300d32a/doc/requirements.md)、[Makefile](https://github.com/NVIDIA/enroot/blob/9367f9d1b1479c51480f97258295723ae300d32a/Makefile)、[common.h](https://github.com/NVIDIA/enroot/blob/9367f9d1b1479c51480f97258295723ae300d32a/bin/common.h)、[runtime.sh](https://github.com/NVIDIA/enroot/blob/9367f9d1b1479c51480f97258295723ae300d32a/src/runtime.sh)、[docker.sh](https://github.com/NVIDIA/enroot/blob/9367f9d1b1479c51480f97258295723ae300d32a/src/docker.sh)、[enroot-mksquashovlfs.c](https://github.com/NVIDIA/enroot/blob/9367f9d1b1479c51480f97258295723ae300d32a/bin/enroot-mksquashovlfs.c)、[98-nvidia.sh](https://github.com/NVIDIA/enroot/blob/9367f9d1b1479c51480f97258295723ae300d32a/conf/hooks/98-nvidia.sh)。初次也读取 main 安装页及 GPU hook，后续技术结论以 v4.2.0 为准。
- bubblewrap v0.6.1：本次 `git ls-remote` 返回 tag object `2965bb00747ece2c0509b05a6e3555ff2154f820`，peeled commit `a362d3c0ee79097ce2e3fca4ad003f1e96dad472`。[bubblewrap.c](https://github.com/containers/bubblewrap/blob/a362d3c0ee79097ce2e3fca4ad003f1e96dad472/bubblewrap.c) 核对实际服务器版本的参数；另读 v0.8.0 README 与源码理解通用挂载语义，不能以新版本参数替代现场版本。
- [Docker Rootless prerequisites](https://docs.docker.com/engine/security/rootless/#prerequisites)：缺失的映射助手和 subordinate ID 是官方明确的先决条件。
- [NuRec 硬件要求](https://docs.nvidia.com/nurec/basics/hardware.html)：Linux x86_64、CUDA12.8+、大于24GB GPU内存，明确列出 L20；不支持 Linux aarch64。
- [NRE NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/nre/containers/nre-ga)：当日展示最新 26.04.01、压缩体积13.31GB；这是另一个标签的信息，不自动替换 AlpaSim 固定的26.04。软件条款链接为 NVIDIA Software License Agreement 与 Omniverse Product-Specific Terms。
- [NuRec 数据集](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)：公开卡片可读，文件 gated，需要已有授权账号接受具体数据协议。主智能体追加实测：Mac既有HF认证访问固定26.01默认USDZ的HEAD为403；元数据可读不等于文件授权通过。
- 可选 Mac 端资产导出工具：[crane export v0.20.3](https://github.com/google/go-containerregistry/blob/v0.20.3/cmd/crane/doc/crane_export.md)、[mutate.Extract](https://github.com/google/go-containerregistry/blob/v0.20.3/pkg/v1/mutate/mutate.go)。源码按倒序层处理 whiteout 与覆盖关系。仅研究，未安装或执行。

## 现场事实与证据边界

以下由主智能体通过既有可信 SSH 路径检查并发回本执行者；完整原始输出在主聊天终端记录。记录时点为2026-10-08，本文件未保存凭据或个人敏感信息。

| 检查 | 主智能体实测 | 可以推导的结论 |
| --- | --- | --- |
| 系统/GPU | Ubuntu22.04、Linux6.8、x86_64、L20、驱动580.95.05、46068MiB总显存 | 硬件类型与官方支持列表相符；空闲量需执行前重查 |
| userns | `unshare -Ur true`退出0；`user.max_user_namespaces=514904`、`unprivileged_userns_clone=1` | 单用户 namespace 可创建 |
| AppArmor | `apparmor_restrict_unprivileged_userns=0` | 未触发该项系统级限制；不能排除其它策略 |
| bwrap | 版本0.6.1；`bwrap --unshare-user --ro-bind / / /usr/bin/true`退出0 | userns内挂载及命令启动通过 |
| 设备权限 | `/dev/nvidia0`、ctl、uvm、uvm-tools、modeset 均 `crw-rw-rw-` | 已有设备节点可访问；不需要新建节点作为首测前提 |
| GPU namespace 查询 | `bwrap --unshare-user --ro-bind / / --dev-bind /dev /dev /usr/bin/nvidia-smi --query-gpu=name,memory.total --format=csv`退出0，返回L20和46068MiB | 宿主rootfs内 NVML 可访问 GPU；不是独立镜像/CUDA/渲染验收 |
| 宿主rootfs CUDA失败 | 上述只读root和dev绑定组合运行已有venv的CUDA ones报`cudaGetDeviceCount Error304 OS call failed unsupported` | NVML通过不足以证明CUDA；只读环境需要进一步适配 |
| 宿主rootfs CUDA修正 | 加`--proc /proc --tmpfs /tmp --tmpfs /dev/shm`后CUDA ones、加1、synchronize/assert2退出0；输出`BWRAP_CUDA_COMPUTE_PASS NVIDIA L20`，peak allocated1024bytes | 该组合具备实际CUDA运算能力；三项一起变更，不能把原因单独归到proc、tmp或shm |
| 工具 | 有 mksquashfs、unsquashfs、fusermount3、curl、zstd、gzip；未发现 jq、fuse-overlayfs、squashfuse | 解包式rootfs无需FUSE；不能直接照跑Enroot全部命令 |
| 驱动库 | ldconfig有 libcuda、libnvidia-ml；主智能体追加查询确认nvoptix、rtcore、EGL、nvvm、ptxjit、encode等库存在；未发现libnvidia-container | 主要驱动库存在，仍需记录具体路径/版本/hash；GPU hook所需cli/库缺少 |
| rootless映射 | 无 newuidmap/newgidmap、无tuqiang subuid/subgid映射 | Docker官方rootless前提未满足 |
| 外网 | 服务器 nvcr/HF curl60 self-signed TLS；Mac公开源码可读 | 离线资产准备可研究；未以`-k`或修改CA绕过 |

## Enroot 的真实边界

`bin/common.h:93-130` 的 `unshare_userns` 只把当前euid/egid各映射1个ID，直接写 `setgroups=deny`、uid_map/gid_map。因此运行解包rootfs不需要借助 subordinate IDs。`src/runtime.sh` 的 `start` 进入user/mount namespace；`create`以unsquashfs展开已完成的 `.sqsh` 后修正权限。采用解包后的rootfs，可避开启动SquashFS文件时的squashfuse/overlay组合。

Makefile支持自定义prefix/libdir/sysconfdir，技术上可部署到用户目录；解包现成deb后也必须核对脚本内路径并显式设置ENROOT_LIBRARY_PATH/ENROOT_SYSCONF_PATH/ENROOT_RUNTIME_PATH等，不能假设解包即能运行。所有目录放在专属任务路径，不能覆盖既有环境。

`src/docker.sh:448-495` 的 **import** 会调用enroot-aufs2ovlfs与enroot-mksquashovlfs。后者源码显式请求CAP_SYS_ADMIN，并把userns替代实现放在禁用的`#if 0`中；Makefile的setcap目标为相关二进制授予能力。故下载一个无cap二进制不能视为import已解决。

v4.2.0的 **load** 则在`src/docker.sh:497-557`先以用户xattrs处理whiteout，再在user/mount namespace使用native overlayfs合并并拷贝rootfs。这是可另行验证的无cap路径，仍需jq、parallel、tar、zstd、native overlayfs/userxattr支持，以及可信registry网络；本机只通过基础namespace，不等于overlayfs已验收。离线合并rootfs在当前条件下少一道服务器导入阻塞。

GPU hook的行为更关键：`conf/hooks/98-nvidia.sh`在启用NVIDIA设备时要求`nvidia-container-cli`，以`--user configure --no-cgroups`挂载匹配驱动资源，默认capability仅utility；compute需显式启用。用户目录部署该cli与libnvidia-container理论上可行，但需要对所选固定版本包执行依赖/加载器检查，不能只拷一个可执行文件。此路线本次未实施；bwrap显式绑定是当前已有二进制的首选验证对象。

## 最小离线资产及rootfs要求

1. 固定 `nvcr.io/nvidia/nre/nre-ga:26.04` 的 **linux/amd64平台manifest digest**、image config digest、全部layer digest。目录标签和平台均需写入manifest；先检查授权/manifest可访问性，后决定大文件下载。
2. 保留官方image config中的Env、WorkingDir、Entrypoint、Cmd和版本信息。AlpaSim的renderer command不代表可省略镜像entrypoint可能做的初始化。解包后逐项核对再形成执行命令。
3. 合并rootfs须正确处理whiteout、opaque目录、symlink、hardlink、文件权限；Docker-save/OCI-archive不是现成rootfs，逐层直接tar解包不合格。可用已审查的`crane --platform linux/amd64 export IMAGE@sha256:DIGEST ROOTFS.tar`导出合并文件系统，同时单独保存config与manifest；该命令是草案，DIGEST仍待实测，禁止`--insecure`。
4. Mac只下载与打包、不执行模型或渲染。保留Linux内容的tar，再由L20非root解包新目录；不要先在macOS文件系统展开再重新打包以免Linux权限/链接/大小写语义丢失。解包前验证成员路径、链接逃逸和设备条目；使用新专属目录，禁止覆盖既有模型、缓存或系统路径。
5. 服务器无需Docker daemon、subuid配置或mknod；最短启动依赖为已有bwrap、可信解包工具、官方rootfs、现有设备与匹配驱动库。FUSE只在采用挂载型SquashFS时才有关，解包路径不需要。
6. 另准备一个与冻结场景表匹配的合法NuRec USDZ、官方所需ego-hood资源及同版AlpaSim代码/协议。已有PhysicalAI原视频不替代这些资产。资产目录记录来源、版本、SHA256、拷贝位置；不把gated资产提交Git或公开分享。
7. 闭环还需官方base服务或独立锁定的CPU Python环境（wizard/runtime/controller/eval）以及physics依赖。uv/Rust/wheels需按官方lock与Linux平台准备到专属环境，不能同步到现有模型venv。原始驾驶模型和checkpoint沿用已核验来源；此文档不更换研究路线。

主智能体追加只读信息：Mac的HF认证存在标志为true（未输出凭据），NuRec26.01元数据列出916个USDZ，官方默认场景`01d503d4-449b-46fc-8d78-9085e70d3554`对应USDZ大小为2,004,837,747bytes（约1.87GiB）。该文件的已认证HEAD实测403，尚未完整下载。NGC首次匿名manifest请求为401，随后标准anonymous pull token流程成功，26.04 manifest实测200，压缩layers合计14,287,662,578bytes；不能把首个401直接认定为镜像账号授权失败。Enroot官方`docker.sh`也实现这种registry认证交换。Mac Docker Desktop daemon未运行不会阻塞独立registry客户端读取/导出。镜像layers未下载，平台digest/config仍需从主智能体实际manifest记录核对后冻结。

## 国内场景镜像与26.04版本决策补充

主智能体报告国内ModelScope同名dataset ID111731，commit `e4a281c0b5809d9425f06c1e5b10fd020003c4e2` 仅含`sample_set/26.04_release`，所查26.01根目录404。同clip的26.04 USDZ元数据大小1,724,125,839bytes，SHA256 `3e2eab16f8cd794f70b7f9b6b0dd290e66ea6f3141304c676a03515ebb0959ee`。这是国内来源的元数据证据，不代表文件已下载或已经在本机算出hash，也不直接证明发布主体和资产许可。

本执行者初次独立读取官方HF26.04目录tree元数据，得到完全相同文件路径/大小1,724,125,839bytes、git pointer blob oid `be67441b85528be71f61fad26463befb40dd7018`，但匿名响应中的LFS oid/xetHash被掩码隐藏。随后主智能体通过Mac已有认证调用`HfApi.dataset_info(revision="26.04", files_metadata=True)`，读取官方HF commit `5fe7a149b40ee83a2d3e09f7fe5dc2f217cabe11` 下指定USDZ的LFS SHA256 `3e2eab16f8cd794f70b7f9b6b0dd290e66ea6f3141304c676a03515ebb0959ee`、size1,724,125,839，均与国内ModelScope元数据完全相同。此追加检查只读元数据、未抓取文件、未输出认证token。现在已建立两来源声明的内容hash/大小一致性；实际下载后仍须在用户文件上验算，才能确认收到的字节正确。

官方AlpaSim固定提交的 [场景README](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/data/scenes/README.md) 明确支持`public_2604`。同clip的官方映射如下，来自 [sim_scenes.csv](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/data/scenes/sim_scenes.csv) 行2/956与 [sim_suites.csv](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/data/scenes/sim_suites.csv) 行2/1682：

| 重建版本 | 套件 | artifact UUID | NRE version_string | 固定相对路径 |
| --- | --- | --- | --- | --- |
| 26.01 | public_2601 | 4f6bf6bb-9696-43b9-a1bf-04e12ccad5cc | 26.1.112-59673f88 | sample_set/26.01_release/01d503d4-449b-46fc-8d78-9085e70d3554/01d503d4-449b-46fc-8d78-9085e70d3554.usdz |
| 26.04 | public_2604 | 23dd34ea-a8d1-410c-aef7-d13f554cc4c9 | 26.4.96-91b06fb8 | sample_set/26.04_release/01d503d4-449b-46fc-8d78-9085e70d3554/01d503d4-449b-46fc-8d78-9085e70d3554.usdz |

因此技术上可以用受支持的26.04同clip开展**新的26.04烟测**，不必修改模型或官方代码；这不是复现26.01。决策为：保留原26.01的403阻塞证据，在确认国内来源使用权限与文件完整性后，单独命名/存放26.04测试。现有renderer官方默认本身为26.04，但仍须固定所用linux/amd64镜像digest并实际渲染验收。不能把26.04文件重命名成26.01 artifact UUID填入旧缓存，也不能把其结论和26.01基线混用。

最小离线接入可直接采用官方已有`scenes.local_usdz_dir`，不新增ModelScope下载适配器：[schema.py:58-64](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/schema.py) 与 [sceneset.py:124-199,230-250](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/scenes/sceneset.py) 已支持扫描本地USDZ的`metadata.yaml`，提取uuid/scene_id/version_string并生成内存local套件。后续用户完成下载并校验后，先确认metadata三项分别为 `23dd34ea-a8d1-410c-aef7-d13f554cc4c9`、`clipgt-01d503d4-449b-46fc-8d78-9085e70d3554`、`26.4.96-91b06fb8`，再使用只含这个USDZ的新26.04目录，配置`scenes.local_usdz_dir=明确绝对路径`、`scenes.scene_ids=null`、`scenes.test_suite_id=local`、`scenes.limit_to_first_n=1`。local目录不能顺带混入其它版本文件；原官方CSV保持原样。

官方本地扫描只读取metadata，没有自动验证下载文件的SHA256。用户执行下载后必须独立验算SHA256、大小与ZIP CRC，并把metadata与官方行进行对照。预期内容SHA256已同时由固定HF commit和固定ModelScope commit元数据交叉确认；未下载时不能称本机资产完整性验收已经完成。`scenes.scene_ids=[同clip]`在多版本CSV中会选择last_modified最新artifact，因此也不能依靠裸scene_id锁定26.01。

另一个重要限制：[context.py:22-95,179-199](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/context.py) 表明`WizardContext.create`会调用`fetch_artifacts`；非local路径进一步调用`create_sceneset_directory`并下载缺少文件。即使`wizard.run_method=NONE`也发生在这之前，故用户下载前不能通过“仅生成配置”命令间接下载USDZ。本轮只给配置草案，待用户自行获取大资产后再启动向本地资产路径绑定的流程。

## 驱动和设备绑定草案

bwrap v0.6.1源码的`--dev-bind`明确允许设备访问，而普通`--ro-bind`不等价。独立rootfs需单独绑定已有`nvidia0`、`nvidiactl`、`nvidia-uvm`及本场景实际要求的其它设备；先用临时`--dev /dev`构造基础设备，再逐个`--dev-bind`已有节点。不新建设备、不加载内核模块、不操作cgroup。

共享宿主网络满足localhost/gRPC通信的技术前提；无需为首测创建新net namespace。服务端只绑定127.0.0.1、端口先检查空闲，不能照搬镜像默认0.0.0.0对外暴露。独立挂载`--proc /proc`、可写tmpfs `/tmp`与`/dev/shm`已经是本机宿主rootfsCUDA通过组合的一部分，须在后续路径保留；不是只把所有路径ro-bind就能运行。sys按用途挂只读视图，持久输出/缓存用明确的任务路径提供写权限。

驱动库从现场ldconfig/readlink解析得到实际文件，库与symlink链需来自同一580.95.05驱动。基本候选包括libcuda.so.1、libnvidia-ml.so.1；JIT可能需要libnvidia-ptxjitcompiler与libnvidia-nvvm；若NRE实际后端使用OptiX/图形/编码，再补对应libnvoptix、rtcore、GLX/EGL、encode/nvcuvid及配置文件。这个列表是核查候选，不能当已经确定的NRE最小依赖。

只把驱动接口库绑定到独立 `/run/nvidia/driver-libs` 并设经过审查的LD_LIBRARY_PATH，不能用宿主整个`/usr/lib`遮盖镜像系统库。CUDA runtime、cuDNN、PyTorch等用户态框架库通常由官方镜像提供；不能用宿主版本随意替换。ldd实际依赖、dlopen错误、PTX支持、OptiX请求均以镜像实测为准，失败即归档后升级。

## 非特权分阶段验证决策草案

编号：D-ALP15-ROOTLESS-SIM-20261008。本执行者提交技术草案；**尚未作为获准部署决策或验收单**。主智能体/项目要求的决策者需复核具体镜像manifest、入口初始化、绑定清单、资产权限和GPU调度后再执行依赖步骤。

- 目标：保留官方模型和NuRec服务行为，验证同一L20的用户目录运行时；不把适配测试替代官方原样基线。
- 风险：中到高，涉及核心运行时/驱动接口/评测环境；阶段0为只读且已执行，后续有下载、用户目录解包及GPU资源占用。
- 文件所有者：本文仅本执行者；服务器实验目录、GPU、实际命令与日志仅主智能体管理。后续脚本另分独占文件，避免同文件并发修改。
- 允许边界：不sudo、不改/etc或CA、不setcap、不修改设备权限、不停止其它项目、不占用现有服务端口、不覆盖已有venv/checkpoint/cache、不下载未授权资产。

| 阶段 | 实际操作/验收 | 当前状态 |
| --- | --- | --- |
| 0 | namespace、bwrap挂载、现有设备、宿主rootfs内NVML和轻量CUDA，保存退出码 | CUDA初次Error304，增加proc/tmp/shm后通过 |
| 1 | 固定镜像linux/amd64 digest/config，确认NuRec授权、资产元数据与大小；准备可信离线rootfs | 未执行 |
| 2 | 独立rootfs内`/bin/true`、加载器/库依赖、镜像Env与入口检查 | 未执行 |
| 3 | 独立rootfs显式驱动/设备绑定；cuInit、device count、轻量CUDA kernel并synchronize；记录库路径与退出码 | 未执行 |
| 4 | 单场景NRE服务；单worker/单backend/小cache；真实render RPC保存图像、日志、GPU/RAM峰值及耗时 | 未执行 |
| 5 | 复用官方session/预处理/physics/controller/runtime，完成单rollout并保存原始ASL/parquet/退出码 | 未执行 |

阶段0的轻量CUDA实测使用既有venv，仅证明宿主rootfs路径；阶段3须在NRE独立rootfs重新验收。主智能体成功命令使用已现场核验的现有venv：

```sh
bwrap --unshare-user --ro-bind / / --dev-bind /dev /dev \
  --proc /proc --tmpfs /tmp --tmpfs /dev/shm \
  /home/USER/lang/Lighten_VLA/alpamayo1.5/.venv/bin/python -c 'import torch; x=torch.ones(1,device="cuda"); y=x+1; torch.cuda.synchronize(); assert y.item()==2; print("BWRAP_CUDA_COMPUTE_PASS",torch.cuda.get_device_name(),torch.cuda.max_memory_allocated())'
```

这条归档为已有轻量检查复现，仍由主智能体统一GPU调度。启动服务但未发送真实render请求、只拿到health/get_version，都不满足阶段4。

官方AlpaSim已有SLURM_ENROOT实现：`wizard.py:71-76`要求活动Slurm allocation，不能在本机伪造SLURM_JOB_ID冒称已有官方Slurm部署。`wizard.py:81-101,115-121`支持NONE生成配置而不启动，但仍生成Docker Compose布局；若把生成命令与地址映射到bwrap进程，应额外逐项核对DNS/localhost、service顺序、卷、env、入口与退出码。NONE不自动提供bubblewrap后端，也不能认为无需检查就一定能生成配置。

阶段4先启动renderer并以合法单场景请求测资源，再根据真实峰值决定是否能与原始模型同卡运行。已有其它项目约15GB占用属于检查瞬时值，不能提前保证renderer+driver共存；不得停止其它项目腾显存。主智能体最新回报完整AWQ任务模型已经加载，此时不得并发启动大渲染。必要时先串行烟测，闭环必须保留实际资源限制和失败证据。

停止/升级条件：userns/mount被策略拒绝、设备或必需driver库无法合法访问、独立rootfs ABI失败、资产gated未授权、镜像版本不匹配、CUDA PTX失败、剩余资源不足或OOM。不得通过降低模型精度/修改模型/替换场景版本使测试“通过”；这需要另行决策。停止仅结束本任务新进程，不影响其他用户进程，产物保留可恢复。

## 本执行者回报

开始/结束：2026-10-08本聊天任务期间；精确工具时间保留于聊天记录。环境：macOS开发机；服务器事实由主智能体提供，本执行者未在L20运行模型。

变更仅新增本文，无服务器变更。公开读取使用curl默认TLS和网页工具；git ls-remote两个版本tag命令成功。部分推测的GitHub文件路径返回404（Enroot src/enroot.c、bubblewrap.xml、AlpaSim wizard/slurm.py等），随后根据官方pyproject定位实际wizard路径；没有把不存在文件当事实。初始Chrome CDP检查node22.23.0通过、Chrome未连接；按web-access静态公开来源工具表继续，未启动浏览器自动化。

已达成：判定Docker rootless当前缺项、核实Enroot不同权限路径、归档bubblewrap宿主rootfs内实际CUDA通过及前一次失败、准备最小资产/驱动绑定/验收方案、核实官方26.04同clip映射和本地资产接入流程、确认国内与官方固定commit元数据所声明的内容hash/大小一致。未达成：大资产由用户下载、实际下载字节与资产metadata验收、真正镜像GPU透传、NuRec烟测、闭环驾驶评测。后续遵守用户大文件只给命令的限制，需要复核资产授权与具体部署命令，而不是再次要求管理员安装或更换服务器。
