# Windows E盘949候选下载：决策及来源核查

决策编号：W-20261011；日期：2026-10-11（Asia/Shanghai）。
主智能体负责来源核查、冻结资产和最终交付；6.1 Sol high执行者独占Windows Python/PowerShell文件。
范围：Windows下载工具及小型文本资产；不下载ZIP，不进行模型、GPU、解压或部署操作。
用户所说E盘属于Windows设备，本机为macOS且未挂载该盘；以拷贝到E根目录的交付包实现，不伪称直接写入。

## 事实依据

- 来源与固定版本：`nvidia/PhysicalAI-Autonomous-Vehicles@33f9bf447ed3bcb7d545ce13f4226f824214fafb`。
- 承接`ALP15-SMALL-TEST-CANDIDATES-20261011.md`冻结10块、949候选ID、50ZIP，54,382,445,102 bytes。
- 主智能体重新查魔搭公开API：
  `https://modelscope.cn/api/v1/datasets/nv-community/PhysicalAI-Autonomous-Vehicles/repo/tree?Revision=master&Recursive=false&PageSize=100`
  HTTP200，Code200，TotalCount4：calibration、.gitattributes、LICENSE.pdf、README.md。
- 同API带`Root=camera`或`Root=labels/egomotion`请求均HTTP404/Code10020101029。
- recursive列表返回TotalCount1888，当前页为calibration子文件；不能把该计数当成所需50ZIP的完整镜像证明。
- 搜索只发现同名仓库引用和NuRec仓库，未确认其他满足本输入的镜像。NuRec不替换相机/egomotion数据。
- `https://pypi.org/pypi/huggingface-hub/1.31.0/json`核实版本存在、requires_python>=3.10.0。
- `https://pypi.tuna.tsinghua.edu.cn/simple/huggingface-hub/`HTTP200，含1.31.0 wheel。
  清华的`/pypi/.../json`HTTP404不代表simple索引不可用；已转查有效索引，不下载wheel。

web-access技能已完整读取。Node22.23.0前置检查通过，Chrome未连接；公开查询使用小型API/curl与搜索，无需操作登录浏览器。
无令牌传给魔搭或其他第三方，不输出账号身份或凭据。

## 执行及验收边界

Windows版为独立包，Python默认仅计划；PowerShell默认仅计划且不安装环境。
`-Download`才允许安装独立E盘venv依赖并启动显式官方HF下载。
认证使用官方工具交互，不硬编码token。不静默替换数据源或数据集。
目录、链接/reparse、已有文件/receipt身份冲突停止并保留；逐文件size与SHA验证。
完成记录绑定50个文件，不复用完整池3165硬编码。
Windows资产CSV标准化为LF，与原CRLF版本949条记录相同，字节SHA另存：
`9fbf80b944f8b810557838da52e7318d9cc7d3855746bc7f39b183dd539e76af`。
计划SHA不变：`7a2356ea86921d535a96ef84e49f1018f8209dd31f7011423c6574372703471f`。

主智能体已离线验证949唯一ID、50唯一路径、准确容量、chunk一致和全部t0_us=5100000。
当前主机Python3.14.6，另有既有Python3.12虚拟环境；未找到pwsh或powershell。
不得把语法/离线mock测试当作Windows实机、真实传输、payload、独立性或精度验收。

## 执行回报与最终有界验收

执行者6.1 Sol high完成独立Python下载器与兼容PS5.1的ASCII启动脚本。
12项安全拒绝场景、模拟50文件下载与完成记录、重跑全部跳过、
第3文件中断保留缓存及重启续跑、venv身份/隔离/版本/位数拒绝检查通过。
主智能体运行默认入口退出0；`git diff --check`退出0。
无真实ZIP传输，无模型运行。

独立6.1 Sol high验收者批准“下载准备包”；socket禁用和禁止HF导入的
默认入口、50文件mock、坏文件保留、foreign receipt与链接拒绝、磁盘余量检查通过。
审查发现README缺少强制ConfirmBytes及既有venv缺真实身份/隔离验证，均已最小修复并复核。
Windows reparse与PS5.1仅静态核查；无Windows实机运行证据。

最终Python SHA：`c4ca28acef72ebb925b164f48863c1411a6071227eb4af9d2bf64ae23a62abb0`。
最终PS SHA：`3b19f16db62537ffb59b9e368ea11b0ac178f4611c731d3f4133ebe5dcc9c0c2`。
ZIP：`deliverables/Alp15_Windows_E_949_20261011.zip`，34,407 bytes。
ZIP SHA：`2d258183440b2e3cdabec744b28ef744fce4e1edc824b3920412d22fb1c93cf2`。
包内5文件平铺，无编译缓存、权重、ZIP数据或凭据；CRC与全部文件逐字节一致性通过。
用户将内容复制至E根目录后手动启动；本机未写实际E盘、未启动大文件下载。

随后用户指定NuRec魔搭仓库，只读核查记录在`ALP15-NUREC-MODELSCOPE-CHECK-20261011.md`。
该仓有1607重建场景，不据此替换当前949四相机候选输入或扩大到仿真执行。
