# Windows E盘下载包

日期：2026-10-11。将本包全部文件放在Windows的`E:\`根目录。
当前编写环境为macOS，未挂载Windows E盘，不能代为保存到真实E盘。

## 数据与镜像

固定949个候选clip、50个ZIP、54,382,445,102 bytes，约54.38 GB。
数据仓：`nvidia/PhysicalAI-Autonomous-Vehicles`。
版本：`33f9bf447ed3bcb7d545ce13f4226f824214fafb`。
初次ZIP下载需约57.11 GB空余；建议预留120 GB供缓存、环境及解压。
不是官方独立“小版本”；drive独立性与完整精度未验证。

魔搭公开仓库：
https://modelscope.cn/datasets/nv-community/PhysicalAI-Autonomous-Vehicles/files

2026-10-11实查公开master根目录，只有calibration、.gitattributes、LICENSE.pdf、README.md。
`Root=camera`及`Root=labels/egomotion`文件列表请求均404。
未确认这批50个ZIP存在可用魔搭镜像；不将NuRec重建数据替代本输入。
下载脚本因此明确使用官方HF源；Python依赖优先清华PyPI镜像，
这不代表测试数据从清华或魔搭下载。

## 运行

需要Windows、E盘、64位Python3.10及以上（推荐3.12）。
在PowerShell执行以下计划命令，不下载数据、不安装依赖：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File E:\Download-Alp15-Test.ps1
```

开始下载（会安装独立环境依赖）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File E:\Download-Alp15-Test.ps1 -Download -ConfirmBytes 54382445102
```

默认虚拟环境：`E:\alp15-download-venv`。
默认数据目录：`E:\alp15-small-test-949`。
只读令牌/仓库授权由官方客户端管理，不把令牌放在脚本或命令中。
若脚本提示未认证，在创建好环境后交互登录，再重跑下载：

```powershell
& E:\alp15-download-venv\Scripts\hf.exe auth login
```

必须在官方数据仓页面申请并获得许可；登录不等于已获数据访问权。
中断后执行同一下载命令；缓存保留，已完整文件先校验再跳过。
已有坏文件、不同名单/计划或不安全路径冲突时停止，不删除或覆盖。
不自动解压、不运行模型、不休眠。
下载后评估只能使用`test-candidates.csv`中的949个ID。

## 版本与验收边界

Python依赖：`huggingface_hub==1.31.0`（官方PyPI版本和清华simple索引已核实）。
冻结计划SHA：`7a2356ea86921d535a96ef84e49f1018f8209dd31f7011423c6574372703471f`。
本包CSV使用LF换行，SHA：`9fbf80b944f8b810557838da52e7318d9cc7d3855746bc7f39b183dd539e76af`。
与此前Mac CSV的949条记录相同，原CRLF字节哈希不同。
源数据文件逐个使用固定官方revision报价中的大小与SHA校验。
默认计划与有界测试仅验证脚本准备，不宣称Windows实机或实际下载通过。
