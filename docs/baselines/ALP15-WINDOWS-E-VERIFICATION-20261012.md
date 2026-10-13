# Windows E盘交付实机检查

日期：2026-10-12（Asia/Shanghai）；只读检查，不启动下载。

## 环境及入口

通过UU端口转发`REDACTED_IP:2223`及用户自行认证的共享SSH连接访问Windows。
Windows版本10.0.26200.9457，E盘存在，检查时空闲112,765,956,096 bytes。
Python launcher列出64位3.14、3.13、3.12及一个3.11环境。
未读取、输出或存储认证密码/令牌。

## 已有文件，不重复上传

E根目录已存在全部5个交付文件，均非reparse point。
远端Get-FileHash SHA256与本地逐项一致：

- Download-Alp15-Test.ps1：3b19f16db62537ffb59b9e368ea11b0ac178f4611c731d3f4133ebe5dcc9c0c2
- download_alp15_windows.py：c4ca28acef72ebb925b164f48863c1411a6071227eb4af9d2bf64ae23a62abb0
- small-download-plan.json：7a2356ea86921d535a96ef84e49f1018f8209dd31f7011423c6574372703471f
- test-candidates.csv：9fbf80b944f8b810557838da52e7318d9cc7d3855746bc7f39b183dd539e76af
- README.md：5f2fdf4e4e940980a9ead8ad939afc50973555ae8f3264167dc87dc20b244f1f

无需上传/覆盖；本轮未修改远端交付文件。
Windows PowerShell Language.Parser ParseFile返回空ParseErrors，退出码0。

## 默认计划实机测试

使用PowerShell EncodedCommand避免跨平台命令引号问题：

```powershell
$env:PYTHON_MANAGER_AUTOMATIC_INSTALL='false'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'E:\Download-Alp15-Test.ps1'
```

PowerShell执行策略仅为本次进程Bypass，未永久修改系统策略。
实际退出码0；完整核心stdout：

```json
{
  "dataset": "nvidia/PhysicalAI-Autonomous-Vehicles",
  "revision": "33f9bf447ed3bcb7d545ce13f4226f824214fafb",
  "files": 50,
  "clip_count": 949,
  "total_bytes": 54382445102,
  "GB": 54.382445102,
  "GiB": 50.64759878627956,
  "output": "E:\\alp15-small-test-949",
  "scope": "small_candidate_pool_not_certified_independent",
  "source": "https://huggingface.co"
}
```

末行：`PLAN ONLY: no network, dependency installation, output writes, or ZIP download.`
网络指下载器未发网络请求；外围SSH仍属于正常远程检查链路。
没有执行-Download、安装依赖、认证HF、下载ZIP、解压或模型操作。
因此仅验收Windows默认入口与交付资产完整，不验收实际下载/续传、独立性或精度。

## 失败留痕

初次远端`cmd /c ver`遇跨平台引号问题，改用`ver`确认版本成功。
本地主目录默认cwd不在仓库，相对包路径检查失败，随后全部使用显式仓库workdir。
一次本地编码拼写utf16-le失败，改为utf-16-le后SSH命令执行正常。
以上均未造成远端文件变更或启动下载。
