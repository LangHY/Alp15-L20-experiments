# Windows下载启动尝试：2026-10-12

用户本轮明确授权“尝试执行下载”，覆盖此前只给下载指令的限制，范围仅为
固定949候选clip/50ZIP/54,382,445,102 bytes，不扩展到全仓、模型或仿真数据。
通过已认证UU SSH共享连接执行，不保存或输出凭据。

## 首次启动

```powershell
& 'E:\Download-Alp15-Test.ps1' -Download -ConfirmBytes 54382445102
```

Windows Python3.14.6/64bit；新建独立`E:\alp15-download-venv`。
计划校验通过。清华pip安装失败退出1：
`No matching distribution found for huggingface_hub==1.31.0`。
没有创建测试数据目录或开始ZIP下载。

## 诊断

使用diagnosing-bugs和web-access技能；直接SSH、只读小型公开API，无浏览器登录。
Windows urllib访问清华simple返回200（189348 bytes，包含1.31.0），
官方PyPI1.31.0 JSON返回200，requires_python>=3.10.0，有通用py3-none-any wheel。
但实际venv pip26.1.2的`pip index versions ... -vv`对同索引请求返回HTTP403，退出1。
无PIP_*环境变量。原因不是固定版本不存在或Python版本不支持，
具体清华对pip请求的拒绝策略未查明。

## 保持版本的受限恢复

6.1 Sol high决策者批准WIN-HF-PIP-20261012-01：
仅一次在现有独立venv使用官方PyPI安装同版本及必要依赖，
不改脚本、数据源、名单、固定版本或模型；认证/权限失败停止。

```powershell
& E:\alp15-download-venv\Scripts\python.exe -m pip install `
  --index-url https://pypi.org/simple --disable-pip-version-check `
  --timeout 20 --retries 0 huggingface_hub==1.31.0
```

实际退出0。huggingface_hub1.31.0及必要依赖安装成功；hf-xet1.7.0为包依赖安装，
下载器仍设置HF_HUB_DISABLE_XET=1，不启用Xet传输。
实际版本核验1.31.0，`pip check`退出0：No broken requirements found。
依赖安装在独立下载环境，未影响模型推理环境。

## 最终阻塞

Windows保存了HF认证，但HfApi官方whoami检查发生ConnectTimeout，退出3。
未打印token或账号身份，不能称为认证有效或无效，目标仓权限尚未验证。
后续匿名HF公开API访问也失败URLError；urllib配置代理列表为空。
两者说明当前Windows到HF的访问受阻，尚未定位网络层具体原因。
没有尝试规避认证、修改系统代理/防火墙，或传递令牌给第三方。

## 当前结果与下一步

- 已建立独立venv并安装、核验依赖；五个交付文件未修改。
- ZIP下载数量0，无下载器后台进程、模型运行或解压。
- 需要Windows具备访问官方Hugging Face的网络条件，或用户提供已配置代理入口。
- 恢复后重新验证whoami与固定目标文件访问，再运行已授权下载。
- PowerShell包装入口仍固定清华索引；本轮官方安装只是受限手动恢复，
  未声称已经修复其未来在清华403下的重跑行为。

详细命令、退出码与脱敏输出保留于本聊天工具记录。
