# Mac下载并流水线传输L20

固定949候选clip、50个ZIP、54.38 GB。官方HF固定revision，按既有冻结清单核验。
本轮仅交付脚本，不由智能体启动大文件。

## 开始

Mac的UU远程启用L20转发：本地2222，目标10.80.156.150，目标端口22。
然后在Mac终端运行下面整段。若SSH要求密码，只在终端输入，不写到脚本或聊天。

```bash
cd /Users/USER/Documents/ChatGPT/Lighten_VLA

ssh -p 2222 -M -S /tmp/alp15-l20-control \
  -o HostKeyAlias=REDACTED_IP -o StrictHostKeyChecking=yes \
  -o ControlPersist=600 -fN tuqiang@REDACTED_IP

caffeinate -i /Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B \
  scripts/download_alp15_mac_l20_pipeline.py \
  --run \
  --control-path /tmp/alp15-l20-control \
  --confirm-bytes 54382445102
```

SSH主机指纹须按真实服务器核对；脚本不绕过主机密钥验证。
`caffeinate -i`仅在命令运行期间防止Mac空闲休眠，不修改永久系统设置。
默认核对L20既有记录的hostname zwl-acloud、用户tuqiang、Linux和NVIDIA L20；不吻合则停止，不默认为正确服务器。
复用本机既有独立Python3.12/huggingface_hub1.31.0环境，不安装模型依赖或改全局环境。

## 行为与输出

- 本地：`/Users/USER/Downloads/alp15-small-test-949-20261013`。
- L20：`/home/USER/lang/datasets/alp15-small-test-949-20261013`。
- 每个ZIP下载完成且本地size/SHA通过后才排队上传；下一文件下载与已完成文件的rsync并行。
- 上传先进入专用`.incoming`，服务器size/SHA通过后才发布最终文件，不能把半文件当成功。
- 初次操作后最多6次网络重试，指数退避上限60秒；401/403、认证、主机密钥、空间、身份、SHA或已有文件冲突停止，不无限重试。
- 中断后执行相同命令：重新真实校验已有本地/远端文件，正确文件跳过，未完成缓存和rsync partial继续使用。
- 不删除本地完整ZIP或缓存，不覆盖冲突最终文件，不自动解压/模型测试/休眠。
- 本地每次运行保存独立`pipeline-*.jsonl`；远端保存绑定身份、逐文件验证与完成记录。

默认不带`--run`仅显示计划：

```bash
/Users/USER/Downloads/alp15-clip-venv-20260917/bin/python3 -B \
  /Users/USER/Documents/ChatGPT/Lighten_VLA/scripts/download_alp15_mac_l20_pipeline.py
```

首次两端各至少约57.11 GB空闲（解压另算）。保持Mac、UU远程和L20在线。
目录或日志文本出现并不代表完成；只有50个ZIP均双端验证通过才记录流水线完成。
949是候选clip名单，不构成独立性或精度已验证；只传原始资产，不运行模型。

脚本依赖同仓库`scripts/alp15_remote_receiver.py`与`deliverables/alp15_windows_949`冻结文本资产，
从现有工作区运行即可，无需把文件复制到Windows。
