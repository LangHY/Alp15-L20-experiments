# Mac-L20流水线交付与终端启动

日期：2026-10-13（Asia/Shanghai）。主协调者独占本文。
实施依据D-20261013-MAC-L20-PIPELINE-01。用户最新授权交付后开终端启动，随后停止。

## 实现与有界验收

一个下载执行者、一个上传工作线程、队列容量2。完整本地ZIP验证后才上传下一阶段，
下一ZIP下载与rsync上传并行。HF缓存续传、rsync partial、固定版本/50文件/949ID，
每操作首次后最多6次网络重试，断线重新连接，认证和校验冲突立即停止，数据保留。
远端专用JSONL接收器持有任务flock，staging完整SHA后hardlink排他发布。

执行者及独立6.1 Sol验收者运行19项小型离线测试退出0；主再运行19项退出0（0.234s）。
验收中发现rsync总时限截断、全量校验短时限、连续重连失败不能继续重试，均已修复并回归。
未把mock或计划输出当作实际传输完成。

冻结文件SHA：

- scripts/download_alp15_mac_l20_pipeline.py：eac42138492acc8340378eb35077cc34af3d25cf842a4d8c20dc62767c9a73dc
- scripts/alp15_remote_receiver.py：63d9a20bbe45480da1c5ce3dceb82b7ce74b233f89bc8c73cf6a3cd0d8f0026e
- tests/test_download_alp15_mac_l20_pipeline.py：3ca2651d0cc09bfd40339d1ece07c3731c79ab3a3f2808723a583ae6dd9d4f25

## 本机预检与连接身份

Mac27.2；既有独立Python3.12环境，HF1.31.0。
本机空闲约90 GB，system openrsync协议29、兼容rsync2.6.9。
小型HF whoami通过，仅在内存核验不输出身份；首个固定目标metadata可访问，未下载ZIP。
真实运行脚本仍将完整检查50个HEAD，不以本次首文件检查替代。
开始2222/2223均拒绝，后续2222恢复监听。

localhost:2222旧主机密钥与当前不同。只读比对发现当前ED25519密钥
与已保存的10.80.156.150 ED25519密钥精确一致。
故采用`HostKeyAlias=REDACTED_IP`及StrictHostKeyChecking=yes；
未删除known_hosts、未自动接受未知密钥、不降低验证。

## 终端启动及停止边界

按用户请求打开Terminal新窗口，启动手册中的SSH共享连接&&caffeinate流水线命令。
osascript返回`tab 1 of window id 6281`，退出0；命令不含密码或令牌。
用户请求代填登录密码，但Computer Use拒绝访问com.apple.Terminal（安全限制），
因此没有通过其他UI注入方式绕过限制；密码未保存到脚本、日志或命令。
必须由用户在已打开终端完成一次SSH密码输入，认证成功后&&链自动执行流水线。
该状态仅为终端命令已启动，不代表认证或真实下载/上传完成。

本轮没有解压、模型运行或数据清理。交付与启动命令见
ALP15-MAC-L20-PIPELINE-COMMANDS-20261013.md。
