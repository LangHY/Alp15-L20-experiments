# 留出集准备 v3：L20 CPU 执行回报

主任务唯一写入。对应决策 v1/v2/v3；v3 SHA e53284d0ebcba64787b4f6ca4f5d60bbef8e779f62f29d7d1845b1910c1731f8。此记录不包含新模型评测结论。

## 环境与部署

指定 tuqiang L20 主机，项目 /home/USER/lang/Lighten_VLA，解释器 alpamayo1.5/.venv/bin/python。实际 pandas2.3.3、pyarrow22.0.0、physical-ai-av0.2.0（1c2480）。本轮只执行 CPU，CUDA_VISIBLE_DEVICES 为空，HF 两项 offline flag 均1。不初始化GPU、不读取模型/PT/ZIP payload、不安装、不下载大文件、不停止其它进程。OS/CUDA/驱动本轮不刷新，不从历史报告冒充当前测量。

新建拒覆盖快照 experiments/precision_holdout_code_20261010_resume_v3（6781b3），SCP 只上传 script/test 和三个决策小文本，传输退出0（6581，f2c3e3/4eb0f0）。远端实际 sha256sum 五文件逐一匹配（2865bf）：

- script：ccebd4d237cf5229a422844579e5871405291e0f01f957f5600fda8148f04b9c。
- test：c102a6f54abe950a7a58acc94aafe459d3b0126aa87b6e419429c890243ed5a5。
- 三决策：3efbfea76b5b336d5199c93337863bff49c120f2b3bd92c9919d6f26e979297c、32b534ea574677928fca9c6e9277498807f00839fe8de852f14f01d7a9e7a23b、e53284d0ebcba64787b4f6ca4f5d60bbef8e779f62f29d7d1845b1910c1731f8。

原远端项目代码、依赖、数据未覆盖。小文本传输不是用户限制的大资产下载。

## 实际测试

快照目录下，以原 venv 和 offline/空CUDA flags 执行 `python -B -m unittest discover -s tests -p test_prepare_alp15_precision_holdout.py -v`，28tests/0.886s/exit0（session8178，2865bf/ab42ec）。完整输出留在上述工具记录。两个文件 `python -m py_compile` exit0（eac325），PYTHONPYCACHEPREFIX=/tmp/alp15-holdout-v3-compile-20261010。

主任务本机同SHA回归28tests/0.911s/exit0（session49608，1c8dfa/ce2de5）。本机测试完整原始文件由执行者在 evidence/holdout_prep_local_20261010/ 归档，见本地执行回报。合成测试成功不代替实际census。

## 实际 census 命令与产物边界

执行命令（0f9eb6），远端 shell 启用 pipefail；完整 stdout/stderr tee 到快照内 census.log（快照目录显式从usage排除，避免日志自引用）。模型/数据原始输入没有写入。

```sh
set -o pipefail
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
/home/USER/lang/Lighten_VLA/alpamayo1.5/.venv/bin/python -B \
  /home/USER/lang/Lighten_VLA/experiments/precision_holdout_code_20261010_resume_v3/scripts/prepare_alp15_precision_holdout.py \
  --root /home/USER/lang/Lighten_VLA \
  --metadata-root /home/USER/lang/Lighten_VLA/datasets/physicalai_chunk0000_20261002_165238/files \
  --usage-root /home/USER/lang/Lighten_VLA/experiments \
  --usage-root /home/USER/lang/Lighten_VLA/datasets \
  --usage-root /home/USER/lang/Lighten_VLA/docs/baselines \
  --usage-root /home/USER/lang/Lighten_VLA/scripts \
  --usage-root /home/USER/lang/Lighten_VLA/compression \
  --usage-root /home/USER/lang/Lighten_VLA/logs \
  --usage-root /home/USER/lang/Lighten_VLA/incoming \
  --usage-root /home/USER/lang/Lighten_VLA/alp15-test-clip-delivery-20260917 \
  --usage-root /home/USER/lang/Lighten_VLA/alp15_l20_audit_20260920_094859 \
  --usage-root /home/USER/lang/Lighten_VLA/Self-realization \
  --output /home/USER/lang/Lighten_VLA/experiments/precision_holdout_prep_20261010_resume_v3 \
  2>&1 | tee /home/USER/lang/Lighten_VLA/experiments/precision_holdout_code_20261010_resume_v3/census.log
```

状态：执行中，session53684 最近多次 wait 返回 live，无退出码（bc6dda/4c66d7/c8b8a6）。不得因无新输出重启或把观察等待当完成。上面是实际命令记录，不是让用户再次执行；相同 output 已存在会拒覆盖，旧日志也需保留。

正式输出目录只存本次新产物，默认无抽样草案。等待终态后检查完整argv/环境/输入前后SHA、usage引用与整chunk排除、eligible计数、缺失统计、扫描稳定、未解决资产和产物SHA。coverage_complete=False、drive_disjoint_verified=False；即便退出0，也需独立范围审定，不能声称独立精度、输入窗口QC或行程不重叠已通过。

## 下一步

只跟随当前句柄到终态，再读取小报告和manifest进行复核。原始模型推理、最终名单与下载均未批准；统计草案独立审查 not_ready 的五项修订与样本量/预算仍待解决。

## 已确认终态与主任务复核

随后同句柄53684返回 `completed_usage_census_pending_coverage_review`、exit0（d4cfbc）。原Python PID71709消失、产物十文件存在（6028e8），没有重新启动。实际UTC开始2026-10-10T15:47:18.150641、结束15:49:47.818446，约149.7秒；对应Asia/Shanghai当日23:47至23:49。

主任务实际读取manifest/ledger/counts、重算所有九项产物SHA（7c4cea）：全部匹配，status/exit0，无诊断，scan_stable、declared_roots_inventory_complete、all_declared_text_parsed均True。真实环境Python3.12.14、Linux6.8.0-87-generic x86_64/glibc2.35，三个offline/空CUDA控制均True，包/source身份与冻结值一致。

注意最初汇总查询对CSV before/after整个字典做相等比较，输出 inputs_unchanged=False；这并非数据变动。before另含git_blob，after只有bytes/SHA，结构不同。主任务按源码门禁进一步核验（b34079/5689f9）：三个metadata前后字典完全相同；CSV7076bytes及SHA61105260827f5cf193c6c3ecbfed13b185432d04899d52bff3a7430ab4c0f304前后相同，source内容未变。原正式manifest和工具观测均不篡改，记录这个汇总查询口径纠正。

|层级|clip|chunk|train / val / test|radar缺失|
|---|---:|---:|---|---:|
|全部 / 合格|306152|3146|153625 / 90928 / 61599|145391|
|仅排除0/3119|305952|3144|153425 / 90928 / 61599|145373|
|追加全部已引用块排除|304075|3125|152534 / 90240 / 61301|144565|
|剩余test候选池|61301|633|0 / 0 / 61301|29771|

实际映射219个引用clip，保守排除21个chunk：0、214、224、276、317、420、727、728、968、982、1519、1657、1984、2277、2368、2372、2447、2599、2634、2868、3119。原仅排除两块不能替代本次结果，剩余61599 test旧计数不可继续当新余额。

清点1182JSON、124JSONL、369CSV、283LOG等。未解决项：1892匿名cache，63177外部路径引用，171dataset revision冲突疑点，247其它revision，4104未验证二进制，20未跟随symlink。缓存来源混合 teacher/prefix cache、decode activations、adapters/checkpoints和scores；不能把1892都当1892条未知clip，也不能只凭名字把它们全部豁免。下一步须按同一冻结实验的显式索引与源码映射分类，不反序列化payload、不猜数组顺序。其它扩展未按声明文本契约读取（如text、ipynb等）亦需范围审查，all_declared_text_parsed不是所有可能文本/未知历史完整覆盖。

只取回四个小证据文件（约0.54MB，SCP退出0，80518/d02847/a0b15f）至 evidence/holdout_prep_remote_20261010_v3/，没有取回45MB候选表、46MBledger或任何ZIP/权重：

- run-manifest.json：282749bytes，SHA022ae526670625ff44c1d69b7507bb1925e0a91cedd25442573cb9ccd29c3ff2。
- metadata-counts.json：248726bytes，SHA64ac2b3035a3be42830c6b769a75f900a099a35dc03985443a22945041c65484。
- report.md：3098bytes，SHA429b279db6ce66c01e5c3a50fe12ddab6b4ff2ebe37a31a71726121c0de71e30。
- census.log：SHAbeeee2467ee85f18833625d5b85e17aeda18b24dbf35bf79456e2c827259f7b4，与服务器实际SHA匹配。

本地工具v3独立验收文档已全文读取，SHA a402127b6acb118a5c3ae943134947f7a2dd14a538bcf7f662c7946bbde671cf。它只验收工具，本段是主任务远端终态复核，尚没有完整独立usage覆盖receipt。

当前准确结论：真实CPU census完成，来源/产物/扫描稳定性主任务复核通过；coverage_complete=False、drive_disjoint_verified=False、最终独立名单未冻结。未知资产分类、范围独立复核、QC、统计草案修订/样本量/报价及新教师/候选采集仍需推进，目标未完成。
