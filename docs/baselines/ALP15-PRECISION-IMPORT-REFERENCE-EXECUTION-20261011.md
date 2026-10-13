# import 指纹引用角色窄域执行回报

- 日期：2026-10-11；执行者：GPT-6.1 Sol high 主任务；对应 `ALP15-PRECISION-USAGE-GAP-DECISION-20261010.md` §4.4。
- 本文和本次查询/证据由主任务独占；不改变旧 census、排除集合、模型或数据。这里只核实引用角色，不验被引用文件的内容，也不声称源码从未读取数据。

## 输入、命令与环境

服务器为已授权 `tuqiang@REDACTED_IP:2222` 端口转发。使用既有 SSH ControlPath 与 known_hosts 严格核验，不记录凭据。远端根 `/home/USER/lang/Lighten_VLA`，使用既有 `alpamayo1.5/.venv/bin/python -B`，仅 stdlib；`CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。未运行模型、安装依赖、访问被引用目标、读取 binary payload 或停止其它任务。

远端新输出目录为既有补充目录 `experiments/precision_usage_gap_20261010_v1`。原始输入 SHA：

|输入|SHA256|
|---|---|
|冻结 unresolved-assets.csv|f398e3f054620d37544217e766fbafba2d6d052f084df75e22df9f41259a09c7|
|冻结 scan-manifest.jsonl|2177978ecf1ab7b3efe8150fb6fcfa7aabe1d60b32b4354584c24aa28b06014b|
|hybrid_full20_v3/result.json|a832c8c845e859349367fecfda53d7d2d212ed9d941a3f21488177127adf68e6|
|实际 hybrid_m1_code_v3 producer|fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd|

逐项读取前后 SHA 相同。producer 的 `imported_source_receipt` 119–152 行扫描真实 `sys.modules` 的 `__file__` 并写文件指纹；嵌套 `source_receipt` 与结果两处赋值绑定 `actual_import_sha256_start`、`actual_import_sha256`。查询 AST 不执行源码。原 CSV 每条 `(source,location,reference_sha256)` 精确匹配同一 JSON 行的路径 key；保留重复出现位置，未按扩展名或路径前缀豁免。

实际远端命令（输出用 `set -C` 拒绝覆盖）：

```bash
cd /home/USER/lang/Lighten_VLA/experiments/precision_usage_gap_20261010_v1
set -C
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 /home/USER/lang/Lighten_VLA/alpamayo1.5/.venv/bin/python -B \
 check_import_reference_roles_v2.py \
 > import-reference-roles-query-v2.json 2> import-reference-roles-stderr-v2.txt
```

本地 current `evidence/usage_gap_20261011_v1/check_import_reference_roles.py` 对应远端 v2，SHA `4095863dfb095bc56fa0b262aabaf4d698b4a88a4cb98bb4f3df00ef3e0723a1`。query UTC `02:34:43.575897`–`02:34:43.934501`（北京时间10:34），SSH exit0（ca94f8）。原 stdout 共3,922,806字节，SHA `0eb4d496584f735366b348938ca9b9f1fffdae31578c82fefbe8f0904c23133d`；本地取回后重算一致（b57291），JSON完整解析。

## 实际结果与计数守恒

两个 import 字典共6271条记录，但 **不是6271条原external记录**：其中26条不属于原external类别。实际精确命中6245条（start2827、final3418），查询候选归类 `proven_non_usage` **仅针对“此引用用于代码指纹而非数据输入”这一角色**。不证明代码资产本身无历史数据曝光，不重哈希 import 目标，也不扩展为全部外部引用已闭合。

- 原external总数63177；本文件涉及6280条。
- 候选已分类6245；本文件其余35仍 pending；其它源56897仍 pending。
- `6245 + 35 = 6280`；`6245 + 56932 = 63177`。
- 匿名650、其它binary来源2860的上一轮余额不受此查询改变；全部4104 binary payload仍未验证。
- `coverage_complete=False`、`drive_disjoint_verified=False`、最终独立名单未冻结、新数据精度未通过。

原 CSV、manifest 和全历史排除集合未改；本查询未审定任何新增/释放chunk。分类须由独立 Sol 验收才进入累计已接受引用余额。

## 失败尝试与限制

首版误将6271个 import 字典条目等同于原external命中数，最后断言失败 exit1（5feaca/4c6b07）；没有输出可用分类。原远端首版源码、空stdout及stderr保留，本地另存 `check_import_reference_roles_v1.py`，SHA `19d96d860dc5a3f7bc60372810e37c98948895fe9a86d56e88e0a8e7a30f379a`。v2改为按CSV实际命中计数，同时预计算结果行避免重复拆分；没有变更输入、放宽精确行/hash匹配或修改原分类。

SCP开始后一次过早解析尚未传完的本地文件报 JSONDecodeError（c4070a）；这不是远端 JSON 损坏。随后全文 SHA 匹配、解析成功后才采用结果。完整 stdout/stderr 在上述新证据目录，原失败产物不删除。

本回报是窄域执行，不是全usage验收、完整独立精度、Orin实机、闭环或300ms达标结论。

### 小日志取回状态补充

完整 v2 stdout 已本地 SHA/JSON验证；但 SCP 在其后 stderr 小文件传输停滞。只终止了本次本地三个传输进程 PID40622/40748/40750（命令身份见2b7859，TERM见071962），没有终止远端任务。后续 SSH 文本取回也尚未完成。上文“完整 stdout/stderr 在证据目录”仅对远端原件成立：本地新出现的0字节 stderr 可能只是中断传输的占位文件，**不得当作已完整取回或真实远端空日志**。v1失败 traceback 此前已由远端只读命令实际显示（4c6b07），但尚无可用本地完整日志；独立验收须保持这个证据层级区别，不把传输占位文件作依据。

随后 SSH 文本取回终态 exit0（848cb5），返回两 stderr 和首版空stdout的完整UTF-8文本及远端 SHA。经 apply_patch 将首版 stderr 归档并重算本地 SHA；其应为 `229126293a2ecac8de9c03d4be3a1b3ece9ce0e36bc655effa83110651880ede`，与远端相符才可采用。v2 stderr 和首版空stdout均为真实0字节，SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`，分别归档为 `import-reference-roles-stderr-v2.txt`、`import-reference-roles-failed-stdout.txt`。前段为传输中状态，不作最终缺失结论；实际取回方式是SSH文本传输并经本地字节SHA验证，不冒称SCP完整成功。
