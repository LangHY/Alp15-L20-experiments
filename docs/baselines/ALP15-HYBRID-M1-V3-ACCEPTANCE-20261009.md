# AWQ M1 v3 动态来源生命周期独立验收

日期：2026-10-09，Asia/Shanghai。决策编号：`D-ALP15-HYBRID-M1-20261008`。验收者及本文唯一写入者：`hybrid_v3_acceptance_20261009`，GPT-6.1 Sol high。验收按决策、当前四个源码文件及执行回报冻结 SHA 进行，不是 Git 分支 code-review。其它文件只读；未连接服务器、未启动 GPU、未加载真实模型或 packed 权重、未安装或下载。

## 结论与进入顺序

**当前 v3 没有发现阻止启动新 `reference2_v3 + decode capture` 的确定代码阻断。** 在主任务完成现场来源/资产/依赖核验与 GPU 串行调度之后，可以启动冻结 code_v3 的新 reference。其 exit0、六条完整请求、252/252 capture、有效独立 final receipt 和持久动态源码快照全部成立，才进入新 `hybrid252_v3`；252 个真实权重的 random + actual 输入 CUDA 桥接全部通过，才进入同代码 `hybrid_smoke2_v3`。smoke 六条 strict 门禁通过后，才可按决策扩20。

本结论是实施准入与 CPU 来源生命周期验收，不是 CUDA、完整轨迹、L20 性能或 Orin 实机验收。Orin NX16GB、每条完整请求 max≤300ms、独立同任务 teacher meanADE 损失≤15% 三项联合目标仍待验证。直接用户后续优化授权以主任务及冻结决策的记录为准；不将历史门禁豁免记作原 NVIDIA 官方入口验收成功，不使用 superpowers。

## 冻结交付与证据身份

五文件实测 SHA 与主任务移交值完全一致；决策 SHA 仍为 `5f567658ef404c2687d629d9051f6595b88651424f14a6c3c2e74a5cdbff3afb`。

| 文件 | SHA256 |
| --- | --- |
| scripts/alp15_awq_hybrid.py | `026eadf0bb1924e0356993e60172d732423e27264862aa751b792c3b6e791672` |
| scripts/probe_alp15_awq_hybrid.py | `7a7e8d8b2118a5487e17d68596e011da4d1d0f6b6b936fa5e7467d2a2f09b197` |
| scripts/run_alp15_awq_hybrid_eval.py | `fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd` |
| tests/test_alp15_awq_hybrid.py | `d22b76035b887f04024e405afea2ade585f5027b41d1e0be3d3c966f8ef56dd8` |
| docs/baselines/ALP15-HYBRID-M1-EXECUTION-20261008.md | `337a23a66f3092ee69933836b04294c4b159a79e862d17709e52f4aa370b1465` |

旧只读文件实测仍为：kernel `3f09aea862990733b9bb1db536c450c79fd6e2ac93c7015eed6b1eef4f191025`；full_packed `150e1310ad3dbcb9f82966146f9627855e87bd12a29f45f410d0b94f1a96de0a`；packed_w4a16 `241edf75d838ad7cba1250374119237864780b95ec81cd8555a7d7c6d15fab12`；KV helper `74886a867a8aae480dd848a6d068d2e3f817674d891774bcdabeb57f412292f8`。本验收未修改它们。

只读核验的 v2 留痕位于 `docs/baselines/evidence/jetson_300ms_20261009/`：

- `hybrid_reference2_v2/result.json` SHA `1b60bab73f88865e873f4bde777c3298f04a47286c6de5532a65df780b5770b8`，completed/exit0，2clip×3seed 共六条完成请求，均64×3；交付声明252/252 capture，独立 receipt 保存上述 result SHA。
- `hybrid252_v2/result.json` SHA `67c3ec6f76c46c7501b9b77fbca4ebef747c450b7451d6bd7bca2cfd3f3ed2d9`，failed/exit1，matrices=0，失败发生在读取 reference import 库存里的 `/tmp/tmpu5d2gwvq/_remote_module_non_scriptable.py`，尚未进入矩阵 CUDA 测试。
- `hybrid252_v2/probe252.log` 与 traceback 均记录同一 FileNotFoundError。v2 reference 的实际 import 记录动态源码 SHA 为 `8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`。本机没有其原临时源码或252 capture tensor，不能据此独立重验服务器文件内容；这些是已传回的历史终态证据。

v3 不为旧 v2 补造源码、改 schema 或换 receipt；必须建立新 code/output 目录与新 reference。仓库现有 `scripts/run_alp15_hybrid_reference2_20261009.sh` 仍指向 code_v2/reference2_v2，属于历史 launcher，主任务启动 v3 时需明确使用新 v3 路径，不能直接复用此旧入口。

## 动态源码与 virtual namespace 复核

`snapshot_torch_remote_module` 只处理名字精确为 `_remote_module_non_scriptable`、Torch 版本核心为2.8.0的实例。生成进程读取实际 `module.__file__` 原字节，要求 module/spec 名称与 origin 一致，源码位于实际 import 的 instantiator 临时目录；生成器、模板和实例字节分别必须匹配冻结 SHA。源码不存在即失败，不调用生成器补造。快照排他创建，已有损坏快照拒绝覆盖。

持久元数据记录原 module/spec 地址、生成器临时目录、快照路径、生成模式、实际 Torch 版本、实际生成器/模板路径与 SHA。稳定身份绑定 module 名称、原字节 SHA、Torch 版本、生成入口和生成器/模板 SHA，排除两个进程各自随机的临时目录。生成模式没有独立进入稳定身份散列，但实例原字节 SHA 已固定为此 non-scriptable 模板的真实产出，本轮不存在以别的生成模式产出不同代码后通过的路径。

消费者核验非空 actual-import 库存及快照本身、快照目录/文件名/完整映射、稳定身份、生成器/模板真实文件与库存 SHA；只有这个已核实实例的原临时地址允许在生产者退出后消失。生产者最终审核 `require_live=True` 仍要求原文件存在且 SHA 正确。任何普通临时文件、未知动态模块和其它缺失来源都按原文件严格核验，没有 `/tmp/*`、包名通配或 missing-file 跳过。

本机实际 Torch2.8 源码证明 instantiator 使用 `TemporaryDirectory`、注册 `atexit.cleanup`，格式化模板再 import；实测生成器 SHA 为 `567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615`，模板为 `0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc`，均匹配冻结值。实际 CPU 子进程退出清理原文件后，v3 持久 receipt 仍可完整校验；两个进程不同临时地址获得相同稳定身份。

`torch.ops`/`torch.classes` 的处理限制为两个精确名字、`torch._ops._Ops`/`torch._classes._Classes` 类模块名和类名、精确文件标记、无 module spec；随后绑定实际 owner 实现文件绝对路径与真实 SHA，写入 actual-import 库存和独立 virtual 映射。消费者校验映射字段与实际实现文件库存；reference/hybrid 要求 virtual 映射完全一致。对于本轮可信 Torch 环境的来源审计，这足以排除将虚拟文件标记当普通磁盘路径产生的假缺失，并保留实现字节漂移门禁。它不是抵抗任意进程内恶意 monkeypatch 的安全签名机制；本决策没有这种威胁模型。两个明确 virtual 身份不是通配 namespace 豁免。

## 成功终态、真实252与严格轨迹门禁

v3 manifest 使用 `alp15-m1-activations-v3`，v2 schema 明确拒绝。reference 审核后才写 manifest，再保存 completed result，最后写独立 receipt；manifest 保存 result/receipt 路径、result 保存 manifest SHA/receipt 路径、receipt 保存 result 和 manifest SHA，没有哈希循环。最后复核失败撤销 completed，result 改 failed/exit1，旧 receipt 的 result SHA 同时失效。

`load_bound_capture` 保留成功 reference/exit0/决策/变体、归属路径、实际交付 manifest/receipt/252覆盖、四种非空 code/runtime/input/actual-import 库存逐文件及声明一致、动态/virtual映射一致、保护摘要前后相同、capture clip/text seed、完整六条轨迹与252唯一名称/文件/SHA门禁。正式 probe 显式绑定 paired reference，要求252真实权重和实际 BF16 M1 activation；random-only状态不能晋升正式候选。probe 来源和 capture/result/receipt SHA绑定在 hybrid 准入时再次核验，扩20允许复用相同完整资产和保护身份的2clip算子探针，仍需20clip的独立严格轨迹对照。

reference/hybrid 的公共 code/runtime/input、请求路由、10steps、text/trajectory seeds 完全比较，候选独有 Triton import 留痕；最终新增动态稳定身份相等和 virtual 实现身份相等核验。本次没有弱化 v2 原有来源门禁。所有保护模块仍前后实测，installer 仍要求恰好252目标并完整预检后安装；M>1与禁用路径仍走同一原 reference，只有实际M1建立共享 fused。

`validate_trajectory_row` 仍要求64×3、坐标/ADE/FDE有限、decision/10step/clip/seed与合法noise/prefix/token/future-start边界；strict gate 比较noise、完整prefix/长度、生成token IDs、future-start、请求身份、ADE差≤1e-4m、NMSE≤1e-3。`validate_smoke` 读取绑定终态后重新逐条计算 strict，不只信保存的 passed 标志。strict失败仍exit2，不允许套用actualKV历史豁免；普通来源/数值失败exit1并保留结果。

## 独立验证留痕

工作目录 `/Users/USER/Documents/ChatGPT/Lighten_VLA`；现有 `/usr/bin/python3` 为Python3.9.6，macOS-27.2-arm64，Torch2.8.0，CUDA available=False。输入仅合成 CPU tensor、小型 Torch 模板生成和 JSON/receipt fixture，无真实模型、无RPC初始化。

| 验证 | 命令或范围 | 结果及日志 |
| --- | --- | --- |
| 四源码、回报和决策冻结 | `shasum -a 256` 指定六文件 | exit0，全部等于移交/决策值；工具输出 `afbaf6` |
| 独立完整CPU回归 | `/usr/bin/python3 -m unittest discover -s tests -p test_alp15_awq_hybrid.py -v` | exit0，39项通过，10.874s；完整输出 `7c4222` |
| v2终态及旧helper SHA | 本机 JSON 解析、六条64×3/receipt/traceback检查、四helper `shasum -a 256` | exit0；工具输出 `0ee88d` |
| 实际Torch生成器源码 | 读取已安装模块源码与SHA，检查临时目录、atexit和模板实例化 | exit0；工具输出 `c68d75` |
| 追加独立virtual/动态负例 | `/usr/bin/python3 -c` 内存fixture，调用生产 `imported_source_receipt`、`verify_import_provenance`、`snapshot_torch_remote_module` | exit0；两个实际virtual正例通过；五个负例均按预期拒绝，工具输出 `03536a` |

五个追加负例是：普通 module 冒充 torch.ops 类身份、virtual 实现SHA声明错误、普通来源原文件缺失、remote module/spec名称不符、已有快照损坏。最后一例还检查损坏快照没有被覆盖。本测试独立于提交的39项测试文件，只在临时fixture目录写入合成产物。

追加fixture第一次通过 stdin 执行时，`__main__.__file__=<stdin>` 被普通来源门禁严格拒绝（FileNotFoundError，exit1，`29c79e`）。这是测试调用方式产生的不存在来源，随后改用 `python -c` 的无文件 main 方式；没有修改生产代码、过滤 stdin 或跳过普通来源检查。

## 现场待验与停止条件

主任务需核验服务器 code_v3 五文件、四旧helper、真实 import 路径、既有离线root/config/overlay/完整20份cache与Torch模板库存，再以新目录串行运行；禁止修改运行中的源码目录。保存 reference/probe/hybrid 完整日志、receipt、全部快照、环境及输入哈希；`--preflight`不等于真实loader/config/model可用性验收。

任何现场生成器/模板SHA或版本不符、未覆盖252、unsupported activation、NaN/OOM、保护或其它来源漂移、strict失败，均保留失败并停止该新候选，升级主任务，不调整库存、补造旧来源或自行替换环境。CUDA数值/逐元素BF16 dequant、每请求252实际分派、完整轨迹和性能仍需实际终态，Orin统一内存与300ms/独立质量必须实机联合验收。

唯一新增工作文件为本文；四源码、执行回报与历史证据只读。本文自身SHA由写完后的主任务另行登记，不写自引用hash。
