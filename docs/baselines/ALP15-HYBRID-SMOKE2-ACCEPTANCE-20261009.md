# AWQ M1 v3 真实 smoke2 独立验收

日期：2026-10-09，Asia/Shanghai。验收者与本文唯一写入者：`hybrid_v3_acceptance_20261009`，GPT-6.1 Sol high。对应决策：`D-ALP15-HYBRID-M1-20261008`。本轮仅本机只读源码、已传回 JSON/receipt，使用现有 Python 重算门禁；未连接服务器、未运行模型、未操作 GPU、未安装或下载，不使用 superpowers。

## 明确结论

**批准按现有决策进入同冻结 code_v3 的 fullprefill10 新 reference20→hybrid20 串行扩评。** 真实 smoke 六条请求均 completed/exit0；独立重算六条 `strict_pair_gate` 全部通过，NMSE、ADE差与最大坐标差逐条均为0；每条请求的252个唯一目标层均实际分派10次M1 fused。来源、保护身份、probe与reference成功终态的已传回记录及可本机重哈希的小文件链条一致，未发现阻断扩20的证据。

扩评维持原决策：新输出目录，fullprefill、Euler10、相同全部资产/代码/seed，先 reference20 完成并核验，再 hybrid20 与其配对。两者均需 `--approved-smoke` 指向本次 smoke2_v3，hybrid20另带本次 `--matrix-probe` 及新 `--paired-reference`。不得改变步数、启用actualKV/其它联合因素或复用不同源码的旧reference。任何来源重哈希或strict失败都停止该新候选并保留终态，不套用旧actualKV豁免。

本次只验收2clip smoke与252算子终态证据及扩评准入；不代表20clip质量完成、独立泛化或Orin NX16GB/300ms联合目标实现。

## 冻结与终态链

四源码实测仍匹配 v3准入文件 `ALP15-HYBRID-M1-V3-ACCEPTANCE-20261009.md`：hybrid `026eadf0bb1924e0356993e60172d732423e27264862aa751b792c3b6e791672`；probe `7a7e8d8b2118a5487e17d68596e011da4d1d0f6b6b936fa5e7467d2a2f09b197`；runner `fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd`；tests `d22b76035b887f04024e405afea2ade585f5027b41d1e0be3d3c966f8ef56dd8`。此前准入文件SHA `d1a9304233867b5068a3e4b5c5fed2bfd59ea94e1f35917a95baef8c7d8d54d3`。本次读取的运行 `code_sha256` 内本机可用源码逐文件SHA也均匹配。

本机证据均位于 `docs/baselines/evidence/jetson_300ms_20261009/`。

| 小文件 | 本机实际 SHA256 | 终态 |
| --- | --- | --- |
| hybrid_reference2_v3/result.json | `02a163ddbadf9aec6c778d36a35e2e7b76515d17b8a384a35e55a9ace75d2356` | completed_awq_hybrid_development_eval / exit0 / reference / 6条 |
| hybrid252_v3/result.json | `3a26316c87508541b69b96a6571def779ed7cdb088b325a02fa25341c4efe4f3` | passed_all252_cuda_m1_probe_with_actual / exit0 / 252条 |
| hybrid_smoke2_v3/result.json | `4aaa6d68fcfa0ef327542e797ef6dcb52a75dbdbd765d1128973069b747db6cb` | completed_awq_hybrid_development_eval / exit0 / hybrid_m1 / 6条 |
| hybrid_reference2_v3/reference_final_receipt.json | `583c5bbcdaecef095f5fefcceed592effff558db49aebbf3b797f08568a7d6df` | 绑定reference result与capture manifest |

smoke的paired-reference/gate-provenance SHA等于本机reference result真实SHA；matrix-probe/gate-provenance SHA等于本机probe result真实SHA。probe的final-receipt SHA等于本机receipt真实SHA。receipt、reference交付声明、probe绑定及smoke verified-activation-binding的manifest路径/记录SHA完全一致：`3975408023cff735c3fcd9edb76e190be654359805fcbab7f6ff15d6a3c4bf04`。所有result/receipt归属路径一致，reference与smoke均provenance_version=3，无v2补造来源。

本机只有传回的小结果/receipt，未复制服务器252张量、capture manifest、runtime快照及完整权重/依赖源。因此上述manifest SHA为多方一致的记录值，未在本机重新对其字节计算；同样不能本机重新读取所有服务器资产。冻结runner在smoke成功前已运行完整来源及capture绑定检查；20clip入口仍需在服务器再次运行生产 `validate_smoke` / `load_bound_capture` /逐文件SHA门禁，本验收不替代或跳过它们。

## 六条 strict 与实际分派

两个clip依次为：A=`06e20fbf-c4c7-4f47-8374-9a54f61f310b`；B=`1f62af5c-2991-47ba-a7c7-0a6f37d8bad3`。reference与smoke都唯一完整覆盖A/B×42/43/44，无重复、遗漏或失败请求。

| clip | seed | NMSE | ADE差 m | 最大xyz差 m | strict | 每层M1 / 全部M1 | hybrid E2E ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 42 | 0 | 0 | 0 | 通过 | 10 / 2520 | 3377.1702 |
| A | 43 | 0 | 0 | 0 | 通过 | 10 / 2520 | 3142.4594 |
| A | 44 | 0 | 0 | 0 | 通过 | 10 / 2520 | 2759.7520 |
| B | 42 | 0 | 0 | 0 | 通过 | 10 / 2520 | 2883.7814 |
| B | 43 | 0 | 0 | 0 | 通过 | 10 / 2520 | 2949.8825 |
| B | 44 | 0 | 0 | 0 | 通过 | 10 / 2520 | 2860.5935 |

重算使用当前冻结生产函数 `strict_pair_gate`，先检查双方64×3、finite、decision/step/clip/seed、noise/prefix哈希及合法token/future-start边界；然后比较noise、prefix SHA/长度、generated token IDs、future-start、请求身份、ADE与NMSE。六条重算字典与result保存门禁逐字段完全相等；不是只读取passed标志。

每条smoke `dispatch.module_count=252`，名字集合恰好为TARGET_NAMES，每层M1均10；全请求M1 fused总2520、M>1 reference总504、disabled总0，汇总值与逐层重新求和一致。六条original-helper门禁保存值也均passed，NMSE/ADE/xyz差均0；本机没有另存的original-helper轨迹，未重跑模型伪称再次验证官方sample。

## 来源、保护与252真实矩阵

reference与smoke的13条code、26条runtime、312条input库存和请求身份字段完全相等，全部20冻结clip的资产集合一致。实际import记录共有3408个相同路径，SHA全部一致；reference独有一份自身目录的remote-module持久快照，smoke额外24条由自身快照、实际AWQ kernel、Triton代码/编译扩展组成，符合不同进程和候选独有导入。没有公共路径hash漂移。

两进程动态身份 SHA 同为 `d567f0ea3e3200e3305c7357c8a25254bb79384409cdc9112293838f7ac40513`，依据metadata重新计算相符；原字节SHA、生成器/模板冻结SHA及库存绑定相符。virtual两项映射完全相同，均绑定真实实现路径/SHA记录。probe activation-provenance中的code/runtime/input/actual-import、动态及virtual映射、快照目录逐字段等于reference；其input库存也完整包含reference来源库存，未改来源门禁。

reference和smoke各自before/after及两variant之间的保护摘要完全一致：`94b2256c7e277915c3b570c6ba941aaf49b56dfa4094b2adac06b70a394093bb`；probe保护声明亦相同。两variant安装前后persistent storage记录均 `10,272,486,512 B`。这是storage库存，不代表Orin全系统内存峰值或运行预算。

独立遍历probe252条：名字唯一完整，逐矩阵SHA与reference/probe input库存中对应packed文件相符；每层均含fixed_random_bf16和actual_decode_bf16，各实际输入dtype=BF16、元素数=该矩阵输入宽度（M1），capture文件库存252。504个输入数值报告均finite、outside-tolerance-count=0，门限rtol=.01/atol=.02；actual187/252、random166/252输出逐元素相等，两组最大绝对误差均0.00390625。全部252 dequant逐元素相等且最大差0，disabled/M2逐元素相等、persistent bytes不增加。算子允许容差与六条完整轨迹恰好相等分别记录，不以算子容差推断驾驶精度。

## 计时与质量边界

两clip同seed的meanADE均1.2327414155m，配对teacher meanADE1.3351951838m，开发子集相对损失记录-7.6733%；这不是独立clip泛化或全20结论。

共享NVIDIA L20记录：reference完整请求mean5503.2893ms；hybrid mean2995.6065ms、max3377.1702ms，6/6请求>300ms；首请求3377.1702ms，后五条mean2919.2937ms/max3142.4594ms。均值比值显示约45.57%减少，仅属于这两个独立进程的共享GPU诊断。环境记录有其它python PID453201常驻15428MiB，前后snapshot不能证明中途没有其它负载。完整请求计时仍排除磁盘I/O、冷加载与审核，不能当目标部署全部开销或无干扰因果测量。

环境记录为L20、Torch2.8.0+cu128、CUDA12.8、Python3.12.14、Triton3.4.0，非Orin；result也明确Orin_gate_passed=False。本次批准扩20用于扩大同代码质量/计算诊断，最终Orin NX16GB、全系统统一内存、完整请求max≤300ms、独立teacher损失≤15%仍须联合实机验收。

## 本轮执行留痕

本机cwd=`/Users/USER/Documents/ChatGPT/Lighten_VLA`，现有 `/usr/bin/python3`；只有JSON/浮点运算和源码函数导入，无模型构建或前向。

- `shasum -a 256` 四源码及v3准入文件、只读当前strict/numerical-gate源码：exit0，工具输出 `ebd3d7`。
- `/usr/bin/python3 -c` 读取三result，生产 `assert_same_provenance` / `assert_dynamic_sources_equal` /逐六条 `strict_pair_gate` /252逐层dispatch断言：exit0，输出 `85244b`。
- `/usr/bin/python3 -c` 逐小文件真实SHA、receipt链、3408公共import、probe来源及252全部数值/反量化/storage断言，保存与重算gate比较：exit0，输出 `3c7eec`。
- `PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -c` 本机可用源码SHA、packed逐矩阵SHA与252实际BF16/M1/文件库存复核：exit0，输出 `498d43`。

唯一新增文件为本文，其它源码/JSON/既有验收只读；本文SHA由写完后另报。
