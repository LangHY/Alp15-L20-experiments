# 504 个 decode activation 的来源链核查

对应决策 D-ALP15-PRECISION-USAGE-GAP-20261010；执行者主 Sol，文件所有者主任务。范围仅两组 activation 的历史数据曝光归因，不验 payload、不运行模型、不替代独立精度或 Jetson 门禁。

## 实际执行

服务器 `/home/USER/lang/Lighten_VLA`，Python3.12.14、Linux6.8.0-87-generic x86_64/glibc2.35，stdlib CPU。环境 `CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。新建拒覆盖目录 `experiments/precision_usage_gap_20261010_v1`；未改冻结 census、旧实验或模型。

实际命令（stdout/stderr 用 noclobber 创建）：

```sh
set -o noclobber
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 /home/USER/lang/Lighten_VLA/alpamayo1.5/.venv/bin/python -B \
 /home/USER/lang/Lighten_VLA/experiments/precision_usage_gap_20261010_v1/check_activation_lineage_v3.py \
 > /home/USER/lang/Lighten_VLA/experiments/precision_usage_gap_20261010_v1/activation-lineage-query-v3.json \
 2> /home/USER/lang/Lighten_VLA/experiments/precision_usage_gap_20261010_v1/activation-lineage-stderr-v3.txt
```

执行退出0（0a2564），开始/结束 UTC 2026-10-10T16:06:39.461565/16:06:40.025864，即北京时间2026-10-11 00:06:39–00:06:40。查询 SHA `67dbb977404eaf3cb9325d0839f3d1660b96b9074e61864a422e778bccf58670`。所有读入文本前后 SHA 一致；四冻结入口 SHA 对照通过。PT仅 lstat，未打开或哈希内容。

本地原始证据目录 `docs/baselines/evidence/usage_gap_20261011_v1/`：完整 v3 JSON、空 stderr、两个冻结 producer 文本和当前查询源码。JSON SHA `0b25714cf65d718781cf9b7c1d42b39bd0e478847ca5aa4ecb85738f9c687e6c`；两组252项 stat 清单完整保存且本地重算其列表哈希通过（935d73）。SCP取回退出0（40d9b7/a5179c/2717e5），源码重算 SHA（80473e）与原实验声明一致。

## 结果及边界

v2 producer SHA `a2ccec42a954be1d8325c89afbc32337193808265121448ec770ec92a792e887`，v3 `fe92978f42488716227563a654d94b483a976a47301ae60908b5efe5b93732fd`。核对各自 result、final receipt、manifest 文本哈希、相互引用路径、dataset revision及code/runtime/input/import哈希字段，全部一致。

每组252个唯一 manifest basename，与实际 PT 集合完全相等，均普通非symlink文件；504个不同路径在原CSV的anonymous/binary两类中分别恰出现一次。各类计数守恒，不把交集相加成新样本量。

两版实际源分别在 v2 421–423/v3 452–454 以当前 clip 载入 data；在 v2 492–493/v3 523–524 仅最后一个 clip 捕获；捕获函数和 manifest 写入函数明确同一 capture 对象的 clip 与文件条目关联。两组均为 `1f62af5c-2991-47ba-a7c7-0a6f37d8bad3`，已在 used_clip_ids。本轮初步归因为**已用 clip 的派生 activation**，不是504个独立clip。当前 query 使用 `candidate_attributed_used`，待独立 Sol 验收；不改原ledger标记。

anonymous记录504已形成候选来源链，剩1388待证据；binary4104条中同504条仅数据来源范围有证据，**全部4104条仍未验payload**。其它3600条来源待分类。external63177、revision171/247、symlink20本次未处置；coverage_complete=False、drive_disjoint_verified=False，最终名单未冻结，没有新精度pass或性能指标。

## 失败尝试与恢复

最初一次查询错误地写了 v3 snapshot 日期20261008，FileNotFoundError退出1（4006af/df5ee3）；没有继续执行或改旧数据。读取该实验 result 的 code_sha256 实际路径，确认为20261009（4880e4）。原远端查询及失败stdout/stderr保留；失败stderr已取回。本地当前查询为修正版，不冒称它是失败版本。修正另存远端 v2 查询，退出0（bdded6）；v3另存版本仅增加两类每路径恰一次和完整stat列表，退出0。

后续取回时旧SSH master过期，BatchMode认证255（3a4fac/3599d7/c04916）；没有重启查询。沿原已信任端点重新认证后只取回已完成的小文本；未改变known_hosts或服务器认证设置。

独立验收尚未签发；下一步先复核本窄域来源链，再核其它 teacher/prefix cache 的同实验生产版本和输入名单。不得依据本回执批准最终独立抽样或GPU实验。

## 追加：expanded20 的40个 prefix 缓存

北京时间2026-10-11 10:08:28执行同决策范围的 `check_prefix_lineage.py`（SHA `1f47739a39510dda165634691c768f52743bffafdde26f5f7dda81d7d1100fc6`），同一远端证据目录、新文件拒覆盖，沿用上面的解释器和offline/空CUDA环境。命令只将脚本名替换为 `check_prefix_lineage.py`，stdout为 `prefix-lineage-query.json`，stderr为 `prefix-lineage-stderr.txt`。退出0（ebb0ce）；UTC02:08:28.412896–02:08:28.884638。

原始JSON已取回本地同证据目录，30306B，SHA `106f600ea0722480ab11808fed773dcd499664c9078931fa8b8848f0e6e421ae`，stderr为空。含完整实际producer源码、所有文本SHA、40条path/role/index/clip/stat映射，当前query SHA已本地核对（68080f）。源码 `controlled_8b_diagnostic.py` SHA `92e858ca556deab9edb26b8f98231d332bdae1376c4907b52853baba27623313` 精确匹配该实验 result。实际argv明确 expanded-validation分支、无resume；其原reference split与实验复制split bytes SHA相同，且原reference SHA被result.inputs记录。不能把另一个同名split的顺序套用。

已读完整producer文本（efc851）：此分支枚举split.expanded_validation的20个clip，生成teacher `validation_{index:02d}.pt`；student从同teacher path/example生成，保存同basename，并继承example.clip_id。各目录恰20个文件、非symlink；按生产者的明确规则保留候选逐项映射。result 240条记录的clip集合与原有序名单集合一致，20clip都在冻结ledger.used_clip_ids，dataset revision一致。40文件在原CSV匿名/binary两类各恰一次；所有读取文本前后SHA一致，没有读取PT。

本次初步累计544个不同文件具有候选曝光归因（504 activation+40 prefix），匿名来源剩1348、其它binary来源剩3560；**payload未验证仍4104条**。未改原CSV或coverage。20clip原已排除，新增排除块为空。独立验收仍待签发。

定位时先查询了一个不存在的expanded20日期目录，查询退出2（bc2e05）并仅返回实际匹配目录；随后依据实际inventory目录 `controlled8_expanded20_20261002_125000` 读取，未生成任何旧实验文件。此定位失败不冒称实验失败或数据变更。

追加取回两组原activation manifest文本（合计约1.59MB，cb85d2/b659f1），本地SHA与原receipt完全一致（68080f），交独立复核逐entry↔stat。原11MB unresolved CSV仍留服务器，独立验收须区分远端完整Counter执行证据和本地独立复算，不能声称已本地逐行验原CSV。
