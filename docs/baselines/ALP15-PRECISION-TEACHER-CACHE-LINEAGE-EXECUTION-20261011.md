# 七组新数据 teacher cache 的有限来源范围核查

对应决策 D-ALP15-PRECISION-USAGE-GAP-20261010；执行和本文唯一所有者为主 Sol。当前结论：700个teacher文件形成同一100clip的**候选有限曝光集合**，待独立验收；不声称每个复制后PT内部clip正确，不验payload，不涵盖adapter/checkpoint、训练输入的所有祖先或完整历史usage。

## 执行身份与证据

沿已授权tuqiang L20 SSH，服务器项目 `/home/USER/lang/Lighten_VLA`，已有Python3.12.14、stdlib CPU。命令：

```sh
set -o noclobber
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 /home/USER/lang/Lighten_VLA/alpamayo1.5/.venv/bin/python -B \
 /home/USER/lang/Lighten_VLA/experiments/precision_usage_gap_20261010_v1/check_newdata_teacher_lineage_v2.py \
 > /home/USER/lang/Lighten_VLA/experiments/precision_usage_gap_20261010_v1/newdata-teacher-lineage-query-v2.json \
 2> /home/USER/lang/Lighten_VLA/experiments/precision_usage_gap_20261010_v1/newdata-teacher-lineage-stderr-v2.txt
```

UTC 2026-10-11T02:20:31.961982–02:20:32.456167，即北京时间10:20:31–10:20:32。实际退出0（05a5e3），stderr空；未启动GPU，未导入模型或读取PT。所有31个文本前后SHA一致，700个文件末次lstat大小/mtime未变化。原census、CSV和旧实验未修改。

本地目录 `docs/baselines/evidence/usage_gap_20261011_v1/`：`check_newdata_teacher_lineage.py` SHA `d291a16e9dda34fbb1c56567746f04de8a0038d96c3a9f5277c6fef47d3e7b7c`，与远端v2执行源码精确一致。完整JSON `newdata-teacher-lineage-query-v2.json` SHA `e062044eab401ab989c6183e505e69d00f663c0996e534ca7d9a8cdc730fbab9`，含七组完整实际producer源码、原有序role clip列表、700条stat、六条donor组边、1398条原CSV受限记录、31个输入文本SHA。SCP取回退出0（f4e556）；本地query SHA复核（c89b36）。原大型CSV/ledger保留远端，仅读取固定SHA文本。

v1查询也退出0（1960e1，UTC02:19:37.871695–02:19:38.356070）；原JSON/源码/空stderr远端保留，本地保留其JSON和空stderr。v2新增donor source_commit/checkpoint_revision/dataset_revision/logit_weight/expert_weight字段一致性、rollout_prefix=True和全部PT末次stat检查，没有改变分类或旧产物。本地当前查询是v2，不冒称它是v1源码。

## 同实验输入与生产者链

七实验：measured6p8_190306、measured7_181113、measured7_182132、newdata8_165238、precision8_balanced_195311、precision8_dense_195955、rebalanced7_210458（完整路径在原JSON）。各自actual code/run_interface_recovery.py SHA均与本次provenance.code_sha256同名字段一致；六个不同source SHA在原JSON中逐组冻结，不以当前同名源码替代。

实际argv的 `--new-training-audit` 原result SHA、复制 `new_data_audit.json` SHA、provenance.new_data_audit_sha256、optimization_split.data_audit_sha256全相等：`472a5d107e3876ce0e81727519c0c67b7c1e6f44fdb6aae90814ce32ecbda349`。该文本同时有records100、training_records80、monitor_records20；先前浅层schema摘要没有展示后两数组，不能因此误判输入名单缺失。没有training-window-audit/resume分支；各组校验80/20唯一、互斥，split角色集合对应，100clip均在冻结ledger.used_clip_ids，dataset revision相同。

主任务静态阅读全部版本相关分支（23873d/757103）：原audit复制→new_data角色覆盖records→enumerate(records)→写入或从donor按(role,record.clip_id)选取并复制到role_index.pt。初始newdata8是直接prepare分支，另外六组用donor；donor仅指这七组中的newdata8或measured7_182132，已绑定同SHA audit/split角色集合与provenance字段。每份reused_teacher_cache声明的100个源路径exact为donor的80 calibration/20 validation文件；声明PT SHA仅历史文本值，不重新计算payload。donor与recipient都在本轮700文件stat集合中。

各版本prepare实际按record.clip_id/t0_us调用官方loader（1fe596）；六版prepare片段同字节SHA，第七为CRLF差异，保持原source bytes，不重写成同SHA。原脚本历史行为中的torch.load、生成或复制仅静态阅读，本轮从未执行。

每组目录PT集合exact为calibration_0..79、validation_0..19，700个不同路径，均regular/non-symlink。因为复制分支的内部身份检查依赖历史PT读取，本轮保守采用**整组100clip范围**，不把这些索引声明为重新验过PT内容的逐文件身份。失败/中断也不释放曝光；本文不宣称七实验全部训练成功。

## 原CSV守恒与当前门禁

query全读原CSV固定SHA `f398e3f054620d37544217e766fbafba2d6d052f084df75e22df9f41259a09c7`、ledger固定SHA `0697cec28cd36ca841fe75ddeb95f1037269d69871bc2bda59ae831947e870d1`。七组700文件对应原binary记录700条、原anonymous记录698条，每类选中source各恰一次；初始newdata8有2文件本就没有anonymous标记，不能为凑700补造这2条。

待独立签发后可将698匿名记录签为bounded_used_group；其它二进制中700条只签有限历史曝光来源范围。与已验收544条无同路径时，累计anonymous可归因1242、剩650；binary来源可归因1244、剩2860。**全部4104条payload仍未验证**，不是只剩2860条payload。当前只是执行候选结果，旧CSV/query/ledger不回写。

该100clip全已used，无新增排除块。旧21块仅保留不释放；coverage_complete=False、drive_disjoint_verified=False，独立名单未冻结，无新数据精度或Jetson性能结论。adapter/checkpoint等当前未归因，不因名字或同目录自动豁免。下一步独立验收本窄域，然后核剩余650匿名来源和其它scope缺口。

## 独立复核指出的 argv 绑定补证

独立Sol指出初始查询没有校验本次argv.root/output/argv[0]对应实验，因此主任务新增只读 `check_teacher_argv_binding.py`，不修改v2查询。实际同环境命令仅替换脚本名，stdout `teacher-argv-binding-query.json`、stderr `teacher-argv-binding-stderr.txt`，noclobber拒覆盖；UTC2026-10-11T02:23:54.709944–02:23:54.716756，exit0（4bf62f），stderr为空。

查询SHA `0d49dd1be30fc7046dbc045ca1b7387c74780a3cdaa1163c5fba01622b52eb05`；JSON SHA `fdbe50b200e148508e65cd022cf934584db10c6d9468d245ed9b855157da5349`，已取回本地同目录（c2f24e/dd7cb5）。它先核父v2 JSON实际SHA，逐组重新核provenance SHA等于父查询读入身份，再校验七组argv中唯一root/output为绝对路径、分别exact项目根与该实验输出。argv[0]明确 `compression/run_interface_recovery.py`；冻结code snapshot实际SHA对应同实验provenance和父查询producer SHA。没有声称现在compression同名源码还等于各次历史版本，也没有从相对argv[0]推定未记录的历史cwd；记录层只签该argv脚本身份、绝对root/output与历史snapshot的关联。文本前后稳定检查通过。

独立验收需纳入此补充回执后才决定700/698有限范围 disposition；原query与原CSV仍不改。
