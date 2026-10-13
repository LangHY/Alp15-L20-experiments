# 全36层FFN敏感性测量：执行回报

对应采前决策：[ALP15-WHITENED-ALL36-MEASURE-20261007](ALP15-FFN-SENSITIVITY-20261007.md)。决策、实现和独立复核为GPT-6.1 Sol high；现场协调为当前主智能体。日期/用户时间统一Asia/Shanghai。

## 当前状态

**全36层敏感性测量及结果复核已完成；不是完整6B模型验收。** 实际控制器开始2026-10-07 20:21:50、结束20:39:58，耗时1088.294秒（约18.14分钟），包装器退出码0。36个child均完成9配置，共324配置。此前20:21:49为包装器发起时间。原模型、缓存和历史结果保持只读。

按用户“这一步完成后就先总结交付”的要求，到此结束本轮，不启动Q/O采集、训练、完整模型组装或下一阶段。

## 启动前验收

- 同一L20服务器、Ubuntu22.04.5，空闲显存41MiB，驱动580.95.05；`alpamayo1.5/.venv/bin/python`推理环境，torch2.8.0+cu128、CUDA12.8、transformers4.57.1、safetensors0.7.0、Python3.12.14。运行时逐child manifest再记录实际环境。
- 36份训练缓存存在、工作盘剩余1.6T；内容来源由每child完整preflight核验，不仅检查存在。
- 新入口SHA256 `80ff1c65cadae6e01cedc988d1d2c2768158a97743ba8d15f6cf27ba97f15665`；测试SHA256 `f5b48b4a4c248b74ad0098abd774afbcf1c85f51075d7000733fc6367e708031`。
- 冻结engine/base哈希分别`fa2e18bf86a768e724ec08536db1a0e6a9ed88125fe199fcebf04317ef86b007`/`c2975113b04a0a5e295ab14f0ba09db4152568fdb751ec957b45c87b3693ad69`，未修改。
- 采前决策SHA256 `0e6b7ef5363729c3a5e6f12cbd7c609ea74f9d1e39bc3d2ff626d557812fbd43`，服务器副本一致。
- 独立审阅发现资源异常和归档异常时真实退出码留痕缺失，已部署前修复；最终9项模拟协议测试全部通过、无跳过。服务器重复运行9项通过，退出码0；它们不运行模型/GPU，不能代替实际测量验收。

## 实际命令与产物

服务器新代码目录 `experiments/ffn_sensitivity_20261007_code/`；CPU测试日志`cpu-tests.log`。CPU命令：

```sh
alpamayo1.5/.venv/bin/python -m unittest discover \
  -s experiments/ffn_sensitivity_20261007_code/tests \
  -p test_alp15_whitened_sensitivity.py -v
```

GPU控制器命令：

```sh
alpamayo1.5/.venv/bin/python -u \
  experiments/ffn_sensitivity_20261007_code/scripts/alp15_whitened_sensitivity.py \
  --root /home/USER/lang/Lighten_VLA \
  --output /home/USER/lang/Lighten_VLA/experiments/ffn_sensitivity_20261007_stage4 \
  --decision /home/USER/lang/Lighten_VLA/experiments/ffn_sensitivity_20261007_code/ALP15-FFN-SENSITIVITY-20261007.md
```

使用nohup独立包装器，stdout/stderr为新代码目录`controller.log`，最终包装器退出码为`exit-code.txt`。每层命令、实际开始/结束时间和退出码保存在root manifest；每层`execution.log`、`resources.csv`和9条结果保存在独立`layer_XX/`。root进度原子保存，有排他锁。资源采样每5秒记录自身child RSS/GPU及其他计算进程，竞争或监测异常仅终止本实验child，禁止碰其他进程。

## 本轮验收与结论

完整交付：[全层指标报告](evidence/ffn_sensitivity_metrics_20261007/REPORT.md)。含324配置、72组fit所选方法与等参数通道配对、5184条逐clip配置指标、1152条逐clip配对；它们共用同一16个校准评估clips，不是独立样本数。

- 主智能体验证720份child声明artifact哈希、root/child manifest链与324行；独立6.1 Sol high复核796次哈希核对、324行/72配对、每clip目标能量/token、8192抽样行属于fit边界均通过。
- 完整报告最终由主智能体再次核对888个输入哈希条目与12份输出文件哈希通过；不同目录副本会重复计数，不能把次数称为独立文件总数。
- 6/12/24/35层全部36配置与旧阶段2/3来源相同，fit/eval聚合数值差均为0。没有选择两次中更好者。
- 汇总器修复旧重复来源未对齐和eval列误命名两项审阅问题；4项标准库测试通过、退出码0。新汇总器SHA256 `c2e8ca4acf84f21a7bc58e56ea68ffe2d4cc9a820a464f13dae93ddaf9a90e79`；测试SHA256 `cdfdb39df3977fa0846014e89eb9d9729b1aa4d7b8d66c9d497d6a1ab9440ff9`。报告SHA256 `be88256154de29b31a829edb30f146ba204800d3b7d6d9fbac478d82dc6a41e8`。

### 直接结论

白化＋压缩特征闭式重建有局部价值，但**不足以支持全层统一低rank直接组装6B**。72组fit-only都选该组合，校准能量NMSE改善52/72组、clip等权NMSE改善54/72组；34/72组的p95或最大误差恶化，仅50/72组在两口径leave-one-clip-out检查中均保持优势。

| 档位 | 能量NMSE改善层数 | 相对误差下降中位数（逐层） | 劣于通道层数 |
| --- | ---: | ---: | ---: |
| rank896 | 27/36 | 14.43% | 9/36 |
| rank1024 | 25/36 | 12.73% | 11/36 |

训练侧只有第3/16层在两个rank均劣于通道；隔离校准侧却有更多层退化，尤其26–32层，提示训练与校准落差。该现象不是因果过拟合证明，也不能据校准失败名单重新分配层/rank。训练侧最高敏感层为10、27、8、18、20、17、19、26、7、28（两档顺序相同）；局部误差不是驾驶轨迹因果重要性。

全层统一低rank方案停止作为直接组装候选。后续方向仅作备忘：先改善训练侧重建稳健性、保留或提高敏感层rank，并另设Q/O测量与参数预算决策；本轮不执行。完整ADE/FDE、自主生成、重载一致性仍待测，不能宣称6B精度保持成功。

### 资源与边界

5秒采样的最大child RSS约1.493GiB、进程显存2410MiB；瞬时峰值可能被采样遗漏，CUDA峰值另在CSV。最大父缓存复现NMSE `8.546675251178866e-7`，低于冻结1e-3门限。结束并重连检查GPU为空闲41MiB、0%利用率。这里是校准/分解资源与耗时，不是完整VLA推理性能。

原始结果与日志完整取回 `evidence/ffn_sensitivity_20261007_stage4/` 和 `evidence/ffn_sensitivity_20261007_code/`。采前计划、源码副本、随机执行顺序、每child环境/argv/退出码、资源CSV、失败/测试记录均保留，没有删除旧实验。

可重复生成报告（输出必须不存在）：

```sh
python3 scripts/summarize_alp15_whitened_sensitivity.py \
  --input docs/baselines/evidence/ffn_sensitivity_20261007_stage4 \
  --output /tmp/alp15-all36-report-NEW \
  --prior-inputs docs/baselines/evidence/whitened_pilot_20261007_stage2 \
                 docs/baselines/evidence/whitened_confirm_20261007_stage3
```

无权重变更、无完整模型组装、无训练、无Q/O采集、无真实ADE/FDE或Jetson测试。fit侧逐clip/SSE/能量缺失明确列出，未补造或按eval改选。
