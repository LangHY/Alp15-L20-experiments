# KV / 官方少步求解：完整20段开发集结果

记录日期：2026-10-09，Asia/Shanghai。主任务唯一写入。对应决策 D-JETSON-KV-CANDIDATE-20261008。目标保持 Orin NX16GB、完整请求 max≤300ms、相较教师 meanADE 损失≤15%；本报告不是实机验收。

## 证据与完整性

服务器结果：`/home/USER/lang/Lighten_VLA/experiments/jetson_300ms_20261008_kv_candidate_full20_v1/result.json`。
本地只读证据：[result.json](evidence/jetson_300ms_20261008/kv_candidate_full20_v1/result.json)，同目录保留 memory_phases.jsonl、pipeline.log。
结果 SHA256：`db04c2ed3226e2c3c988c3f6ebd9218bd6150759d3334d25d7e193c7693ef84e`，已核对本机/服务器一致。
终态 `completed_kv_steps_development_probe`；pipeline于2026-10-08T14:22:12Z（22:22:12 CST）记录 COMPLETE，set-e链路退出成功。旧 launcher1124038、Python1128980结束，未重启。2026-10-09重新确认GPU仅其它项目PID453201占15428MiB，本任务未停止它。

冻结代码目录 `experiments/jetson_300ms_20261008_kv_code_v3`。入口 SHA256 `74886a867a8aae480dd848a6d068d2e3f817674d891774bcdabeb57f412292f8`；完整命令、源/权重/数据/依赖身份与实际保护摘要见结果与执行计划。20clip×seed42/43/44×5variant=300条全部completed，各variant60条，64×3坐标有限；原helper严格门禁60/60通过，保护模块前后摘要相同，来源最终复验通过。

actualKV10对同代码fullprefill10严格数值门禁仅2/60通过、58/60失败。因此actualKV是显式批准的新精度候选，不是无损替换；前缀/噪声/来源硬配对仍保持。不能给后续混合内核自动套用此豁免。

## 指标

同20clip/seed教师meanADE=1.2323994789272548m，15%上限1.4172594007663428m。

| 方案 | meanADE m | 相对教师ADE变化 | meanFDE m | clipADE p95 m | E2E mean / p50 / p95 / max ms | >300ms |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| fullprefill10 | 1.282724 | +4.0835% | 3.653804 | 2.504728 | 6429.29 / 6592.43 / 7513.05 / 7706.72 | 60/60 |
| actualKV10 | 1.282572 | +4.0711% | 3.653678 | 2.502071 | 5477.40 / 5659.29 / 6564.55 / 6598.96 | 60/60 |
| actualKV6 | 1.258088 | +2.0844% | 3.601106 | 2.454229 | 5320.97 / 5380.49 / 6442.01 / 6852.84 | 60/60 |
| actualKV4 | 1.245025 | +1.0245% | 3.595224 | 2.412862 | 5260.32 / 5267.97 / 6386.81 / 6495.00 | 60/60 |
| actualKV2 | 1.249218 | +1.3647% | 3.694152 | 2.672191 | 5213.36 / 5183.68 / 6148.74 / 6351.99 | 60/60 |

四步目前meanADE/FDE/clip p95更好，可作为后续联合验证优先候选；两步保留为性能候选。没有做显著性/置信区间检验，不称统计最优。独立单位为20clip，不是60seed或300请求；反复使用开发集不能证明独立泛化。共享L20有其它训练，时间不是无干扰因果比较，更不能外推Orin。

四步generation均值约4981.60ms，占完整请求大部分；减少Expert步数不足以消除语言生成瓶颈。所有方案未达300ms。

## 内存与下一步

同次记录 BF16-first cold load CUDA allocated峰值22,256,622,080B，reserved22,502,440,960B；packed替换阶段allocated峰值16,064,608,768B、reserved22,565,355,520B。最终persistent storage为10,272,486,512B。这些是L20阶段观测，不是Orin统一内存；不能相加RSS与CUDA独立峰值、也不能由裸权重断言16GB足够。

优先执行M1混合解码完整模型验证，降低逐token反量化成本；低峰值meta/分块packed-only loader并行开发，但实体加载需先通过真实CPU-meta门禁。FlashDrive Expert/静态KV/动作缓存及LightVLA Qwen视觉筛选需独立因素验收，不能相乘论文加速比。候选冻结后仍需独立clip、真实Orin ARM64依赖、冷启动/稳态全系统统一内存与完整请求最大延迟联合验证。
