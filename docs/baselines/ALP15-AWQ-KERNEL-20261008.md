# AWQ packed 算子成本诊断与内核门禁

决策编号 `D-JETSON-KERNEL-PROBE-20261008`，2026-10-08。执行者仅拥有 `scripts/alp15_awq_kernel.py`、`scripts/benchmark_alp15_awq_kernel.py`、`tests/test_alp15_awq_kernel.py` 和本文件；主任务统一上传、调度 L20 并独立验收。本阶段先交 reference 基线工具，收到实际成本证据后才进入候选内核实现。

目标仍为 Jetson16GB、端到端延迟不大于300ms、相教师精度损失不超过15%。本阶段只回答既有 packed forward 的成本来源，不能用 isolated L20 算子速度验收上述目标。既有 AWQ full20 真实权重留在服务器，不下载、不拷贝大矩阵到开发机，也不重新量化或校准。

## 冻结数学与来源

参考文件 `scripts/alp15_awq_full_packed.py`，SHA256 `150e1310ad3dbcb9f82966146f9627855e87bd12a29f45f410d0b94f1a96de0a`；signed nibble 文件 `scripts/alp15_packed_w4a16.py`，SHA256 `241edf75d838ad7cba1250374119237864780b95ec81cd8555a7d7c6d15fab12`。原完整 AWQ runner SHA256 `1ba06bf2c461633a2d743960c22b0dd02bbe25c831617fbae771028a70758357`。三文件均不修改。

参考 `PackedAWQLinear.forward` 的语义是先按原 dtype 执行 `inputs * pre_scale`，按 scaled dtype 保留反量化舍入，再执行 `F.linear`。反量化为 signed INT4 [-7,7]、group128、FP32 中 `q / (7 / amax)`，零amax组输出0；不是 `q * (amax / 7)`，不重新计算 RTN scale，不丢弃原 pre_scale。原 unpack 的 `.any().item()` 引入 host 同步，该检查保留在原 reference 成本中。

full20 `result.json` 的第一份真实矩阵为 `language_model.layers.0.mlp.gate_proj`，5376×4096，stored amax/pre_scale 均为 BF16；对应文件 SHA256 `eb67b5705a49e08eab3eadecdafa4113a7e80a72a423d0b7bbef97e6f3d5b15b`。执行 CLI 在加载前要求此 SHA吻合，并由冻结 loader 校验格式和nibble；不接受只看文件名。

## 本机 RED→GREEN 与验证

完整读取 tdd/diagnosing-bugs 及 tdd tests/mocking。公开 seam 按主任务决策确认：真实 safetensors load→原 reference forward、隔离 CUDA cost measurement CLI。第一轮 `load_reference_matrix` 模块缺失 RED→GREEN；第二轮 `benchmark_matrix` API缺失 RED→GREEN。数学 fixture 采用手算二补码0x79（低位-7、高位+7）、零amax、bias、M=5奇数尾部，原buffer逐字节保持。CPU benchmark 明确拒绝，import 不引入 Triton。冻结SHA不符拒绝加载。

命令：

```sh
/opt/anaconda3/bin/python -m unittest discover -s tests -p test_alp15_awq_kernel.py -v
/opt/anaconda3/bin/python -m py_compile scripts/benchmark_alp15_awq_kernel.py tests/test_alp15_awq_kernel.py
```

环境：macOS开发机、Python3.13、torch2.13.0、safetensors0.8.0。最终4项CPU测试通过、无跳过、退出0；静态编译退出0。这些只证明 reference 诊断接口与格式数学行为，CUDA从未运行，不能称 CUDA 测试通过。

## L20 reference 基线命令（主任务上传与执行）

建议将新 benchmark/tests 放入独立 `experiments/awq_kernel_20261008_code_v1/`，通过后续 PYTHONPATH 复用冻结AWQpacked文件。先对单个真实矩阵采样M=1、17、256；不会加载完整 VLA。此 CLI 已禁止覆盖现有输出文件。

```sh
PYTHONPATH=experiments/awq_kernel_20261008_code_v1:experiments/awq_full_20261008_code_v4:scripts \
alpamayo1.5/.venv/bin/python -u experiments/awq_kernel_20261008_code_v1/benchmark_alp15_awq_kernel.py \
  --matrix /home/USER/lang/Lighten_VLA/experiments/awq_full_20261008_best8_full20_v4/packed/language_model_layers_0_mlp_gate_proj.safetensors \
  --sha256 eb67b5705a49e08eab3eadecdafa4113a7e80a72a423d0b7bbef97e6f3d5b15b \
  --output /home/USER/lang/Lighten_VLA/experiments/awq_kernel_20261008_gate_reference_bf16_v1.json \
  --device cuda:0 --dtype bf16 --m 1 17 256 --warmup 3 --repeats 10
```

首次调用、full_reference、dequant_only、prescale_only、dense_linear_only 分别记录 CUDA event 及含Python/同步的 wall 时间 median/p95/min/max/raw samples，baseline/peak allocated/reserved、额外allocated峰值、环境与源码SHA。dense_linear_only临时持有一份单矩阵dense，仅用于分解对照，明确记录额外驻留字节，结束即释放；没有在packed模块缓存dense。所有阶段独立测量，时间与峰值不可直接相加。输入采用固定seed随机激活并记录hash，不能作为驾驶精度检查或推断实际视觉prefill的M。

可选 `--inspect-only` 仅在CPU读取格式并保存receipt，状态为 `passed_reference_format_inspection_only`；正式 CUDA 命令无 CUDA/BF16支持即报错，绝不默默转CPU。FP16需独立输出及再测；原amax/pre_scale不随 `--dtype fp16` 转换，若它们提升 scaled dtype，结果明确记录实际dtype。

## 后续候选门禁

等待现场 reference 基线、环境与独立调度证据，再依据成本占比决定small-M GEMV/prefill候选。候选须保留signed nibble、零amax、FP32除法顺序、BF16 scaled与weight舍入、bias和奇数M/N尾部；不得缓存全dense，不修改既有模型，不用silent fallback。公开算子 API 需显式拒绝不支持的设备/dtype/shape，CPU import 不要求Triton，CUDA使用当前设备/流正确。

内核准确性须在真实矩阵与独立手算fixture上同原 reference 对拍并记录bitwise权重/激活桥接及输出差异；禁止跳过CUDA测试而计为通过。数值不符合、非法nibble、OOM或改变参考来源即停止并升级，不以单算子提速宣称Jetson部署达标。

## 已完成 reference 反馈环与v2候选

主任务现场回传：L20 gate5376×4096真实矩阵 reference 基线退出0，输出 `experiments/jetson_300ms_20261008_kernel_probe/gate_reference_bf16.json`，待小型JSON取回本地证据目录 `docs/baselines/evidence/jetson_300ms_20261008/kernel_probe/`。

| M | full wall median(ms) | dequant median(ms) | dense median(ms) | full extra allocated(B) |
|---:|---:|---:|---:|---:|
|1|1.2679|1.2333|0.0332|353878016|
|17|1.2891|1.2330|0.0426|待原JSON独验|
|256|1.3590|1.2335|0.1257|待原JSON独验|
|3086|6.4405|1.6934|3.5016|379150336|

M1 dequant约占full的97%，M1 prescale为0.0152ms，支持优先消除小M临时完整反量化。M3086来自主任务现场诊断需要，不能推断任何所有层都具有该M。具体百分比仅作成本定位，阶段不可简单相加。

原benchmark SHA `ab8fe9704b0a3b08a96570f9571fbc89bd93be9822f73a0321933824680631ae` 已冻结在服务器 `experiments/jetson_300ms_20261008_code_v1/scripts/benchmark_alp15_awq_kernel.py`；同目录冻结原两个packed helper，原4CPU测试在 `code_v1/tests`，服务器也4项通过0skip退出0。新候选上传独立 `experiments/jetson_300ms_20261008_code_v2/`，不覆盖原代码与JSON。

v2候选 `FusedAWQLinear.from_reference` 仅接受已经原loader验真的PackedAWQLinear，共享同一份原qweight/amax/prescale/bias，不量化、不重新计算scale、不产生full dense cache。forward要求同设备连续BF16、无输入梯度、NVIDIA CUDA capability≥8；CPU/其它dtype显式报错。M≤8用每8个output行的GEMV，K按128循环，BF16 scaled和weight在tile内舍入后FP32累加与归约；M>8使用BM32/BN64/BK128 BF16 `tl.dot`并FP32 accumulator。CUDA默认使用调用者当前stream，forward没有 `.item()`、CPUdata copy或资源采样；Python只检查shape/dtype/device/stride。

实际Torch数学追加核实：`TorchDispatchMode` 显示参考 `7. / amax.float()` 实际为 `aten.reciprocal.default`→`aten.mul.Tensor`。因此候选用FP32精确 `div_rn(1,amax)` 再乘7、再 `div_rn(q,inverse)`，不是合并为单次 `div_rn(7,amax)`，也不是 `q*amax/7`。关闭fp_fusion避免中间舍入被合并。官方[Triton div_rn](https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html)规定IEEE round-to-nearest精确除法；官方[矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)提供tile/FP32 accumulator/masked tails语义；[Triton3.4 JIT源码](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/runtime/jit.py)确认lazy builder的nested JIT closure会捕获nonlocals。公开网页静态核验，不依赖Chrome登录或额外下载。

独立 `dequantize_for_validation` 仅在显式门禁时临时构造单矩阵BF16输出，必须与原 `matrix.dequantize(BF16)` 整矩阵 `torch.equal`；forward不会调用它。GEMV/GEMM采用不同FP32归约顺序，输出不提前声称bitwise相同；候选benchmark逐M报告exact_equal/max_abs/mean_abs/outside_count，并按此前AWQ Linear桥接的rtol0.01/atol0.02拒绝超限。这是算子数值桥接门禁，不代替教师精度15%门禁。

本机第三轮公开candidate API模块缺失RED→GREEN；现在5项CPU测试通过0skip、py_compile退出0。新增4CUDA测试由 `ALP15_KERNEL_REQUIRE_CUDA=1` 显式启用：手算非平凡prescale/零amax/signed nibbles/bias/M1/5/17/33及3Dshape；随机group128/math与精确dequant；调用者non-default stream；不同同shape权重不存在cached dense污染。没有CUDA时require flag会ERROR，绝不skip当通过。当前CUDA测试未运行，由主任务统一调度：

```sh
ALP15_KERNEL_REQUIRE_CUDA=1 \
PYTHONPATH=experiments/jetson_300ms_20261008_code_v2/scripts:experiments/jetson_300ms_20261008_code_v1/scripts \
alpamayo1.5/.venv/bin/python -m unittest discover \
  -s experiments/jetson_300ms_20261008_code_v2/tests -p test_alp15_awq_kernel.py -v
```

9项全部通过、无skip后再运行：

```sh
PYTHONPATH=experiments/jetson_300ms_20261008_code_v2/scripts:experiments/jetson_300ms_20261008_code_v1/scripts \
alpamayo1.5/.venv/bin/python -u experiments/jetson_300ms_20261008_code_v2/scripts/benchmark_alp15_awq_kernel.py \
  --matrix /home/USER/lang/Lighten_VLA/experiments/awq_full_20261008_best8_full20_v4/packed/language_model_layers_0_mlp_gate_proj.safetensors \
  --sha256 eb67b5705a49e08eab3eadecdafa4113a7e80a72a423d0b7bbef97e6f3d5b15b \
  --output /home/USER/lang/Lighten_VLA/experiments/jetson_300ms_20261008_kernel_probe/gate_candidate_bf16_v2.json \
  --device cuda:0 --dtype bf16 --m 1 17 256 3086 --warmup 3 --repeats 10 --candidate
```

冻结v2：kernel SHA `3f09aea862990733b9bb1db536c450c79fd6e2ac93c7015eed6b1eef4f191025`，benchmark SHA `229ad91e2d8553282da7afa3380187a6ebbb4e0b572d72c9074332da8760844f`，tests SHA `45834d6486b206840bf4d225d56cb4eda59588b2366b359d63507efdf7f834a1`。候选首次JIT单列记录，不与steady耗时混淆。尚无候选CUDA实测，亦未替换全模型。
