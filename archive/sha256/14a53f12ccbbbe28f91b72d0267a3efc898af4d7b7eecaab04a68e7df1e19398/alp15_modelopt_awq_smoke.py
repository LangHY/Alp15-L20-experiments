"""ModelOpt 0.39 AWQ CUDA BF16 API 桥接；仅随机 Linear，不作 VLA 验收。

用已有 isolated vendor 经 PYTHONPATH 导入，不安装或修改依赖环境。
固定源码依据：NVIDIA/Model-Optimizer f329b19d4d0512b4a0774838a94f684e20347cb7。
权重仍由 ModelOpt fake quant 管理；state_dict 不代表真实 packed INT4 或部署产物。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path


SEED = 20261008
SOURCE = ("https://github.com/NVIDIA/Model-Optimizer/blob/"
          "f329b19d4d0512b4a0774838a94f684e20347cb7/"
          "modelopt/torch/quantization/model_calib.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    started_monotonic = time.monotonic()
    # 拒绝已有目录，包括失败产物；每次运行须新建独立路径。
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "seed": SEED, "official_source": SOURCE,
              "scope": "random CUDA BF16 Linear 256->64; no checkpoint or VLA data",
              "representation": "ModelOpt AWQ fake quant state; not packed deployment",
              "output": str(args.output.resolve()), "callback_count": 0,
              "calibration_batches": 4, "calibration_rows": 256,
              "started_utc": started_utc,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "python": sys.version, "platform": platform.platform()}
    observed_warnings = []
    exit_code = 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            import torch
            from safetensors.torch import save_file
            import modelopt
            import modelopt.torch.quantization as mtq

            version = importlib.metadata.version("nvidia-modelopt")
            report.update(modelopt_version=version, modelopt_source=modelopt.__file__,
                          torch_version=torch.__version__, cuda_version=torch.version.cuda)
            if version != "0.39.0":
                raise RuntimeError(f"需要固定 ModelOpt 0.39.0，实际 {version}")
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("需要支持 BF16 的 CUDA GPU")
            report["gpu"] = torch.cuda.get_device_name(0)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            linear = torch.nn.Linear(256, 64, bias=True, device="cuda", dtype=torch.bfloat16)
            model = torch.nn.Sequential(linear).eval()
            # 每个 group128 内高/低方差交错，保证缩放可以改变组内相对精度。
            # 固定随机数据；不使用任何真实 clip。
            channel_std = torch.tensor([8., .05], device="cuda").repeat(128)
            calibration = (torch.randn(256, 256, device="cuda") * channel_std).to(torch.bfloat16)
            evaluation = (torch.randn(64, 256, device="cuda") * channel_std).to(torch.bfloat16)
            source_weight = linear.weight.detach().clone()
            source_bias = linear.bias.detach().clone()

            def finite(tensor, name):
                if not torch.isfinite(tensor).all().item():
                    raise RuntimeError(f"{name} 包含非有限值")

            def describe(tensor, values=False):
                finite(tensor, "diagnostic tensor")
                data = tensor.detach().float().cpu()
                result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                          "min": data.min().item(), "max": data.max().item(),
                          "finite": True}
                if values:
                    result["values"] = data.tolist()
                return result

            def save_tensors(path, tensors):
                if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
                    raise RuntimeError("state_dict 含非 tensor；停止而不静默丢弃状态")
                for name, tensor in tensors.items():
                    finite(tensor, name)
                save_file({name: value.detach().cpu().contiguous().clone()
                           for name, value in tensors.items()}, str(path))

            save_tensors(args.output / "source.safetensors",
                         {"weight": source_weight, "bias": source_bias,
                          "calibration": calibration, "evaluation": evaluation,
                          "channel_std": channel_std})
            report["source_saved_before_quantize"] = True
            empirical_std = calibration.float().std(dim=0)
            report["input_groups_std"] = {
                "layout": "each group128: even-index high8 / odd-index low0.05",
                "high_even_mean_std": empirical_std[0::2].mean().item(),
                "low_odd_mean_std": empirical_std[1::2].mean().item(),
                "ratio": (empirical_std[0::2].mean() / empirical_std[1::2].mean()).item(),
                "group128_ratios": [
                    (group[0::2].mean() / group[1::2].mean()).item()
                    for group in empirical_std.chunk(2)]}
            config = copy.deepcopy(mtq.INT4_AWQ_CFG)
            report["actual_config"] = config
            algorithm = config.get("algorithm", {})
            if (not isinstance(algorithm, dict) or algorithm.get("method") != "awq_lite" or
                    algorithm.get("alpha_step") != .1):
                raise RuntimeError("固定 INT4_AWQ_CFG 的 algorithm 与核查决策不符")

            @torch.no_grad()
            def forward_loop(calibration_model):
                report["callback_count"] += 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for batch in calibration.chunk(4):
                        finite(calibration_model(batch), "calibration output")

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                original_output = model(evaluation).detach().clone()
                finite(original_output, "original output")
                model = mtq.quantize(model, config, forward_loop=forward_loop)
                awq_fallback_warnings = [str(item.message) for item in caught
                                         if "awq" in str(item.message).lower() and
                                         any(word in str(item.message).lower()
                                             for word in ("fallback", "fall back", "skipping", "cannot perform"))]
                if awq_fallback_warnings:
                    raise RuntimeError("ModelOpt 报告 AWQ fallback/跳过，停止: " +
                                       " | ".join(awq_fallback_warnings))
                if report["callback_count"] < 2:
                    raise RuntimeError("AWQ 校准 callback 少于两次，未验证两 pass")
                quant_linear = model[0]
                report["quant_linear_is_original_linear"] = quant_linear is linear
                if quant_linear is not linear:
                    raise RuntimeError("ModelOpt 未原位复用原 Linear 对象，适配器桥接不符")
                input_quantizer = quant_linear.input_quantizer
                weight_quantizer = quant_linear.weight_quantizer
                # 固定 0.39 公开 property，经 vendor TensorQuantizer 源码核查：
                # num_bits L183、block_sizes L293、is_enabled L395、step_size L266。
                block_sizes = weight_quantizer.block_sizes
                report["actual_quantizer_state"] = {
                    "input_is_enabled": input_quantizer.is_enabled,
                    "weight_is_enabled": weight_quantizer.is_enabled,
                    "weight_num_bits": weight_quantizer.num_bits,
                    "weight_block_sizes": block_sizes}
                if input_quantizer.is_enabled or not weight_quantizer.is_enabled:
                    raise RuntimeError("实际状态须为 activation quant disabled / weight quant enabled")
                if (weight_quantizer.num_bits != 4 or not isinstance(block_sizes, dict) or
                        block_sizes.get(-1) != 128 or block_sizes.get("type") != "static"):
                    raise RuntimeError("实际 weight quantizer 须为 INT4 / static group128")
                pre_scale = input_quantizer.pre_quant_scale
                amax = weight_quantizer.amax
                if not isinstance(pre_scale, torch.Tensor) or pre_scale.numel() == 0:
                    raise RuntimeError("AWQ 未生成实际 input pre_quant_scale，拒绝 fallback")
                finite(pre_scale, "pre_quant_scale")
                if not (pre_scale > 0).all().item():
                    raise RuntimeError("AWQ pre_quant_scale 必须为正")
                if torch.allclose(pre_scale.float(), torch.ones_like(pre_scale).float()):
                    raise RuntimeError("AWQ pre_quant_scale 全一，未验证实际异方差缩放")
                if not isinstance(amax, torch.Tensor) or amax.numel() == 0:
                    raise RuntimeError("weight_quantizer 未产生实际 amax")
                finite(amax, "weight_quantizer.amax")
                if not (amax >= 0).all().item():
                    raise RuntimeError("weight_quantizer.amax 不能为负")
                step_size = weight_quantizer.step_size
                if not isinstance(step_size, torch.Tensor):
                    raise RuntimeError("weight_quantizer 未产生实际 step_size")
                finite(step_size, "weight_quantizer.step_size")
                torch.testing.assert_close(step_size, amax / 7, rtol=0, atol=0)
                report["quant_linear_class"] = str(type(quant_linear))
                report["pre_quant_scale"] = describe(pre_scale, values=True)
                report["weight_quantizer_amax"] = describe(amax, values=True)
                report["weight_quantizer_step_size"] = describe(step_size, values=True)
                scaled_input = evaluation * pre_scale
                quantizer_input = input_quantizer(evaluation)
                quant_weight = weight_quantizer(quant_linear.weight)
                for name, tensor in (("scaled_input", scaled_input), ("quantizer_input", quantizer_input),
                                     ("quantized_weight", quant_weight)):
                    report[name] = describe(tensor)
                # 0.39 TensorQuantizer.forward 直接 inputs * pre_quant_scale；
                # 用同一 CUDA BF16 autocast 保持与离线 runner 的 matmul 口径一致。
                torch.testing.assert_close(quantizer_input, scaled_input, rtol=0, atol=0)
                quant_output = quant_linear(evaluation)
                manual_output = torch.nn.functional.linear(scaled_input, quant_weight, quant_linear.bias)
                omitted_scale_output = torch.nn.functional.linear(evaluation, quant_weight, quant_linear.bias)
                for name, tensor in (("quant_output", quant_output), ("manual_output", manual_output),
                                     ("omitted_scale_output", omitted_scale_output)):
                    report[name] = describe(tensor)
                torch.testing.assert_close(quant_output, manual_output, rtol=.01, atol=.02)
                manual_error = (quant_output.float() - manual_output.float()).abs().max().item()
                omitted_error = (quant_output.float() - omitted_scale_output.float()).abs().max().item()
                change = (quant_output.float() - original_output.float()).abs().max().item()
                report.update(manual_reconstruction_max_abs_error=manual_error,
                              omitted_pre_scale_max_abs_error=omitted_error,
                              quantization_output_max_abs_change=change,
                              quantization_output_rmse=(quant_output.float() - original_output.float())
                              .square().mean().sqrt().item())
                if change <= 0:
                    raise RuntimeError("量化前后 forward 未改变，停止")
                if omitted_error <= manual_error + .001:
                    raise RuntimeError("遗漏 pre_scale 未产生可测额外误差，桥接验证不足")
                # 保存 ModelOpt fake quant tensor state；没有 packed storage 声称。
                save_tensors(args.output / "awq_state.safetensors", model.state_dict())
                save_tensors(args.output / "diagnostics.safetensors",
                             {"pre_quant_scale": pre_scale, "weight_quantizer_amax": amax,
                              "weight_quantizer_step_size": step_size,
                              "quantized_weight": quant_weight, "source_output": original_output,
                              "quant_output": quant_output, "manual_output": manual_output,
                              "omitted_scale_output": omitted_scale_output})
                torch.cuda.synchronize()
            report["status"] = "passed_api_smoke_only"
            exit_code = 0
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc())
        finally:
            observed_warnings = [{"category": item.category.__name__, "message": str(item.message),
                                  "filename": item.filename, "lineno": item.lineno} for item in caught]
    report["warnings"] = observed_warnings
    report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    report["elapsed_s"] = time.monotonic() - started_monotonic
    (args.output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8")
    print(json.dumps({"status": report["status"], "result": str(args.output / "result.json"),
                      "callback_count": report["callback_count"], "warning_count": len(observed_warnings)},
                     ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
