"""实体评测入口预留；默认reference也需后续正式执行单，PhaseA不启动GPU。"""
from __future__ import annotations
import argparse
from alp15_packed_loader import GateError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", choices=("bf16_reference", "packed_only"), default="bf16_reference")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-audit-dir")
    args = parser.parse_args(argv)
    raise GateError(f"{args.loader}评测未实施：PhaseA真实meta验收未完成；禁止实体load/GPU/质量伪通过")


if __name__ == "__main__":
    main()
