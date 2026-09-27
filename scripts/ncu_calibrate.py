#!/usr/bin/env python3
"""Calibrate GPU FLOP counting on workloads with analytically known FLOPs.

Runs a matrix multiply (exactly 2*M*N*K FLOPs) or a 3x3 convolution (analytic
direct-convolution FLOPs) in one precision mode, inside an NVTX range called
"measure", and prints a JSON line with the analytic FLOPs and achieved TFLOP/s.

Profile it under Nsight Compute restricted to the NVTX range, then divide the
counted operations by the analytic FLOPs to see which counters capture the work:

    ncu --nvtx --nvtx-include "measure/" --metrics <...> --csv \
        python ncu_calibrate.py --op mm --mode tf32

Note: cuDNN may pick Winograd/FFT algorithms for convolutions, which execute
fewer operations than the direct-convolution count. The matmul is the clean
reference; the convolution is the representative case for a CNN.
"""

from __future__ import annotations

import argparse
import contextlib
import json

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--op", choices=["mm", "conv"], required=True)
    p.add_argument("--mode", choices=["fp32", "tf32", "fp16"], required=True,
                   help="fp32: TF32 disabled; tf32: TF32 enabled; fp16: autocast to float16")
    p.add_argument("--iters", type=int, default=5, help="Iterations inside the measured NVTX range")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--size", type=int, default=4096, help="Matrix dimension for --op mm")
    p.add_argument("--batch", type=int, default=16, help="Batch size for --op conv")
    p.add_argument("--channels", type=int, default=64, help="In/out channels for --op conv")
    p.add_argument("--hw", type=int, default=256, help="Spatial size (H=W) for --op conv")
    return p.parse_args()


def configure_precision(mode: str) -> contextlib.AbstractContextManager:
    allow_tf32 = mode == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = False
    if mode == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def build_workload(args: argparse.Namespace):
    dev = "cuda"
    if args.op == "mm":
        n = args.size
        a = torch.randn(n, n, device=dev)
        b = torch.randn(n, n, device=dev)
        flops = 2 * n * n * n

        def fn():
            return a @ b

        desc = {"M": n, "N": n, "K": n}
    else:
        n, c, hw, k, pad = args.batch, args.channels, args.hw, 3, 1
        x = torch.randn(n, c, hw, hw, device=dev)
        w = torch.randn(c, c, k, k, device=dev)
        hout = wout = hw + 2 * pad - k + 1
        flops = 2 * n * c * hout * wout * c * k * k

        def fn():
            return F.conv2d(x, w, padding=pad)

        desc = {"batch": n, "cin": c, "cout": c, "hw": hw, "kernel": k}
    return fn, flops, desc


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - run this on a GPU compute node.")

    ctx = configure_precision(args.mode)
    fn, flops_per_iter, desc = build_workload(args)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad(), ctx:
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()

        torch.cuda.nvtx.range_push("measure")
        start.record()
        for _ in range(args.iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    elapsed_s = start.elapsed_time(end) / 1000.0
    total_flops = flops_per_iter * args.iters
    result = {
        "op": args.op,
        "mode": args.mode,
        "shape": desc,
        "iters": args.iters,
        "analytic_flops_per_iter": flops_per_iter,
        "analytic_flops_total": total_flops,
        "elapsed_s": elapsed_s,
        "achieved_tflops": total_flops / elapsed_s / 1e12 if elapsed_s > 0 else None,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
