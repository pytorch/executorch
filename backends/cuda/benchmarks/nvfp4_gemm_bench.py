# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CUDA graph microbench for Muse Glimmer NVFP4 projection shapes.

Run with --output PATH; reports end-to-end activation quantization for W4A4.
TFLOPS counts logical multiply-adds, TB/s counts packed weight + scale bytes.
Theoretical RTX5090 memory bandwidth is 1.792 TB/s (stock GDDR7).
Dense tensor-core reference peaks are approximately 1676 TFLOPS for FP4
and 209.5 TFLOPS for BF16, at the nominal clock. Small cached weights may
exceed DRAM bandwidth; these are effective throughput ratios, not hardware
counter measurements.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import triton
from executorch.backends.cuda.autotune.cuda_graph_timing import cuda_graph_timing
from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_params,
    clear_launch_param_cache,
)
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm as kernels
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    nvfp4_gemv,
    nvfp4_w4a16,
    nvfp4_w4a4,
)
from torch._library.triton import set_wrap_triton_enabled

SHAPES = {
    "q_proj": (4096, 6656),
    "kv_proj": (256, 6656),
    "o_proj": (6656, 4096),
    "output_gate": (4096, 6656),
    "gate_up": (39936, 6656),
    "down_proj": (6656, 19968),
    "gate": (19968, 6656),
    "qko": (8448, 6656),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--m", type=int, nargs="+", default=[1, 4096])
    p.add_argument("--shapes", nargs="+", choices=list(SHAPES), default=list(SHAPES))
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "triton": triton.__version__,
                "gpu": torch.cuda.get_device_name(),
            }
        ),
        flush=True,
    )
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    fields = [
        "shape",
        "m",
        "n",
        "k",
        "mode",
        "latency_us",
        "effective_tb_s",
        "bandwidth_pct_5090",
        "tflops",
        "compute_pct_5090",
        "split_k",
        "config",
    ]
    with open(args.output, "w", newline="") as f, Path(args.output).with_suffix(
        ".tuning.jsonl"
    ).open("w") as tuning:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for shape in args.shapes:
            n, k = SHAPES[shape]
            q = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
            s = (
                (0.02 + torch.rand(n, k // 16, device="cuda") * 0.03)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            pts = torch.tensor(1.0, device="cuda")
            w = CudaNVFP4Tensor(q, s, pts)
            dense = w.dequantize()
            for m in args.m:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                funcs = {
                    "bf16_cublas": lambda: x @ dense.T,
                    "w4a4": lambda: nvfp4_w4a4(x, q, s, pts),
                    "w4a16": lambda: nvfp4_w4a16(x, q, s, pts),
                }
                if m <= 4:
                    funcs["w4a16"] = lambda: nvfp4_gemv(x, q, s, pts)
                splits = {"bf16_cublas": 1}
                for mode in ["w4a4", "w4a16"]:
                    clear_launch_param_cache()
                    with cuda_graph_timing(), autotune_launch_params() as stats:
                        funcs[mode]()
                    if not stats.picks:
                        raise RuntimeError("NVFP4 split-K selection was not recorded")
                    split = stats.picks[-1].choice
                    if stats.out_of_memory:
                        raise RuntimeError("NVFP4 candidates were not actually timed")
                    splits[mode] = split
                    if stats.picks:
                        pick = stats.picks[-1]
                        tuning.write(
                            json.dumps(
                                dict(
                                    shape=shape,
                                    m=m,
                                    n=n,
                                    k=k,
                                    mode=mode,
                                    split_k=split,
                                    times_us=pick.times_us,
                                    rejected=pick.rejected,
                                )
                            )
                            + "\n"
                        )
                        tuning.flush()
                    if mode == "w4a4":
                        funcs[mode] = lambda split=split: nvfp4_w4a4(
                            x, q, s, pts, split_k=split
                        )
                    elif m <= 4:
                        funcs[mode] = lambda split=split: nvfp4_gemv(
                            x, q, s, pts, split_k=split
                        )
                    else:
                        funcs[mode] = lambda split=split: nvfp4_w4a16(
                            x, q, s, pts, split_k=split
                        )
                for mode, fn in funcs.items():
                    fn()
                    torch.cuda.synchronize()
                    us = (
                        triton.testing.do_bench_cudagraph(
                            fn, rep=30 if args.quick else 200
                        )
                        * 1000
                    )
                    data_bytes = n * k * (2 if mode == "bf16_cublas" else 0.5625)
                    kernel = (
                        kernels._w4a4
                        if mode == "w4a4"
                        else (kernels._gemv if m <= 4 else kernels._w4a16)
                    )
                    config = None
                    if mode != "bf16_cublas":
                        matches = [
                            config
                            for key, config in kernel.cache.items()
                            if key[:4] == (m, n, k, splits[mode])
                        ]
                        if len(matches) != 1:
                            raise RuntimeError(
                                "Selected NVFP4 tile is missing or ambiguous"
                            )
                        config = matches[0]
                    selected_config = (
                        {}
                        if config is None
                        else dict(
                            config.kwargs,
                            num_warps=config.num_warps,
                            num_stages=config.num_stages,
                        )
                    )
                    row = dict(
                        shape=shape,
                        m=m,
                        n=n,
                        k=k,
                        mode=mode,
                        latency_us=us,
                        effective_tb_s=data_bytes / us / 1e6,
                        bandwidth_pct_5090=data_bytes / us / 1e6 / 1.792 * 100,
                        tflops=2 * m * n * k / us / 1e6,
                        compute_pct_5090=(
                            2
                            * m
                            * n
                            * k
                            / us
                            / 1e6
                            / (1676 if mode == "w4a4" else 209.5)
                            * 100
                            if m > 4
                            else ""
                        ),
                        split_k=splits[mode],
                        config=json.dumps(selected_config, sort_keys=True),
                    )
                    writer.writerow(row)
                    f.flush()
                    print(json.dumps(row), flush=True)
            del dense, q, s, w
            torch.cuda.empty_cache()


if __name__ == "__main__":
    with set_wrap_triton_enabled(False):
        main()
