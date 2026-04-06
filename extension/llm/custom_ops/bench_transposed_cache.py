# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Benchmark to compare performance of transposed vs standard KV cache layout
for custom_sdpa and update_cache ops.

Standard layout:    [Batch, Seq, Heads, HeadDim]  (is_seq_dim_2=False)
Transposed layout:  [Batch, Heads, Seq, HeadDim]  (is_seq_dim_2=True)

The hypothesis is that transposed cache may improve GEMM performance in
custom_sdpa because:
  - In attn_score @ V: V's stride along the S_kv dimension changes from
    H*D (strided) to D (contiguous), improving memory access patterns.
  - In Q @ K^T: K's stride similarly improves from H*D to D.
"""

import argparse
import time
from typing import Dict, List, Tuple

import torch

from executorch.extension.llm.custom_ops import custom_ops  # noqa


def benchmark_fn(fn, warmup: int = 10, iterations: int = 100) -> float:
    """Run fn for warmup iterations, then measure average time over iterations."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()

    return (end - start) / iterations


# Model configurations: (num_heads_kv, num_heads_q, head_dim, max_seq_len)
MODEL_CONFIGS = {
    "llama3_8b": (8, 32, 128, 2048),
    "llama3_70b": (8, 64, 128, 2048),
    "llama2_7b": (32, 32, 128, 2048),
    "small": (4, 8, 64, 512),
}


def bench_custom_sdpa(
    batch_size: int,
    num_heads_kv: int,
    num_heads_q: int,
    head_dim: int,
    max_seq_len: int,
    start_pos: int,
    seq_len: int,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark custom_sdpa with both cache layouts.

    Returns dict with times for standard and transposed layouts.
    """
    results = {}

    # Standard layout: [B, S, H, D]
    q_std = torch.rand(batch_size, seq_len, num_heads_q, head_dim)
    k_cache_std = torch.rand(batch_size, max_seq_len, num_heads_kv, head_dim)
    v_cache_std = torch.rand(batch_size, max_seq_len, num_heads_kv, head_dim)

    def run_std():
        torch.ops.llama.custom_sdpa(
            q_std, k_cache_std, v_cache_std,
            start_pos, None, 0.0, True, None, False,
        )

    results["standard"] = benchmark_fn(run_std, warmup, iterations)

    # Transposed layout: [B, H, S, D]
    q_trans = q_std.transpose(1, 2).contiguous()
    k_cache_trans = k_cache_std.transpose(1, 2).contiguous()
    v_cache_trans = v_cache_std.transpose(1, 2).contiguous()

    def run_trans():
        torch.ops.llama.custom_sdpa(
            q_trans, k_cache_trans, v_cache_trans,
            start_pos, None, 0.0, True, None, True,
        )

    results["transposed"] = benchmark_fn(run_trans, warmup, iterations)

    return results


def bench_update_cache(
    batch_size: int,
    num_heads_kv: int,
    head_dim: int,
    max_seq_len: int,
    start_pos: int,
    seq_len: int,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark update_cache with both cache layouts.

    Returns dict with times for standard and transposed layouts.
    """
    results = {}

    # Standard layout: [B, S, H, D]
    value_std = torch.rand(batch_size, seq_len, num_heads_kv, head_dim)
    cache_std = torch.zeros(batch_size, max_seq_len, num_heads_kv, head_dim)

    def run_std():
        torch.ops.llama.update_cache(value_std, cache_std, start_pos, False)

    results["standard"] = benchmark_fn(run_std, warmup, iterations)

    # Transposed layout: [B, H, S, D]
    value_trans = value_std.transpose(1, 2).contiguous()
    cache_trans = cache_std.transpose(1, 2).contiguous()

    def run_trans():
        torch.ops.llama.update_cache(value_trans, cache_trans, start_pos, True)

    results["transposed"] = benchmark_fn(run_trans, warmup, iterations)

    return results


def bench_combined_update_and_sdpa(
    batch_size: int,
    num_heads_kv: int,
    num_heads_q: int,
    head_dim: int,
    max_seq_len: int,
    start_pos: int,
    seq_len: int,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark combined update_cache + custom_sdpa to simulate a full attention
    step, which is the real-world usage pattern.
    """
    results = {}

    # Standard layout
    q_std = torch.rand(batch_size, seq_len, num_heads_q, head_dim)
    k_proj_std = torch.rand(batch_size, seq_len, num_heads_kv, head_dim)
    v_proj_std = torch.rand(batch_size, seq_len, num_heads_kv, head_dim)
    k_cache_std = torch.rand(batch_size, max_seq_len, num_heads_kv, head_dim)
    v_cache_std = torch.rand(batch_size, max_seq_len, num_heads_kv, head_dim)

    def run_std():
        torch.ops.llama.update_cache(k_proj_std, k_cache_std, start_pos, False)
        torch.ops.llama.update_cache(v_proj_std, v_cache_std, start_pos, False)
        torch.ops.llama.custom_sdpa(
            q_std, k_cache_std, v_cache_std,
            start_pos, None, 0.0, True, None, False,
        )

    results["standard"] = benchmark_fn(run_std, warmup, iterations)

    # Transposed layout
    q_trans = q_std.transpose(1, 2).contiguous()
    k_proj_trans = k_proj_std.transpose(1, 2).contiguous()
    v_proj_trans = v_proj_std.transpose(1, 2).contiguous()
    k_cache_trans = k_cache_std.transpose(1, 2).contiguous()
    v_cache_trans = v_cache_std.transpose(1, 2).contiguous()

    def run_trans():
        torch.ops.llama.update_cache(k_proj_trans, k_cache_trans, start_pos, True)
        torch.ops.llama.update_cache(v_proj_trans, v_cache_trans, start_pos, True)
        torch.ops.llama.custom_sdpa(
            q_trans, k_cache_trans, v_cache_trans,
            start_pos, None, 0.0, True, None, True,
        )

    results["transposed"] = benchmark_fn(run_trans, warmup, iterations)

    return results


def format_results(
    label: str,
    results: Dict[str, float],
) -> str:
    """Format benchmark results into a readable string."""
    std_us = results["standard"] * 1e6
    trans_us = results["transposed"] * 1e6
    speedup = results["standard"] / results["transposed"]
    return (
        f"  {label:40s}  std={std_us:10.1f} us  trans={trans_us:10.1f} us  "
        f"speedup={speedup:.3f}x"
    )


def run_benchmarks(
    config_name: str = "llama3_8b",
    batch_size: int = 1,
    warmup: int = 20,
    iterations: int = 200,
    num_threads: int = 1,
):
    """Run all benchmarks for a given model configuration."""
    if config_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )

    num_heads_kv, num_heads_q, head_dim, max_seq_len = MODEL_CONFIGS[config_name]

    # Set thread count to isolate from OMP variability
    torch.set_num_threads(num_threads)

    print(f"\n{'=' * 90}")
    print(f"Config: {config_name}  B={batch_size}  H_kv={num_heads_kv}  "
          f"H_q={num_heads_q}  D={head_dim}  max_S={max_seq_len}  "
          f"threads={num_threads}")
    print(f"Warmup={warmup}  Iterations={iterations}")
    print(f"{'=' * 90}")

    # Decode phase (seq_len=1) at various cache positions
    print("\n--- custom_sdpa: Decode (seq_len=1) ---")
    for start_pos in [0, 64, 256, 512, 1024]:
        if start_pos >= max_seq_len:
            continue
        results = bench_custom_sdpa(
            batch_size, num_heads_kv, num_heads_q, head_dim,
            max_seq_len, start_pos, seq_len=1,
            warmup=warmup, iterations=iterations,
        )
        print(format_results(f"start_pos={start_pos}", results))

    # Prefill phase (various seq_len) at start_pos=0
    print("\n--- custom_sdpa: Prefill (start_pos=0) ---")
    for seq_len in [32, 64, 128, 256, 512]:
        if seq_len >= max_seq_len:
            continue
        results = bench_custom_sdpa(
            batch_size, num_heads_kv, num_heads_q, head_dim,
            max_seq_len, start_pos=0, seq_len=seq_len,
            warmup=warmup, iterations=iterations,
        )
        print(format_results(f"seq_len={seq_len}", results))

    # update_cache: Decode (seq_len=1)
    print("\n--- update_cache: Decode (seq_len=1) ---")
    for start_pos in [0, 64, 256, 512, 1024]:
        if start_pos >= max_seq_len:
            continue
        results = bench_update_cache(
            batch_size, num_heads_kv, head_dim,
            max_seq_len, start_pos, seq_len=1,
            warmup=warmup, iterations=iterations,
        )
        print(format_results(f"start_pos={start_pos}", results))

    # update_cache: Prefill
    print("\n--- update_cache: Prefill (start_pos=0) ---")
    for seq_len in [32, 64, 128, 256, 512]:
        if seq_len >= max_seq_len:
            continue
        results = bench_update_cache(
            batch_size, num_heads_kv, head_dim,
            max_seq_len, start_pos=0, seq_len=seq_len,
            warmup=warmup, iterations=iterations,
        )
        print(format_results(f"seq_len={seq_len}", results))

    # Combined: update_cache + custom_sdpa (realistic attention step)
    print("\n--- Combined (update_cache + custom_sdpa): Decode ---")
    for start_pos in [0, 64, 256, 512, 1024]:
        if start_pos >= max_seq_len:
            continue
        results = bench_combined_update_and_sdpa(
            batch_size, num_heads_kv, num_heads_q, head_dim,
            max_seq_len, start_pos, seq_len=1,
            warmup=warmup, iterations=iterations,
        )
        print(format_results(f"start_pos={start_pos}", results))

    print("\n--- Combined (update_cache + custom_sdpa): Prefill ---")
    for seq_len in [32, 128]:
        if seq_len >= max_seq_len:
            continue
        results = bench_combined_update_and_sdpa(
            batch_size, num_heads_kv, num_heads_q, head_dim,
            max_seq_len, start_pos=0, seq_len=seq_len,
            warmup=warmup, iterations=iterations,
        )
        print(format_results(f"seq_len={seq_len}", results))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark transposed vs standard KV cache layout"
    )
    parser.add_argument(
        "--config", type=str, default="llama3_8b",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model configuration to benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--num-threads", type=int, default=1,
        help="Number of threads for torch operations",
    )
    parser.add_argument(
        "--all-configs", action="store_true",
        help="Run benchmarks for all model configurations",
    )
    args = parser.parse_args()

    if args.all_configs:
        for config_name in MODEL_CONFIGS:
            run_benchmarks(
                config_name, args.batch_size,
                args.warmup, args.iterations, args.num_threads,
            )
    else:
        run_benchmarks(
            args.config, args.batch_size,
            args.warmup, args.iterations, args.num_threads,
        )


if __name__ == "__main__":
    main()
