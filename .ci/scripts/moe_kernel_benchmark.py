"""Standalone MoE kernel benchmark for tuning block sizes.

Imports the actual Triton kernels from executorch.backends.cuda and sweeps
(BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages) on real Qwen3.5 MoE
dimensions with INT4 quantized weights.

GEMM1: M=1, N=1024 (2*intermediate), K=2048 (hidden), 8 experts
GEMM2: M=1, N=2048 (hidden), K=512 (intermediate), 8 experts
"""

import itertools
import time

import torch
import triton
import triton.language as tl
from executorch.backends.cuda.triton.kernels.fused_moe import (
    _fused_moe_kernel,
    _fused_moe_silu_kernel,
)

# Qwen3.5 MoE dimensions
HIDDEN = 2048
INTERMEDIATE = 512
NUM_EXPERTS = 256
TOP_K = 8
GROUP_SIZE = 128  # HQQ group size


def bench_gemm1(N, K, num_pairs, top_k, group_size, block_n, block_k, warps, stages):
    A = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randint(
        -128, 127, (NUM_EXPERTS, N, K // 2), dtype=torch.int8, device="cuda"
    )
    C = torch.empty(num_pairs, N, dtype=torch.bfloat16, device="cuda")
    B_scale = torch.randn(
        NUM_EXPERTS, N, K // group_size, dtype=torch.bfloat16, device="cuda"
    )
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (num_pairs,), dtype=torch.int64, device="cuda"
    )
    topk_weights = torch.randn(num_pairs, dtype=torch.float32, device="cuda")

    grid = (num_pairs * triton.cdiv(N, block_n),)

    def run():
        _fused_moe_kernel[grid](
            A,
            B,
            C,
            B_scale,
            topk_ids,
            topk_weights,
            N=N,
            K=K,
            num_token_expert_pairs=num_pairs,
            stride_am=A.stride(0),
            stride_ak=A.stride(1),
            stride_be=B.stride(0),
            stride_bk=B.stride(2),
            stride_bn=B.stride(1),
            stride_cm=C.stride(0),
            stride_cn=C.stride(1),
            stride_bse=B_scale.stride(0),
            stride_bsk=B_scale.stride(2),
            stride_bsn=B_scale.stride(1),
            group_size=group_size,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            MUL_ROUTED_WEIGHT=False,
            top_k=top_k,
            compute_type=tl.bfloat16,
            num_warps=warps,
            num_stages=stages,
        )

    # Warmup
    for _ in range(10):
        run()
    torch.cuda.synchronize()

    # Benchmark
    iters = 200
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6  # us


def bench_gemm2(N, K, num_pairs, top_k, group_size, block_n, block_k, warps, stages):
    A = torch.randn(num_pairs, 2 * K, dtype=torch.bfloat16, device="cuda")
    B = torch.randint(
        -128, 127, (NUM_EXPERTS, N, K // 2), dtype=torch.int8, device="cuda"
    )
    C = torch.empty(num_pairs, N, dtype=torch.bfloat16, device="cuda")
    B_scale = torch.randn(
        NUM_EXPERTS, N, K // group_size, dtype=torch.bfloat16, device="cuda"
    )
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (num_pairs,), dtype=torch.int64, device="cuda"
    )
    topk_weights = torch.randn(num_pairs, dtype=torch.float32, device="cuda")

    grid = (num_pairs * triton.cdiv(N, block_n),)

    def run():
        _fused_moe_silu_kernel[grid](
            A,
            B,
            C,
            B_scale,
            topk_ids,
            topk_weights,
            N=N,
            K=K,
            num_token_expert_pairs=num_pairs,
            stride_am=A.stride(0),
            stride_ak=A.stride(1),
            stride_be=B.stride(0),
            stride_bk=B.stride(2),
            stride_bn=B.stride(1),
            stride_cm=C.stride(0),
            stride_cn=C.stride(1),
            stride_bse=B_scale.stride(0),
            stride_bsk=B_scale.stride(2),
            stride_bsn=B_scale.stride(1),
            group_size=group_size,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            compute_type=tl.bfloat16,
            num_warps=warps,
            num_stages=stages,
        )

    for _ in range(10):
        run()
    torch.cuda.synchronize()

    iters = 200
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6


def main():
    N1 = 2 * INTERMEDIATE  # 1024
    K1 = HIDDEN  # 2048
    N2 = HIDDEN  # 2048
    K2 = INTERMEDIATE  # 512
    num_pairs = TOP_K  # 8

    # Search space (including small sizes 8, 16 per user request)
    block_ns = [8, 16, 32, 64, 128, 256]
    block_ks = [8, 16, 32, 64, 128, 256]
    warp_counts = [2, 4, 8]
    stage_counts = [2, 3, 4, 5]

    print(f"GEMM1: M=1, N={N1}, K={K1}, pairs={num_pairs}, group_size={GROUP_SIZE}")
    print(f"GEMM2: M=1, N={N2}, K={K2}, pairs={num_pairs}, group_size={GROUP_SIZE}")
    print()

    # GEMM1
    print("=== GEMM1 (_fused_moe_kernel) ===")
    print(f"{'N':>4} {'K':>4} {'warps':>5} {'stages':>6} {'time_us':>8}")
    best1 = (float("inf"), None)
    results1 = []
    for bn, bk, w, s in itertools.product(
        block_ns, block_ks, warp_counts, stage_counts
    ):
        if bk > K1 or bn > N1:
            continue
        try:
            t = bench_gemm1(N1, K1, num_pairs, TOP_K, GROUP_SIZE, bn, bk, w, s)
            results1.append((t, bn, bk, w, s))
            if t < best1[0]:
                best1 = (t, (bn, bk, w, s))
            print(f"{bn:>4} {bk:>4} {w:>5} {s:>6} {t:>8.1f}")
        except Exception as e:
            print(f"{bn:>4} {bk:>4} {w:>5} {s:>6} FAILED: {e}")

    print(f"\nBest GEMM1: {best1[1]} -> {best1[0]:.1f} us")

    # GEMM2
    print("\n=== GEMM2 (_fused_moe_silu_kernel) ===")
    print(f"{'N':>4} {'K':>4} {'warps':>5} {'stages':>6} {'time_us':>8}")
    best2 = (float("inf"), None)
    results2 = []
    for bn, bk, w, s in itertools.product(
        block_ns, block_ks, warp_counts, stage_counts
    ):
        if bk > K2 or bn > N2:
            continue
        try:
            t = bench_gemm2(N2, K2, num_pairs, TOP_K, GROUP_SIZE, bn, bk, w, s)
            results2.append((t, bn, bk, w, s))
            if t < best2[0]:
                best2 = (t, (bn, bk, w, s))
            print(f"{bn:>4} {bk:>4} {w:>5} {s:>6} {t:>8.1f}")
        except Exception as e:
            print(f"{bn:>4} {bk:>4} {w:>5} {s:>6} FAILED: {e}")

    print(f"\nBest GEMM2: {best2[1]} -> {best2[0]:.1f} us")

    # Summary — extract best configs
    t1_best, (bn1, bk1, w1, s1) = best1
    t2_best, (bn2, bk2, w2, s2) = best2

    print("\n=== SUMMARY ===")
    t1_base = bench_gemm1(N1, K1, num_pairs, TOP_K, GROUP_SIZE, 32, 32, 4, 2)
    t2_base = bench_gemm2(N2, K2, num_pairs, TOP_K, GROUP_SIZE, 32, 32, 4, 2)
    print(
        f"Baseline (32,32): GEMM1={t1_base:.1f}us, GEMM2={t2_base:.1f}us, "
        f"total={t1_base+t2_base:.1f}us"
    )
    print(
        f"Best GEMM1 ({bn1},{bk1},w{w1},s{s1}): {t1_best:.1f}us "
        f"({(1-t1_best/t1_base)*100:.1f}% faster)"
    )
    print(
        f"Best GEMM2 ({bn2},{bk2},w{w2},s{s2}): {t2_best:.1f}us "
        f"({(1-t2_best/t2_base)*100:.1f}% faster)"
    )

    if (bn1, bk1, w1, s1) != (bn2, bk2, w2, s2):
        t2_with_g1 = bench_gemm2(N2, K2, num_pairs, TOP_K, GROUP_SIZE, bn1, bk1, w1, s1)
        t1_with_g2 = bench_gemm1(N1, K1, num_pairs, TOP_K, GROUP_SIZE, bn2, bk2, w2, s2)
        unified_a = t1_best + t2_with_g1
        unified_b = t1_with_g2 + t2_best
        print(
            f"\nUnified option A (GEMM1-best {bn1},{bk1},w{w1},s{s1}): "
            f"GEMM1={t1_best:.1f}+GEMM2={t2_with_g1:.1f}={unified_a:.1f}us"
        )
        print(
            f"Unified option B (GEMM2-best {bn2},{bk2},w{w2},s{s2}): "
            f"GEMM1={t1_with_g2:.1f}+GEMM2={t2_best:.1f}={unified_b:.1f}us"
        )
        print(
            f"Separate configs: GEMM1={t1_best:.1f}+GEMM2={t2_best:.1f}"
            f"={t1_best+t2_best:.1f}us"
        )

    # Overall improvement
    total_base = t1_base + t2_base
    total_best = t1_best + t2_best
    print(
        f"\nOverall: baseline total={total_base:.1f}us, best total={total_best:.1f}us, "
        f"improvement={((1-total_best/total_base)*100):.1f}%"
    )

    results1.sort()
    results2.sort()
    print("\nTop 5 GEMM1:")
    for t, bn, bk, w, s in results1[:5]:
        print(f"  N={bn}, K={bk}, warps={w}, stages={s}: {t:.1f} us")
    print("\nTop 5 GEMM2:")
    for t, bn, bk, w, s in results2[:5]:
        print(f"  N={bn}, K={bk}, warps={w}, stages={s}: {t:.1f} us")


if __name__ == "__main__":
    main()
