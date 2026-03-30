"""Standalone MoE kernel benchmark for tuning block sizes.

Sweeps (BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages) on the actual
Qwen3.5 MoE dimensions with INT4 quantized weights.

GEMM1: M=1, N=1024 (2*intermediate), K=2048 (hidden), 8 experts
GEMM2: M=1, N=2048 (hidden), K=512 (intermediate), 8 experts
"""

import torch
import triton
import triton.language as tl
import itertools

# Qwen3.5 MoE dimensions
HIDDEN = 2048
INTERMEDIATE = 512
NUM_EXPERTS = 256
TOP_K = 8
GROUP_SIZE = 128  # HQQ group size

# GEMM1: N=2*INTERMEDIATE=1024, K=HIDDEN=2048
# GEMM2: N=HIDDEN=2048, K=INTERMEDIATE=512


@triton.jit
def _fused_moe_kernel(
    A, B, C, B_scale, topk_ids, topk_weights,
    N: tl.constexpr, K: tl.constexpr,
    num_token_expert_pairs,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_bse, stride_bsk, stride_bsn,
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    pair_idx = pid // num_n_blocks
    n_block = pid % num_n_blocks
    if pair_idx >= num_token_expert_pairs:
        return
    expert_id = tl.load(topk_ids + pair_idx).to(tl.int64)
    token_idx = pair_idx // top_k
    offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + token_idx * stride_am + offs_k * stride_ak
    b_ptrs = B + expert_id * stride_be + (offs_k[:, None] // 2) * stride_bk + offs_n[None, :] * stride_bn
    b_shifter = (offs_k[:, None] % 2) * 4
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k_step in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k_step * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        b = (b >> b_shifter) & 0xF
        scale_ptrs = B_scale + expert_id * stride_bse + offs_n[None, :] * stride_bsn + ((offs_k[:, None] + BLOCK_SIZE_K * k_step) // group_size) * stride_bsk
        b_scale = tl.load(scale_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
        b_dequant = ((b.to(tl.float32) - 8.0) * b_scale).to(compute_type)
        acc += tl.sum(a[:, None].to(compute_type) * b_dequant, axis=0)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights + pair_idx)
        acc = acc * weight
    c_ptrs = C + pair_idx * stride_cm + offs_n * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=n_mask)


@triton.jit
def _fused_moe_silu_kernel(
    A, B, C, B_scale, topk_ids, topk_weights,
    N: tl.constexpr, K: tl.constexpr,
    num_token_expert_pairs,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_bse, stride_bsk, stride_bsn,
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    pair_idx = pid // num_n_blocks
    n_block = pid % num_n_blocks
    if pair_idx >= num_token_expert_pairs:
        return
    expert_id = tl.load(topk_ids + pair_idx).to(tl.int64)
    offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_gate_ptrs = A + pair_idx * stride_am + offs_k * stride_ak
    a_up_ptrs = a_gate_ptrs + K * stride_ak
    b_ptrs = B + expert_id * stride_be + (offs_k[:, None] // 2) * stride_bk + offs_n[None, :] * stride_bn
    b_shifter = (offs_k[:, None] % 2) * 4
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k_step in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k_step * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining
        gate = tl.load(a_gate_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        up = tl.load(a_up_ptrs, mask=k_mask, other=0.0)
        a = (gate * tl.sigmoid(gate) * up).to(compute_type)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        b = (b >> b_shifter) & 0xF
        scale_ptrs = B_scale + expert_id * stride_bse + offs_n[None, :] * stride_bsn + ((offs_k[:, None] + BLOCK_SIZE_K * k_step) // group_size) * stride_bsk
        b_scale = tl.load(scale_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
        b_dequant = ((b.to(tl.float32) - 8.0) * b_scale).to(compute_type)
        acc += tl.sum(a[:, None].to(compute_type) * b_dequant, axis=0)
        a_gate_ptrs += BLOCK_SIZE_K * stride_ak
        a_up_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
    weight = tl.load(topk_weights + pair_idx)
    acc = acc * weight
    c_ptrs = C + pair_idx * stride_cm + offs_n * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=n_mask)


def bench_gemm1(N, K, num_pairs, top_k, group_size, block_n, block_k, warps, stages):
    A = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randint(-128, 127, (NUM_EXPERTS, N, K // 2), dtype=torch.int8, device='cuda')
    C = torch.empty(num_pairs, N, dtype=torch.bfloat16, device='cuda')
    B_scale = torch.randn(NUM_EXPERTS, N, K // group_size, dtype=torch.bfloat16, device='cuda')
    topk_ids = torch.randint(0, NUM_EXPERTS, (num_pairs,), dtype=torch.int64, device='cuda')
    topk_weights = torch.randn(num_pairs, dtype=torch.float32, device='cuda')

    grid = (num_pairs * triton.cdiv(N, block_n),)

    def run():
        _fused_moe_kernel[grid](
            A, B, C, B_scale, topk_ids, topk_weights,
            N=N, K=K, num_token_expert_pairs=num_pairs,
            stride_am=A.stride(0), stride_ak=A.stride(1),
            stride_be=B.stride(0), stride_bk=B.stride(2), stride_bn=B.stride(1),
            stride_cm=C.stride(0), stride_cn=C.stride(1),
            stride_bse=B_scale.stride(0), stride_bsk=B_scale.stride(2), stride_bsn=B_scale.stride(1),
            group_size=group_size, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
            MUL_ROUTED_WEIGHT=False, top_k=top_k, compute_type=tl.bfloat16,
            num_warps=warps, num_stages=stages,
        )

    # Warmup
    for _ in range(10):
        run()
    torch.cuda.synchronize()

    # Benchmark
    import time
    iters = 200
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6  # us


def bench_gemm2(N, K, num_pairs, top_k, group_size, block_n, block_k, warps, stages):
    A = torch.randn(num_pairs, 2 * K, dtype=torch.bfloat16, device='cuda')
    B = torch.randint(-128, 127, (NUM_EXPERTS, N, K // 2), dtype=torch.int8, device='cuda')
    C = torch.empty(num_pairs, N, dtype=torch.bfloat16, device='cuda')
    B_scale = torch.randn(NUM_EXPERTS, N, K // group_size, dtype=torch.bfloat16, device='cuda')
    topk_ids = torch.randint(0, NUM_EXPERTS, (num_pairs,), dtype=torch.int64, device='cuda')
    topk_weights = torch.randn(num_pairs, dtype=torch.float32, device='cuda')

    grid = (num_pairs * triton.cdiv(N, block_n),)

    def run():
        _fused_moe_silu_kernel[grid](
            A, B, C, B_scale, topk_ids, topk_weights,
            N=N, K=K, num_token_expert_pairs=num_pairs,
            stride_am=A.stride(0), stride_ak=A.stride(1),
            stride_be=B.stride(0), stride_bk=B.stride(2), stride_bn=B.stride(1),
            stride_cm=C.stride(0), stride_cn=C.stride(1),
            stride_bse=B_scale.stride(0), stride_bsk=B_scale.stride(2), stride_bsn=B_scale.stride(1),
            group_size=group_size, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
            compute_type=tl.bfloat16,
            num_warps=warps, num_stages=stages,
        )

    for _ in range(10):
        run()
    torch.cuda.synchronize()

    import time
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
    K1 = HIDDEN            # 2048
    N2 = HIDDEN            # 2048
    K2 = INTERMEDIATE      # 512
    num_pairs = TOP_K      # 8

    # Search space
    block_ns = [32, 64, 128, 256]
    block_ks = [32, 64, 128, 256]
    warp_counts = [2, 4, 8]
    stage_counts = [2, 3, 4, 5]

    print(f"GEMM1: M=1, N={N1}, K={K1}, pairs={num_pairs}, group_size={GROUP_SIZE}")
    print(f"GEMM2: M=1, N={N2}, K={K2}, pairs={num_pairs}, group_size={GROUP_SIZE}")
    print()

    # GEMM1
    print("=== GEMM1 (_fused_moe_kernel) ===")
    print(f"{'N':>4} {'K':>4} {'warps':>5} {'stages':>6} {'time_us':>8}")
    best1 = (float('inf'), None)
    results1 = []
    for bn, bk, w, s in itertools.product(block_ns, block_ks, warp_counts, stage_counts):
        # Skip invalid: K must be divisible, and block_k must divide group_size evenly or vice versa
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
    best2 = (float('inf'), None)
    results2 = []
    for bn, bk, w, s in itertools.product(block_ns, block_ks, warp_counts, stage_counts):
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

    # Summary
    print("\n=== SUMMARY ===")
    # Baseline (N=32, K=32, default warps/stages)
    t1_base = bench_gemm1(N1, K1, num_pairs, TOP_K, GROUP_SIZE, 32, 32, 4, 2)
    t2_base = bench_gemm2(N2, K2, num_pairs, TOP_K, GROUP_SIZE, 32, 32, 4, 2)
    bn1, bk1, w1, s1 = best1[1]
    bn2, bk2, w2, s2 = best2[1]
    t1_best = best1[0]
    t2_best = best2[0]
    print(f"Baseline (32,32): GEMM1={t1_base:.1f}us, GEMM2={t2_base:.1f}us, total={t1_base+t2_base:.1f}us")
    print(f"Best GEMM1 ({bn1},{bk1},w{w1},s{s1}): {t1_best:.1f}us ({(1-t1_best/t1_base)*100:.1f}% faster)")
    print(f"Best GEMM2 ({bn2},{bk2},w{w2},s{s2}): {t2_best:.1f}us ({(1-t2_best/t2_base)*100:.1f}% faster)")

    # If GEMM1 and GEMM2 best configs differ, also show a unified config
    if (bn1, bk1, w1, s1) != (bn2, bk2, w2, s2):
        # Try best GEMM1 config on GEMM2 and vice versa
        t2_with_g1 = bench_gemm2(N2, K2, num_pairs, TOP_K, GROUP_SIZE, bn1, bk1, w1, s1)
        t1_with_g2 = bench_gemm1(N1, K1, num_pairs, TOP_K, GROUP_SIZE, bn2, bk2, w2, s2)
        unified_a = t1_best + t2_with_g1
        unified_b = t1_with_g2 + t2_best
        print(f"\nUnified option A (GEMM1-best {bn1},{bk1},w{w1},s{s1}): GEMM1={t1_best:.1f}+GEMM2={t2_with_g1:.1f}={unified_a:.1f}us")
        print(f"Unified option B (GEMM2-best {bn2},{bk2},w{w2},s{s2}): GEMM1={t1_with_g2:.1f}+GEMM2={t2_best:.1f}={unified_b:.1f}us")
        print(f"Separate configs: GEMM1={t1_best:.1f}+GEMM2={t2_best:.1f}={t1_best+t2_best:.1f}us")

    # Top 5 for each
    results1.sort()
    results2.sort()
    print("\nTop 5 GEMM1:")
    for t, bn, bk, w, s in results1[:5]:
        print(f"  N={bn}, K={bk}, warps={w}, stages={s}: {t:.1f} us")
    print("\nTop 5 GEMM2:")
    for t, bn, bk, w, s in results2[:5]:
        print(f"  N={bn}, K={bk}, warps={w}, stages={s}: {t:.1f} us")


if __name__ == '__main__':
    main()
