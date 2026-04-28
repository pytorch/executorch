#!/usr/bin/env python3
"""Benchmark dedicated INT4 matvec kernels for M=1 decode."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def _int4_matvec_v1(
    X,
    W,
    Out,
    W_scale,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """V1: Each CTA computes BLOCK_N outputs, loops over K."""
    nb = tl.program_id(0)
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for ks in range(tl.cdiv(K, BLOCK_K)):
        km = offs_k < (K - ks * BLOCK_K)
        x_val = tl.load(X + ks * BLOCK_K + offs_k, mask=km, other=0.0).to(tl.float32)
        w_ptrs = (
            W
            + offs_n[:, None] * stride_wn
            + ((ks * BLOCK_K + offs_k[None, :]) // 2) * stride_wk
        )
        w_shift = ((ks * BLOCK_K + offs_k[None, :]) % 2) * 4
        w_raw = tl.load(w_ptrs, mask=nm[:, None] & km[None, :], other=0)
        w_uint4 = (w_raw >> w_shift) & 0xF

        gi = (ks * BLOCK_K) // group_size
        s_ptrs = W_scale + offs_n * stride_sn + gi * stride_sk
        scale = tl.load(s_ptrs, mask=nm, other=0.0).to(tl.float32)

        w_dq = (w_uint4.to(tl.float32) - 8.0) * scale[:, None]  # [BN, BK]
        acc += tl.sum(w_dq * x_val[None, :], axis=1)  # [BN]

        offs_k += BLOCK_K

    tl.store(Out + offs_n, acc.to(tl.bfloat16), mask=nm)


@triton.jit
def _int4_matvec_v2(
    X,
    W,
    Out,
    W_scale,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """V2: Transposed load — iterate N-first, K-second for better coalescing on W."""
    nb = tl.program_id(0)
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base = X
    w_base = W + offs_n[:, None] * stride_wn
    s_base = W_scale + offs_n * stride_sn

    for ks in range(tl.cdiv(K, BLOCK_K)):
        abs_k = ks * BLOCK_K + offs_k
        km = abs_k < K

        x_val = tl.load(x_base + abs_k, mask=km, other=0.0).to(tl.float32)

        w_ptrs = w_base + (abs_k[None, :] // 2) * stride_wk
        w_shift = (abs_k[None, :] % 2) * 4
        w_raw = tl.load(w_ptrs, mask=nm[:, None] & km[None, :], other=0)
        w_uint4 = (w_raw >> w_shift) & 0xF

        gi = (ks * BLOCK_K) // group_size
        scale = tl.load(s_base + gi * stride_sk, mask=nm, other=0.0).to(tl.float32)

        w_dq = (w_uint4.to(tl.float32) - 8.0) * scale[:, None]
        acc += tl.sum(w_dq * x_val[None, :], axis=1)

    tl.store(Out + offs_n, acc.to(tl.bfloat16), mask=nm)


@triton.jit
def _int4_matvec_splitk(
    X,
    W,
    Out,
    W_scale,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    group_size: tl.constexpr,
    K_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """V3: Split-K matvec — more CTAs, atomic accumulate."""
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    nb = pid // K_SPLITS
    kid = pid % K_SPLITS

    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = offs_n < N

    k_per_split = tl.cdiv(K, K_SPLITS)
    k_start = kid * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    num_steps = tl.cdiv(k_end - k_start, BLOCK_K)
    for step in range(num_steps):
        abs_k = k_start + step * BLOCK_K + offs_k
        km = abs_k < k_end

        x_val = tl.load(X + abs_k, mask=km, other=0.0).to(tl.float32)

        w_ptrs = W + offs_n[:, None] * stride_wn + (abs_k[None, :] // 2) * stride_wk
        w_shift = (abs_k[None, :] % 2) * 4
        w_raw = tl.load(w_ptrs, mask=nm[:, None] & km[None, :], other=0)
        w_uint4 = (w_raw >> w_shift) & 0xF

        gi = (k_start + step * BLOCK_K) // group_size
        scale = tl.load(
            W_scale + offs_n * stride_sn + gi * stride_sk, mask=nm, other=0.0
        ).to(tl.float32)

        w_dq = (w_uint4.to(tl.float32) - 8.0) * scale[:, None]
        acc += tl.sum(w_dq * x_val[None, :], axis=1)

    if K_SPLITS == 1:
        tl.store(Out + offs_n, acc.to(tl.bfloat16), mask=nm)
    else:
        tl.atomic_add(Out + offs_n, acc.to(tl.bfloat16), mask=nm)


def main():
    import executorch.backends.cuda.triton.kernels  # noqa: F401 — register ops
    import torch.nn as nn
    from executorch.extension.llm.export.quantize import quantize_model_
    from torchao.quantization.quant_primitives import (
        choose_qparams_affine,
        MappingType,
        quantize_affine,
    )

    gs = 128
    shapes = [
        (2048, 2048, "q/o_proj"),
        (12352, 2048, "shared_g+u"),
        (256, 2048, "k/v_proj"),
    ]

    for N, K, label in shapes:
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        sc, zp = choose_qparams_affine(
            w.float(),
            MappingType.SYMMETRIC,
            (1, gs),
            target_dtype=torch.int8,
            quant_min=-8,
            quant_max=7,
        )
        idata = quantize_affine(
            w.float(),
            (1, gs),
            sc,
            zp,
            output_dtype=torch.int8,
            quant_min=-8,
            quant_max=7,
        )
        u4 = (idata + 8).to(torch.int16)
        packed = (u4[:, 0::2] | (u4[:, 1::2] << 4)).to(torch.int8).cuda()
        w_scale = sc.reshape(N, -1).to(torch.bfloat16).cuda()

        linear = nn.Linear(K, N, bias=False, dtype=torch.bfloat16, device="cuda")
        wr = nn.ModuleDict({"linear": linear})
        quantize_model_(
            wr,
            qlinear_config="4w",
            qlinear_group_size=gs,
            qlinear_packing_format="tile_packed_to_4d",
        )
        tw = wr.linear.weight

        x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
        x_flat = x.squeeze(0)

        t_tiny = (
            do_bench(
                lambda: nn.functional.linear(x, tw),
                warmup=50,
                rep=200,
                return_mode="median",
            )
            * 1000
        )

        t_i4mm = (
            do_bench(
                lambda: torch.ops.triton.int4_matmul(x, packed, w_scale, gs),
                warmup=50,
                rep=200,
                return_mode="median",
            )
            * 1000
        )

        print(f"\n{'='*70}")
        print(
            f"[{N}x{K}] {label} — tinygemm={t_tiny:.1f}us, int4_matmul={t_i4mm:.1f}us"
        )
        print(f"{'='*70}")

        out = torch.empty(N, dtype=torch.bfloat16, device="cuda")
        best_t, best_cfg = float("inf"), ""

        # V1: basic matvec
        print("\n--- V1: basic matvec ---")
        for BN, BK, warps, stages in [
            (16, 128, 4, 3),
            (16, 256, 4, 3),
            (32, 128, 4, 3),
            (32, 256, 4, 3),
            (8, 128, 2, 3),
            (8, 256, 2, 3),
            (16, 128, 2, 3),
            (4, 128, 2, 3),
            (4, 256, 2, 3),
        ]:
            grid = ((N + BN - 1) // BN,)

            def run(_BN=BN, _BK=BK, _w=warps, _s=stages, _g=grid):
                _int4_matvec_v1[_g](
                    x_flat,
                    packed,
                    out,
                    w_scale,
                    N,
                    K,
                    packed.stride(0),
                    packed.stride(1),
                    w_scale.stride(0),
                    w_scale.stride(1),
                    gs,
                    BLOCK_N=_BN,
                    BLOCK_K=_BK,
                    num_warps=_w,
                    num_stages=_s,
                )

            try:
                run()
                t = do_bench(run, warmup=50, rep=200, return_mode="median") * 1000
                tag = " <<<" if t < best_t else ""
                if t < best_t:
                    best_t, best_cfg = t, f"v1 BN={BN} BK={BK} w={warps}"
                print(
                    f"  BN={BN:2d} BK={BK:3d} w={warps}: {t:6.1f}us ({t/t_tiny:.2f}x) grid={grid[0]:5d}{tag}"
                )
            except Exception as e:
                print(f"  BN={BN:2d} BK={BK:3d} w={warps}: FAIL {str(e)[:50]}")

        # V3: split-K matvec
        print("\n--- V3: split-K matvec ---")
        for BN, BK, splits, warps, stages in [
            (16, 128, 4, 4, 3),
            (16, 128, 8, 4, 3),
            (8, 128, 4, 2, 3),
            (8, 128, 8, 2, 3),
            (8, 128, 16, 2, 3),
            (4, 128, 4, 2, 3),
            (4, 128, 8, 2, 3),
            (4, 128, 16, 2, 3),
            (16, 64, 8, 4, 3),
            (8, 64, 16, 2, 3),
        ]:
            grid = (((N + BN - 1) // BN) * splits,)
            out_sk = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

            def run(_BN=BN, _BK=BK, _sp=splits, _w=warps, _s=stages, _g=grid):
                out_sk.zero_()
                _int4_matvec_splitk[_g](
                    x_flat,
                    packed,
                    out_sk,
                    w_scale,
                    N,
                    K,
                    packed.stride(0),
                    packed.stride(1),
                    w_scale.stride(0),
                    w_scale.stride(1),
                    gs,
                    K_SPLITS=_sp,
                    BLOCK_N=_BN,
                    BLOCK_K=_BK,
                    num_warps=_w,
                    num_stages=_s,
                )

            try:
                run()
                t = do_bench(run, warmup=50, rep=200, return_mode="median") * 1000
                tag = " <<<" if t < best_t else ""
                if t < best_t:
                    best_t, best_cfg = t, f"v3 BN={BN} BK={BK} sp={splits} w={warps}"
                print(
                    f"  BN={BN:2d} BK={BK:3d} sp={splits:2d} w={warps}: {t:6.1f}us ({t/t_tiny:.2f}x) grid={grid[0]:5d}{tag}"
                )
            except Exception as e:
                print(
                    f"  BN={BN:2d} BK={BK:3d} sp={splits:2d} w={warps}: FAIL {str(e)[:50]}"
                )

        print(f"\nBest: {best_t:.1f}us ({best_t/t_tiny:.2f}x tinygemm) — {best_cfg}")

        del wr, tw, packed, w_scale
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
