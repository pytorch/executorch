#!/usr/bin/env python3
"""Benchmark INT4 matmul strategies for M=1 decode."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


# Strategy 1: tl.dot with BLOCK_M=16 padding (current approach)
@triton.jit
def _int4_dot_kernel(
    A,
    B,
    C,
    B_scale,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bsn,
    stride_bsk,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_SIZE_N)
    mb = pid // num_n
    nb = pid % num_n
    offs_m = mb * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = nb * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mm = offs_m < M
    nm = offs_n < N
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_n[None, :] * stride_bn + (offs_k[:, None] // 2) * stride_bk
    b_shift = (offs_k[:, None] % 2) * 4
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ks in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        kr = K - ks * BLOCK_SIZE_K
        km = offs_k < kr
        a = tl.load(a_ptrs, mask=mm[:, None] & km[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=km[:, None] & nm[None, :], other=0)
        b = (b >> b_shift) & 0xF
        gi = (BLOCK_SIZE_K * ks) // group_size
        sp = B_scale + offs_n[None, :] * stride_bsn + gi * stride_bsk
        bs = tl.load(sp, mask=nm[None, :], other=0.0).to(tl.float32)
        bd = ((b.to(tl.float32) - 8.0) * bs).to(tl.bfloat16)
        acc += tl.dot(a.to(tl.bfloat16), bd)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mm[:, None] & nm[None, :])


# Strategy 2: vec-mat with tl.sum (no tl.dot, no M padding waste)
@triton.jit
def _int4_vecmat_kernel(
    A,
    B,
    C,
    B_scale,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_bn,
    stride_bk,
    stride_bsn,
    stride_bsk,
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    nb = tl.program_id(0)
    offs_n = nb * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    nm = offs_n < N
    b_ptrs = B + offs_n[None, :] * stride_bn + (offs_k[:, None] // 2) * stride_bk
    b_shift = (offs_k[:, None] % 2) * 4
    a_ptrs = A + offs_k
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for ks in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        kr = K - ks * BLOCK_SIZE_K
        km = offs_k < kr
        a = tl.load(a_ptrs, mask=km, other=0.0).to(tl.float32)  # [BK]
        b = tl.load(b_ptrs, mask=km[:, None] & nm[None, :], other=0)
        b = (b >> b_shift) & 0xF
        gi = (BLOCK_SIZE_K * ks) // group_size
        sp = B_scale + offs_n * stride_bsn + gi * stride_bsk
        bs = tl.load(sp, mask=nm, other=0.0).to(tl.float32)  # [BN]
        bd = (b.to(tl.float32) - 8.0) * bs[None, :]  # [BK, BN]
        acc += tl.sum(a[:, None] * bd, axis=0)  # [BN]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
    c_ptrs = C + offs_n
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=nm)


# Strategy 3: split-K with tl.dot — more CTAs, then atomic reduce
@triton.jit
def _int4_splitk_kernel(
    A,
    B,
    C,
    B_scale,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bsn,
    stride_bsk,
    group_size: tl.constexpr,
    K_SPLITS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_nk = num_n * K_SPLITS
    mb = pid // num_nk
    nk = pid % num_nk
    nb = nk // K_SPLITS
    ks_id = nk % K_SPLITS

    offs_m = mb * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = nb * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mm = offs_m < M
    nm = offs_n < N

    k_per_split = tl.cdiv(K, K_SPLITS)
    k_start = ks_id * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    a_ptrs = A + offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak
    b_ptrs = (
        B + offs_n[None, :] * stride_bn + ((k_start + offs_k[:, None]) // 2) * stride_bk
    )
    b_shift = ((k_start + offs_k[:, None]) % 2) * 4

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_steps = tl.cdiv(k_end - k_start, BLOCK_SIZE_K)
    for step in range(0, num_steps):
        abs_k = k_start + step * BLOCK_SIZE_K + offs_k
        km = abs_k < k_end
        a = tl.load(a_ptrs, mask=mm[:, None] & km[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=km[:, None] & nm[None, :], other=0)
        b = (b >> b_shift) & 0xF
        gi = (k_start + step * BLOCK_SIZE_K) // group_size
        sp = B_scale + offs_n[None, :] * stride_bsn + gi * stride_bsk
        bs = tl.load(sp, mask=nm[None, :], other=0.0).to(tl.float32)
        bd = ((b.to(tl.float32) - 8.0) * bs).to(tl.bfloat16)
        acc += tl.dot(a.to(tl.bfloat16), bd)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        b_shift = (offs_k[:, None] % 2) * 4  # reset shift after first step

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if K_SPLITS == 1:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mm[:, None] & nm[None, :])
    else:
        tl.atomic_add(c_ptrs, acc.to(tl.bfloat16), mask=mm[:, None] & nm[None, :])


def main():
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
        t_tiny = (
            do_bench(
                lambda: nn.functional.linear(x, tw),
                warmup=50,
                rep=200,
                return_mode="median",
            )
            * 1000
        )

        print(f"\n{'='*70}")
        print(f"[{N}x{K}] {label} — M=1, tinygemm={t_tiny:.1f}us")
        print(f"{'='*70}")

        # Strategy 1: tl.dot with various configs
        print("\n--- Strategy 1: tl.dot (BLOCK_M=16 padding) ---")
        out = torch.empty(1, N, dtype=torch.bfloat16, device="cuda")
        for BN, BK, warps, stages in [
            (16, 128, 4, 5),
            (32, 128, 4, 5),
            (32, 256, 4, 3),
            (16, 128, 2, 5),
            (32, 128, 2, 5),
        ]:
            grid = ((N + BN - 1) // BN,)

            def run(_BN=BN, _BK=BK, _w=warps, _s=stages, _g=grid):
                _int4_dot_kernel[_g](
                    x,
                    packed,
                    out,
                    w_scale,
                    1,
                    N,
                    K,
                    x.stride(0),
                    x.stride(1),
                    packed.stride(0),
                    packed.stride(1),
                    out.stride(0),
                    out.stride(1),
                    w_scale.stride(0),
                    w_scale.stride(1),
                    gs,
                    BLOCK_SIZE_M=16,
                    BLOCK_SIZE_N=_BN,
                    BLOCK_SIZE_K=_BK,
                    num_warps=_w,
                    num_stages=_s,
                )

            try:
                run()
                t = do_bench(run, warmup=50, rep=200, return_mode="median") * 1000
                print(
                    f"  BN={BN:3d} BK={BK:3d} w={warps} s={stages}: {t:6.1f}us ({t/t_tiny:.2f}x) grid={grid[0]}"
                )
            except Exception as e:
                print(
                    f"  BN={BN:3d} BK={BK:3d} w={warps} s={stages}: FAIL {str(e)[:50]}"
                )

        # Strategy 2: vec-mat with tl.sum (no padding waste)
        print("\n--- Strategy 2: vec-mat tl.sum (no M padding) ---")
        for BN, BK, warps, stages in [
            (16, 128, 4, 5),
            (32, 128, 4, 5),
            (64, 128, 4, 5),
            (16, 256, 4, 3),
            (32, 256, 4, 3),
            (16, 128, 2, 5),
            (32, 128, 2, 5),
            (16, 64, 2, 5),
            (32, 64, 2, 5),
        ]:
            grid = ((N + BN - 1) // BN,)
            out1d = torch.empty(N, dtype=torch.bfloat16, device="cuda")

            def run(_BN=BN, _BK=BK, _w=warps, _s=stages, _g=grid):
                _int4_vecmat_kernel[_g](
                    x,
                    packed,
                    out1d,
                    w_scale,
                    N,
                    K,
                    packed.stride(0),
                    packed.stride(1),
                    w_scale.stride(0),
                    w_scale.stride(1),
                    gs,
                    BLOCK_SIZE_N=_BN,
                    BLOCK_SIZE_K=_BK,
                    num_warps=_w,
                    num_stages=_s,
                )

            try:
                run()
                t = do_bench(run, warmup=50, rep=200, return_mode="median") * 1000
                print(
                    f"  BN={BN:3d} BK={BK:3d} w={warps} s={stages}: {t:6.1f}us ({t/t_tiny:.2f}x) grid={grid[0]}"
                )
            except Exception as e:
                print(
                    f"  BN={BN:3d} BK={BK:3d} w={warps} s={stages}: FAIL {str(e)[:50]}"
                )

        # Strategy 3: split-K with tl.dot
        print("\n--- Strategy 3: split-K tl.dot ---")
        for BN, BK, splits, warps, stages in [
            (32, 128, 4, 4, 3),
            (32, 128, 8, 4, 3),
            (32, 128, 16, 4, 3),
            (16, 128, 4, 4, 3),
            (16, 128, 8, 4, 3),
            (16, 128, 16, 4, 3),
            (64, 128, 4, 4, 3),
            (64, 128, 8, 4, 3),
        ]:
            grid = (((N + BN - 1) // BN) * splits,)
            out_sk = torch.zeros(1, N, dtype=torch.bfloat16, device="cuda")

            def run(_BN=BN, _BK=BK, _sp=splits, _w=warps, _s=stages, _g=grid):
                out_sk.zero_()
                _int4_splitk_kernel[_g](
                    x,
                    packed,
                    out_sk,
                    w_scale,
                    1,
                    N,
                    K,
                    x.stride(0),
                    x.stride(1),
                    packed.stride(0),
                    packed.stride(1),
                    out_sk.stride(0),
                    out_sk.stride(1),
                    w_scale.stride(0),
                    w_scale.stride(1),
                    gs,
                    K_SPLITS=_sp,
                    BLOCK_SIZE_M=16,
                    BLOCK_SIZE_N=_BN,
                    BLOCK_SIZE_K=_BK,
                    num_warps=_w,
                    num_stages=_s,
                )

            try:
                run()
                t = do_bench(run, warmup=50, rep=200, return_mode="median") * 1000
                print(
                    f"  BN={BN:3d} BK={BK:3d} sp={splits:2d} w={warps} s={stages}: {t:6.1f}us ({t/t_tiny:.2f}x) grid={grid[0]}"
                )
            except Exception as e:
                print(
                    f"  BN={BN:3d} BK={BK:3d} sp={splits:2d} w={warps} s={stages}: FAIL {str(e)[:50]}"
                )

        del wr, tw, packed, w_scale
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
