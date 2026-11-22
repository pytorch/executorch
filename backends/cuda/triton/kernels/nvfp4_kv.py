# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 KV cache for the CUDA backend: fused write + attention that reads
NVFP4 K/V directly.  There is no standalone quantize or dequantize pass.

Cache layout (per layer, per K and V), head_dim D (power of two, >= 32):

    q : uint8   [B, H_kv, L, D/2]   E2M1 codes, two per byte, low nibble first
                                     (element 2m in the low nibble of byte m)
    s : uint8   [B, H_kv, L, D/16]  E4M3 block scale per 16 consecutive
                                     head_dim elements (bit pattern)

    x[d] ~= e2m1(q[d]) * e4m3(s[d // 16])

The NVFP4 global scale is fixed at 1.0 (as in SGLang/flashinfer and
DeepSeek-V4.1): ``s = e4m3(min(amax(|block|) / 6, 448))`` and the codes
saturate at +-6, so magnitudes above 6 * 448 = 2688 clip to 2688.  Storage is
4.5 bits / element (D=128: 72 bytes per token-head vs 256 for BF16, 28.1%).

Kernels
  * ``triton::nvfp4_kv_write``  -- fused (optional) K RMS-norm + (optional)
    interleaved RoPE + NVFP4 quantization, written straight into the cache slot
    (``pos`` or ``pos % ring`` for sliding-window ring buffers).  Mutates the
    four cache tensors in place; replaces the BF16 ``index_copy_``.
  * ``triton::nvfp4_sdpa``      -- attention over the NVFP4 cache:
      - L_q == 1           : split-K flash decoding (+ reduce)
      - 2 <= L_q <= 16     : small-query split-K (statically known L_q)
      - otherwise          : chunked-prefill flash attention.
    Every path decodes K and V exactly from E2M1 x E4M3 to FP16 in registers
    (the products have <= 6 significant bits and fit FP16), converts the BF16
    query to FP16, and runs Q.K^T and P.V on FP16 MMAs with FP32 accumulation.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from executorch.backends.cuda.autotune.inputs import autotune_inputs
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    _e4m3_to_f32,
    _SOFT_E2M1_HALF2,
)
from executorch.backends.cuda.triton.kernels.sdpa import (
    _decode_splitk_config,
    _kv_len_scenarios,
    _next_power_of_2_unclamped,
    _sdpa_small_query_reduce_kernel,
    _SMALL_QUERY_MAX_LQ,
    _visible_mask_scenarios,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice


# ---------------------------------------------------------------------------
# helpers
#
# E4M3 conversions are native on sm_89+ and E2M1 on sm_100+; older GPUs (A100)
# take the bit-exact software paths.  Kernels receive the current device's
# compute capability as the ARCH constexpr: tl.target_info queries the current
# CUDA device while compiling, which fails in Inductor's forked compile workers.
# ---------------------------------------------------------------------------
def cuda_arch() -> int:
    if torch.version.hip is not None:
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@triton.jit
def _e2m1_code(v):
    """fp32 -> E2M1 code, round to nearest even, saturating, sign from the sign
    bit (so tiny negatives give -0 like ``cvt.rn.satfinite``)."""
    a = tl.abs(v)
    c = tl.where(a > 0.25, 1, 0)
    c = tl.where(a >= 0.75, 2, c)
    c = tl.where(a > 1.25, 3, c)
    c = tl.where(a >= 1.75, 4, c)
    c = tl.where(a > 2.5, 5, c)
    c = tl.where(a >= 3.5, 6, c)
    c = tl.where(a > 5.0, 7, c)
    sign = (v.to(tl.uint32, bitcast=True) >> 28) & 8
    return c.to(tl.uint32) | sign


@triton.jit
def _e2m1x2(lo, hi, ARCH: tl.constexpr):
    """Pack two fp32 tensors into E2M1 pairs (lo -> low nibble), RN, satfinite."""
    if ARCH >= 100:
        return tl.inline_asm_elementwise(
            "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; mov.b32 $0, {tmp,0,0,0}; }",
            constraints="=r,f,f",
            args=[lo, hi],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        ).to(tl.uint8)
    else:
        return (_e2m1_code(lo) | (_e2m1_code(hi) << 4)).to(tl.uint8)


@triton.jit
def _e4m3_encode_nonneg(x, ARCH: tl.constexpr):
    """fp32 in [0, 448] -> E4M3 byte, round to nearest even (``cvt.rn.satfinite``)."""
    if ARCH >= 89:
        return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        u = x.to(tl.uint32, bitcast=True)
        normal = ((u + 0x7FFFF + ((u >> 20) & 1)) >> 20).to(tl.int32) - (120 << 3)
        sub = libdevice.rint(x * 512.0).to(tl.int32)
        code = tl.where(x < 0.015625, sub, normal)
        return tl.minimum(code, 0x7E).to(tl.uint8)


@triton.jit
def _load_kv_tile_bf16(
    Q_ptr,
    S_ptr,
    base_q,
    base_s,
    offs_n,
    n_valid,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    ARCH: tl.constexpr,
    SF_GATHER: tl.constexpr = False,
):
    """Load a [BLOCK_N, D] NVFP4 tile and decode it in registers.

    Returns (lo, hi) as FP16 [BLOCK_N, D/2]: lo[:, m] = x[:, 2m] and
    hi[:, m] = x[:, 2m+1].  e2m1 * e4m3 has <= 6 significant bits and a
    nonzero magnitude range [2**-10, 2688], so the FP16 values are exact.  The
    decode stays in packed f16x2 (one cvt + one mul per byte).
    """
    half = tl.arange(0, D // 2)
    grp = tl.arange(0, D // 16)
    q = tl.load(
        Q_ptr + base_q + offs_n[:, None] * (D // 2) + half[None, :],
        mask=n_valid[:, None],
        other=0,
    )
    sf_idx = (half // 8)[None, :] if SF_GATHER else grp[None, :]
    sf_raw = tl.load(
        S_ptr + base_s + offs_n[:, None] * (D // 16) + sf_idx,
        mask=n_valid[:, None],
        other=0,
    )
    if ARCH >= 89:
        sf16 = sf_raw.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    else:
        sf16 = _e4m3_to_f32(sf_raw, ARCH >= 89).to(tl.float16)
    sf_bits = sf16.to(tl.uint16, bitcast=True).to(tl.uint32)
    sf_pair = sf_bits | (sf_bits << 16)
    if not SF_GATHER:
        # [BLOCK_N, D/16] -> [BLOCK_N, D/2]: broadcast each scale over 8 bytes.
        sf_pair = tl.broadcast_to(sf_pair[:, :, None], (BLOCK_N, D // 16, 8)).reshape(
            (BLOCK_N, D // 2)
        )
    if ARCH >= 100:
        prod = tl.inline_asm_elementwise(
            "{ .reg .b8 tmp,z1,z2,z3; .reg .b32 h; mov.b32 {tmp,z1,z2,z3}, $1; "
            "cvt.rn.f16x2.e2m1x2 h, tmp; mul.rn.f16x2 $0, h, $2; }",
            constraints="=r,r,r",
            args=[q.to(tl.uint32), sf_pair],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
    else:
        # Same exact products as the native path: FP16 E2M1 * 2**-14, then
        # * 2**14 and * scale in two packed multiplies.
        u = q.to(tl.uint32)
        bits = (
            ((u & 8) << 12) | ((u & 7) << 9) | ((u & 0x80) << 24) | ((u & 0x70) << 21)
        )
        prod = tl.inline_asm_elementwise(
            "{ .reg .b32 h; mul.rn.f16x2 h, $1, $2; mul.rn.f16x2 $0, h, $3; }",
            constraints="=r,r,r,r",
            args=[bits, tl.full(bits.shape, _SOFT_E2M1_HALF2, tl.uint32), sf_pair],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
    lo = (prod & 65535).to(tl.uint16).to(tl.float16, bitcast=True)
    hi = (prod >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    return lo, hi


# ---------------------------------------------------------------------------
# fused write: [k-norm] + [RoPE] + NVFP4 quantize + cache store
# ---------------------------------------------------------------------------
@triton.jit
def _quantize_row_store(
    even, odd, Q_ptr, S_ptr, row, in_range, D: tl.constexpr, ARCH: tl.constexpr
):
    """Quantize one head row given as even/odd fp32 halves [D/2] and store it."""
    e2 = tl.reshape(even, (D // 16, 8))
    o2 = tl.reshape(odd, (D // 16, 8))
    bmax = tl.maximum(tl.max(tl.abs(e2), 1), tl.max(tl.abs(o2), 1))  # [D/16]
    # IEEE division: bf16 values over E4M3 scales often land exactly on an E2M1
    # rounding tie, which div.full.f32 (Triton's ``/``) does not resolve to even.
    sf_bits = _e4m3_encode_nonneg(tl.minimum(tl.div_rn(bmax, 6.0), 448.0), ARCH)
    sff = _e4m3_to_f32(sf_bits, ARCH >= 89)
    den = tl.where(sff > 0, sff, 1.0)[:, None]
    qb = _e2m1x2(tl.div_rn(e2, den), tl.div_rn(o2, den), ARCH)  # [D/16, 8]
    qb = tl.reshape(qb, (D // 2,))
    tl.store(Q_ptr + row * (D // 2) + tl.arange(0, D // 2), qb, mask=in_range)
    tl.store(S_ptr + row * (D // 16) + tl.arange(0, D // 16), sf_bits, mask=in_range)


@triton.jit
def _nvfp4_kv_write_kernel(
    K_ptr,
    V_ptr,
    POS_ptr,
    INVF_ptr,
    KQ,
    KS,
    VQ,
    VS,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    H,
    L,
    eps: tl.float32,
    RING: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    NORM_K: tl.constexpr,
    D: tl.constexpr,
    ARCH: tl.constexpr,
):
    t = tl.program_id(0)
    bh = tl.program_id(1)
    eps = eps.to(tl.float32)  # AOTI passes Python floats as f64
    b = bh // H
    h = bh % H
    pos = tl.load(POS_ptr + t)
    if RING:
        slot = pos % L
    else:
        slot = pos
    # Negative positions, and positions past the end of a global cache, are
    # dropped.
    in_range = (pos >= 0) & (slot < L)
    row = (b * H + h).to(tl.int64) * L + slot
    half = tl.arange(0, D // 2)

    kbase = K_ptr + b * stride_kb + t * stride_kt + h * stride_kh
    ke = tl.load(kbase + (2 * half) * stride_kd).to(tl.float32)
    ko = tl.load(kbase + (2 * half + 1) * stride_kd).to(tl.float32)
    if NORM_K:
        ms = (tl.sum(ke * ke, 0) + tl.sum(ko * ko, 0)) / D
        r = 1.0 / tl.sqrt(ms + eps)
        ke = ke * r
        ko = ko * r
    if HAS_ROPE:
        inv = tl.load(INVF_ptr + half)
        ang = pos.to(tl.float32) * inv
        c = tl.cos(ang)
        s = tl.sin(ang)
        r1 = ke * c - ko * s
        r2 = ke * s + ko * c
        ke = r1
        ko = r2
    _quantize_row_store(ke, ko, KQ, KS, row, in_range, D, ARCH)

    vbase = V_ptr + b * stride_vb + t * stride_vt + h * stride_vh
    ve = tl.load(vbase + (2 * half) * stride_vd).to(tl.float32)
    vo = tl.load(vbase + (2 * half + 1) * stride_vd).to(tl.float32)
    _quantize_row_store(ve, vo, VQ, VS, row, in_range, D, ARCH)


def _nvfp4_kv_write_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    k_q: torch.Tensor,
    k_s: torch.Tensor,
    v_q: torch.Tensor,
    v_s: torch.Tensor,
    inv_freq: Optional[torch.Tensor] = None,
    ring: bool = False,
    k_norm_eps: float = -1.0,
) -> None:
    """Launch body of ``triton::nvfp4_kv_write``.

    Also registered as the op's decomposition in the CUDA backend: inside
    AOTInductor the in-place op is retraced to the user Triton kernel that
    mutates the cache buffers, so Inductor compiles it (no fallback kernel).
    """
    B, T, H, D = k.shape
    L = k_q.shape[2]
    has_rope = inv_freq is not None
    wrap_triton(_nvfp4_kv_write_kernel)[(T, B * H)](
        k,
        v,
        input_pos,
        inv_freq if has_rope else 0,
        k_q,
        k_s,
        v_q,
        v_s,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        H,
        L,
        float(k_norm_eps),
        RING=ring,
        HAS_ROPE=has_rope,
        NORM_K=k_norm_eps > 0,
        D=D,
        ARCH=cuda_arch(),
        num_warps=1,
    )


@triton_op(
    "triton::nvfp4_kv_write",
    mutates_args={"k_q", "k_s", "v_q", "v_s"},
)
def nvfp4_kv_write(
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    k_q: torch.Tensor,
    k_s: torch.Tensor,
    v_q: torch.Tensor,
    v_s: torch.Tensor,
    inv_freq: Optional[torch.Tensor] = None,
    ring: bool = False,
    k_norm_eps: float = -1.0,
) -> None:
    """Write new K/V tokens into an NVFP4 cache.

    k, v: [B, T, H_kv, D] (any strides; bf16/fp16/fp32), *pre* k-norm/RoPE when
    ``k_norm_eps > 0`` / ``inv_freq`` are given (those are fused here).
    input_pos: [T] absolute positions.  ring: store at ``pos % L``.  Tokens at
    negative positions, or at ``pos >= L`` without ``ring``, are not written.
    """
    _nvfp4_kv_write_impl(
        k, v, input_pos, k_q, k_s, v_q, v_s, inv_freq, ring, k_norm_eps
    )


@nvfp4_kv_write.register_fake
def _nvfp4_kv_write_fake(
    k,
    v,
    input_pos,
    k_q,
    k_s,
    v_q,
    v_s,
    inv_freq=None,
    ring=False,
    k_norm_eps=-1.0,
) -> None:
    return None


# ---------------------------------------------------------------------------
# split-K attention (decode L_q == 1 and small query blocks)
# ---------------------------------------------------------------------------
@autotune_inputs(KV_LEN_ptr=_kv_len_scenarios, Mask_ptr=_visible_mask_scenarios)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=2),
    ],
    key=["Lk", "HEAD_DIM", "NUM_GROUPS", "LQ", "HAS_MASK", "MASK_IS_CAUSAL", "WINDOW"],
)
@triton.jit
def _nvfp4_splitk_kernel(
    Q_ptr,
    KQ,
    KS,
    VQ,
    VS,
    O_partial_ptr,
    M_partial_ptr,
    L_partial_ptr,
    Mask_ptr,
    KV_LEN_ptr,
    H_kv,
    Lk,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_op_s,
    stride_op_b,
    stride_op_h,
    stride_op_m,
    stride_mp_s,
    stride_mp_b,
    stride_mp_h,
    stride_mp_m,
    stride_mb,
    stride_mq,
    stride_mk,
    sm_scale: tl.float32,
    chunk_size,
    HAS_MASK: tl.constexpr,
    HAS_KV_LEN: tl.constexpr,
    MASK_IS_CAUSAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    LQ: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ARCH: tl.constexpr,
    WINDOW: tl.constexpr = 0,
):
    D: tl.constexpr = HEAD_DIM
    sm_scale = sm_scale.to(tl.float32)
    split_id = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H_kv
    h_kv = pid_bh % H_kv

    if HAS_KV_LEN:
        kv_len = tl.load(KV_LEN_ptr)
    else:
        kv_len = Lk
    if WINDOW > 0:
        # Sliding window over a ring of Lk slots: sweep only the absolute
        # positions some query row can see and that the ring still holds.
        range_lo = tl.maximum(tl.maximum(kv_len - LQ - WINDOW + 1, kv_len - Lk), 0)
    else:
        range_lo = 0
    start_n = range_lo + split_id * chunk_size
    end_n = tl.minimum(start_n + chunk_size, kv_len)

    offs_m = tl.arange(0, BLOCK_M)
    query_rows = offs_m // BLOCK_G
    group_heads = offs_m % BLOCK_G
    row_valid = (query_rows < LQ) & (group_heads < NUM_GROUPS)
    h_q = h_kv * NUM_GROUPS + group_heads
    half = tl.arange(0, D // 2)
    q_base = (
        Q_ptr
        + b * stride_qb
        + h_q[:, None] * stride_qh
        + query_rows[:, None] * stride_qm
    )
    q_even = tl.load(
        q_base + (2 * half[None, :]) * stride_qd, mask=row_valid[:, None], other=0.0
    ).to(tl.float16)
    q_odd = tl.load(
        q_base + (2 * half[None, :] + 1) * stride_qd, mask=row_valid[:, None], other=0.0
    ).to(tl.float16)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_lo = tl.zeros([BLOCK_M, D // 2], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_M, D // 2], dtype=tl.float32)

    head_row = (b * H_kv + h_kv).to(tl.int64) * Lk
    base_q = head_row * (D // 2)
    base_s = head_row * (D // 16)
    offs_n_init = tl.arange(0, BLOCK_N)

    for tile_start in tl.range(start_n, end_n, BLOCK_N):
        offs_n = tile_start + offs_n_init
        n_valid = offs_n < end_n
        if WINDOW > 0:
            slots = offs_n % Lk
        else:
            slots = offs_n
        visible = 1
        if HAS_MASK:
            mask_block = tl.load(
                Mask_ptr
                + b * stride_mb
                + query_rows[:, None] * stride_mq
                + slots[None, :] * stride_mk,
                mask=row_valid[:, None] & n_valid[None, :],
                other=False,
            )
            visible = tl.max(tl.max(mask_block.to(tl.int32), axis=1), axis=0)
        if visible > 0:
            k_lo, k_hi = _load_kv_tile_bf16(
                KQ, KS, base_q, base_s, slots, n_valid, BLOCK_N, D, ARCH
            )
            qk = tl.dot(q_even, tl.trans(k_lo))
            qk = tl.dot(q_odd, tl.trans(k_hi), qk)
            qk = qk * sm_scale
            qk = tl.where(n_valid[None, :], qk, -float("inf"))
            if HAS_MASK:
                qk = tl.where(mask_block, qk, -float("inf"))
            if MASK_IS_CAUSAL:
                absolute_q = (kv_len - LQ) + query_rows
                qk = tl.where(offs_n[None, :] > absolute_q[:, None], -float("inf"), qk)
            if WINDOW > 0:
                row_pos = (kv_len - LQ) + query_rows
                vis = (offs_n[None, :] <= row_pos[:, None]) & (
                    offs_n[None, :] > row_pos[:, None] - WINDOW
                )
                qk = tl.where(vis, qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            safe_m = tl.where(m_ij > -float("inf"), m_ij, 0.0)
            p = tl.exp(qk - safe_m[:, None])
            alpha = tl.exp(m_i - safe_m)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v_lo, v_hi = _load_kv_tile_bf16(
                VQ, VS, base_q, base_s, slots, n_valid, BLOCK_N, D, ARCH
            )
            pv = p.to(tl.float16)
            acc_lo = tl.dot(pv, v_lo, acc_lo * alpha[:, None])
            acc_hi = tl.dot(pv, v_hi, acc_hi * alpha[:, None])
            m_i = m_ij

    acc = tl.reshape(tl.join(acc_lo, acc_hi), (BLOCK_M, D))
    offs_d = tl.arange(0, D)
    tl.store(
        O_partial_ptr
        + split_id * stride_op_s
        + b * stride_op_b
        + h_q[:, None] * stride_op_h
        + query_rows[:, None] * stride_op_m
        + offs_d[None, :],
        acc,
        mask=row_valid[:, None],
    )
    mp = (
        split_id * stride_mp_s
        + b * stride_mp_b
        + h_q * stride_mp_h
        + query_rows * stride_mp_m
    )
    tl.store(M_partial_ptr + mp, m_i, mask=row_valid)
    tl.store(L_partial_ptr + mp, l_i, mask=row_valid)


def _launch_nvfp4_splitk(
    query,
    k_q,
    k_s,
    v_q,
    v_s,
    out,
    mask,
    kv_len_t,
    sm_scale,
    mask_is_causal,
    window=0,
):
    B, H_q, L_q, D = query.shape
    H_kv, L_kv = k_q.shape[1], k_q.shape[2]
    num_groups = H_q // H_kv
    num_splits, chunk_size = _decode_splitk_config(L_kv, B * H_kv, query.device)
    if window > 0:
        chunk_size = triton.cdiv(window + L_q - 1, num_splits)
    O_partial = torch.empty(
        (num_splits, B, H_q, L_q, D), device=query.device, dtype=torch.float32
    )
    M_partial = torch.empty(
        (num_splits, B, H_q, L_q), device=query.device, dtype=torch.float32
    )
    L_partial = torch.empty(
        (num_splits, B, H_q, L_q), device=query.device, dtype=torch.float32
    )
    HAS_MASK = mask is not None
    if HAS_MASK:
        smb, smq, smk = mask.stride(0), mask.stride(2), mask.stride(3)
    else:
        smb = smq = smk = 0
    sq = query.stride()
    sop = O_partial.stride()
    smp = M_partial.stride()
    block_g = _next_power_of_2_unclamped(num_groups)
    wrap_triton(_nvfp4_splitk_kernel)[(num_splits, B * H_kv)](
        query,
        k_q,
        k_s,
        v_q,
        v_s,
        O_partial,
        M_partial,
        L_partial,
        mask if HAS_MASK else 0,
        kv_len_t if kv_len_t is not None else 0,
        H_kv,
        L_kv,
        sq[0],
        sq[1],
        sq[2],
        sq[3],
        sop[0],
        sop[1],
        sop[2],
        sop[3],
        smp[0],
        smp[1],
        smp[2],
        smp[3],
        smb,
        smq,
        smk,
        sm_scale,
        chunk_size,
        HAS_MASK=HAS_MASK,
        HAS_KV_LEN=kv_len_t is not None,
        MASK_IS_CAUSAL=mask_is_causal,
        HEAD_DIM=D,
        NUM_GROUPS=num_groups,
        LQ=L_q,
        BLOCK_G=block_g,
        BLOCK_M=max(16, _next_power_of_2_unclamped(L_q * block_g)),
        ARCH=cuda_arch(),
        WINDOW=window,
    )
    so = out.stride()
    wrap_triton(_sdpa_small_query_reduce_kernel)[(B * H_q * L_q,)](
        O_partial,
        M_partial,
        L_partial,
        out,
        num_splits,
        sop[0],
        sop[1],
        sop[2],
        sop[3],
        sop[4],
        smp[0],
        smp[1],
        smp[2],
        smp[3],
        smp[0],
        smp[1],
        smp[2],
        smp[3],
        so[0],
        so[1],
        so[2],
        so[3],
        H_q,
        BLOCK_S=_next_power_of_2_unclamped(num_splits),
        LQ=L_q,
        HEAD_DIM=D,
        num_warps=4,
        num_stages=1,
    )


# ---------------------------------------------------------------------------
# chunked-prefill flash attention over the NVFP4 cache
# ---------------------------------------------------------------------------
_PREFILL_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=3),
]


@autotune_inputs(KV_LEN_ptr=_kv_len_scenarios, Mask_ptr=_visible_mask_scenarios)
@triton.autotune(
    configs=_PREFILL_CONFIGS,
    key=["Lq", "Lk", "HEAD_DIM", "HAS_MASK", "MASK_IS_CAUSAL", "NUM_GROUPS", "WINDOW"],
)
@triton.jit
def _nvfp4_prefill_kernel(
    Q_ptr,
    KQ,
    KS,
    VQ,
    VS,
    O_ptr,
    Mask_ptr,
    KV_LEN_ptr,
    H_q,
    Lq,
    Lk,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_mb,
    stride_mq,
    stride_mk,
    sm_scale: tl.float32,
    HAS_MASK: tl.constexpr,
    HAS_KV_LEN: tl.constexpr,
    MASK_IS_CAUSAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ARCH: tl.constexpr,
    WINDOW: tl.constexpr = 0,
):
    D: tl.constexpr = HEAD_DIM
    sm_scale = sm_scale.to(tl.float32)
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H_q
    h_q = pid_bh % H_q
    H_kv = H_q // NUM_GROUPS
    h_kv = h_q // NUM_GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_valid = offs_m < Lq
    half = tl.arange(0, D // 2)
    q_base = Q_ptr + b * stride_qb + h_q * stride_qh + offs_m[:, None] * stride_qm
    q_even = tl.load(
        q_base + (2 * half[None, :]) * stride_qd, mask=row_valid[:, None], other=0.0
    ).to(tl.float16)
    q_odd = tl.load(
        q_base + (2 * half[None, :] + 1) * stride_qd, mask=row_valid[:, None], other=0.0
    ).to(tl.float16)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_lo = tl.zeros([BLOCK_M, D // 2], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_M, D // 2], dtype=tl.float32)

    if HAS_KV_LEN:
        kv_len = tl.load(KV_LEN_ptr)
        if WINDOW == 0:
            kv_len = tl.minimum(kv_len, Lk)
    else:
        kv_len = Lk
    end_n = kv_len
    start_lo = 0
    if MASK_IS_CAUSAL or WINDOW > 0:
        # Rows are absolute positions (kv_len - Lq) + offs_m: stop at the last
        # key the last row of this tile can see.
        end_n = tl.minimum(kv_len, (kv_len - Lq) + (pid_m + 1) * BLOCK_M)
    if WINDOW > 0:
        # Ring of Lk slots: start at the oldest position the first row of the
        # tile can see that the ring still holds.
        start_lo = tl.maximum(
            tl.maximum((kv_len - Lq) + pid_m * BLOCK_M - WINDOW + 1, kv_len - Lk), 0
        )

    head_row = (b * H_kv + h_kv).to(tl.int64) * Lk
    base_q = head_row * (D // 2)
    base_s = head_row * (D // 16)
    offs_n_init = tl.arange(0, BLOCK_N)

    for start_n in tl.range(start_lo, end_n, BLOCK_N):
        offs_n = start_n + offs_n_init
        n_valid = offs_n < end_n
        if WINDOW > 0:
            slots = offs_n % Lk
        else:
            slots = offs_n
        active = 1
        if HAS_MASK:
            mask_block = tl.load(
                Mask_ptr
                + b * stride_mb
                + offs_m[:, None] * stride_mq
                + slots[None, :] * stride_mk,
                mask=row_valid[:, None] & n_valid[None, :],
                other=False,
            )
            active = tl.max(tl.max(mask_block.to(tl.int32), axis=1), axis=0)
        if active > 0:
            k_lo, k_hi = _load_kv_tile_bf16(
                KQ, KS, base_q, base_s, slots, n_valid, BLOCK_N, D, ARCH, True
            )
            qk = tl.dot(q_even, tl.trans(k_lo))
            qk = tl.dot(q_odd, tl.trans(k_hi), qk)
            qk = qk * sm_scale
            qk = tl.where(n_valid[None, :], qk, -float("inf"))
            if HAS_MASK:
                qk = tl.where(mask_block, qk, -float("inf"))
            if MASK_IS_CAUSAL or WINDOW > 0:
                absolute_q = (kv_len - Lq) + offs_m
                qk = tl.where(offs_n[None, :] > absolute_q[:, None], -float("inf"), qk)
            if WINDOW > 0:
                qk = tl.where(
                    offs_n[None, :] <= absolute_q[:, None] - WINDOW, -float("inf"), qk
                )
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            safe_m = tl.where(m_ij > -float("inf"), m_ij, 0.0)
            p = tl.exp(qk - safe_m[:, None])
            alpha = tl.exp(m_i - safe_m)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v_lo, v_hi = _load_kv_tile_bf16(
                VQ, VS, base_q, base_s, slots, n_valid, BLOCK_N, D, ARCH, True
            )
            pv = p.to(tl.float16)
            acc_lo = tl.dot(pv, v_lo, acc_lo * alpha[:, None])
            acc_hi = tl.dot(pv, v_hi, acc_hi * alpha[:, None])
            m_i = m_ij

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc = tl.reshape(tl.join(acc_lo, acc_hi), (BLOCK_M, D)) * inv_l[:, None]
    offs_d = tl.arange(0, D)
    tl.store(
        O_ptr
        + b * stride_ob
        + h_q * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=row_valid[:, None],
    )


def _launch_nvfp4_prefill(
    query,
    k_q,
    k_s,
    v_q,
    v_s,
    out,
    mask,
    kv_len_t,
    sm_scale,
    mask_is_causal,
    window=0,
):
    B, H_q, L_q, D = query.shape
    H_kv, L_kv = k_q.shape[1], k_q.shape[2]
    HAS_MASK = mask is not None
    if HAS_MASK:
        smb, smq, smk = mask.stride(0), mask.stride(2), mask.stride(3)
    else:
        smb = smq = smk = 0
    sq = query.stride()
    so = out.stride()

    def grid(meta):
        return (triton.cdiv(L_q, meta["BLOCK_M"]), B * H_q)

    wrap_triton(_nvfp4_prefill_kernel)[grid](
        query,
        k_q,
        k_s,
        v_q,
        v_s,
        out,
        mask if HAS_MASK else 0,
        kv_len_t if kv_len_t is not None else 0,
        H_q,
        L_q,
        L_kv,
        sq[0],
        sq[1],
        sq[2],
        sq[3],
        so[0],
        so[1],
        so[2],
        so[3],
        smb,
        smq,
        smk,
        sm_scale,
        HAS_MASK=HAS_MASK,
        HAS_KV_LEN=kv_len_t is not None,
        MASK_IS_CAUSAL=mask_is_causal,
        HEAD_DIM=D,
        NUM_GROUPS=H_q // H_kv,
        ARCH=cuda_arch(),
        WINDOW=window,
    )


# ---------------------------------------------------------------------------
# public op
# ---------------------------------------------------------------------------
@triton_op("triton::nvfp4_sdpa", mutates_args={})
def nvfp4_sdpa(
    query: torch.Tensor,
    k_q: torch.Tensor,
    k_s: torch.Tensor,
    v_q: torch.Tensor,
    v_s: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: float = 0.0,
    kv_len: Optional[torch.Tensor] = None,
    window: int = 0,
) -> torch.Tensor:
    """GQA attention of BF16 ``query`` [B, H_q, L_q, D] over an NVFP4 KV cache.

    ``attn_mask``: optional bool [B, 1, L_q, L_kv] over cache slots (True =
    attend); with ``window`` it is indexed by ring slot and combined with the
    in-kernel window.
    ``is_causal`` requires ``kv_len`` and uses bottom-right alignment (query
    row i is absolute position kv_len - L_q + i).  ``kv_len``: device int
    scalar bounding the KV sweep (CUDA-graph safe).  ``window`` > 0: sliding-window layer
    over a ring of L_kv slots (slot = pos % L_kv); ``kv_len`` is then the
    absolute number of tokens written, and query row i (position
    kv_len - L_q + i) sees positions (pos - window, pos] that the ring still
    holds -- the same visibility as the model's dense sliding mask, computed
    in-kernel so only ~window keys are swept.
    """
    B, H_q, L_q, D = query.shape
    L_kv = k_q.shape[2]
    out = torch.empty((B, H_q, L_q, D), device=query.device, dtype=query.dtype)
    sm_scale = 1.0 / math.sqrt(D) if scale == 0.0 else scale
    if window > 0:
        if kv_len is None:
            raise RuntimeError("nvfp4_sdpa: a sliding window requires kv_len")
        kv_len_t = kv_len.reshape(1).to(torch.int32).contiguous()
    elif kv_len is not None:
        kv_len_t = torch.clamp(
            kv_len.reshape(1).to(torch.int32), max=int(L_kv)
        ).contiguous()
    else:
        kv_len_t = None
    mask_is_causal = bool(is_causal) and kv_len_t is not None
    if is_causal and kv_len_t is None:
        raise RuntimeError("nvfp4_sdpa: is_causal requires kv_len")
    if statically_known_true(L_q == 1) or (
        statically_known_true(L_q >= 2)
        and statically_known_true(L_q <= _SMALL_QUERY_MAX_LQ)
    ):
        _launch_nvfp4_splitk(
            query,
            k_q,
            k_s,
            v_q,
            v_s,
            out,
            attn_mask,
            kv_len_t,
            sm_scale,
            mask_is_causal and not statically_known_true(L_q == 1),
            window,
        )
    else:
        _launch_nvfp4_prefill(
            query,
            k_q,
            k_s,
            v_q,
            v_s,
            out,
            attn_mask,
            kv_len_t,
            sm_scale,
            mask_is_causal,
            window,
        )
    return out


@nvfp4_sdpa.register_fake
def _nvfp4_sdpa_fake(
    query,
    k_q,
    k_s,
    v_q,
    v_s,
    attn_mask=None,
    is_causal=False,
    scale=0.0,
    kv_len=None,
    window=0,
):
    return torch.empty_like(query)


# ---------------------------------------------------------------------------
# reference helpers (tests)
# ---------------------------------------------------------------------------
def nvfp4_kv_dequantize_reference(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch decode of a cache tensor pair to fp32 [..., D]."""
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=q.device,
    )
    lo = lut[(q & 15).long()]
    hi = lut[(q >> 4).long()]
    x = torch.stack([lo, hi], dim=-1).flatten(-2)
    sf = s.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return x * sf
