# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mid-M flash SDPA: a length-bounded, split-K attention kernel for a few query rows.

A companion to the kernels in ``sdpa.py``. The shared ``_sdpa_fwd_kernel_m64/m32``
path scans the whole K/V buffer (mask width = max_seq_len), so a speculative
verify forward (M = chain+1, a few rows) over a 64K-context export pays a 64K-wide
attention pass even when only a few hundred positions are valid.

This kernel specializes that mid-M regime (analogous to the small-M weight-
stationary INT4 GEMM): it keeps all M query rows resident, streams K/V once, and
**bounds the key range to the actual valid length**. Crucially it also **splits
the key range across CTAs** (flash-decoding / split-K): a single (B, H) grid puts
only one CTA per head looping the whole KV serially, so at long context the verify
attention is occupancy-starved and grows linearly with length. Split-K partitions
[0, valid_len) into NUM_SPLITS chunks computed in parallel, then a reduce kernel
combines the per-split online-softmax partials -- the same trick that keeps the
M=1 decode path (``_sdpa_decode_splitk``) flat out to 64K. Since attention here is
bandwidth-bound on the K/V read, the M query rows ride along for free and the
verify attention approaches the decode floor regardless of context.

Causal masking is computed per row from ``input_pos`` (each of the M rows has its
own cutoff) -- no materialized max_seq_len mask. scale / GQA / bf16-in-fp32-
accumulate match ``F.scaled_dot_product_attention(..., is_causal=False,
enable_gqa=True)`` (Gemma 4 uses scale=1; QK-norm absorbs 1/sqrt(d)). Sliding-
window layers use a ring cache and stay on F.sdpa; this targets the flat full-
attention layers.

Lives in the CUDA backend (imported only during CUDA lowering / by the model's
mid-M dispatch), so triton is imported unconditionally -- same as ``sdpa.py``.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

# Verify windows up to this M route to the mid-M kernel; above it the prefill
# path is appropriate (enough rows to amortize a tiled kernel).
MIDM_MAX_M = 8

# Number of key-range partitions for split-K. B / H / D are static for the
# exported verify method; M is the dynamic verify length (bounded by MIDM_MAX_M,
# BLOCK_M covers it), so the grid (NUM_SPLITS x B*H) is static-shaped; the
# per-split chunk size (derived from the dynamic valid_len) is a runtime scalar.
# 32 splits x (B*H) heads gives ~1K CTAs at the gemma4 global shape -- ample
# occupancy on an A100 while keeping the fp32 partials small.
NUM_SPLITS = 32


@triton.jit
def _sdpa_midm_splitk_kernel(
    Q,
    K,
    V,
    POS,
    Opart,
    Lpart,
    Mpart,
    sqb,
    sqh,
    sqm,
    sqd,
    skb,
    skh,
    skn,
    skd,
    svb,
    svh,
    svn,
    svd,
    sops,
    sopb,
    soph,
    sopm,
    sopd,
    slps,
    slpb,
    slph,
    slpm,
    smps,
    smpb,
    smph,
    smpm,
    valid_len,
    chunk_size,
    scale,
    M,
    H: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    split_id = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    kv_h = pid_h // (H // HKV)

    # This CTA owns keys [start_n, end_n) of the valid range. Splits whose range
    # falls entirely past valid_len run an empty loop and emit a null partial
    # (m=-inf, l=0, acc=0), which the reduce discards.
    start_n = split_id * chunk_size
    end_n = tl.minimum(start_n + chunk_size, valid_len)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(
        Q + pid_b * sqb + pid_h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=offs_m[:, None] < M,
        other=0.0,
    )
    qpos = tl.load(POS + offs_m, mask=offs_m < M, other=0).to(tl.int32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    kbase = K + pid_b * skb + kv_h * skh
    vbase = V + pid_b * svb + kv_h * svh
    for sn in tl.range(start_n, end_n, BLOCK_N):
        n = sn + offs_n
        nmask = n < end_n
        k = tl.load(
            kbase + n[:, None] * skn + offs_d[None, :] * skd,
            mask=nmask[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        causal = (n[None, :] <= qpos[:, None]) & nmask[None, :]
        # Keep fp32: a bare -inf python literal promotes the loop-carried softmax
        # state to fp64, which AOTI's Triton compile rejects.
        qk = tl.where(causal, qk, float("-inf")).to(tl.float32)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # A split whose whole tile is causal-masked for a row leaves m_ij=-inf;
        # guard exp(-inf - -inf)=NaN (mirrors sdpa.py). Such null tiles then yield
        # a (m=-inf, l=0, acc=0) partial that the reduce discards.
        safe_qk = tl.where(
            m_ij[:, None] > -float("inf"), qk - m_ij[:, None], -float("inf")
        )
        p = tl.exp(safe_qk)
        safe_alpha = tl.where(m_ij > -float("inf"), m_i - m_ij, 0.0)
        alpha = tl.exp(safe_alpha)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            vbase + n[:, None] * svn + offs_d[None, :] * svd,
            mask=nmask[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # Store the un-normalized partial (acc, running max, running denom). The
    # reduce kernel rescales across splits and divides by the global denom.
    pbase = split_id * sops + pid_b * sopb + pid_h * soph
    tl.store(
        Opart + pbase + offs_m[:, None] * sopm + offs_d[None, :] * sopd,
        acc,
        mask=offs_m[:, None] < M,
    )
    tl.store(
        Lpart + split_id * slps + pid_b * slpb + pid_h * slph + offs_m * slpm,
        l_i,
        mask=offs_m < M,
    )
    tl.store(
        Mpart + split_id * smps + pid_b * smpb + pid_h * smph + offs_m * smpm,
        m_i,
        mask=offs_m < M,
    )


@triton.jit
def _sdpa_midm_reduce_kernel(
    Opart,
    Lpart,
    Mpart,
    OUT,
    sops,
    sopb,
    soph,
    sopm,
    sopd,
    slps,
    slpb,
    slph,
    slpm,
    smps,
    smpb,
    smph,
    smpm,
    sob,
    soh,
    som,
    sod,
    M,
    NUM_SPLITS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    m_g = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_g = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    for s in range(0, NUM_SPLITS):
        m_s = tl.load(
            Mpart + s * smps + pid_b * smpb + pid_h * smph + offs_m * smpm,
            mask=offs_m < M,
            other=-float("inf"),
        )
        l_s = tl.load(
            Lpart + s * slps + pid_b * slpb + pid_h * slph + offs_m * slpm,
            mask=offs_m < M,
            other=0.0,
        )
        o_s = tl.load(
            Opart
            + s * sops
            + pid_b * sopb
            + pid_h * soph
            + offs_m[:, None] * sopm
            + offs_d[None, :] * sopd,
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        m_new = tl.maximum(m_g, m_s)
        # Guard the all-empty case (m_new = -inf): -inf - -inf is NaN; where
        # selects the safe value and discards it (mirrors sdpa.py).
        finite = m_new > -float("inf")
        alpha_g = tl.where(finite, tl.exp(m_g - m_new), 1.0)
        alpha_s = tl.where(finite, tl.exp(m_s - m_new), 0.0)
        l_g = l_g * alpha_g + l_s * alpha_s
        acc = acc * alpha_g[:, None] + o_s * alpha_s[:, None]
        m_g = m_new

    inv = tl.where(l_g > 0, 1.0 / l_g, 0.0)
    acc = acc * inv[:, None]
    tl.store(
        OUT + pid_b * sob + pid_h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
        acc.to(OUT.dtype.element_ty),
        mask=offs_m[:, None] < M,
    )


@triton_op("triton::sdpa_midm", mutates_args={})
def _sdpa_midm_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    valid_len: int,
    scale: float,
) -> torch.Tensor:
    """Length-bounded, split-K mid-M flash SDPA (triton_op so AOTI codegens it).

    ``valid_len`` (max valid position + 1) bounds the key range; it is split into
    NUM_SPLITS chunks of ``chunk_size`` keys computed in parallel, then reduced.
    B / H / D are static for the exported verify method; M is the dynamic verify
    length (bounded by MIDM_MAX_M). chunk_size (from the dynamic valid_len) is a
    runtime (backed-SymInt) scalar; the grid (NUM_SPLITS x B*H) is static.
    """
    B, H, M, D = q.shape
    HKV = k.shape[1]
    out = torch.empty_like(q)
    # M <= MIDM_MAX_M (8) => next_pow2(M) <= 8 => max(16, .) is always 16. Hardcode
    # so M can be a runtime (dynamic verify) dim -- next_power_of_2 can't take a
    # SymInt, and M is a kernel runtime arg used only for the offs_m < M masks.
    BLOCK_M = 16
    # gemma4 global layers use D=512; a wide key tile + pipelining overflow SMEM
    # there, so shrink both. Small D can afford more.
    BLOCK_N, num_stages = (32, 1) if D >= 512 else (64, 2)
    chunk_size = (valid_len + NUM_SPLITS - 1) // NUM_SPLITS

    Opart = torch.empty((NUM_SPLITS, B, H, M, D), device=q.device, dtype=torch.float32)
    Lpart = torch.empty((NUM_SPLITS, B, H, M), device=q.device, dtype=torch.float32)
    Mpart = torch.empty((NUM_SPLITS, B, H, M), device=q.device, dtype=torch.float32)

    wrap_triton(_sdpa_midm_splitk_kernel)[(NUM_SPLITS, B, H)](
        q,
        k,
        v,
        input_pos,
        Opart,
        Lpart,
        Mpart,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *Opart.stride(),
        *Lpart.stride(),
        *Mpart.stride(),
        valid_len,
        chunk_size,
        scale,
        H=H,
        HKV=HKV,
        M=M,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=num_stages,
        num_warps=4,
    )
    wrap_triton(_sdpa_midm_reduce_kernel)[(B, H)](
        Opart,
        Lpart,
        Mpart,
        out,
        *Opart.stride(),
        *Lpart.stride(),
        *Mpart.stride(),
        *out.stride(),
        NUM_SPLITS=NUM_SPLITS,
        M=M,
        D=D,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return out


@_sdpa_midm_op.register_fake
def _sdpa_midm_abstract(q, k, v, input_pos, valid_len, scale):
    return torch.empty_like(q)


def sdpa_midm(q, k, v, input_pos, scale=1.0, valid_len=None):
    """Eager/convenience wrapper. ``valid_len`` defaults to max(input_pos)+1
    clamped to the buffer; callers in a traced graph should pass a single
    precomputed ``valid_len`` to avoid per-layer SymInts."""
    if valid_len is None:
        valid_len = min(int(input_pos[-1]) + 1, k.shape[2])
    return torch.ops.triton.sdpa_midm(q, k, v, input_pos, valid_len, scale)


def sdpa_midm_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reference: F.sdpa over the valid prefix with a causal additive mask.

    Mirrors the gemma4 full-attention call (is_causal=False, enable_gqa=True,
    explicit additive mask) sliced to the valid length, so it equals what the
    model computes over the full buffer (the rest is masked to -inf anyway).
    """
    valid_len = int(input_pos.max().item()) + 1
    key_idx = torch.arange(valid_len, device=q.device)
    keep = key_idx[None, :] <= input_pos[:, None]
    attn_mask = torch.where(keep, 0.0, float("-inf")).to(q.dtype)
    return F.scaled_dot_product_attention(
        q,
        k[:, :, :valid_len],
        v[:, :, :valid_len],
        attn_mask=attn_mask,
        is_causal=False,
        enable_gqa=True,
        scale=scale,
    )


def midm_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float = 1.0,
    enable: bool = False,
    valid_len=None,
) -> torch.Tensor:
    """Dispatch: the mid-M op for a small query window when enabled; otherwise
    the standard F.sdpa the model already uses (which the replacement pass swaps
    for triton::sdpa). M (q.shape[2]) is the dynamic verify length; its exported
    range [2, MIDM_MAX_M] satisfies this guard, so the branch resolves at export.
    ``valid_len`` is the shared per-forward key bound."""
    M = q.shape[2]
    if enable and 2 <= M <= MIDM_MAX_M:
        return sdpa_midm(q, k, v, input_pos, scale, valid_len=valid_len)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=True, scale=scale
    )
