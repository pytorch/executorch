# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""W4A8 prefill GEMM for ``CudaCoalescedInt4Tensor`` weights (no decode bucket).

The weight is consumed in its existing storage, shared with the decode
buckets (``int4_quantized_gemm.py``); nothing is re-packed or duplicated:

* qdata: [N, K / 2] uint8, low nibble is even K, high nibble is odd K;
* scale/zero: [N, K / 32] uint8 codes; scale_step/zero_point_step: [N, K / 256]
  fp16;
* weight = (q - zero_code * zero_step) * (scale_code * scale_step).

Activations are quantized to symmetric INT8 with one FP32 scale ``a`` per K32
group (round half to even, IEEE division), and ``a * sum(xq)`` is kept per
group. A group whose ``127 / amax`` overflows FP32 (amax below ~3.7e-37) is
quantized as ``x / amax * 127`` instead. A group holding a NaN or an infinity
gets a NaN scale, so every output of that row is NaN; dequant + F.linear would
give NaN or an infinity there.

The GEMM unpacks the nibbles in registers, runs one INT8 tensor-core dot
``d = sum(xq * q)`` per K32 group, and accumulates ``d * a * scale`` in FP32.
The asymmetric zero term ``sum_g (a * sum(xq)) * (zero * scale)`` is a
K/32-deep TF32 tensor-core product after the K loop (``ZERO_TAIL_TF32``, from
``zero * scale`` precomputed in the prep launch); every table config uses it.
``ZERO_TAIL_TF32X3`` and ``ZERO_EPI_CODES`` (per group in the epilogue,
straight from the codes) are alternatives for ``w4a8_prefill_gemm`` only.
INT8 tensor-core dots need an SM80+ CUDA target; ROCm is not supported.

``triton::int4_w4a8_prefill_gemm`` takes a static launch-config index. Each
config fixes the tile, warps, pipeline depth, split-K factor and zero mode;
split-K writes FP32 partials reduced in a fixed order (deterministic).
``int4_w4a8_prefill_linear`` picks the config: a static M takes the config of
its M class; a symbolic M (dynamic-shape export) takes the large-M class
config, so the exported graph has one launch and no data-dependent branch.
"""

from typing import Callable, NamedTuple, Optional

import torch
import triton
import triton.language as tl
from torch._subclasses.fake_tensor import is_fake
from torch.library import wrap_triton

_GROUP = 32
_SUPER = 256

ZERO_TAIL_TF32X3 = 4
ZERO_TAIL_TF32 = 5
ZERO_EPI_CODES = 8


@triton.jit
def _quantize_act_block(
    x,
    xq,
    xscale,
    xga,
    M,
    stride_xm,
    pid_m,
    sb,
    K: tl.constexpr,
    ACT_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    rows = offs_m.to(tl.int64)
    offs_k = sb * 256 + tl.arange(0, 256)
    v = tl.load(
        x + rows[:, None] * stride_xm + offs_k[None, :], mask=m_mask[:, None], other=0.0
    ).to(tl.float32)
    NA: tl.constexpr = 256 // ACT_GROUP
    v3 = tl.reshape(v, (BLOCK_M, NA, ACT_GROUP))
    finite = tl.abs(v3) < float("inf")
    non_finite_group = tl.min(finite.to(tl.int32), axis=2) == 0
    v3 = tl.where(finite, v3, 0.0)
    amax = tl.max(tl.abs(v3), axis=2)
    c127 = tl.full(amax.shape, 127.0, tl.float32)
    inv = tl.where(amax > 0.0, tl.math.div_rn(c127, amax), 0.0)
    scale = tl.where(non_finite_group, float("nan"), tl.math.div_rn(amax, c127))
    scaled = tl.where(
        (inv == float("inf"))[:, :, None],
        tl.math.div_rn(v3, amax[:, :, None]) * 127.0,
        v3 * inv[:, :, None],
    )
    q3 = tl.extra.cuda.libdevice.rint(scaled)
    q3 = tl.minimum(tl.maximum(q3, -127.0), 127.0)
    q = tl.reshape(q3, (BLOCK_M, 256))
    tl.store(
        xq + rows[:, None] * K + offs_k[None, :], q.to(tl.int8), mask=m_mask[:, None]
    )
    KA: tl.constexpr = K // ACT_GROUP
    offs_a = sb * NA + tl.arange(0, NA)
    tl.store(xscale + rows[:, None] * KA + offs_a[None, :], scale, mask=m_mask[:, None])
    KG: tl.constexpr = K // 32
    GPA: tl.constexpr = ACT_GROUP // 32
    gsum = tl.sum(tl.reshape(q, (BLOCK_M, NA, GPA, 32)), axis=3)
    ga = tl.reshape(gsum * scale[:, :, None], (BLOCK_M, 8))
    offs_g = sb * 8 + tl.arange(0, 8)
    tl.store(xga + rows[:, None] * KG + offs_g[None, :], ga, mask=m_mask[:, None])


@triton.jit
def _zero_weight_block(
    zcode,
    zstep,
    scode,
    sstep,
    zw,
    N,
    pid_n,
    pid_g,
    KG: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """zw[n, g] = (zero_code * zero_step) * (scale_code * scale_step) in FP32."""
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    g = pid_g * 32 + tl.arange(0, 32)
    mask = (offs_n < N)[:, None] & (g < KG)[None, :]
    n = offs_n.to(tl.int64)[:, None]
    gg = g[None, :]
    zq = tl.load(zcode + n * KG + gg, mask=mask, other=0).to(tl.float32) * tl.load(
        zstep + n * (KG // 8) + gg // 8, mask=mask, other=0.0
    ).to(tl.float32)
    ws = tl.load(scode + n * KG + gg, mask=mask, other=0).to(tl.float32) * tl.load(
        sstep + n * (KG // 8) + gg // 8, mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(zw + n * KG + gg, zq * ws, mask=mask)


# M is never specialized, as in a dynamic-M export.
@triton.jit(do_not_specialize=["M"])
def _quantize_act_kernel(
    x,
    xq,
    xscale,
    xga,
    M,
    stride_xm,
    K: tl.constexpr,
    ACT_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    _quantize_act_block(
        x,
        xq,
        xscale,
        xga,
        M,
        stride_xm,
        tl.program_id(0),
        tl.program_id(1),
        K,
        ACT_GROUP,
        BLOCK_M,
    )


@triton.jit
def _zero_weight_kernel(
    zcode, zstep, scode, sstep, zw, N, KG: tl.constexpr, BLOCK_N: tl.constexpr
):
    _zero_weight_block(
        zcode,
        zstep,
        scode,
        sstep,
        zw,
        N,
        tl.program_id(0),
        tl.program_id(1),
        KG,
        BLOCK_N,
    )


# M is never specialized, as in a dynamic-M export.
@triton.jit(do_not_specialize=["M", "act_programs"])
def _prep_kernel(
    x,
    xq,
    xscale,
    xga,
    M,
    stride_xm,
    act_programs,
    zcode,
    zstep,
    scode,
    sstep,
    zw,
    N,
    K: tl.constexpr,
    ACT_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One launch: activation quantization programs, then zero-weight programs."""
    pid = tl.program_id(0)
    NSB: tl.constexpr = K // 256
    KG: tl.constexpr = K // 32
    if pid < act_programs:
        _quantize_act_block(
            x,
            xq,
            xscale,
            xga,
            M,
            stride_xm,
            pid // NSB,
            pid % NSB,
            K,
            ACT_GROUP,
            BLOCK_M,
        )
    else:
        z = pid - act_programs
        GB: tl.constexpr = (KG + 31) // 32
        _zero_weight_block(
            zcode, zstep, scode, sstep, zw, N, z // GB, z % GB, KG, BLOCK_N
        )


# M is never specialized, as in a dynamic-M export.
@triton.jit(do_not_specialize=["M"])
def _w4a8_prefill_kernel(
    xq,
    xscale,
    xga,
    qdata,
    scode,
    sstep,
    zcode,
    zstep,
    zw_pre,
    out,
    M,
    N,
    stride_om,
    K: tl.constexpr,
    ZERO_MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    STAGES: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_s = pid % SPLIT_K
    pid = pid // SPLIT_K
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    per_group = GROUP_M * num_n
    first_m = (pid // per_group) * GROUP_M
    rows_in_group = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % per_group) % rows_in_group
    pid_n = (pid % per_group) // rows_in_group

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Out-of-range rows/columns are clamped (loads stay unmasked) and never stored.
    rows = tl.minimum(offs_m, M - 1).to(tl.int64)
    cols = tl.minimum(offs_n, N - 1).to(tl.int64)
    KG: tl.constexpr = K // 32
    NSB: tl.constexpr = K // 256
    g_lo = pid_s * KG // SPLIT_K
    g_hi = (pid_s + 1) * KG // SPLIT_K

    x_ptrs = xq + rows[:, None] * K + tl.arange(0, 32)[None, :]
    q_ptrs = qdata + cols[:, None] * (K // 2) + tl.arange(0, 16)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g in tl.range(g_lo, g_hi, num_stages=STAGES):
        x = tl.load(x_ptrs + g * 32)
        p = tl.load(q_ptrs + g * 16)
        w = tl.interleave(p & 0xF, p >> 4).to(tl.int8)
        d = tl.dot(x, tl.trans(w), out_dtype=tl.int32)
        ws = tl.load(scode + cols * KG + g).to(tl.float32) * tl.load(
            sstep + cols * NSB + g // 8
        ).to(tl.float32)
        a = tl.load(xscale + rows * KG + g)
        if ZERO_MODE == 8:
            zq = tl.load(zcode + cols * KG + g).to(tl.float32) * tl.load(
                zstep + cols * NSB + g // 8
            ).to(tl.float32)
            ga = tl.load(xga + rows * KG + g)
            acc += (d.to(tl.float32) * a[:, None] - ga[:, None] * zq[None, :]) * ws[
                None, :
            ]
        else:
            acc += d.to(tl.float32) * (a[:, None] * ws[None, :])

    if ZERO_MODE == 4 or ZERO_MODE == 5:
        for g0 in tl.range(g_lo, g_hi, 32, num_stages=1):
            gi = g0 + tl.arange(0, 32)
            g_mask = (gi < g_hi)[None, :]
            ga = tl.load(xga + rows[:, None] * KG + gi[None, :], mask=g_mask, other=0.0)
            zw = tl.load(
                zw_pre + cols[:, None] * KG + gi[None, :], mask=g_mask, other=0.0
            )
            if ZERO_MODE == 4:
                acc = tl.dot(-ga, tl.trans(zw), acc, input_precision="tf32x3")
            else:
                acc = tl.dot(-ga, tl.trans(zw), acc, input_precision="tf32")

    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    if SPLIT_K == 1:
        tl.store(
            out + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :],
            acc.to(tl.bfloat16),
            mask=mask,
        )
    else:
        part = out + pid_s.to(tl.int64) * M * N
        tl.store(
            part + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :], acc, mask=mask
        )


# M is never specialized, as in a dynamic-M export.
@triton.jit(do_not_specialize=["M"])
def _splitk_sum_kernel(
    part,
    out,
    M,
    N,
    stride_om,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """out[m, n] = bf16(sum_s part[s, m, n]) in fixed order; int64 offsets."""
    offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    idx = offs_m.to(tl.int64)[:, None] * N + offs_n[None, :]
    plane = tl.full((), M, tl.int64) * N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(part + s * plane + idx, mask=mask, other=0.0)
    tl.store(
        out + offs_m.to(tl.int64)[:, None] * stride_om + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=mask,
    )


class LaunchConfig(NamedTuple):
    block_m: int
    block_n: int
    num_warps: int
    stages: int
    split_k: int
    zero_mode: int


def _c(bm, bn, w, st, sk, zm=ZERO_TAIL_TF32):
    return LaunchConfig(bm, bn, w, st, sk, zm)


# Runtime-M classes: inclusive upper bound of M (None = the rest).
M_CLASSES = (16, 32, 64, 128, 192, 256, 384, 512, 640, 768, 1024, 1536, None)

# Per Q4_K projection shape (N, K): launch config per M class, from a dense
# runtime-M sweep on an RTX 5090 (SM120) with M never specialized, as in a
# dynamic export. Other GPUs run the same configs correctly, but their
# performance is untested. Small M is weight-bandwidth bound (small BLOCK_M,
# split-K); mid M trades tile size against SM-wave quantization; large M uses
# the 128x256 tile.
_TABLE = {
    (8448, 6656): (  # qko_proj
        _c(32, 128, 4, 3, 4),
        _c(32, 128, 4, 3, 4),
        _c(64, 128, 4, 3, 4),
        _c(128, 256, 8, 3, 4),
        _c(64, 256, 8, 3, 4),
        _c(64, 256, 8, 3, 1),
        _c(64, 256, 8, 3, 2),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 2),
        _c(128, 256, 8, 3, 4),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
    ),
    (6656, 4096): (  # o_proj
        _c(32, 64, 4, 3, 4),
        _c(32, 64, 4, 3, 4),
        _c(64, 128, 4, 3, 4),
        _c(128, 256, 8, 3, 4),
        _c(64, 256, 8, 3, 2),
        _c(64, 128, 4, 3, 4),
        _c(64, 256, 8, 3, 1),
        _c(64, 128, 4, 3, 4),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
        _c(64, 128, 4, 3, 2),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
    ),
    (39936, 6656): (  # gate_up_proj
        _c(32, 256, 4, 3, 8),
        _c(32, 256, 4, 3, 8),
        _c(64, 128, 4, 3, 4),
        _c(64, 256, 8, 3, 1),
        _c(64, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
    ),
    (6656, 19968): (  # down_proj
        _c(32, 128, 4, 3, 4),
        _c(32, 128, 4, 3, 4),
        _c(64, 128, 4, 3, 4),
        _c(128, 256, 8, 3, 4),
        _c(64, 256, 8, 3, 2),
        _c(128, 128, 8, 3, 4),
        _c(64, 256, 8, 3, 1),
        _c(64, 256, 8, 3, 4),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
        _c(64, 256, 8, 3, 2),
        _c(128, 256, 8, 3, 1),
        _c(128, 256, 8, 3, 1),
    ),
}
# Other shapes with N >= 1024; narrower outputs keep dequant + F.linear.
_DEFAULT_ROW = (
    _c(32, 128, 4, 3, 4),
    _c(32, 128, 4, 3, 4),
    _c(64, 128, 4, 3, 4),
    _c(128, 256, 8, 3, 4),
    _c(64, 256, 8, 3, 2),
    _c(64, 256, 8, 3, 1),
    _c(64, 256, 8, 3, 1),
    _c(128, 256, 8, 3, 1),
    _c(128, 256, 8, 3, 1),
    _c(128, 256, 8, 3, 1),
    _c(128, 256, 8, 3, 1),
    _c(128, 256, 8, 3, 1),
    _c(128, 256, 8, 3, 1),
)
assert all(len(row) == len(M_CLASSES) for row in (*_TABLE.values(), _DEFAULT_ROW))

# Finite launch-config table (index = static op argument). Entry 0 is the
# large-M default.
CONFIGS = tuple(
    dict.fromkeys(
        (_c(128, 256, 8, 3, 1),)
        + _DEFAULT_ROW
        + tuple(c for row in _TABLE.values() for c in row)
    )
)


def quantize_activations_w4a8_prefill(x: torch.Tensor, block_m: int = 16):
    """BF16 [M, K] -> INT8 [M, K], FP32 scale and FP32 a * sum(xq), each [M, K/32]."""
    M, K = x.shape
    xq = torch.empty((M, K), dtype=torch.int8, device=x.device)
    xscale = torch.empty((M, K // _GROUP), dtype=torch.float32, device=x.device)
    xga = torch.empty((M, K // _GROUP), dtype=torch.float32, device=x.device)
    wrap_triton(_quantize_act_kernel)[(triton.cdiv(M, block_m), K // _SUPER)](
        x,
        xq,
        xscale,
        xga,
        M,
        x.stride(0),
        K=K,
        ACT_GROUP=_GROUP,
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=1,
    )
    return xq, xscale, xga


def zero_weight_w4a8_prefill(scale, scale_step, zero, zero_point_step):
    """(zero * scale) per weight group, [N, K/32] FP32, from the stored codes."""
    N, KG = scale.shape
    zw = torch.empty((N, KG), dtype=torch.float32, device=scale.device)
    wrap_triton(_zero_weight_kernel)[(triton.cdiv(N, 64), triton.cdiv(KG, 32))](
        zero, zero_point_step, scale, scale_step, zw, N, KG=KG, BLOCK_N=64, num_warps=4
    )
    return zw


def _prepare(
    x, scale, scale_step, zero, zero_point_step, block_m: int = 16, block_n: int = 64
):
    """Activation quantization and per-group zero*scale in one launch."""
    M, K = x.shape
    N, KG = scale.shape
    xq = torch.empty((M, K), dtype=torch.int8, device=x.device)
    xscale = torch.empty((M, KG), dtype=torch.float32, device=x.device)
    xga = torch.empty((M, KG), dtype=torch.float32, device=x.device)
    zw = torch.empty((N, KG), dtype=torch.float32, device=x.device)
    act_programs = triton.cdiv(M, block_m) * (K // _SUPER)
    zero_programs = triton.cdiv(N, block_n) * triton.cdiv(KG, 32)
    wrap_triton(_prep_kernel)[(act_programs + zero_programs,)](
        x,
        xq,
        xscale,
        xga,
        M,
        x.stride(0),
        act_programs,
        zero,
        zero_point_step,
        scale,
        scale_step,
        zw,
        N,
        K=K,
        ACT_GROUP=_GROUP,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )
    return xq, xscale, xga, zw


def w4a8_prefill_gemm_quantized(
    xq,
    xscale,
    xga,
    zw,
    qdata,
    scale,
    scale_step,
    zero,
    zero_point_step,
    cfg: LaunchConfig,
    out: Optional[torch.Tensor] = None,
    group_m: int = 8,
):
    """GEMM launch (+ fixed-order split-K reduce) on prepared activations."""
    M, K = xq.shape
    N = qdata.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=xq.device)
    dst = (
        out
        if cfg.split_k == 1
        else torch.empty((cfg.split_k, M, N), dtype=torch.float32, device=xq.device)
    )
    zw = xga if zw is None else zw
    grid = (triton.cdiv(M, cfg.block_m) * triton.cdiv(N, cfg.block_n) * cfg.split_k,)
    wrap_triton(_w4a8_prefill_kernel)[grid](
        xq,
        xscale,
        xga,
        qdata,
        scale,
        scale_step,
        zero,
        zero_point_step,
        zw,
        dst,
        M,
        N,
        out.stride(0),
        K=K,
        ZERO_MODE=cfg.zero_mode,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        GROUP_M=group_m,
        STAGES=cfg.stages,
        SPLIT_K=cfg.split_k,
        num_warps=cfg.num_warps,
    )
    if cfg.split_k > 1:
        wrap_triton(_splitk_sum_kernel)[(triton.cdiv(N, 128), triton.cdiv(M, 16))](
            dst,
            out,
            M,
            N,
            out.stride(0),
            SPLIT_K=cfg.split_k,
            BLOCK_M=16,
            BLOCK_N=128,
            num_warps=4,
        )
    return out


def _run_config(x, qdata, scale, scale_step, zero, zero_point_step, cfg: LaunchConfig):
    if cfg.zero_mode in (ZERO_TAIL_TF32, ZERO_TAIL_TF32X3):
        xq, xscale, xga, zw = _prepare(x, scale, scale_step, zero, zero_point_step)
    else:
        xq, xscale, xga = quantize_activations_w4a8_prefill(x)
        zw = None
    return w4a8_prefill_gemm_quantized(
        xq, xscale, xga, zw, qdata, scale, scale_step, zero, zero_point_step, cfg
    )


def w4a8_prefill_gemm(
    x, qdata, scale, scale_step, zero, zero_point_step, cfg: LaunchConfig = CONFIGS[0]
):
    """Eager helper (tests/benchmarks): any LaunchConfig, not only table entries."""
    return _run_config(
        x.contiguous(), qdata, scale, scale_step, zero, zero_point_step, cfg
    )


def reference_quantize_activations(x: torch.Tensor, act_group: int = 32):
    """Torch replica of the activation quantizer (same FP32 operations)."""
    M, K = x.shape
    v = x.float().reshape(M, K // act_group, act_group)
    finite = v.isfinite()
    v = torch.where(finite, v, torch.zeros_like(v))
    amax = v.abs().amax(dim=2)
    c127 = torch.full_like(amax, 127.0)
    inv = torch.where(amax > 0, torch.div(c127, amax), torch.zeros_like(amax))
    scale = torch.div(amax, c127).masked_fill(~finite.all(dim=2), float("nan"))
    scaled = torch.where(
        inv.isinf()[:, :, None],
        torch.div(v, amax[:, :, None]) * 127.0,
        v * inv[:, :, None],
    )
    q = torch.round(scaled).clamp_(-127, 127)
    xq = q.reshape(M, K).to(torch.int8)
    gsum = q.reshape(M, K // _GROUP, _GROUP).sum(dim=2)
    return xq, scale, gsum * scale.repeat_interleave(act_group // _GROUP, dim=1)


def reference_w4a8_prefill(
    x, qdata, scale, scale_step, zero, zero_point_step, act_group: int = 32
):
    """Exact (FP64) evaluation of the quantized algorithm the kernel implements."""
    M, K = x.shape
    N = qdata.shape[0]
    xq, xscale, _ = reference_quantize_activations(x, act_group)
    qu = qdata.to(torch.uint8)
    q = torch.stack([qu & 0xF, qu >> 4], dim=-1).reshape(N, K).double()
    gps = _SUPER // _GROUP
    s = scale.double() * scale_step.double().repeat_interleave(gps, dim=1)
    z = zero.double() * zero_point_step.double().repeat_interleave(gps, dim=1)
    w = (q - z.repeat_interleave(_GROUP, dim=1)) * s.repeat_interleave(_GROUP, dim=1)
    xd = xq.double() * xscale.double().repeat_interleave(act_group, dim=1)
    return xd @ w.t()


# ---------------------------------------------------------------------------
# Config selection
# ---------------------------------------------------------------------------


def class_configs(N: int, K: int) -> list:
    """[(m_upper, CONFIGS index or None for dequant + F.linear)] per runtime-M class."""
    if N < 1024:
        return [(None, None)]
    row = _TABLE.get((N, K), _DEFAULT_ROW)
    out = []
    for upper, cfg in zip(M_CLASSES, row):
        if (K // _GROUP) // cfg.split_k < 1:
            cfg = cfg._replace(split_k=1)
        out.append((upper, CONFIGS.index(cfg) if cfg in CONFIGS else 0))
    return out


def config_for_m(M: int, N: int, K: int) -> Optional[int]:
    for upper, cfg in class_configs(N, K):
        if upper is None or M <= upper:
            return cfg
    return None


def _compile_target_sm() -> int:
    from torch._inductor.codegen.cuda.compile_utils import _nvcc_arch_as_compile_option

    target = (_nvcc_arch_as_compile_option() or "").removesuffix("a")
    return int(target) if target.isdigit() else 0


def _target_unsupported_reason(x: torch.Tensor) -> Optional[str]:
    """INT8 tensor-core dots and CUDA libdevice need an SM80+ CUDA target.

    Fake tensors (export) check the CUDA target being compiled for, so the
    route is fixed when the program is traced: a program compiled for an SM80+
    target holds the W4A8 kernel and does not fall back at run time on an older
    GPU.
    """
    if torch.version.hip is not None:
        return "ROCm targets are not supported"
    if is_fake(x):
        sm = _compile_target_sm()
    else:
        major, minor = torch.cuda.get_device_capability(x.device)
        sm = 10 * major + minor
    if sm < 80:
        return f"needs an SM80+ CUDA target, got sm_{sm}"
    return None


def prefill_unsupported_reason(
    x, qdata, scale, scale_step, zero, zero_point_step, group_size
) -> Optional[str]:
    """Why the prefill op cannot serve these arguments, or None. Never raises."""
    if group_size != _GROUP:
        return f"group_size {group_size} != 32"
    if x.dim() != 2 or x.dtype != torch.bfloat16:
        return "activation must be 2-D bfloat16"
    K = x.shape[1]
    if not isinstance(K, int) or K <= 0 or K % _SUPER != 0:
        return "K must be a static positive multiple of 256"
    if qdata.dim() != 2 or not isinstance(qdata.shape[0], int) or qdata.shape[0] <= 0:
        return "qdata must be [N, K/2] with static N > 0"
    N = qdata.shape[0]
    expected = (
        (qdata, torch.uint8, (N, K // 2)),
        (scale, torch.uint8, (N, K // _GROUP)),
        (zero, torch.uint8, (N, K // _GROUP)),
        (scale_step, torch.float16, (N, K // _SUPER)),
        (zero_point_step, torch.float16, (N, K // _SUPER)),
    )
    for t, dtype, shape in expected:
        if t.dtype != dtype or tuple(t.shape) != shape:
            return (
                f"weight buffer must be {dtype} {shape}, got {t.dtype} {tuple(t.shape)}"
            )
        if not t.is_contiguous():
            return "weight buffers must be contiguous"
        if t.device != x.device:
            return f"weight buffer on {t.device}, activation on {x.device}"
    if x.device.type != "cuda" and not is_fake(x):
        return "real tensors must be on CUDA"
    return _target_unsupported_reason(x)


# ---------------------------------------------------------------------------
# triton::int4_w4a8_prefill_gemm
# ---------------------------------------------------------------------------


@torch.library.triton_op("triton::int4_w4a8_prefill_gemm", mutates_args={})
def int4_w4a8_prefill_gemm(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    scale_step: torch.Tensor,
    zero: torch.Tensor,
    zero_point_step: torch.Tensor,
    config: int = 0,
) -> torch.Tensor:
    """BF16 [M, K] x CudaCoalescedInt4Tensor storage -> BF16 [M, N].

    W4A8 with per-K32 activation scales.

    ``config`` indexes ``CONFIGS`` (static): launch geometry only, same arithmetic.
    """
    reason = prefill_unsupported_reason(
        x, qdata, scale, scale_step, zero, zero_point_step, _GROUP
    )
    if reason is not None:
        raise RuntimeError(f"int4_w4a8_prefill_gemm: {reason}")
    if not 0 <= config < len(CONFIGS):
        raise RuntimeError(f"int4_w4a8_prefill_gemm: config {config} out of range")
    M = x.shape[0]
    if isinstance(M, int) and M == 0:
        return x.new_empty((0, qdata.shape[0]), dtype=torch.bfloat16)
    return _run_config(
        x.contiguous(), qdata, scale, scale_step, zero, zero_point_step, CONFIGS[config]
    )


@int4_w4a8_prefill_gemm.register_fake
def _(x, qdata, scale, scale_step, zero, zero_point_step, config=0):
    return x.new_empty((x.shape[0], qdata.shape[0]), dtype=torch.bfloat16)


def int4_w4a8_prefill_linear(
    x, weight_args, fallback: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """F.linear for a prefill activation: a W4A8 config or ``fallback``.

    A static M takes its M class's choice; a symbolic M takes the large-M
    class's choice.
    """
    N, K = weight_args[0].shape[0], x.shape[1]
    M = x.shape[0]
    cfg = config_for_m(M, N, K) if isinstance(M, int) else class_configs(N, K)[-1][1]
    if cfg is None:
        return fallback(x)
    return int4_w4a8_prefill_gemm(x, *weight_args[:5], config=cfg)
