# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in LICENSE.
"""Sliding attention reads prior ring and packed current K/V before committing.

History is [window, columns, kv_heads, head_dim]. Current Q/K/V remain BHSD.
seq_info retains the current seven-field ABI; q_start/q_len/column/kv_end
name the real rows. Padded rows have no segment and must never commit.
This initial implementation is a qualification candidate, not a promoted ET
kernel. The full model/native lowering integration is a separate required gate.
"""
import math
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _attention(Q, KH, VH, KN, VN, Info, Out,
               sqh, sqt, skhr, skhc, skhh, svhr, svhc, svhh,
               sknh, sknt, svnh, svnt, soh, sot, scale,
               W: tl.constexpr, D: tl.constexpr, GROUPS: tl.constexpr,
               INFO: tl.constexpr, SEGMENTS: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    segs = tl.arange(0, triton.next_power_of_2(SEGMENTS))
    lengths = tl.load(Info + segs * INFO + 1, segs < SEGMENTS, other=0)
    counts = tl.cdiv(lengths * GROUPS, BM)
    ends = tl.cumsum(counts)
    if tile >= tl.sum(counts):
        return
    seg = tl.sum(((tile >= ends) & (segs < SEGMENTS)).to(tl.int32))
    before = tl.sum(tl.where(segs < seg, counts, 0))
    qb = tile - before
    qstart = tl.load(Info + seg * INFO)
    qlen = tl.load(Info + seg * INFO + 1)
    col = tl.load(Info + seg * INFO + 2).to(tl.int64)
    end = tl.load(Info + seg * INFO + 3)
    columns = tl.load(Info + seg * INFO + 4).to(tl.int64)
    skhr = columns * skhc
    svhr = columns * svhc
    first_new = end - qlen
    flat = qb * BM + tl.arange(0, BM)
    qr = flat // GROUPS
    qhead = head * GROUPS + flat % GROUPS
    validq = qr < qlen
    qt = qstart + qr
    qpos = first_new + qr
    ds = tl.arange(0, D)
    q = tl.load(Q + qhead[:, None] * sqh + qt[:, None] * sqt + ds[None, :],
                validq[:, None], other=0)
    first_q = (qb * BM) // GROUPS
    last_q = tl.minimum(qlen - 1, (qb * BM + BM - 1) // GROUPS)
    lo = tl.maximum(0, first_new + first_q - W + 1)
    hi = first_new + last_q + 1
    m = tl.full((BM,), -float('inf'), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for n0 in tl.range(lo, hi, BN):
        n = n0 + tl.arange(0, BN)
        new = n >= first_new
        nvalid = n < hi
        current_t = qstart + n - first_new
        hp = KH + col * skhc + head * skhh + (n % W)[:, None] * skhr + ds[None, :]
        np = KN + head * sknh + current_t[:, None] * sknt + ds[None, :]
        k = tl.load(tl.where(new[:, None], np, hp), nvalid[:, None], other=0)
        score = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        visible = nvalid[None, :] & validq[:, None] & (n[None, :] <= qpos[:, None]) & (n[None, :] > qpos[:, None] - W)
        score = tl.where(visible, score, -float('inf'))
        mn = tl.maximum(m, tl.max(score, 1))
        alpha = tl.where(mn > -float('inf'), tl.exp(m - mn), 0.0)
        p = tl.where(mn[:, None] > -float('inf'), tl.exp(score - mn[:, None]), 0.0)
        hvp = VH + col * svhc + head * svhh + (n % W)[:, None] * svhr + ds[None, :]
        nvp = VN + head * svnh + current_t[:, None] * svnt + ds[None, :]
        v = tl.load(tl.where(new[:, None], nvp, hvp), nvalid[:, None], other=0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l = l * alpha + tl.sum(p, 1)
        m = mn
    result = acc * tl.where(l > 0, 1.0 / l, 0.0)[:, None]
    tl.store(Out + qhead[:, None] * soh + qt[:, None] * sot + ds[None, :],
             result.to(Out.dtype.element_ty), validq[:, None])


@triton.jit
def _commit(KH, VH, KN, VN, Info,
            skhc, skhh, svhc, svhh, sknh, sknt, svnh, svnt,
            T: tl.constexpr, W: tl.constexpr, D: tl.constexpr,
            INFO: tl.constexpr, SEGMENTS: tl.constexpr):
    t = tl.program_id(0)
    head = tl.program_id(1)
    segs = tl.arange(0, triton.next_power_of_2(SEGMENTS))
    starts = tl.load(Info + segs * INFO, segs < SEGMENTS, other=0)
    lens = tl.load(Info + segs * INFO + 1, segs < SEGMENTS, other=0)
    match = (segs < SEGMENTS) & (t >= starts) & (t < starts + lens)
    if tl.sum(match.to(tl.int32)) == 0:
        return
    seg = tl.sum(tl.where(match, segs, 0))
    qstart = tl.load(Info + seg * INFO)
    qlen = tl.load(Info + seg * INFO + 1)
    # Only the last W writes can survive. Earlier writes may alias these rows.
    if t < qstart + tl.maximum(qlen - W, 0):
        return
    col = tl.load(Info + seg * INFO + 2).to(tl.int64)
    end = tl.load(Info + seg * INFO + 3)
    columns = tl.load(Info + seg * INFO + 4).to(tl.int64)
    pos = end - qlen + t - qstart
    ds = tl.arange(0, D)
    k = tl.load(KN + head * sknh + t * sknt + ds)
    v = tl.load(VN + head * svnh + t * svnt + ds)
    tl.store(KH + (pos % W) * columns * skhc + col * skhc + head * skhh + ds, k)
    tl.store(VH + (pos % W) * columns * svhc + col * svhc + head * svhh + ds, v)


@triton_op('triton::sdpa_ring_current', mutates_args={})
def sdpa_ring_current(query: torch.Tensor, history_key: torch.Tensor,
                      history_value: torch.Tensor, current_key: torch.Tensor,
                      current_value: torch.Tensor, seq_info: torch.Tensor,
                      window: int, scale: float) -> torch.Tensor:
    heads, total, dim = query.shape[1:]
    kvheads = current_key.shape[1]
    groups = heads // kvheads
    # Native cache allocations can be smaller than declared tensor shapes.
    # Runtime columns in seq_info define the actual pitch for history reads.
    out = torch.zeros_like(query)
    bm = 64 if dim <= 128 else 16
    bn = 64 if dim <= 128 else 32
    tiles = triton.cdiv(total * groups, bm) + seq_info.shape[0]
    wrap_triton(_attention)[(tiles, kvheads)](
        query, history_key, history_value, current_key, current_value, seq_info, out,
        query.stride(1), query.stride(2),
        history_key.stride(0), history_key.stride(1), history_key.stride(2),
        history_value.stride(0), history_value.stride(1), history_value.stride(2),
        current_key.stride(1), current_key.stride(2), current_value.stride(1), current_value.stride(2),
        out.stride(1), out.stride(2), scale,
        window, dim, groups, seq_info.shape[1], seq_info.shape[0], bm, bn,
        num_warps=4, num_stages=2 if dim <= 128 else 1)
    return out


@triton_op('triton::ring_commit_current', mutates_args={'history_key','history_value'})
def ring_commit_current(history_key: torch.Tensor, history_value: torch.Tensor,
                        current_key: torch.Tensor, current_value: torch.Tensor,
                        seq_info: torch.Tensor, window: int) -> None:
    wrap_triton(_commit)[(current_key.shape[2], current_key.shape[1])](
        history_key, history_value, current_key, current_value, seq_info,
        history_key.stride(1), history_key.stride(2), history_value.stride(1), history_value.stride(2),
        current_key.stride(1), current_key.stride(2), current_value.stride(1), current_value.stride(2),
        current_key.shape[2], window, current_key.shape[3], seq_info.shape[1], seq_info.shape[0],
        num_warps=4)
