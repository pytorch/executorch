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
    # Inductor can pass the Python scalar as FP64; keep attention math FP32.
    scale = scale.to(tl.float32)
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
    segment_columns = tl.load(Info + segs * INFO + 2, segs < SEGMENTS, other=-1)
    segment_ends = tl.load(Info + segs * INFO + 3, segs < SEGMENTS, other=0)
    segment_starts = tl.load(Info + segs * INFO, segs < SEGMENTS, other=0)
    segment_first = segment_ends - lengths
    owner_first = tl.min(tl.where((segment_columns == col) & (lengths > 0), segment_first, 2147483647))
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
        new = n >= owner_first
        nvalid = n < hi
        current_t = qstart + n - first_new
        if owner_first != first_new:
            belongs = (segment_columns[None, :] == col) & (lengths[None, :] > 0) & (n[:, None] >= segment_first[None, :]) & (n[:, None] < segment_ends[None, :])
            current_t = tl.sum(tl.where(belongs, segment_starts[None, :] + n[:, None] - segment_first[None, :], 0), 1)
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
def _decode(
    Q_ptr,
    K_ptr,
    V_ptr,
    KN_ptr,
    VN_ptr,
    stride_knh,
    stride_knt,
    stride_vnh,
    stride_vnt,
    O_ptr,
    O_part_ptr,
    M_part_ptr,
    L_part_ptr,
    Info_ptr,
    Hq,
    T,
    stride_qh,
    stride_qt,
    stride_kr,
    stride_kc,
    stride_kh,
    stride_vr,
    stride_vc,
    stride_vh,
    stride_oh,
    stride_ot,
    sm_scale,
    rows_available,
    NUM_SEGMENTS: tl.constexpr,
    INFO_WIDTH: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    CONTEXT_WINDOW: tl.constexpr,
    WINDOW: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    sm_scale = sm_scale.to(tl.float32)
    scale = scale.to(tl.float32)
    t = tl.program_id(0)
    h_kv = tl.program_id(1)
    split = tl.program_id(2)

    owned = False
    q_start = 0
    q_len = 0
    col = 0
    kv_end = 0
    for seg in range(NUM_SEGMENTS):
        seg_q_start = tl.load(Info_ptr + seg * INFO_WIDTH)
        seg_q_len = tl.load(Info_ptr + seg * INFO_WIDTH + 1)
        take = (t >= seg_q_start) & (t < seg_q_start + seg_q_len)
        owned |= take
        q_start = tl.where(take, seg_q_start, q_start)
        q_len = tl.where(take, seg_q_len, q_len)
        if INFO_WIDTH == 7 and RING == 0:
            col = tl.where(take, tl.load(Info_ptr + seg * INFO_WIDTH + 5), col)
        else:
            col = tl.where(take, tl.load(Info_ptr + seg * INFO_WIDTH + 2), col)
        kv_end = tl.where(take, tl.load(Info_ptr + seg * INFO_WIDTH + 3), kv_end)
    if INFO_WIDTH >= 5:
        columns = tl.load(Info_ptr + 4).to(tl.int64)
        stride_kr = columns * stride_kc
        stride_vr = columns * stride_vc

    if INFO_WIDTH == 7 and RING == 0:
        rows_available = kv_end
        stride_kr = stride_kc
        stride_vr = stride_vc
    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, HEAD_DIM)
    g_valid = offs_g < NUM_GROUPS
    row_valid = owned & g_valid
    h_q = h_kv * NUM_GROUPS + offs_g
    q = tl.load(
        Q_ptr + h_q[:, None] * stride_qh + t * stride_qt + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )

    first_new = kv_end - q_len
    slots = tl.arange(0, triton.next_power_of_2(NUM_SEGMENTS))
    sc = tl.load(Info_ptr + slots * INFO_WIDTH + 2, slots < NUM_SEGMENTS, other=-1)
    sl = tl.load(Info_ptr + slots * INFO_WIDTH + 1, slots < NUM_SEGMENTS, other=0)
    se = tl.load(Info_ptr + slots * INFO_WIDTH + 3, slots < NUM_SEGMENTS, other=0)
    ss = tl.load(Info_ptr + slots * INFO_WIDTH, slots < NUM_SEGMENTS, other=0)
    sf = se - sl
    owner_first = tl.min(tl.where((sc == col) & (sl > 0), sf, 2147483647))
    pos = first_new + t - q_start
    lo = tl.maximum(0, pos - WINDOW + 1)
    hi = pos + 1
    span = tl.maximum(hi - lo, 0)
    chunk = tl.cdiv(tl.cdiv(span, NUM_SPLITS), BLOCK_N) * BLOCK_N
    start = lo + split * chunk
    stop = tl.minimum(start + chunk, hi)

    m_i = tl.full([BLOCK_G], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, HEAD_DIM], dtype=tl.float32)
    k_base = K_ptr + col * stride_kc + h_kv * stride_kh
    v_base = V_ptr + col * stride_vc + h_kv * stride_vh
    for n0 in tl.range(start, stop, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        physical_n = offs_n % RING if RING > 0 else offs_n
        n_valid = offs_n < stop
        use_new = offs_n >= owner_first
        current_t = q_start + offs_n - first_new
        if owner_first != first_new:
            belongs = (sc[None, :] == col) & (sl[None, :] > 0) & (offs_n[:, None] >= sf[None, :]) & (offs_n[:, None] < se[None, :])
            current_t = tl.sum(tl.where(belongs, ss[None, :] + offs_n[:, None] - sf[None, :], 0), 1)
        k = tl.load(
            tl.where(use_new[:, None], KN_ptr + h_kv * stride_knh + current_t[:, None] * stride_knt + offs_d[None, :],
                     k_base + physical_n[:, None] * stride_kr + offs_d[None, :]),
            mask=n_valid[:, None] & owned,
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        kv_pos = offs_n
        visible = n_valid[None, :] & row_valid[:, None] & (kv_pos[None, :] >= 0)
        if IS_CAUSAL:
            visible &= kv_pos[None, :] <= pos
        else:
            visible &= kv_pos[None, :] < kv_end
        if WINDOW > 0:
            if CONTEXT_WINDOW:
                visible &= kv_pos[None, :] >= kv_end - q_len - WINDOW
            else:
                visible &= kv_pos[None, :] > pos - WINDOW
        qk = tl.where(visible, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.where(m_ij[:, None] > -float("inf"), tl.exp(qk - m_ij[:, None]), 0.0)
        alpha = tl.where(m_ij > -float("inf"), tl.exp(m_i - m_ij), 0.0)
        v = tl.load(
            tl.where(use_new[:, None], VN_ptr + h_kv * stride_vnh + current_t[:, None] * stride_vnt + offs_d[None, :],
                     v_base + physical_n[:, None] * stride_vr + offs_d[None, :]),
            mask=n_valid[:, None] & owned,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    if NUM_SPLITS == 1:
        out = acc * tl.where(l_i > 0, 1.0 / l_i, 0.0)[:, None]
        tl.store(
            O_ptr + h_q[:, None] * stride_oh + t * stride_ot + offs_d[None, :],
            out.to(O_ptr.dtype.element_ty),
            mask=g_valid[:, None],
        )
    else:
        part = (split * Hq + h_q) * T + t
        tl.store(
            O_part_ptr + part[:, None] * HEAD_DIM + offs_d[None, :],
            acc,
            mask=g_valid[:, None],
        )
        tl.store(M_part_ptr + part, m_i, mask=g_valid)
        tl.store(L_part_ptr + part, l_i, mask=g_valid)


@triton.jit
def _reduce(
    O_part_ptr,
    M_part_ptr,
    L_part_ptr,
    O_ptr,
    Hq,
    T,
    stride_oh,
    stride_ot,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    h_q = pid // T
    t = pid % T
    offs_s = tl.arange(0, BLOCK_S)
    s_valid = offs_s < NUM_SPLITS
    part = (offs_s * Hq + h_q) * T + t
    m = tl.load(M_part_ptr + part, mask=s_valid, other=-float("inf"))
    m_max = tl.max(m, axis=0)
    alpha = tl.where(s_valid & (m > -float("inf")), tl.exp(m - m_max), 0.0)
    l_sum = tl.sum(tl.load(L_part_ptr + part, mask=s_valid, other=0.0) * alpha, 0)
    offs_d = tl.arange(0, HEAD_DIM)
    o = tl.load(
        O_part_ptr + part[:, None] * HEAD_DIM + offs_d[None, :],
        mask=s_valid[:, None],
        other=0.0,
    )
    out = tl.sum(o * alpha[:, None], axis=0) * tl.where(l_sum > 0, 1.0 / l_sum, 0.0)
    tl.store(
        O_ptr + h_q * stride_oh + t * stride_ot + offs_d,
        out.to(O_ptr.dtype.element_ty),
    )



@triton.jit
def _commit(KH, VH, KN, VN, Info, KJ, VJ,
            JOURNAL: tl.constexpr,
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
    ds = tl.arange(0, D)
    if JOURNAL:
        # One packed staging row per real token, shared by all session owners.
        k = tl.load(KN + head * sknh + t * sknt + ds)
        v = tl.load(VN + head * svnh + t * svnt + ds)
        tl.store(KJ + (t * tl.num_programs(1) + head) * D + ds, k)
        tl.store(VJ + (t * tl.num_programs(1) + head) * D + ds, v)
    # Only the last W writes can survive. Earlier writes may alias these rows.
    col = tl.load(Info + seg * INFO + 2).to(tl.int64)
    end = tl.load(Info + seg * INFO + 3)
    columns_of_segments = tl.load(Info + segs * INFO + 2, segs < SEGMENTS, other=-1)
    ends_of_segments = tl.load(Info + segs * INFO + 3, segs < SEGMENTS, other=0)
    final_end = tl.max(tl.where((columns_of_segments == col) & (lens > 0), ends_of_segments, 0))
    if end - qlen + t - qstart < final_end - W:
        return
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
    # Keep the existing row-complete split-K route for native small decode.
    # The wrapper uses only static shapes; graph replay still reads device metadata.
    use_decode = isinstance(total, int) and total <= (16 if dim <= 128 else 4)
    if use_decode:
        splits = min(16, triton.cdiv(window, 256))
        out = torch.empty_like(query)
        partial = torch.empty((splits, heads, total, dim), device=query.device, dtype=torch.float32)
        maximum = torch.empty((splits, heads, total), device=query.device, dtype=torch.float32)
        normalizer = torch.empty_like(maximum)
        wrap_triton(_decode)[(total, kvheads, splits)](
            query, history_key, history_value, current_key, current_value,
            current_key.stride(1), current_key.stride(2), current_value.stride(1), current_value.stride(2),
            out, partial, maximum, normalizer, seq_info, heads, total,
            query.stride(1), query.stride(2), history_key.stride(0), history_key.stride(1), history_key.stride(2),
            history_value.stride(0), history_value.stride(1), history_value.stride(2), out.stride(1), out.stride(2),
            scale, window, seq_info.shape[0], seq_info.shape[1], splits, True, False,
            window, window, 128 if dim <= 128 else 64, dim, groups, triton.next_power_of_2(groups),
            num_warps=4, num_stages=2 if dim <= 128 else 1)
        wrap_triton(_reduce)[(heads * total,)](partial, maximum, normalizer, out, heads, total,
            out.stride(1), out.stride(2), splits, triton.next_power_of_2(splits), dim, num_warps=4)
        return out
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
        history_key, history_value, current_key, current_value, seq_info, history_key, history_value, False,
        history_key.stride(1), history_key.stride(2), history_value.stride(1), history_value.stride(2),
        current_key.stride(1), current_key.stride(2), current_value.stride(1), current_value.stride(2),
        current_key.shape[2], window, current_key.shape[3], seq_info.shape[1], seq_info.shape[0],
        num_warps=4)


@triton_op('triton::ring_commit_journal', mutates_args={'history_key', 'history_value', 'staging'})
def ring_commit_journal(history_key: torch.Tensor, history_value: torch.Tensor,
                        current_key: torch.Tensor, current_value: torch.Tensor,
                        seq_info: torch.Tensor, window: int,
                        staging: torch.Tensor, ring_layer: int) -> None:
    # staging is [ring_layer, K_or_V, max_write, kv_heads, head_dim].
    stage_key = staging[ring_layer, 0]
    stage_value = staging[ring_layer, 1]
    wrap_triton(_commit)[(current_key.shape[2], current_key.shape[1])](
        history_key, history_value, current_key, current_value, seq_info, stage_key, stage_value, True,
        history_key.stride(1), history_key.stride(2), history_value.stride(1), history_value.stride(2),
        current_key.stride(1), current_key.stride(2), current_value.stride(1), current_value.stride(2),
        current_key.shape[2], window, current_key.shape[3], seq_info.shape[1], seq_info.shape[0],
        num_warps=4)
