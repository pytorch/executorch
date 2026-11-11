# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)

"""Length-aware SDPA over a batched-sequence off-graph KV cache.

A step packs the tokens of several sequences on one axis; each sequence keeps
its K/V in its own column of a ``[rows, columns, H_kv, D]`` storage, row = its
logical position (or ``position % ring`` for a sliding-window ring). Each
sequence's query rows attend only that sequence's column, bounded by its own
length, so a step costs the sum of the contexts rather than a sweep of every
sequence's cells.

The runtime describes the step in ``seq_info`` (``Int32[S, 4]``), one row per
segment of consecutive query rows of one sequence:

* ``q_start``, ``q_len``: the segment's rows on the packed query axis
  (``q_len == 0`` marks an unused row);
* ``col``: the sequence's storage column;
* ``kv_end``: one past the logical position of the segment's last row, i.e. the
  sequence's length once this step's writes land.

Rows are the segment's newest positions: row ``r`` sits at
``kv_end - q_len + r``. Active segments must be disjoint and in bounds;
``seq_info`` is read on the device, so a captured CUDA graph replays it with new
contents. Head dimension ``D`` must be a power of two in ``[16, 256]``.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

_SPLITK_MAX_QUERY = 4
_SPLITK_ROWS_PER_SPLIT = 256
_SPLITK_MAX_SPLITS = 16


@triton.jit
def _sdpa_varlen_kernel(  # noqa: C901
    Q_ptr,
    K_ptr,
    V_ptr,
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
    num_qb,
    NUM_SPLITS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    CONTEXT_WINDOW: tl.constexpr,
    WINDOW: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    sm_scale = sm_scale.to(tl.float32)
    pid0 = tl.program_id(0)
    h_kv = tl.program_id(1)
    split = tl.program_id(2)
    seg = pid0 // num_qb
    qb = pid0 % num_qb

    q_start = tl.load(Info_ptr + seg * 4 + 0)
    q_len = tl.load(Info_ptr + seg * 4 + 1)
    col = tl.load(Info_ptr + seg * 4 + 2)
    kv_end = tl.load(Info_ptr + seg * 4 + 3)
    if qb * BLOCK_M >= q_len * NUM_GROUPS:
        return

    offs_p = qb * BLOCK_M + tl.arange(0, BLOCK_M)
    r = offs_p // NUM_GROUPS
    h_q = h_kv * NUM_GROUPS + offs_p % NUM_GROUPS
    row_valid = r < q_len
    t = q_start + r
    pos = kv_end - q_len + r
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q_ptr + h_q[:, None] * stride_qh + t[:, None] * stride_qt + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )

    # The rows this tile may see, before the split divides them.
    r_last = tl.minimum(q_len - 1, (qb * BLOCK_M + BLOCK_M - 1) // NUM_GROUPS)
    r_first = (qb * BLOCK_M) // NUM_GROUPS
    if RING > 0:
        lo = 0
        hi = tl.minimum(kv_end, RING)
    else:
        hi = kv_end
        if IS_CAUSAL:
            hi = kv_end - q_len + r_last + 1
        lo = 0
        if WINDOW > 0:
            if CONTEXT_WINDOW:
                lo = tl.maximum(kv_end - q_len - WINDOW, 0)
            else:
                lo = tl.maximum(kv_end - q_len + r_first - WINDOW + 1, 0)
    hi = tl.minimum(hi, rows_available)
    span = tl.maximum(hi - lo, 0)
    chunk = tl.cdiv(tl.cdiv(span, NUM_SPLITS), BLOCK_N) * BLOCK_N
    start = lo + split * chunk
    stop = tl.minimum(start + chunk, hi)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    k_base = K_ptr + col * stride_kc + h_kv * stride_kh
    v_base = V_ptr + col * stride_vc + h_kv * stride_vh

    for n0 in tl.range(start, stop, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_valid = offs_n < stop
        k = tl.load(
            k_base + offs_n[:, None] * stride_kr + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        if RING > 0:
            # The newest logical position that landed in each ring row.
            kv_pos = offs_n + RING * ((kv_end - 1 - offs_n) // RING)
        else:
            kv_pos = offs_n
        visible = n_valid[None, :] & row_valid[:, None] & (kv_pos[None, :] >= 0)
        if IS_CAUSAL:
            visible = visible & (kv_pos[None, :] <= pos[:, None])
        else:
            visible = visible & (kv_pos[None, :] < kv_end)
        if WINDOW > 0:
            if CONTEXT_WINDOW:
                visible = visible & (kv_pos[None, :] >= kv_end - q_len - WINDOW)
            else:
                visible = visible & (kv_pos[None, :] > pos[:, None] - WINDOW)
        qk = tl.where(visible, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.where(m_ij[:, None] > -float("inf"), tl.exp(qk - m_ij[:, None]), 0.0)
        alpha = tl.where(m_ij > -float("inf"), tl.exp(m_i - m_ij), 0.0)
        v = tl.load(
            v_base + offs_n[:, None] * stride_vr + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    if NUM_SPLITS == 1:
        out = acc * tl.where(l_i > 0, 1.0 / l_i, 0.0)[:, None]
        tl.store(
            O_ptr + h_q[:, None] * stride_oh + t[:, None] * stride_ot + offs_d[None, :],
            out.to(O_ptr.dtype.element_ty),
            mask=row_valid[:, None],
        )
    else:
        part = (split * Hq + h_q) * T + t
        tl.store(
            O_part_ptr + part[:, None] * HEAD_DIM + offs_d[None, :],
            acc,
            mask=row_valid[:, None],
        )
        tl.store(M_part_ptr + part, m_i, mask=row_valid)
        tl.store(L_part_ptr + part, l_i, mask=row_valid)


@triton.jit
def _sdpa_varlen_decode_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
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
    t = tl.program_id(0)
    h_kv = tl.program_id(1)
    split = tl.program_id(2)

    owned = False
    q_start = 0
    q_len = 0
    col = 0
    kv_end = 0
    for seg in range(NUM_SEGMENTS):
        seg_q_start = tl.load(Info_ptr + seg * 4)
        seg_q_len = tl.load(Info_ptr + seg * 4 + 1)
        take = (t >= seg_q_start) & (t < seg_q_start + seg_q_len)
        owned |= take
        q_start = tl.where(take, seg_q_start, q_start)
        q_len = tl.where(take, seg_q_len, q_len)
        col = tl.where(take, tl.load(Info_ptr + seg * 4 + 2), col)
        kv_end = tl.where(take, tl.load(Info_ptr + seg * 4 + 3), kv_end)

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

    pos = kv_end - q_len + t - q_start
    if RING > 0:
        lo = 0
        hi = tl.minimum(kv_end, RING)
    else:
        hi = pos + 1 if IS_CAUSAL else kv_end
        lo = 0
        if WINDOW > 0:
            if CONTEXT_WINDOW:
                lo = tl.maximum(kv_end - q_len - WINDOW, 0)
            else:
                lo = tl.maximum(pos - WINDOW + 1, 0)
    hi = tl.minimum(hi, rows_available)
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
        n_valid = offs_n < stop
        k = tl.load(
            k_base + offs_n[:, None] * stride_kr + offs_d[None, :],
            mask=n_valid[:, None] & owned,
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        if RING > 0:
            kv_pos = offs_n + RING * ((kv_end - 1 - offs_n) // RING)
        else:
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
            v_base + offs_n[:, None] * stride_vr + offs_d[None, :],
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
def _sdpa_varlen_reduce_kernel(
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


def _num_splits(T: int, rows: int) -> int:
    if T > _SPLITK_MAX_QUERY:
        return 1
    return max(1, min(_SPLITK_MAX_SPLITS, triton.cdiv(rows, _SPLITK_ROWS_PER_SPLIT)))


@triton_op("triton::sdpa_varlen", mutates_args={})
def sdpa_varlen(  # noqa: C901
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_info: torch.Tensor,
    scale: float,
    is_causal: bool = True,
    window: int = 0,
    ring: int = 0,
    rows_available: int = -1,
    context_window: bool = False,
) -> torch.Tensor:
    """Attend each segment of ``query`` to its own sequence's K/V.

    Args:
        query: ``[1, H_q, T, D]`` bfloat16, the step's queries in packed order.
        key, value: ``[rows, columns, H_kv, D]`` bfloat16 storage; one column per
            sequence. Only the head dimension must be contiguous.
        seq_info: ``Int32[S, 4]`` on the query's device, ``(q_start, q_len,
            col, kv_end)`` per segment; see the module docstring.
        scale: softmax scale.
        is_causal: row ``r`` sees positions up to its own; otherwise every
            row sees all of ``[0, kv_end)``.
        window: a sliding window: a row sees only its last ``window``
            positions. Zero for full attention.
        ring: the storage holds a ring of this many rows, indexed by
            ``position % ring``. Zero when the row is the position.
        rows_available: rows actually backed by storage, when fewer than its
            declared extent; negative means all of them.
        context_window: non-causal draft queries share the last ``window``
            context positions before ``kv_end - q_len`` and see the entire
            query block. Ordinary attention keeps its per-query window.
    Returns:
        ``[1, H_q, T, D]``; rows no segment covers are zero.
    """
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be rank 4")
    if query.shape[0] != 1:
        raise ValueError("query must have shape [1, H_q, T, D]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    _, Hq, T, D = query.shape
    rows, _, Hkv, Dkv = key.shape
    if D != Dkv:
        raise ValueError("query, key, and value must have the same head dimension")
    if (
        query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
    ):
        raise ValueError("query, key, and value must be bfloat16")
    if (
        query.device.type != "cuda"
        or key.device != query.device
        or value.device != query.device
    ):
        raise ValueError("query, key, and value must be on the same CUDA device")
    if Hkv == 0 or Hq % Hkv != 0:
        raise ValueError(f"H_q={Hq} is not a multiple of H_kv={Hkv}")
    if D < 16 or D > 256 or D & (D - 1):
        raise ValueError("head dimension must be a power of two in [16, 256]")
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("sdpa_varlen needs a contiguous head dimension")
    if seq_info.dtype != torch.int32 or seq_info.dim() != 2 or seq_info.shape[1] != 4:
        raise ValueError("seq_info must be Int32[S, 4]")
    if seq_info.device != query.device or not seq_info.is_contiguous():
        raise ValueError("seq_info must be contiguous on the query device")
    if context_window and is_causal:
        raise ValueError("context_window requires non-causal attention")
    groups = Hq // Hkv
    available = rows if rows_available < 0 else min(rows_available, rows)
    # Branching on a symbolic T would guard (and AOTI then pin) the exported
    # range, so a dynamic step takes the generic path, valid for any T.
    static = isinstance(T, int)
    use_decode = static and T <= _SPLITK_MAX_QUERY
    block_m = 16 if static and T * groups <= 16 else 64
    num_qb = (T * groups + block_m - 1) // block_m
    generic_num_stages = 3 if static and T <= 64 else 2
    splits = _num_splits(T, rows) if static else 1
    # The row-complete decode path writes uncovered rows and empty split slots.
    out = torch.empty_like(query) if use_decode else torch.zeros_like(query)
    if splits > 1:
        o_part = torch.empty(
            (splits, Hq, T, D), device=query.device, dtype=torch.float32
        )
        m_part = torch.empty((splits, Hq, T), device=query.device, dtype=torch.float32)
        l_part = torch.empty((splits, Hq, T), device=query.device, dtype=torch.float32)
    else:
        o_part = m_part = l_part = out
    if use_decode:
        grid = (T, Hkv, splits)
        wrap_triton(_sdpa_varlen_decode_kernel)[grid](
            query,
            key,
            value,
            out,
            o_part,
            m_part,
            l_part,
            seq_info,
            Hq,
            T,
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            out.stride(1),
            out.stride(2),
            scale,
            available,
            NUM_SEGMENTS=seq_info.shape[0],
            NUM_SPLITS=splits,
            IS_CAUSAL=is_causal,
            CONTEXT_WINDOW=context_window,
            WINDOW=window,
            RING=ring,
            BLOCK_N=128,
            HEAD_DIM=D,
            NUM_GROUPS=groups,
            BLOCK_G=triton.next_power_of_2(groups),
            num_warps=4,
            num_stages=2,
        )
    else:
        grid = (seq_info.shape[0] * num_qb, Hkv, splits)
        wrap_triton(_sdpa_varlen_kernel)[grid](
            query,
            key,
            value,
            out,
            o_part,
            m_part,
            l_part,
            seq_info,
            Hq,
            T,
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            out.stride(1),
            out.stride(2),
            scale,
            available,
            num_qb,
            NUM_SPLITS=splits,
            IS_CAUSAL=is_causal,
            CONTEXT_WINDOW=context_window,
            WINDOW=window,
            RING=ring,
            BLOCK_M=block_m,
            BLOCK_N=64,
            HEAD_DIM=D,
            NUM_GROUPS=groups,
            num_warps=4,
            num_stages=generic_num_stages,
        )
    if splits > 1:
        wrap_triton(_sdpa_varlen_reduce_kernel)[(Hq * T,)](
            o_part,
            m_part,
            l_part,
            out,
            Hq,
            T,
            out.stride(1),
            out.stride(2),
            NUM_SPLITS=splits,
            BLOCK_S=triton.next_power_of_2(splits),
            HEAD_DIM=D,
        )
    return out


@sdpa_varlen.register_fake
def _sdpa_varlen_fake(
    query,
    key,
    value,
    seq_info,
    scale,
    is_causal=True,
    window=0,
    ring=0,
    rows_available=-1,
    context_window=False,
):
    return torch.empty_like(query)
