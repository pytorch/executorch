# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BF16 runtime-owned KV-cache primitives for the AOTI CUDA backend."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from executorch.backends.cuda.triton.kernels.sdpa import (
    _decode_splitk_config,
    _next_power_of_2_unclamped,
    _sdpa_decode_reduce_kernel,
    _sdpa_fwd_kernel_body,
    _should_pack_gqa,
)
from torch.library import triton_op, wrap_triton


FLAT_CACHE = 0
RING_CACHE = 1


def ring_physical_capacity(window: int, max_write: int) -> int:
    """Slots a ring layer needs to serve one step of up to ``max_write`` tokens.

    A step writes all its tokens before attending, and its earliest query still
    reads back to ``position - window + 1``, so ``window + max_write - 1``
    positions must be live at once. Sizing the ring to the window alone lets a
    step overwrite cells its own earlier queries still attend to.

    Matches ``RingPolicy`` in executorch/extension/llm/cache/sequence_cache.h;
    the CUDA runtime applies the same formula when it allocates.
    """
    if window <= 0:
        raise ValueError("ring cache requires a positive window")
    if max_write <= 0:
        raise ValueError("ring cache requires a positive max_write")
    return window + max_write - 1


@triton.jit
def _update_cache_kernel(
    K,
    V,
    Position,
    KStorage,
    VStorage,
    PhysicalCapacity,
    H,
    T,
    D,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    CACHE_POLICY: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_input = (offs_t[:, None] < T) & (offs_d[None, :] < D)
    logical_pos = tl.load(Position + offs_t, mask=offs_t < T, other=0)
    capacity = tl.load(PhysicalCapacity)
    if pid_bh == 0 and pid_t == 0 and pid_d == 0:
        tl.store(PhysicalCapacity, capacity)

    if CACHE_POLICY == 1:
        physical_pos = logical_pos % capacity
        valid_store = valid_input
    else:
        physical_pos = logical_pos
        valid_store = valid_input & (logical_pos[:, None] < capacity)

    k = tl.load(
        K
        + b * stride_kb
        + h * stride_kh
        + offs_t[:, None] * stride_kt
        + offs_d[None, :] * stride_kd,
        mask=valid_input,
        other=0.0,
    )
    v = tl.load(
        V
        + b * stride_vb
        + h * stride_vh
        + offs_t[:, None] * stride_vt
        + offs_d[None, :] * stride_vd,
        mask=valid_input,
        other=0.0,
    )
    storage_offset = (
        (b * H + h) * capacity * D + physical_pos[:, None] * D + offs_d[None, :]
    )
    tl.store(KStorage + storage_offset, k, mask=valid_store)
    tl.store(VStorage + storage_offset, v, mask=valid_store)


@triton.jit
def _offgraph_sdpa_kernel(
    Q,
    KStorage,
    VStorage,
    Out,
    Position,
    PhysicalCapacity,
    B,
    HGrid,
    HKV,
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
    sm_scale: tl.float32,
    HEAD_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACK_GQA: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _sdpa_fwd_kernel_body(
        Q,
        KStorage,
        VStorage,
        Out,
        0,
        0,
        Position,
        PhysicalCapacity,
        B,
        HGrid,
        HKV,
        Lq,
        Lk,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        0,
        0,
        0,
        sm_scale,
        HAS_MASK=False,
        IS_CAUSAL=True,
        HAS_KV_LEN=False,
        OFFGRAPH_KV=True,
        WINDOW_SIZE=WINDOW_SIZE,
        HEAD_DIM=HEAD_DIM,
        NUM_GROUPS=NUM_GROUPS,
        PACK_GQA=PACK_GQA,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


@triton.jit
def _offgraph_decode_splitk_kernel(
    Q,
    KStorage,
    VStorage,
    Position,
    PhysicalCapacity,
    OPartial,
    MPartial,
    LPartial,
    H_kv,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_op_s,
    stride_op_b,
    stride_op_h,
    stride_op_d,
    stride_mp_s,
    stride_mp_b,
    stride_mp_h,
    stride_lp_s,
    stride_lp_b,
    stride_lp_h,
    sm_scale: tl.float32,
    chunk_size,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    split_id = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H_kv
    h_kv = pid_bh % H_kv
    capacity = tl.load(PhysicalCapacity)
    kv_len = tl.load(Position) + 1
    loop_start = tl.maximum(0, kv_len - WINDOW_SIZE) if WINDOW_SIZE else 0
    start_n = loop_start + split_id * chunk_size
    end_n = tl.minimum(start_n + chunk_size, kv_len)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_g = tl.arange(0, BLOCK_G)
    g_valid = offs_g < NUM_GROUPS
    h_q = h_kv * NUM_GROUPS + offs_g
    q_ptrs = Q + (
        b * stride_qb
        + h_q[:, None] * stride_qh
        + 0 * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=g_valid[:, None], other=0.0).to(tl.bfloat16)
    m_i = tl.full([BLOCK_G], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, HEAD_DIM], dtype=tl.float32)
    offs_n_init = tl.arange(0, BLOCK_N)

    for tile_start in tl.range(start_n, end_n, BLOCK_N):
        offs_n = tile_start + offs_n_init
        n_valid = offs_n < end_n
        physical_n = offs_n % capacity if WINDOW_SIZE else offs_n
        storage_offset = (
            (b * H_kv + h_kv) * capacity * HEAD_DIM
            + physical_n[:, None] * HEAD_DIM
            + offs_d[None, :]
        )
        k = tl.load(KStorage + storage_offset, mask=n_valid[:, None], other=0.0)
        qk = (tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale).to(tl.float32)
        qk = tl.where(
            n_valid[None, :],
            qk,
            tl.full(qk.shape, -float("inf"), dtype=tl.float32),
        )
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1).to(tl.float32))
        safe_diff = tl.where(
            m_ij[:, None] > -float("inf"), qk - m_ij[:, None], -float("inf")
        )
        p = tl.exp(safe_diff).to(tl.float32)
        l_ij = tl.sum(p, axis=1).to(tl.float32)
        alpha = tl.exp(tl.where(m_ij > -float("inf"), m_i - m_ij, 0.0)).to(tl.float32)
        v = tl.load(VStorage + storage_offset, mask=n_valid[:, None], other=0.0)
        acc = (acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)).to(tl.float32)
        l_i = (l_i * alpha + l_ij).to(tl.float32)
        m_i = m_ij

    o_ptrs = OPartial + (
        split_id * stride_op_s
        + b * stride_op_b
        + h_q[:, None] * stride_op_h
        + offs_d[None, :] * stride_op_d
    )
    m_ptrs = MPartial + split_id * stride_mp_s + b * stride_mp_b + h_q * stride_mp_h
    l_ptrs = LPartial + split_id * stride_lp_s + b * stride_lp_b + h_q * stride_lp_h
    tl.store(o_ptrs, acc, mask=g_valid[:, None])
    tl.store(m_ptrs, m_i, mask=g_valid)
    tl.store(l_ptrs, l_i, mask=g_valid)


def _launch_offgraph_decode_splitk(
    q: torch.Tensor,
    k_storage: torch.Tensor,
    v_storage: torch.Tensor,
    position: torch.Tensor,
    physical_capacity: torch.Tensor,
    out: torch.Tensor,
    h_kv: int,
    max_capacity: int,
    window_size: int,
    scale: float,
) -> None:
    B, h_q, _, head_dim = q.shape
    sweep_length = window_size if window_size else max_capacity
    num_splits, chunk_size = _decode_splitk_config(sweep_length, B * h_kv, q.device)
    o_partial = torch.empty(
        (num_splits, B, h_q, head_dim), device=q.device, dtype=torch.float32
    )
    m_partial = torch.empty((num_splits, B, h_q), device=q.device, dtype=torch.float32)
    l_partial = torch.empty_like(m_partial)
    groups = h_q // h_kv
    wrap_triton(_offgraph_decode_splitk_kernel)[(num_splits, B * h_kv)](
        q,
        k_storage,
        v_storage,
        position,
        physical_capacity,
        o_partial,
        m_partial,
        l_partial,
        h_kv,
        *q.stride(),
        *o_partial.stride(),
        *m_partial.stride(),
        *l_partial.stride(),
        scale,
        chunk_size,
        WINDOW_SIZE=window_size,
        BLOCK_N=128,
        HEAD_DIM=head_dim,
        NUM_GROUPS=groups,
        BLOCK_G=_next_power_of_2_unclamped(groups),
        num_warps=4,
        num_stages=2,
    )
    wrap_triton(_sdpa_decode_reduce_kernel)[(B * h_q,)](
        o_partial,
        m_partial,
        l_partial,
        out,
        num_splits,
        o_partial.stride(0),
        o_partial.stride(2),
        o_partial.stride(3),
        m_partial.stride(0),
        m_partial.stride(2),
        l_partial.stride(0),
        l_partial.stride(2),
        out.stride(1),
        out.stride(3),
        BLOCK_S=_next_power_of_2_unclamped(num_splits),
        HEAD_DIM=head_dim,
        num_warps=4,
        num_stages=1,
    )


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position: torch.Tensor,
    k_storage: torch.Tensor,
    v_storage: torch.Tensor,
    physical_capacity: torch.Tensor,
    cache_policy: int,
    window_size: int,
    out_dtype: torch.dtype,
) -> None:
    if not all(
        tensor.is_cuda
        for tensor in (q, k, v, position, k_storage, v_storage, physical_capacity)
    ):
        raise RuntimeError("off-graph KV tensors must be CUDA tensors")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError("off-graph KV cache requires BF16 Q, K, and V")
    if k_storage.dtype != q.dtype or v_storage.dtype != q.dtype:
        raise RuntimeError("off-graph KV storage must be BF16")
    if out_dtype != torch.bfloat16:
        raise RuntimeError("off-graph KV cache currently supports BF16 output only")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise RuntimeError("Q, K, and V must be BHSD tensors")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise RuntimeError("off-graph KV cache currently supports batch size one")
    if k.shape != v.shape or q.shape[2:] != k.shape[2:]:
        raise RuntimeError("Q, K, and V sequence/head dimensions must match")
    if q.shape[1] % k.shape[1] != 0:
        raise RuntimeError("Q heads must be divisible by KV heads")
    if position.dim() != 1 or position.shape[0] != q.shape[2]:
        raise RuntimeError("position must contain one entry per query token")
    if physical_capacity.numel() != 1:
        raise RuntimeError("physical_capacity must be a scalar tensor")
    if cache_policy not in (FLAT_CACHE, RING_CACHE):
        raise RuntimeError(f"unsupported cache policy {cache_policy}")
    if cache_policy == RING_CACHE and window_size <= 0:
        raise RuntimeError("ring cache requires a positive window size")
    if cache_policy == FLAT_CACHE and window_size != 0:
        raise RuntimeError("flat cache window size must be zero")
    if not torch.compiler.is_compiling():
        capacity = int(physical_capacity.item())
        if capacity <= 0:
            raise RuntimeError("physical_capacity must be positive")
        if cache_policy == FLAT_CACHE and int(position[-1].item()) >= capacity:
            raise RuntimeError("flat KV write exceeds physical capacity")


@triton_op(
    "triton::cuda_offgraph_update_and_attend",
    mutates_args={"k_storage", "v_storage", "physical_capacity"},
)
def cuda_offgraph_update_and_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position: torch.Tensor,
    k_storage: torch.Tensor,
    v_storage: torch.Tensor,
    physical_capacity: torch.Tensor,
    cache_policy: int,
    window_size: int,
    scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Append BF16 K/V to runtime storage and attend over valid history."""
    _validate_inputs(
        q,
        k,
        v,
        position,
        k_storage,
        v_storage,
        physical_capacity,
        cache_policy,
        window_size,
        out_dtype,
    )
    B, H_q, L_q, D = q.shape
    H_kv = k.shape[1]
    max_capacity = k_storage.numel() // (B * H_kv * D)
    if max_capacity <= 0:
        raise RuntimeError("off-graph KV storage metadata has zero capacity")

    def update_grid(meta):
        return (
            B * H_kv,
            triton.cdiv(L_q, meta["BLOCK_T"]),
            triton.cdiv(D, meta["BLOCK_D"]),
        )

    wrap_triton(_update_cache_kernel)[update_grid](
        k,
        v,
        position,
        k_storage,
        v_storage,
        physical_capacity,
        H_kv,
        L_q,
        D,
        *k.stride(),
        *v.stride(),
        CACHE_POLICY=cache_policy,
        BLOCK_T=16,
        BLOCK_D=64,
    )

    out = torch.empty((B, H_q, L_q, D), device=q.device, dtype=out_dtype)
    sm_scale = 1.0 / math.sqrt(D) if scale == 0.0 else scale
    if L_q == 1 and max_capacity >= 256:
        _launch_offgraph_decode_splitk(
            q,
            k_storage,
            v_storage,
            position,
            physical_capacity,
            out,
            H_kv,
            max_capacity,
            window_size,
            sm_scale,
        )
        return out

    num_groups = H_q // H_kv
    pack_gqa = _should_pack_gqa(L_q, num_groups, 64)
    h_grid = H_kv if pack_gqa else H_q
    lq_packed = L_q * num_groups if pack_gqa else L_q

    def attention_grid(meta):
        return (triton.cdiv(lq_packed, meta["BLOCK_M"]), B * h_grid)

    wrap_triton(_offgraph_sdpa_kernel)[attention_grid](
        q,
        k_storage,
        v_storage,
        out,
        position,
        physical_capacity,
        B,
        h_grid,
        H_kv,
        L_q,
        max_capacity,
        *q.stride(),
        *out.stride(),
        sm_scale,
        HEAD_DIM=D,
        NUM_GROUPS=num_groups,
        PACK_GQA=pack_gqa,
        WINDOW_SIZE=window_size,
        BLOCK_M=32,
        BLOCK_N=64,
        num_warps=4,
        num_stages=2,
    )
    return out
