# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scatter packed step K/V to independent owner spans without ring races."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _private_kv_stage_kernel(
    K_ptr,
    V_ptr,
    Rows_ptr,
    K_stage_ptr,
    V_stage_ptr,
    T,
    H,
    D,
    stride_kh,
    stride_kt,
    stride_vh,
    stride_vt,
    BLOCK: tl.constexpr,
):
    lanes = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = lanes // (H * D)
    head = (lanes // D) % H
    dim = lanes % D
    valid = token < T
    row = tl.load(Rows_ptr + token, mask=valid, other=0)
    k = tl.load(
        K_ptr + head * stride_kh + token * stride_kt + dim, mask=valid, other=0.0
    )
    v = tl.load(
        V_ptr + head * stride_vh + token * stride_vt + dim, mask=valid, other=0.0
    )
    dst = row * H * D + head * D + dim
    tl.store(K_stage_ptr + dst, k, mask=valid)
    tl.store(V_stage_ptr + dst, v, mask=valid)


@triton_op("triton::private_kv_stage", mutates_args={"key_stage", "value_stage"})
def private_kv_stage(
    key: torch.Tensor,
    value: torch.Tensor,
    stage_rows: torch.Tensor,
    key_stage: torch.Tensor,
    value_stage: torch.Tensor,
) -> None:
    """One launch copies both K/V planes into the runtime's unique step slots."""
    if key.dim() != 4 or key.shape[0] != 1 or value.shape != key.shape:
        raise ValueError("private staging requires matching BHSD K/V")
    _, heads, tokens, dim = key.shape
    if (
        key.dtype != torch.bfloat16
        or value.dtype != key.dtype
        or key.stride(-1) != 1
        or value.stride(-1) != 1
        or key_stage.dim() != 4
        or key_stage.shape[0] != 1
        or key_stage.shape[1] < tokens
        or key_stage.shape[2:] != (heads, dim)
        or value_stage.shape != key_stage.shape
        or key_stage.dtype != key.dtype
        or value_stage.dtype != key.dtype
        or not key_stage.is_contiguous()
        or not value_stage.is_contiguous()
        or stage_rows.dim() != 1
        or stage_rows.shape[0] < tokens
        or stage_rows.dtype != torch.int64
        or not stage_rows.is_contiguous()
    ):
        raise ValueError("invalid private staging tensors")
    if key.device.type != "cuda" or any(
        t.device != key.device for t in (value, stage_rows, key_stage, value_stage)
    ):
        raise ValueError("private staging tensors must share a CUDA device")
    wrap_triton(_private_kv_stage_kernel)[(triton.cdiv(tokens * heads * dim, 256),)](
        key,
        value,
        stage_rows,
        key_stage,
        value_stage,
        tokens,
        heads,
        dim,
        key.stride(1),
        key.stride(2),
        value.stride(1),
        value.stride(2),
        BLOCK=256,
    )


@private_kv_stage.register_fake
def _private_kv_stage_fake(key, value, stage_rows, key_stage, value_stage):
    return None


@triton.jit
def _private_kv_write_kernel(
    K_ptr,
    V_ptr,
    Position_ptr,
    Cols_ptr,
    Layout_ptr,
    Step_ptr,
    K_cache_ptr,
    V_cache_ptr,
    T,
    H,
    D,
    stride_kh,
    stride_kt,
    stride_vh,
    stride_vt,
    OWNERS: tl.constexpr,
    RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lanes = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = lanes // (H * D)
    head = (lanes // D) % H
    dim = lanes % D
    in_step = token < T
    col = tl.load(Cols_ptr + token, mask=in_step, other=OWNERS)
    pos = tl.load(Position_ptr + token, mask=in_step, other=0)
    base = tl.load(Layout_ptr + col * 2).to(tl.int64)
    capacity = tl.load(Layout_ptr + col * 2 + 1)
    valid = in_step & (col < OWNERS) & (capacity > 0) & (pos >= 0)
    start = tl.load(Step_ptr + col * 3 + 1)
    end = start + tl.load(Step_ptr + col * 3 + 2)
    valid &= (pos >= start) & (pos < end)
    if RING:
        # A chunk can wrap repeatedly. Only its newest write to each slot
        # survives, so no two programs race for the same K/V element.
        valid &= pos >= end - capacity
        row = pos % tl.maximum(capacity, 1)
    else:
        valid &= pos < capacity
        row = pos
    k = tl.load(
        K_ptr + head * stride_kh + token * stride_kt + dim, mask=valid, other=0.0
    )
    v = tl.load(
        V_ptr + head * stride_vh + token * stride_vt + dim, mask=valid, other=0.0
    )
    dst = (base + row) * H * D + head * D + dim
    tl.store(K_cache_ptr + dst, k, mask=valid)
    tl.store(V_cache_ptr + dst, v, mask=valid)


@triton_op("triton::private_kv_write", mutates_args={"key_cache", "value_cache"})
def private_kv_write(
    key: torch.Tensor,
    value: torch.Tensor,
    position: torch.Tensor,
    seq_cols: torch.Tensor,
    owner_layout: torch.Tensor,
    owner_step: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    ring: bool = False,
) -> None:
    """BHSD step to packed BSHD storage; padding has the final scratch owner."""
    if key.dim() != 4 or key.shape[0] != 1 or value.shape != key.shape:
        raise ValueError("private writes require matching BHSD K/V with batch one")
    _, heads, tokens, dim = key.shape
    if (
        key.dtype != torch.bfloat16
        or value.dtype != key.dtype
        or key.stride(-1) != 1
        or value.stride(-1) != 1
        or key.device.type != "cuda"
    ):
        raise ValueError("private writes require CUDA BF16 K/V with dense head dim")
    if (
        key_cache.dim() != 4
        or key_cache.shape[0] != 1
        or key_cache.shape[2:] != (heads, dim)
        or value_cache.shape != key_cache.shape
        or key_cache.dtype != key.dtype
        or value_cache.dtype != key.dtype
        or not key_cache.is_contiguous()
        or not value_cache.is_contiguous()
    ):
        raise ValueError("private cache must be dense BSHD K/V")
    if (
        position.dim() != 1
        or position.shape[0] != tokens
        or position.dtype != torch.int64
        or not position.is_contiguous()
        or seq_cols.dim() != 1
        or seq_cols.shape[0] < tokens
        or seq_cols.dtype != torch.int64
        or not seq_cols.is_contiguous()
        or owner_layout.dim() != 2
        or owner_layout.shape[1] != 2
        or owner_layout.dtype != torch.int32
        or not owner_layout.is_contiguous()
        or owner_step.shape != (owner_layout.shape[0], 3)
        or owner_step.dtype != torch.int32
        or not owner_step.is_contiguous()
    ):
        raise ValueError("private write metadata has invalid shape or storage")
    for tensor in (
        value,
        position,
        seq_cols,
        owner_layout,
        owner_step,
        key_cache,
        value_cache,
    ):
        if tensor.device != key.device:
            raise ValueError("private write tensors must share a CUDA device")
    wrap_triton(_private_kv_write_kernel)[(triton.cdiv(tokens * heads * dim, 256),)](
        key,
        value,
        position,
        seq_cols,
        owner_layout,
        owner_step,
        key_cache,
        value_cache,
        tokens,
        heads,
        dim,
        key.stride(1),
        key.stride(2),
        value.stride(1),
        value.stride(2),
        OWNERS=owner_layout.shape[0] - 1,
        RING=ring,
        BLOCK=256,
    )


@private_kv_write.register_fake
def _private_kv_write_fake(
    key,
    value,
    position,
    seq_cols,
    owner_layout,
    owner_step,
    key_cache,
    value_cache,
    ring=False,
):
    return None
