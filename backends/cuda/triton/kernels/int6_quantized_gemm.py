# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)

"""INT6 GEMM for M <= 64 on planar GGUF Q6_K weights.

The kernels consume ``CudaDp4aPlanarInt6Tensor`` storage directly: ``ql`` holds
nibble-packed low bits, ``qh`` holds the two high bits in K32 even/odd planes,
``scale`` holds signed K16 scale codes, and ``steps`` holds FP16 K256 scale
steps. Buckets 1-4 retain W6A8 DP4A with FP32 scaling and accumulation. Buckets
8-64 decode K256 weight tiles to BF16 and use tensor-core W6A16 with FP32
accumulation. Outputs are BF16.

One op is registered for every bucket in M={1,2,3,4,8,16,32,64}. Every kernel
masks rows at or above runtime M. Small buckets keep their existing generic
W6A8 configuration space and split rule. Large buckets independently autotune
generic BLOCK_M/BLOCK_N/warp/stage tiles with M in the key and K-loop-trip
pruning; split-K is selected outside autotune, writes disjoint FP32 partials,
and uses the shared deterministic reduction. Phase 1 defaults large buckets to
split 1.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from executorch.backends.cuda.autotune.launch_params import autotune_launch_param
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    launch_split_k_gemm,
    QuantizedGemmFamily,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import (
    _dp4a_u8_s8,
    _warp_sum_f32,
    autotune_configs,
    check_contiguous,
    check_device,
    check_dtypes,
    check_k,
    check_rows,
    check_split_k,
    first_reason,
    prune_by_main_loop_trips,
    quantize_activations_q8,
    SPLIT_K_CANDIDATES,
    tile_autotune_configs,
)

_GROUP_SIZE = 16
_SUPER_BLOCK = 256
_TL_GROUP_SIZE = tl.constexpr(16)
_TL_Q8_BLOCK = tl.constexpr(32)
_TL_SUPER_BLOCK = tl.constexpr(256)

_SMALL_BUCKETS = (1, 2, 3, 4)
_LARGE_BUCKETS = (8, 16, 32, 64)
# The tile kernels split K in 256-wide super-blocks and keep gaining from more
# splits on narrow N (e.g. N = 256), so they try one more power of two.
TILE_SPLIT_K_CANDIDATES = (*SPLIT_K_CANDIDATES, 32)
SUPPORTED_BUCKETS = (*_SMALL_BUCKETS, *_LARGE_BUCKETS)


@triton.jit
def _spread_high2(high_byte):
    """Spread four packed two-bit fields into four uint8 lanes."""
    value = (high_byte | (high_byte << 12)) & 0x000F000F
    return (value | (value << 6)) & 0x03030303


@triton.jit
def _signed_byte_to_f32(value):
    """Interpret an int8 or uint8 tensor element as a signed byte."""
    return value.to(tl.int8, bitcast=True).to(tl.float32)


@triton.jit
def _load_int6_k32(
    ql,
    qh,
    scale,
    steps,
    offs_n,
    block,
    block_mask,
    stride_qln: tl.constexpr,
    stride_qlk: tl.constexpr,
    stride_qhn: tl.constexpr,
    stride_qhk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_stn: tl.constexpr,
    stride_stk: tl.constexpr,
):
    """Load and reconstruct one K32 block for every lane."""
    word = tl.arange(0, 4)
    ql_base = ql + offs_n * stride_qln + block * 16 * stride_qlk
    packed_low = tl.load(
        ql_base.to(tl.pointer_type(tl.uint32))[:, None] + word[None, :],
        mask=block_mask[:, None],
        other=0,
    )
    low_even = packed_low & 0x0F0F0F0F
    low_odd = (packed_low >> 4) & 0x0F0F0F0F

    qh_base = qh + offs_n * stride_qhn + block * 8 * stride_qhk
    high_even_word = tl.load(
        qh_base.to(tl.pointer_type(tl.uint32)), mask=block_mask, other=0
    )
    high_odd_word = tl.load(
        qh_base.to(tl.pointer_type(tl.uint32)) + 1, mask=block_mask, other=0
    )
    shift = word * 8
    high_even = (high_even_word[:, None] >> shift[None, :]) & 0xFF
    high_odd = (high_odd_word[:, None] >> shift[None, :]) & 0xFF
    weight_even = low_even | (_spread_high2(high_even) << 4)
    weight_odd = low_odd | (_spread_high2(high_odd) << 4)

    group = block[:, None] * 2 + tl.arange(0, 2)[None, :]
    scale_code = tl.load(
        scale + offs_n[:, None] * stride_sn + group * stride_sk,
        mask=block_mask[:, None],
        other=0,
    )
    scale_code = _signed_byte_to_f32(scale_code)
    step = tl.load(
        steps + offs_n * stride_stn + (block // 8) * stride_stk,
        mask=block_mask,
        other=0.0,
    ).to(tl.float32)
    return weight_even, weight_odd, scale_code * step[:, None]


@triton.jit
def _int6_row_contribution(
    qwords,
    x_scale,
    weight_even,
    weight_odd,
    weight_scale,
    block,
    block_mask,
    row,
    blocks32: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Return one activation row's K32 contribution for every lane."""
    word = tl.arange(0, 4)
    qword_base = (row * blocks32 + block) * 8
    activation_even = tl.load(
        qwords + qword_base[:, None] + word[None, :],
        mask=block_mask[:, None],
        other=0,
    )
    activation_odd = tl.load(
        qwords + qword_base[:, None] + 4 + word[None, :],
        mask=block_mask[:, None],
        other=0,
    )
    dot_words = _dp4a_u8_s8(
        weight_even,
        activation_even,
        tl.zeros((BLOCK_N * 32, 4), dtype=tl.int32),
    )
    dot_words = _dp4a_u8_s8(weight_odd, activation_odd, dot_words)
    ones = tl.full((BLOCK_N * 32, 4), 0x01010101, dtype=tl.uint32)
    sum_words = _dp4a_u8_s8(
        ones,
        activation_even,
        tl.zeros((BLOCK_N * 32, 4), dtype=tl.int32),
    )
    sum_words = _dp4a_u8_s8(ones, activation_odd, sum_words)
    corrected_words = dot_words - 32 * sum_words
    corrected = tl.sum(
        tl.reshape(corrected_words, (BLOCK_N * 32, 2, 2), can_reorder=False),
        axis=2,
    ).to(tl.float32)
    activation_scale = tl.load(
        x_scale + row * blocks32 + block, mask=block_mask, other=0.0
    ).to(tl.float32)
    contribution = tl.sum(corrected * weight_scale, axis=1)
    return tl.where(block_mask, contribution * activation_scale, 0.0)


@triton.jit
def _int6_row_contribution_if_active(
    qwords,
    x_scale,
    weight_even,
    weight_odd,
    weight_scale,
    block,
    block_mask,
    row,
    M,
    blocks32: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    """Dynamic exact rows are active; mask a static inactive row."""
    if DYNAMIC_M:
        return _int6_row_contribution(
            qwords,
            x_scale,
            weight_even,
            weight_odd,
            weight_scale,
            block,
            block_mask,
            row,
            blocks32,
            BLOCK_N,
        )
    else:
        return _int6_row_contribution(
            qwords,
            x_scale,
            weight_even,
            weight_odd,
            weight_scale,
            block,
            block_mask & (row < M),
            row,
            blocks32,
            BLOCK_N,
        )


@triton.jit
def _int6_w6a8_explicit_kernel(
    qwords,
    x_scale,
    ql,
    qh,
    scale,
    steps,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qln: tl.constexpr,
    stride_qlk: tl.constexpr,
    stride_qhn: tl.constexpr,
    stride_qhk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_stn: tl.constexpr,
    stride_stk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BUCKET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    """Row-count-specialized W6A8 kernel with explicit accumulators."""
    pid_n = tl.program_id(0)
    split_id = tl.program_id(2)
    thread = tl.arange(0, BLOCK_N * 32)
    warp = thread // 32
    lane = thread % 32
    offs_n = pid_n * BLOCK_N + warp
    n_mask = offs_n < N
    blocks32: tl.constexpr = K // _TL_Q8_BLOCK
    blocks_per_split: tl.constexpr = tl.cdiv(blocks32, SPLIT_K)
    first_block = split_id * blocks_per_split
    last_block = tl.minimum(first_block + blocks_per_split, blocks32)
    partial0 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial1 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial2 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial3 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)

    for block_offset in tl.range(0, blocks_per_split, 32, num_stages=PIPELINE_STAGES):
        block = first_block + block_offset + lane
        block_mask = n_mask & (block < last_block)
        weight_even, weight_odd, weight_scale = _load_int6_k32(
            ql,
            qh,
            scale,
            steps,
            offs_n,
            block,
            block_mask,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_stn,
            stride_stk,
        )
        partial0 += _int6_row_contribution(
            qwords,
            x_scale,
            weight_even,
            weight_odd,
            weight_scale,
            block,
            block_mask,
            0,
            blocks32,
            BLOCK_N,
        )
        if BUCKET >= 2:
            partial1 += _int6_row_contribution_if_active(
                qwords,
                x_scale,
                weight_even,
                weight_odd,
                weight_scale,
                block,
                block_mask,
                1,
                M,
                blocks32,
                BLOCK_N,
                DYNAMIC_M,
            )
        if BUCKET >= 3:
            partial2 += _int6_row_contribution_if_active(
                qwords,
                x_scale,
                weight_even,
                weight_odd,
                weight_scale,
                block,
                block_mask,
                2,
                M,
                blocks32,
                BLOCK_N,
                DYNAMIC_M,
            )
        if BUCKET >= 4:
            partial3 += _int6_row_contribution_if_active(
                qwords,
                x_scale,
                weight_even,
                weight_odd,
                weight_scale,
                block,
                block_mask,
                3,
                M,
                blocks32,
                BLOCK_N,
                DYNAMIC_M,
            )

    result0 = _warp_sum_f32(partial0)
    result1 = _warp_sum_f32(partial1)
    result2 = _warp_sum_f32(partial2)
    result3 = _warp_sum_f32(partial3)
    out_base = out + split_id * stride_os + offs_n * stride_on
    store_mask = n_mask & (lane == 0)
    tl.store(
        out_base,
        result0.to(tl.float32) if SPLIT_K > 1 else result0.to(tl.bfloat16),
        mask=store_mask,
    )
    if BUCKET >= 2:
        tl.store(
            out_base + stride_om,
            result1.to(tl.float32) if SPLIT_K > 1 else result1.to(tl.bfloat16),
            mask=store_mask if DYNAMIC_M else store_mask & (1 < M),
        )
    if BUCKET >= 3:
        tl.store(
            out_base + 2 * stride_om,
            result2.to(tl.float32) if SPLIT_K > 1 else result2.to(tl.bfloat16),
            mask=store_mask if DYNAMIC_M else store_mask & (2 < M),
        )
    if BUCKET >= 4:
        tl.store(
            out_base + 3 * stride_om,
            result3.to(tl.float32) if SPLIT_K > 1 else result3.to(tl.bfloat16),
            mask=store_mask if DYNAMIC_M else store_mask & (3 < M),
        )


@triton.jit
def _load_int6_k16(
    ql,
    qh,
    scale,
    steps,
    offs_n,
    group,
    group_mask,
    stride_qln: tl.constexpr,
    stride_qlk: tl.constexpr,
    stride_qhn: tl.constexpr,
    stride_qhk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_stn: tl.constexpr,
    stride_stk: tl.constexpr,
):
    """Load and reconstruct one K16 group for every lane."""
    word = tl.arange(0, 2)
    ql_base = ql + offs_n * stride_qln + group * 8 * stride_qlk
    packed_low = tl.load(
        ql_base.to(tl.pointer_type(tl.uint32))[:, None] + word[None, :],
        mask=group_mask[:, None],
        other=0,
    )
    low_even = packed_low & 0x0F0F0F0F
    low_odd = (packed_low >> 4) & 0x0F0F0F0F

    block = group // 2
    word_in_block = (group % 2)[:, None] * 2 + word[None, :]
    qh_base = qh + offs_n * stride_qhn + block * 8 * stride_qhk
    high_even = (
        tl.load(
            qh_base[:, None] + word_in_block * stride_qhk,
            mask=group_mask[:, None],
            other=0,
        )
        .to(tl.uint8)
        .to(tl.uint32)
    )
    high_odd = (
        tl.load(
            qh_base[:, None] + (4 + word_in_block) * stride_qhk,
            mask=group_mask[:, None],
            other=0,
        )
        .to(tl.uint8)
        .to(tl.uint32)
    )
    weight_even = low_even | (_spread_high2(high_even) << 4)
    weight_odd = low_odd | (_spread_high2(high_odd) << 4)

    scale_code = tl.load(
        scale + offs_n * stride_sn + group * stride_sk,
        mask=group_mask,
        other=0,
    )
    step = tl.load(
        steps + offs_n * stride_stn + (block // 8) * stride_stk,
        mask=group_mask,
        other=0.0,
    ).to(tl.float32)
    return (
        weight_even,
        weight_odd,
        _signed_byte_to_f32(scale_code) * step,
        block,
        word_in_block,
    )


@triton.jit
def _int6_k16_row_contribution(
    qwords,
    x_scale,
    weight_even,
    weight_odd,
    weight_scale,
    block,
    word_in_block,
    group_mask,
    row,
    blocks32: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    qword_base = (row * blocks32 + block) * 8
    activation_even = tl.load(
        qwords + qword_base[:, None] + word_in_block,
        mask=group_mask[:, None],
        other=0,
    )
    activation_odd = tl.load(
        qwords + qword_base[:, None] + 4 + word_in_block,
        mask=group_mask[:, None],
        other=0,
    )
    dot_words = _dp4a_u8_s8(
        weight_even,
        activation_even,
        tl.zeros((BLOCK_N * 32, 2), dtype=tl.int32),
    )
    dot_words = _dp4a_u8_s8(weight_odd, activation_odd, dot_words)
    ones = tl.full((BLOCK_N * 32, 2), 0x01010101, dtype=tl.uint32)
    sum_words = _dp4a_u8_s8(
        ones,
        activation_even,
        tl.zeros((BLOCK_N * 32, 2), dtype=tl.int32),
    )
    sum_words = _dp4a_u8_s8(ones, activation_odd, sum_words)
    corrected = tl.sum(dot_words - 32 * sum_words, axis=1).to(tl.float32)
    activation_scale = tl.load(
        x_scale + row * blocks32 + block,
        mask=group_mask,
        other=0.0,
    ).to(tl.float32)
    return tl.where(
        group_mask,
        corrected * weight_scale * activation_scale,
        0.0,
    )


@triton.jit
def _int6_k16_row_contribution_if_active(
    qwords,
    x_scale,
    weight_even,
    weight_odd,
    weight_scale,
    block,
    word_in_block,
    group_mask,
    row,
    M,
    blocks32: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    if DYNAMIC_M:
        return _int6_k16_row_contribution(
            qwords,
            x_scale,
            weight_even,
            weight_odd,
            weight_scale,
            block,
            word_in_block,
            group_mask,
            row,
            blocks32,
            BLOCK_N,
        )
    else:
        return _int6_k16_row_contribution(
            qwords,
            x_scale,
            weight_even,
            weight_odd,
            weight_scale,
            block,
            word_in_block,
            group_mask & (row < M),
            row,
            blocks32,
            BLOCK_N,
        )


@triton.jit
def _int6_w6a8_k16_kernel(
    qwords,
    x_scale,
    ql,
    qh,
    scale,
    steps,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qln: tl.constexpr,
    stride_qlk: tl.constexpr,
    stride_qhn: tl.constexpr,
    stride_qhk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_stn: tl.constexpr,
    stride_stk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BUCKET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    """Row-count-specialized K16 schedule with smaller per-lane vectors."""
    pid_n = tl.program_id(0)
    split_id = tl.program_id(2)
    thread = tl.arange(0, BLOCK_N * 32)
    warp = thread // 32
    lane = thread % 32
    offs_n = pid_n * BLOCK_N + warp
    n_mask = offs_n < N
    groups16: tl.constexpr = K // _TL_GROUP_SIZE
    blocks32: tl.constexpr = K // _TL_Q8_BLOCK
    groups_per_split: tl.constexpr = tl.cdiv(groups16, SPLIT_K)
    first_group = split_id * groups_per_split
    last_group = tl.minimum(first_group + groups_per_split, groups16)
    partial0 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial1 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial2 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial3 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)

    for group_offset in tl.range(0, groups_per_split, 32, num_stages=PIPELINE_STAGES):
        group = first_group + group_offset + lane
        group_mask = n_mask & (group < last_group)
        weight_even, weight_odd, weight_scale, block, word_in_block = _load_int6_k16(
            ql,
            qh,
            scale,
            steps,
            offs_n,
            group,
            group_mask,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_stn,
            stride_stk,
        )
        partial0 += _int6_k16_row_contribution(
            qwords,
            x_scale,
            weight_even,
            weight_odd,
            weight_scale,
            block,
            word_in_block,
            group_mask,
            0,
            blocks32,
            BLOCK_N,
        )
        if BUCKET >= 2:
            partial1 += _int6_k16_row_contribution_if_active(
                qwords,
                x_scale,
                weight_even,
                weight_odd,
                weight_scale,
                block,
                word_in_block,
                group_mask,
                1,
                M,
                blocks32,
                BLOCK_N,
                DYNAMIC_M,
            )
        if BUCKET >= 3:
            partial2 += _int6_k16_row_contribution_if_active(
                qwords,
                x_scale,
                weight_even,
                weight_odd,
                weight_scale,
                block,
                word_in_block,
                group_mask,
                2,
                M,
                blocks32,
                BLOCK_N,
                DYNAMIC_M,
            )
        if BUCKET >= 4:
            partial3 += _int6_k16_row_contribution_if_active(
                qwords,
                x_scale,
                weight_even,
                weight_odd,
                weight_scale,
                block,
                word_in_block,
                group_mask,
                3,
                M,
                blocks32,
                BLOCK_N,
                DYNAMIC_M,
            )

    result0 = _warp_sum_f32(partial0)
    result1 = _warp_sum_f32(partial1)
    result2 = _warp_sum_f32(partial2)
    result3 = _warp_sum_f32(partial3)
    out_base = out + split_id * stride_os + offs_n * stride_on
    store_mask = n_mask & (lane == 0)
    tl.store(
        out_base,
        result0.to(tl.float32) if SPLIT_K > 1 else result0.to(tl.bfloat16),
        mask=store_mask,
    )
    if BUCKET >= 2:
        tl.store(
            out_base + stride_om,
            result1.to(tl.float32) if SPLIT_K > 1 else result1.to(tl.bfloat16),
            mask=store_mask if DYNAMIC_M else store_mask & (1 < M),
        )
    if BUCKET >= 3:
        tl.store(
            out_base + 2 * stride_om,
            result2.to(tl.float32) if SPLIT_K > 1 else result2.to(tl.bfloat16),
            mask=store_mask if DYNAMIC_M else store_mask & (2 < M),
        )
    if BUCKET >= 4:
        tl.store(
            out_base + 3 * stride_om,
            result3.to(tl.float32) if SPLIT_K > 1 else result3.to(tl.bfloat16),
            mask=store_mask if DYNAMIC_M else store_mask & (3 < M),
        )


@triton.jit
def _int6_w6a8_rows(
    qwords,
    x_scale,
    ql,
    qh,
    scale,
    steps,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qln: tl.constexpr,
    stride_qlk: tl.constexpr,
    stride_qhn: tl.constexpr,
    stride_qhk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_stn: tl.constexpr,
    stride_stk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
    K_TILE: tl.constexpr,
):
    """The explicit ROWS-row kernel for the selected K tile."""
    if K_TILE == 32:
        _int6_w6a8_explicit_kernel(
            qwords,
            x_scale,
            ql,
            qh,
            scale,
            steps,
            out,
            M,
            N,
            K,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_stn,
            stride_stk,
            stride_os,
            stride_om,
            stride_on,
            ROWS,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
        )
    else:
        _int6_w6a8_k16_kernel(
            qwords,
            x_scale,
            ql,
            qh,
            scale,
            steps,
            out,
            M,
            N,
            K,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_stn,
            stride_stk,
            stride_os,
            stride_om,
            stride_on,
            ROWS,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
        )


@triton.jit
def _int6_w6a8_exact_rows(
    qwords,
    x_scale,
    ql,
    qh,
    scale,
    steps,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qln: tl.constexpr,
    stride_qlk: tl.constexpr,
    stride_qhn: tl.constexpr,
    stride_qhk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_stn: tl.constexpr,
    stride_stk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
    K_TILE: tl.constexpr,
):
    """Dynamic M: branch uniformly to the explicit kernel with exactly M rows."""
    if M == 1:
        _int6_w6a8_rows(
            qwords,
            x_scale,
            ql,
            qh,
            scale,
            steps,
            out,
            M,
            N,
            K,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_stn,
            stride_stk,
            stride_os,
            stride_om,
            stride_on,
            1,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            K_TILE,
        )
    elif ROWS >= 2 and M == 2:
        _int6_w6a8_rows(
            qwords,
            x_scale,
            ql,
            qh,
            scale,
            steps,
            out,
            M,
            N,
            K,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_stn,
            stride_stk,
            stride_os,
            stride_om,
            stride_on,
            2,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            K_TILE,
        )
    elif ROWS >= 3 and M == 3:
        _int6_w6a8_rows(
            qwords,
            x_scale,
            ql,
            qh,
            scale,
            steps,
            out,
            M,
            N,
            K,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_stn,
            stride_stk,
            stride_os,
            stride_om,
            stride_on,
            3,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            K_TILE,
        )
    else:
        _int6_w6a8_rows(
            qwords,
            x_scale,
            ql,
            qh,
            scale,
            steps,
            out,
            M,
            N,
            K,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_stn,
            stride_stk,
            stride_os,
            stride_om,
            stride_on,
            ROWS,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            K_TILE,
        )


@triton.jit
def _int6_w6a8_bucket_kernel(
    qwords,
    x_scale,
    ql,
    qh,
    scale,
    steps,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qln: tl.constexpr,
    stride_qlk: tl.constexpr,
    stride_qhn: tl.constexpr,
    stride_qhk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_stn: tl.constexpr,
    stride_stk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BUCKET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
    K_TILE: tl.constexpr,
):
    if DYNAMIC_M:
        _int6_w6a8_exact_rows(
            qwords,
            x_scale,
            ql,
            qh,
            scale,
            steps,
            out,
            M,
            N,
            K,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_stn,
            stride_stk,
            stride_os,
            stride_om,
            stride_on,
            BUCKET,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            K_TILE,
        )
    else:
        _int6_w6a8_rows(
            qwords,
            x_scale,
            ql,
            qh,
            scale,
            steps,
            out,
            M,
            N,
            K,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_stn,
            stride_stk,
            stride_os,
            stride_om,
            stride_on,
            BUCKET,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            K_TILE,
        )


@triton.jit
def _decode_int6_w6a16_superblock(
    ql,
    qh,
    scale,
    steps,
    offs_n,
    n_mask,
    super_block,
    stride_qln: tl.constexpr,
    stride_qlk: tl.constexpr,
    stride_qhn: tl.constexpr,
    stride_qhk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_stn: tl.constexpr,
    stride_stk: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Decode one planar Q6_K K256 tile to BF16."""
    offs_k = tl.arange(0, 256)
    packed_low = tl.load(
        ql
        + offs_n[:, None] * stride_qln
        + (super_block * 128 + offs_k[None, :] // 2) * stride_qlk,
        mask=n_mask[:, None],
        other=0,
    ).to(tl.uint32)
    low = (packed_low >> ((offs_k[None, :] & 1) * 4)) & 0xF

    k_in_chunk = offs_k % 32
    qh_byte = (
        super_block * 64 + (offs_k // 32) * 8 + (k_in_chunk % 2) * 4 + k_in_chunk // 8
    )
    high_shift = ((k_in_chunk % 8) // 2) * 2
    packed_high = tl.load(
        qh + offs_n[:, None] * stride_qhn + qh_byte[None, :] * stride_qhk,
        mask=n_mask[:, None],
        other=0,
    ).to(tl.uint32)
    high = (packed_high >> high_shift[None, :]) & 0x3
    q = (low | (high << 4)).to(tl.int32) - 32

    offs_group = tl.arange(0, 16)
    group = super_block * 16 + offs_group
    scale_code = tl.load(
        scale + offs_n[:, None] * stride_sn + group[None, :] * stride_sk,
        mask=n_mask[:, None],
        other=0,
    )
    scale_code = _signed_byte_to_f32(scale_code).to(tl.bfloat16)
    step = tl.load(
        steps + offs_n * stride_stn + super_block * stride_stk,
        mask=n_mask,
        other=0.0,
    ).to(tl.bfloat16)
    group_scale = (scale_code * step[:, None]).to(tl.bfloat16)
    expanded_scale = tl.reshape(
        tl.broadcast_to(group_scale[:, :, None], (BLOCK_N, 16, 16)),
        (BLOCK_N, 256),
        can_reorder=False,
    )
    return (q.to(tl.bfloat16) * expanded_scale).to(tl.bfloat16)


@triton.jit
def _int6_w6a16_tile_kernel(
    x,
    ql,
    qh,
    scale,
    steps,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_qln: tl.constexpr,
    stride_qlk: tl.constexpr,
    stride_qhn: tl.constexpr,
    stride_qhk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_stn: tl.constexpr,
    stride_stk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """W6A16 K256 tiles with direct or disjoint deterministic split output."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    split_id = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_mask = offs_n < N
    m_mask = offs_m < M
    offs_k = tl.arange(0, 256)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_super_blocks: tl.constexpr = K // _TL_SUPER_BLOCK
    super_blocks_per_split: tl.constexpr = tl.cdiv(num_super_blocks, SPLIT_K)
    first_super = split_id * super_blocks_per_split
    last_super = tl.minimum(first_super + super_blocks_per_split, num_super_blocks)
    for super_block in tl.range(first_super, last_super, num_stages=PIPELINE_STAGES):
        weight = _decode_int6_w6a16_superblock(
            ql,
            qh,
            scale,
            steps,
            offs_n,
            n_mask,
            super_block,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_stn,
            stride_stk,
            BLOCK_N,
        )
        activation = tl.load(
            x
            + offs_m[:, None] * stride_xm
            + (super_block * 256 + offs_k[None, :]) * stride_xk,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        acc += tl.dot(activation, tl.trans(weight), out_dtype=tl.float32)

    tl.store(
        out
        + split_id * stride_os
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on,
        acc.to(tl.float32) if SPLIT_K > 1 else acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _main_loop_trips(args) -> int:
    """K_TILE units are consumed by 32 lanes per main-loop trip."""
    units_per_split = triton.cdiv(
        int(args["K"]) // int(args["K_TILE"]), int(args["SPLIT_K"])
    )
    return triton.cdiv(units_per_split, 32)


def int6_autotune_configs() -> list[triton.Config]:
    return autotune_configs([{"K_TILE": 32}, {"K_TILE": 16}])


def _prune(configs, named_args, **kwargs):
    kept = []
    for k_tile in (32, 16):
        subset = [config for config in configs if config.kwargs["K_TILE"] == k_tile]
        if subset:
            kept.extend(
                prune_by_main_loop_trips(
                    _main_loop_trips,
                    subset,
                    named_args,
                    K_TILE=k_tile,
                    **kwargs,
                )
            )
    return kept


def _tile_main_loop_trips(args) -> int:
    return max(
        1,
        triton.cdiv(
            int(args["K"]) // _SUPER_BLOCK,
            int(args["SPLIT_K"]),
        ),
    )


def _prune_tiles(configs, named_args, **kwargs):
    return prune_by_main_loop_trips(
        _tile_main_loop_trips, configs, named_args, **kwargs
    )


_BUCKET_KERNELS = {
    bucket: triton.autotune(
        configs=int6_autotune_configs(),
        key=["N", "K", "SPLIT_K", "DYNAMIC_M"],
        prune_configs_by={"early_config_prune": _prune},
    )(_int6_w6a8_bucket_kernel)
    for bucket in _SMALL_BUCKETS
}
_BUCKET_KERNELS.update(
    {
        bucket: triton.autotune(
            configs=tile_autotune_configs(bucket),
            key=["M", "N", "K", "SPLIT_K"],
            prune_configs_by={"early_config_prune": _prune_tiles},
        )(_int6_w6a16_tile_kernel)
        for bucket in _LARGE_BUCKETS
    }
)


def _unsupported_reason(
    bucket: int,
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    steps: torch.Tensor,
    group_size: int,
) -> Optional[str]:
    weights = (ql, qh, scale, steps)
    if any(tensor.dim() != 2 for tensor in (x, *weights)):
        return "expects rank-2 activation and weight tensors"
    M, K = x.shape
    N, K_half = ql.shape
    if isinstance(K, int) and K <= 0:
        return f"K must be positive, got {K}"
    reason = first_reason(
        check_dtypes(
            (
                ("activation", x, (torch.bfloat16,)),
                ("ql", ql, (torch.uint8, torch.int8)),
                ("qh", qh, (torch.uint8, torch.int8)),
                ("scale codes", scale, (torch.int8, torch.uint8)),
                ("steps", steps, (torch.float16,)),
            )
        ),
        (
            None
            if group_size == _GROUP_SIZE
            else f"group_size must be {_GROUP_SIZE}, got {group_size}"
        ),
        check_k(K, _SUPER_BLOCK),
        check_rows(M, bucket),
        check_contiguous((x, *weights)),
        check_device(x, weights),
    )
    if reason is not None:
        return reason
    if K_half * 2 != K:
        return f"ql K/2 mismatch: x K={K}, ql K/2={K_half}"
    if tuple(qh.shape) != (N, K // 4):
        return "qh shape does not match [N, K/4]"
    if tuple(scale.shape) != (N, K // group_size):
        return "scale shape does not match [N, K/group_size]"
    if tuple(steps.shape) != (N, K // _SUPER_BLOCK):
        return "steps shape does not match [N, K/256]"
    return None


def _launch_w6a8(
    bucket: int,
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    steps: torch.Tensor,
    group_size: int,
    **launch_params,
) -> torch.Tensor:
    """Runs the bucket kernel. Whether M is dynamic is read here, before
    split-K timing replaces ``x`` with a concrete tensor, so a dynamic-M call
    is timed on the dynamic path and cached apart from a static M."""
    return _launch_rows(
        bucket,
        x,
        ql,
        qh,
        scale,
        steps,
        group_size,
        not isinstance(x.shape[0], int),
        **launch_params,
    )


@autotune_launch_param("split_k", SPLIT_K_CANDIDATES)
def _launch_rows(
    bucket: int,
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    steps: torch.Tensor,
    group_size: int,
    dynamic_m: bool,
    *,
    split_k: int,
) -> torch.Tensor:
    del group_size
    M, K = x.shape
    N = ql.shape[0]
    check_split_k(split_k, int(K), _SUPER_BLOCK)
    qwords, x_scale, _ = quantize_activations_q8(x, store_sum=False)
    block_m = triton.next_power_of_2(bucket)
    return launch_split_k_gemm(
        _BUCKET_KERNELS[bucket],
        bucket=bucket,
        m=M,
        n=N,
        device=x.device,
        split_k=split_k,
        block_m=block_m,
        inputs=(qwords, x_scale, ql, qh, scale, steps),
        shape_args=(
            M,
            N,
            K,
            ql.stride(0),
            ql.stride(1),
            qh.stride(0),
            qh.stride(1),
            scale.stride(0),
            scale.stride(1),
            steps.stride(0),
            steps.stride(1),
        ),
        BUCKET=bucket,
        BLOCK_M=block_m,
        DYNAMIC_M=dynamic_m,
    )


@autotune_launch_param("split_k", TILE_SPLIT_K_CANDIDATES)
def _launch_w6a16(
    bucket: int,
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    steps: torch.Tensor,
    group_size: int,
    *,
    split_k: int,
    config: Optional[triton.Config] = None,
) -> torch.Tensor:
    del group_size
    M, K = x.shape
    N = ql.shape[0]
    check_split_k(split_k, int(K), _SUPER_BLOCK)
    if config is None:
        kernel = _BUCKET_KERNELS[bucket]
        launch_options = {}
    else:
        kernel = _int6_w6a16_tile_kernel
        launch_options = {
            **config.kwargs,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
    return launch_split_k_gemm(
        kernel,
        bucket=bucket,
        m=M,
        n=N,
        device=x.device,
        split_k=split_k,
        block_m=min(bucket, 32),
        inputs=(x, ql, qh, scale, steps),
        shape_args=(
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            ql.stride(0),
            ql.stride(1),
            qh.stride(0),
            qh.stride(1),
            scale.stride(0),
            scale.stride(1),
            steps.stride(0),
            steps.stride(1),
        ),
        **launch_options,
    )


def _launch(
    bucket: int,
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    steps: torch.Tensor,
    group_size: int,
    **launch_params,
) -> torch.Tensor:
    launch = _launch_w6a8 if bucket in _SMALL_BUCKETS else _launch_w6a16
    return launch(bucket, x, ql, qh, scale, steps, group_size, **launch_params)


def _prototype(
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    steps: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    raise NotImplementedError


def _fake(bucket: int, x: torch.Tensor, ql: torch.Tensor, *args) -> torch.Tensor:
    return torch.empty((x.shape[0], ql.shape[0]), dtype=torch.bfloat16, device=x.device)


INT6_QUANTIZED_GEMM = QuantizedGemmFamily(
    "int6_quantized_gemm",
    SUPPORTED_BUCKETS,
    _prototype,
    _launch,
    _fake,
    _unsupported_reason,
)


__all__ = [
    "INT6_QUANTIZED_GEMM",
    "SUPPORTED_BUCKETS",
    "int6_autotune_configs",
]
