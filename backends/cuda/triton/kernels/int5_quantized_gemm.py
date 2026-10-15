# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)

"""INT5 GEMM on planar Q5_K weights for M <= 64.

``ql`` stores even/odd low nibbles and ``qh`` stores one high bit per weight:
each K32 block has four ``qh`` bytes whose low/high nibbles describe the four
even/odd bytes of one DP4A word. Scale and zero codes use K/group_size groups;
their FP16 steps use K256 super-blocks.

For M <= 4, activations are quantized to signed INT8 in even/odd K32 order.
The kernels reconstruct unsigned five-bit weights and accumulate
``(dot - activation_sum * wz) * activation_scale * ws`` in FP32, where
``ws = scale_code * scale_step`` and ``wz = zero_code * zero_point_step``.
Split-K partials are FP32 and the output is BF16.

Each M bucket independently autotunes a generic row-blocked implementation and
explicit-row implementations for the bucket and its next bucket. Dynamic M
branches uniformly to exactly the runtime row count; static extra rows are
masked.

Buckets 8, 16, 32, and 64 keep BF16 activations and unpack Q5_K tiles inside
the GEMM. They decode K256 super-blocks in K256 tiles, plus one measured K64
M64 candidate, and use an exact exponent-pattern INT5-to-BF16 conversion. These
buckets require group_size=32 and preserve BF16 dequant rounding before FP32
accumulation, without a full BF16 weight buffer.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from executorch.backends.cuda.autotune.launch_params import autotune_launch_param
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    _compile_every_candidate_at_once,
    launch_split_k_gemm,
    QuantizedGemmFamily,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import (
    _dp4a_u8_s8,
    _warp_sum_f32,
    autotune_configs,
    autotune_do_bench,
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

_GROUP_SIZE = 32
_SUPER_BLOCK = 256
_TL_GROUP_SIZE = tl.constexpr(32)

_SMALL_BUCKETS = (1, 2, 3, 4)
_LARGE_BUCKETS = (8, 16, 32, 64)
TILE_SPLIT_K_CANDIDATES = (1, 2, 4, 8, 16)
SUPPORTED_BUCKETS = _SMALL_BUCKETS + _LARGE_BUCKETS
_TL_SUPER_BLOCK = tl.constexpr(256)


@triton.jit
def _spread_high1(nibble):
    """Spread nibble bit j into bit zero of byte lane j."""
    return (nibble * 0x00204081) & 0x01010101


@triton.jit
def _load_int5_k32(
    ql,
    qh,
    scale,
    scale_step,
    zero,
    zero_point_step,
    offs_n,
    group,
    group_mask,
    stride_qln: tl.constexpr,
    stride_qlk: tl.constexpr,
    stride_qhn: tl.constexpr,
    stride_qhk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_ssn: tl.constexpr,
    stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr,
    stride_zsk: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Load one K32 group and reconstruct four even/odd DP4A words."""
    word = tl.arange(0, 4)
    ql_base = ql + offs_n * stride_qln + group * 16 * stride_qlk
    packed_low = tl.load(
        ql_base.to(tl.pointer_type(tl.uint32))[:, None] + word[None, :],
        mask=group_mask[:, None],
        other=0,
    )
    low_even = packed_low & 0x0F0F0F0F
    low_odd = (packed_low >> 4) & 0x0F0F0F0F

    qh_base = qh + offs_n * stride_qhn + group * 4 * stride_qhk
    packed_high = tl.load(
        qh_base.to(tl.pointer_type(tl.uint32)), mask=group_mask, other=0
    )
    high_byte = (packed_high[:, None] >> (word[None, :] * 8)) & 0xFF
    weight_even = low_even | (_spread_high1(high_byte & 0x0F) << 4)
    weight_odd = low_odd | (_spread_high1(high_byte >> 4) << 4)

    metadata_group = group * _TL_GROUP_SIZE // GROUP_SIZE
    scale_code = tl.load(
        scale + offs_n * stride_sn + metadata_group * stride_sk,
        mask=group_mask,
        other=0,
    ).to(tl.float32)
    zero_code = tl.load(
        zero + offs_n * stride_zn + metadata_group * stride_zk,
        mask=group_mask,
        other=0,
    ).to(tl.float32)
    step_group = group // 8
    scale_step_value = tl.load(
        scale_step + offs_n * stride_ssn + step_group * stride_ssk,
        mask=group_mask,
        other=0.0,
    ).to(tl.float32)
    zero_step_value = tl.load(
        zero_point_step + offs_n * stride_zsn + step_group * stride_zsk,
        mask=group_mask,
        other=0.0,
    ).to(tl.float32)
    return (
        weight_even,
        weight_odd,
        scale_code * scale_step_value,
        zero_code * zero_step_value,
    )


@triton.jit
def _int5_w5a8_dp4a_kernel(
    qwords,
    x_scale,
    x_sum,
    ql,
    qh,
    scale,
    scale_step,
    zero,
    zero_point_step,
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
    stride_ssn: tl.constexpr,
    stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr,
    stride_zsk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    """Generic row-blocked W5A8 kernel."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    split_id = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    thread = tl.arange(0, BLOCK_N * 32)
    warp = thread // 32
    lane = thread % 32
    offs_n = pid_n * BLOCK_N + warp
    n_mask = offs_n < N
    groups: tl.constexpr = K // _TL_GROUP_SIZE
    groups_per_split: tl.constexpr = tl.cdiv(groups, SPLIT_K)
    first_group = split_id * groups_per_split
    last_group = tl.minimum(first_group + groups_per_split, groups)
    partial = tl.zeros((BLOCK_M, BLOCK_N * 32), dtype=tl.float32)

    for group_offset in tl.range(0, groups_per_split, 32, num_stages=PIPELINE_STAGES):
        group = first_group + group_offset + lane
        group_mask = n_mask & (group < last_group)
        activation_mask = (offs_m[:, None] < M) & group_mask[None, :]
        weight_even, weight_odd, ws, wz = _load_int5_k32(
            ql,
            qh,
            scale,
            scale_step,
            zero,
            zero_point_step,
            offs_n,
            group,
            group_mask,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_ssn,
            stride_ssk,
            stride_zn,
            stride_zk,
            stride_zsn,
            stride_zsk,
            GROUP_SIZE,
        )
        word = tl.arange(0, 4)
        qword_base = (offs_m[:, None] * groups + group[None, :]) * 8
        activation_even = tl.load(
            qwords + qword_base[:, :, None] + word[None, None, :],
            mask=activation_mask[:, :, None],
            other=0,
        )
        activation_odd = tl.load(
            qwords + qword_base[:, :, None] + 4 + word[None, None, :],
            mask=activation_mask[:, :, None],
            other=0,
        )
        dot_words = _dp4a_u8_s8(
            weight_even[None, :, :],
            activation_even,
            tl.zeros((BLOCK_M, BLOCK_N * 32, 4), dtype=tl.int32),
        )
        dot_words = _dp4a_u8_s8(weight_odd[None, :, :], activation_odd, dot_words)
        dot = tl.sum(dot_words, axis=2).to(tl.float32)
        activation_scale = tl.load(
            x_scale + offs_m[:, None] * groups + group[None, :],
            mask=activation_mask,
            other=0.0,
        ).to(tl.float32)
        activation_sum = tl.load(
            x_sum + offs_m[:, None] * groups + group[None, :],
            mask=activation_mask,
            other=0,
        ).to(tl.float32)
        contribution = (
            (dot - activation_sum * wz[None, :]) * activation_scale * ws[None, :]
        )
        partial += tl.where(activation_mask, contribution, 0.0)

    partial = tl.reshape(partial, (BLOCK_M, BLOCK_N, 32), can_reorder=False)
    result = tl.sum(partial, axis=2)
    store_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    store_mask = (offs_m[:, None] < M) & (store_n[None, :] < N)
    tl.store(
        out
        + split_id * stride_os
        + offs_m[:, None] * stride_om
        + store_n[None, :] * stride_on,
        result.to(tl.float32) if SPLIT_K > 1 else result.to(tl.bfloat16),
        mask=store_mask,
    )


@triton.jit
def _int5_row_contribution(
    qwords,
    x_scale,
    x_sum,
    weight_even,
    weight_odd,
    ws,
    wz,
    group,
    activation_mask,
    row,
    groups: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Return one activation row's K32 contribution for every lane."""
    word = tl.arange(0, 4)
    qword_base = (row * groups + group) * 8
    activation_even = tl.load(
        qwords + qword_base[:, None] + word[None, :],
        mask=activation_mask[:, None],
        other=0,
    )
    activation_odd = tl.load(
        qwords + qword_base[:, None] + 4 + word[None, :],
        mask=activation_mask[:, None],
        other=0,
    )
    dot_words = _dp4a_u8_s8(
        weight_even,
        activation_even,
        tl.zeros((BLOCK_N * 32, 4), dtype=tl.int32),
    )
    dot_words = _dp4a_u8_s8(weight_odd, activation_odd, dot_words)
    dot = tl.sum(dot_words, axis=1).to(tl.float32)
    activation_scale = tl.load(
        x_scale + row * groups + group, mask=activation_mask, other=0.0
    ).to(tl.float32)
    activation_sum = tl.load(
        x_sum + row * groups + group, mask=activation_mask, other=0
    ).to(tl.float32)
    return tl.where(
        activation_mask,
        (dot - activation_sum * wz) * activation_scale * ws,
        0.0,
    )


@triton.jit
def _int5_row_contribution_if_active(
    qwords,
    x_scale,
    x_sum,
    weight_even,
    weight_odd,
    ws,
    wz,
    group,
    group_mask,
    row,
    M,
    groups: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    """Uniformly skip a dynamic inactive row; mask a static inactive row."""
    if DYNAMIC_M:
        contribution = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
        if row < M:
            contribution = _int5_row_contribution(
                qwords,
                x_scale,
                x_sum,
                weight_even,
                weight_odd,
                ws,
                wz,
                group,
                group_mask,
                row,
                groups,
                BLOCK_N,
            )
        return contribution
    else:
        return _int5_row_contribution(
            qwords,
            x_scale,
            x_sum,
            weight_even,
            weight_odd,
            ws,
            wz,
            group,
            group_mask & (row < M),
            row,
            groups,
            BLOCK_N,
        )


@triton.jit
def _int5_w5a8_dp4a_explicit_kernel(
    qwords,
    x_scale,
    x_sum,
    ql,
    qh,
    scale,
    scale_step,
    zero,
    zero_point_step,
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
    stride_ssn: tl.constexpr,
    stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr,
    stride_zsk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
    ROWS: tl.constexpr,
):
    """W5A8 DP4A with ROWS explicit per-thread accumulators."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    split_id = tl.program_id(2)
    thread = tl.arange(0, BLOCK_N * 32)
    warp = thread // 32
    lane = thread % 32
    offs_n = pid_n * BLOCK_N + warp
    n_mask = offs_n < N
    groups: tl.constexpr = K // _TL_GROUP_SIZE
    groups_per_split: tl.constexpr = tl.cdiv(groups, SPLIT_K)
    first_group = split_id * groups_per_split
    last_group = tl.minimum(first_group + groups_per_split, groups)
    row0 = pid_m * BLOCK_M
    partial0 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    if ROWS >= 2:
        row1 = row0 + 1
        partial1 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    if ROWS >= 3:
        row2 = row0 + 2
        partial2 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    if ROWS >= 4:
        row3 = row0 + 3
        partial3 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)

    for group_offset in tl.range(0, groups_per_split, 32, num_stages=PIPELINE_STAGES):
        group = first_group + group_offset + lane
        group_mask = n_mask & (group < last_group)
        weight_even, weight_odd, ws, wz = _load_int5_k32(
            ql,
            qh,
            scale,
            scale_step,
            zero,
            zero_point_step,
            offs_n,
            group,
            group_mask,
            stride_qln,
            stride_qlk,
            stride_qhn,
            stride_qhk,
            stride_sn,
            stride_sk,
            stride_ssn,
            stride_ssk,
            stride_zn,
            stride_zk,
            stride_zsn,
            stride_zsk,
            GROUP_SIZE,
        )
        partial0 += _int5_row_contribution(
            qwords,
            x_scale,
            x_sum,
            weight_even,
            weight_odd,
            ws,
            wz,
            group,
            group_mask & (row0 < M),
            row0,
            groups,
            BLOCK_N,
        )
        if ROWS >= 2:
            partial1 += _int5_row_contribution_if_active(
                qwords,
                x_scale,
                x_sum,
                weight_even,
                weight_odd,
                ws,
                wz,
                group,
                group_mask,
                row1,
                M,
                groups,
                BLOCK_N,
                DYNAMIC_M,
            )
        if ROWS >= 3:
            partial2 += _int5_row_contribution_if_active(
                qwords,
                x_scale,
                x_sum,
                weight_even,
                weight_odd,
                ws,
                wz,
                group,
                group_mask,
                row2,
                M,
                groups,
                BLOCK_N,
                DYNAMIC_M,
            )
        if ROWS >= 4:
            partial3 += _int5_row_contribution_if_active(
                qwords,
                x_scale,
                x_sum,
                weight_even,
                weight_odd,
                ws,
                wz,
                group,
                group_mask,
                row3,
                M,
                groups,
                BLOCK_N,
                DYNAMIC_M,
            )

    result0 = _warp_sum_f32(partial0)
    out_base = out + split_id * stride_os + offs_n * stride_on
    store_mask = n_mask & (lane == 0)
    tl.store(
        out_base + row0 * stride_om,
        result0.to(tl.float32) if SPLIT_K > 1 else result0.to(tl.bfloat16),
        mask=store_mask & (row0 < M),
    )
    if ROWS >= 2:
        result1 = _warp_sum_f32(partial1)
        tl.store(
            out_base + row1 * stride_om,
            result1.to(tl.float32) if SPLIT_K > 1 else result1.to(tl.bfloat16),
            mask=store_mask & (row1 < M),
        )
    if ROWS >= 3:
        result2 = _warp_sum_f32(partial2)
        tl.store(
            out_base + row2 * stride_om,
            result2.to(tl.float32) if SPLIT_K > 1 else result2.to(tl.bfloat16),
            mask=store_mask & (row2 < M),
        )
    if ROWS >= 4:
        result3 = _warp_sum_f32(partial3)
        tl.store(
            out_base + row3 * stride_om,
            result3.to(tl.float32) if SPLIT_K > 1 else result3.to(tl.bfloat16),
            mask=store_mask & (row3 < M),
        )


@triton.jit
def _int5_w5a8_dp4a_rows(
    qwords,
    x_scale,
    x_sum,
    ql,
    qh,
    scale,
    scale_step,
    zero,
    zero_point_step,
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
    stride_ssn: tl.constexpr,
    stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr,
    stride_zsk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Dispatch a compile-time explicit row count."""
    _int5_w5a8_dp4a_explicit_kernel(
        qwords,
        x_scale,
        x_sum,
        ql,
        qh,
        scale,
        scale_step,
        zero,
        zero_point_step,
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
        stride_ssn,
        stride_ssk,
        stride_zn,
        stride_zk,
        stride_zsn,
        stride_zsk,
        stride_os,
        stride_om,
        stride_on,
        GROUP_SIZE,
        BLOCK_M,
        BLOCK_N,
        SPLIT_K,
        PIPELINE_STAGES,
        DYNAMIC_M,
        ROWS,
    )


@triton.jit
def _int5_w5a8_dp4a_exact_rows(
    qwords,
    x_scale,
    x_sum,
    ql,
    qh,
    scale,
    scale_step,
    zero,
    zero_point_step,
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
    stride_ssn: tl.constexpr,
    stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr,
    stride_zsk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Dynamic M: branch uniformly to the implementation with exactly M rows."""
    if M == 1:
        _int5_w5a8_dp4a_rows(
            qwords,
            x_scale,
            x_sum,
            ql,
            qh,
            scale,
            scale_step,
            zero,
            zero_point_step,
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
            stride_ssn,
            stride_ssk,
            stride_zn,
            stride_zk,
            stride_zsn,
            stride_zsk,
            stride_os,
            stride_om,
            stride_on,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            1,
        )
    elif ROWS >= 2 and M == 2:
        _int5_w5a8_dp4a_rows(
            qwords,
            x_scale,
            x_sum,
            ql,
            qh,
            scale,
            scale_step,
            zero,
            zero_point_step,
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
            stride_ssn,
            stride_ssk,
            stride_zn,
            stride_zk,
            stride_zsn,
            stride_zsk,
            stride_os,
            stride_om,
            stride_on,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            2,
        )
    elif ROWS >= 3 and M == 3:
        _int5_w5a8_dp4a_rows(
            qwords,
            x_scale,
            x_sum,
            ql,
            qh,
            scale,
            scale_step,
            zero,
            zero_point_step,
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
            stride_ssn,
            stride_ssk,
            stride_zn,
            stride_zk,
            stride_zsn,
            stride_zsk,
            stride_os,
            stride_om,
            stride_on,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            3,
        )
    else:
        _int5_w5a8_dp4a_rows(
            qwords,
            x_scale,
            x_sum,
            ql,
            qh,
            scale,
            scale_step,
            zero,
            zero_point_step,
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
            stride_ssn,
            stride_ssk,
            stride_zn,
            stride_zk,
            stride_zsn,
            stride_zsk,
            stride_os,
            stride_om,
            stride_on,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            ROWS,
        )


@triton.jit
def _int5_w5a8_bucket_kernel(
    qwords,
    x_scale,
    x_sum,
    ql,
    qh,
    scale,
    scale_step,
    zero,
    zero_point_step,
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
    stride_ssn: tl.constexpr,
    stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr,
    stride_zsk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BUCKET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Select the generic kernel or an explicit-row implementation."""
    if ROWS == 0:
        _int5_w5a8_dp4a_kernel(
            qwords,
            x_scale,
            x_sum,
            ql,
            qh,
            scale,
            scale_step,
            zero,
            zero_point_step,
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
            stride_ssn,
            stride_ssk,
            stride_zn,
            stride_zk,
            stride_zsn,
            stride_zsk,
            stride_os,
            stride_om,
            stride_on,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
        )
    elif DYNAMIC_M:
        _int5_w5a8_dp4a_exact_rows(
            qwords,
            x_scale,
            x_sum,
            ql,
            qh,
            scale,
            scale_step,
            zero,
            zero_point_step,
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
            stride_ssn,
            stride_ssk,
            stride_zn,
            stride_zk,
            stride_zsn,
            stride_zsk,
            stride_os,
            stride_om,
            stride_on,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            ROWS,
        )
    else:
        _int5_w5a8_dp4a_rows(
            qwords,
            x_scale,
            x_sum,
            ql,
            qh,
            scale,
            scale_step,
            zero,
            zero_point_step,
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
            stride_ssn,
            stride_ssk,
            stride_zn,
            stride_zk,
            stride_zsn,
            stride_zsk,
            stride_os,
            stride_om,
            stride_on,
            GROUP_SIZE,
            BLOCK_M,
            BLOCK_N,
            SPLIT_K,
            PIPELINE_STAGES,
            DYNAMIC_M,
            ROWS,
        )


def _main_loop_trips(args) -> int:
    """Main-loop trips at 32 K32 groups per trip for one split."""
    groups_per_split = triton.cdiv(int(args["K"]) // _GROUP_SIZE, int(args["SPLIT_K"]))
    return triton.cdiv(groups_per_split, 32)


@triton.jit
def _decode_w5a16_tile(
    ql,
    qh,
    scale,
    scale_step,
    zero,
    zero_point_step,
    offs_n,
    n_mask,
    k_start,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_hn: tl.constexpr,
    stride_hk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_ssn: tl.constexpr,
    stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr,
    stride_zsk: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Decode one aligned BLOCK_K tile, expanding each metadata code once."""
    offs_packed = tl.arange(0, BLOCK_K // 2)
    packed = tl.load(
        ql
        + offs_n[:, None] * stride_qn
        + (k_start // 2 + offs_packed[None, :]) * stride_qk,
        mask=n_mask[:, None],
        other=0,
    ).to(tl.uint32)
    high = tl.load(
        qh
        + offs_n[:, None] * stride_hn
        + (k_start // 8 + offs_packed[None, :] // 4) * stride_hk,
        mask=n_mask[:, None],
        other=0,
    ).to(tl.uint32)
    shift = offs_packed[None, :] % 4
    even = (packed & 0xF) | (((high >> shift) & 1) << 4)
    odd = ((packed >> 4) & 0xF) | (((high >> (shift + 4)) & 1) << 4)
    q = tl.interleave(even, odd)
    # BF16 represents every integer in [128, 159] exactly at unit spacing.
    q = (q.to(tl.uint16) | 0x4300).to(tl.bfloat16, bitcast=True) - 128.0
    q = q.to(tl.bfloat16)

    offs_group = tl.arange(0, BLOCK_K // 32)
    group = k_start // 32 + offs_group
    scale_code = (
        tl.load(
            scale + offs_n[:, None] * stride_sn + group[None, :] * stride_sk,
            mask=n_mask[:, None],
            other=0,
        )
        .to(tl.float32)
        .to(tl.bfloat16)
    )
    zero_code = (
        tl.load(
            zero + offs_n[:, None] * stride_zn + group[None, :] * stride_zk,
            mask=n_mask[:, None],
            other=0,
        )
        .to(tl.float32)
        .to(tl.bfloat16)
    )
    step = k_start // 256
    scale_step_value = tl.load(
        scale_step + offs_n * stride_ssn + step * stride_ssk,
        mask=n_mask,
        other=0.0,
    ).to(tl.bfloat16)
    zero_step_value = tl.load(
        zero_point_step + offs_n * stride_zsn + step * stride_zsk,
        mask=n_mask,
        other=0.0,
    ).to(tl.bfloat16)
    scale_group = (scale_code * scale_step_value[:, None]).to(tl.bfloat16)
    zero_group = (zero_code * zero_step_value[:, None]).to(tl.bfloat16)
    scale_expanded = tl.reshape(
        tl.broadcast_to(scale_group[:, :, None], (BLOCK_N, BLOCK_K // 32, 32)),
        (BLOCK_N, BLOCK_K),
        can_reorder=False,
    )
    zero_expanded = tl.reshape(
        tl.broadcast_to(zero_group[:, :, None], (BLOCK_N, BLOCK_K // 32, 32)),
        (BLOCK_N, BLOCK_K),
        can_reorder=False,
    )
    return ((q - zero_expanded).to(tl.bfloat16) * scale_expanded).to(tl.bfloat16)


@triton.jit
def _int5_w5a16_direct_kernel(
    x,
    ql,
    qh,
    scale,
    scale_step,
    zero,
    zero_point_step,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_hn: tl.constexpr,
    stride_hk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_ssn: tl.constexpr,
    stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr,
    stride_zsk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    STATIC_M: tl.constexpr,
    BUCKET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """Direct W5A16 producer without split-range or workspace arithmetic."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if BLOCK_M >= BUCKET:
        offs_m = tl.arange(0, BLOCK_M)
    else:
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    n_mask = offs_n < N
    if STATIC_M > 0:
        m_mask = offs_m < STATIC_M
    else:
        m_mask = offs_m < M
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for super_block in tl.range(0, K // _TL_SUPER_BLOCK, num_stages=PIPELINE_STAGES):
        for tile in tl.static_range(0, 256 // BLOCK_K):
            k_start = super_block * 256 + tile * BLOCK_K
            weight = _decode_w5a16_tile(
                ql,
                qh,
                scale,
                scale_step,
                zero,
                zero_point_step,
                offs_n,
                n_mask,
                k_start,
                stride_qn,
                stride_qk,
                stride_hn,
                stride_hk,
                stride_sn,
                stride_sk,
                stride_ssn,
                stride_ssk,
                stride_zn,
                stride_zk,
                stride_zsn,
                stride_zsk,
                BLOCK_N,
                BLOCK_K,
            )
            activation = tl.load(
                x
                + offs_m[:, None] * stride_xm
                + (k_start + offs_k[None, :]) * stride_xk,
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            acc += tl.dot(activation, tl.trans(weight), out_dtype=tl.float32)
    tl.store(
        out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _int5_w5a16_splitk_kernel(
    x,
    ql,
    qh,
    scale,
    scale_step,
    zero,
    zero_point_step,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_hn: tl.constexpr,
    stride_hk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_ssn: tl.constexpr,
    stride_ssk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
    stride_zsn: tl.constexpr,
    stride_zsk: tl.constexpr,
    stride_os: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    STATIC_M: tl.constexpr,
    BUCKET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """Produce disjoint FP32 W5A16 K-partials for fixed-order reduction."""
    pid_n = tl.program_id(0)
    split_id = tl.program_id(1)
    if BLOCK_M >= BUCKET:
        offs_m = tl.arange(0, BLOCK_M)
    else:
        offs_m = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    if STATIC_M > 0:
        m_mask = offs_m < STATIC_M
    else:
        m_mask = offs_m < M
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_super_blocks: tl.constexpr = K // _TL_SUPER_BLOCK
    super_blocks_per_split: tl.constexpr = num_super_blocks // SPLIT_K
    extra_super_blocks: tl.constexpr = num_super_blocks % SPLIT_K
    first_super = split_id * super_blocks_per_split + tl.minimum(
        split_id, extra_super_blocks
    )
    split_super_blocks = super_blocks_per_split + (split_id < extra_super_blocks)
    last_super = first_super + split_super_blocks
    for super_block in tl.range(first_super, last_super, num_stages=PIPELINE_STAGES):
        for tile in tl.static_range(0, 256 // BLOCK_K):
            k_start = super_block * 256 + tile * BLOCK_K
            weight = _decode_w5a16_tile(
                ql,
                qh,
                scale,
                scale_step,
                zero,
                zero_point_step,
                offs_n,
                n_mask,
                k_start,
                stride_qn,
                stride_qk,
                stride_hn,
                stride_hk,
                stride_sn,
                stride_sk,
                stride_ssn,
                stride_ssk,
                stride_zn,
                stride_zk,
                stride_zsn,
                stride_zsk,
                BLOCK_N,
                BLOCK_K,
            )
            activation = tl.load(
                x
                + offs_m[:, None] * stride_xm
                + (k_start + offs_k[None, :]) * stride_xk,
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            acc += tl.dot(activation, tl.trans(weight), out_dtype=tl.float32)

    tl.store(
        out
        + split_id * stride_os
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _int5_large_autotune_configs(bucket: int) -> list[triton.Config]:
    """Default K256 tiles plus the measured bucket-64 K64 specialization."""
    configs = [
        triton.Config(
            {**config.kwargs, "BLOCK_K": _SUPER_BLOCK},
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        for config in tile_autotune_configs(bucket)
    ]
    if bucket == 64:
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 64,
                    "PIPELINE_STAGES": 1,
                },
                num_warps=4,
                num_stages=1,
            )
        )
    return configs


def int5_autotune_configs(bucket: int) -> list[triton.Config]:
    """DP4A row implementations for small M, BF16 tiles for larger M."""
    if bucket in _LARGE_BUCKETS:
        return _int5_large_autotune_configs(bucket)
    rows = sorted({bucket, min(bucket + 1, max(_SMALL_BUCKETS))})
    return autotune_configs(
        implementations=[{"ROWS": 0}] + [{"ROWS": row} for row in rows]
    )


def _prune(configs, named_args, **kwargs):
    return prune_by_main_loop_trips(_main_loop_trips, configs, named_args, **kwargs)


_BUCKET_KERNELS = {
    bucket: triton.autotune(
        configs=int5_autotune_configs(bucket),
        key=["N", "K", "SPLIT_K", "DYNAMIC_M"],
        prune_configs_by={"early_config_prune": _prune},
        do_bench=autotune_do_bench,
    )(_int5_w5a8_bucket_kernel)
    for bucket in _SMALL_BUCKETS
}


def _tile_loop_trips(args):
    return max(1, triton.cdiv(int(args["K"]) // _SUPER_BLOCK, int(args["SPLIT_K"])))


def _prune_tiles(configs, named_args, **kwargs):
    return prune_by_main_loop_trips(_tile_loop_trips, configs, named_args, **kwargs)


_W5A16_DIRECT_KERNELS = {
    b: triton.autotune(
        configs=_int5_large_autotune_configs(b),
        key=["M", "N", "K", "SPLIT_K"],
        prune_configs_by={"early_config_prune": _prune_tiles},
        do_bench=autotune_do_bench,
    )(_int5_w5a16_direct_kernel)
    for b in _LARGE_BUCKETS
}
_W5A16_SPLIT_KERNELS = {
    b: triton.autotune(
        configs=_int5_large_autotune_configs(b),
        key=["M", "N", "K", "SPLIT_K"],
        prune_configs_by={"early_config_prune": _prune_tiles},
        do_bench=autotune_do_bench,
    )(_int5_w5a16_splitk_kernel)
    for b in _LARGE_BUCKETS
}


def _unsupported_reason(
    bucket: int,
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    scale_step: torch.Tensor,
    zero: torch.Tensor,
    zero_point_step: torch.Tensor,
    group_size: int,
) -> Optional[str]:
    if bucket in _LARGE_BUCKETS and group_size != _GROUP_SIZE:
        return "large INT5 buckets require group_size=32"
    weights = (ql, qh, scale, scale_step, zero, zero_point_step)
    if any(tensor.dim() != 2 for tensor in (x, *weights)):
        return "expects rank-2 activation and weight tensors"

    M, K = x.shape
    N, K_half = ql.shape
    if isinstance(K, int) and K <= 0:
        return f"K must be positive, got {K}"

    weight_dims = tuple(dim for tensor in weights for dim in tensor.shape)
    weight_shape_reason = (
        None
        if all(isinstance(dim, int) for dim in weight_dims)
        else "weight shapes must be static"
    )
    valid_group_size = (
        isinstance(group_size, int)
        and not isinstance(group_size, bool)
        and group_size > 0
        and group_size % _GROUP_SIZE == 0
        and (group_size & (group_size - 1)) == 0
    )
    group_reason = (
        None
        if valid_group_size
        else (
            "group_size must be a positive power-of-two multiple of "
            f"{_GROUP_SIZE}, got {group_size}"
        )
    )
    reason = first_reason(
        check_dtypes(
            (
                ("activation", x, (torch.bfloat16,)),
                ("ql", ql, (torch.uint8, torch.int8)),
                ("qh", qh, (torch.uint8, torch.int8)),
                ("scale codes", scale, (torch.uint8,)),
                ("scale_step", scale_step, (torch.float16,)),
                ("zero codes", zero, (torch.uint8,)),
                ("zero_point_step", zero_point_step, (torch.float16,)),
            )
        ),
        group_reason,
        check_k(K, _SUPER_BLOCK),
        weight_shape_reason,
        check_rows(M, bucket),
        check_contiguous((x, *weights)),
        check_device(x, weights),
    )
    if reason is not None:
        return reason
    if K_half * 2 != K:
        return f"ql K/2 mismatch: x K={K}, ql K/2={K_half}"
    if tuple(qh.shape) != (N, K // 8):
        return "qh shape does not match [N, K/8]"
    if K % group_size != 0:
        return f"group_size={group_size} must divide K={K}"
    metadata_shape = (N, K // group_size)
    if tuple(scale.shape) != metadata_shape or tuple(zero.shape) != metadata_shape:
        return "scale/zero shape does not match [N, K/group_size]"
    step_shape = (N, K // _SUPER_BLOCK)
    if (
        tuple(scale_step.shape) != step_shape
        or tuple(zero_point_step.shape) != step_shape
    ):
        return "step shape does not match [N, K/256]"
    return None


@autotune_launch_param(
    "split_k",
    TILE_SPLIT_K_CANDIDATES,
    candidate_prepass=_compile_every_candidate_at_once,
)
def _launch_w5a16(
    bucket: int,
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    scale_step: torch.Tensor,
    zero: torch.Tensor,
    zero_point_step: torch.Tensor,
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
        kernels = _W5A16_DIRECT_KERNELS if split_k == 1 else _W5A16_SPLIT_KERNELS
        kernel = kernels[bucket]
        launch_options = {}
    else:
        kernel = (
            _int5_w5a16_direct_kernel if split_k == 1 else _int5_w5a16_splitk_kernel
        )
        launch_options = {
            **config.kwargs,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        launch_options.setdefault("BLOCK_K", _SUPER_BLOCK)
    launch_options["STATIC_M"] = M if isinstance(M, int) and M < bucket else 0
    return launch_split_k_gemm(
        kernel,
        bucket=bucket,
        m=M,
        n=N,
        device=x.device,
        split_k=split_k,
        block_m=bucket,
        reduce_block_n=32 if bucket >= 32 else 64,
        inputs=(x, ql, qh, scale, scale_step, zero, zero_point_step),
        split_k_on_axis_1=True,
        BUCKET=bucket,
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
            scale_step.stride(0),
            scale_step.stride(1),
            zero.stride(0),
            zero.stride(1),
            zero_point_step.stride(0),
            zero_point_step.stride(1),
        ),
        **launch_options,
    )


def _launch(
    bucket: int,
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    scale_step: torch.Tensor,
    zero: torch.Tensor,
    zero_point_step: torch.Tensor,
    group_size: int,
    **launch_params,
) -> torch.Tensor:
    """Runs the bucket kernel. Whether M is dynamic is read here, before
    split-K timing replaces ``x`` with a concrete tensor, so a dynamic-M call
    is timed on the dynamic path and cached apart from a static M."""
    if bucket in _LARGE_BUCKETS:
        return _launch_w5a16(
            bucket,
            x,
            ql,
            qh,
            scale,
            scale_step,
            zero,
            zero_point_step,
            group_size,
            **launch_params,
        )
    return _launch_rows(
        bucket,
        x,
        ql,
        qh,
        scale,
        scale_step,
        zero,
        zero_point_step,
        group_size,
        not isinstance(x.shape[0], int),
        **launch_params,
    )


@autotune_launch_param(
    "split_k",
    SPLIT_K_CANDIDATES,
    candidate_prepass=_compile_every_candidate_at_once,
)
def _launch_rows(
    bucket: int,
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    scale_step: torch.Tensor,
    zero: torch.Tensor,
    zero_point_step: torch.Tensor,
    group_size: int,
    dynamic_m: bool,
    *,
    split_k: int,
) -> torch.Tensor:
    M, K = x.shape
    N = ql.shape[0]
    check_split_k(split_k, int(K), _SUPER_BLOCK)
    qwords, x_scale, x_sum = quantize_activations_q8(x)
    block_m = triton.next_power_of_2(bucket)
    return launch_split_k_gemm(
        _BUCKET_KERNELS[bucket],
        bucket=bucket,
        m=M,
        n=N,
        device=x.device,
        split_k=split_k,
        block_m=block_m,
        inputs=(
            qwords,
            x_scale,
            x_sum,
            ql,
            qh,
            scale,
            scale_step,
            zero,
            zero_point_step,
        ),
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
            scale_step.stride(0),
            scale_step.stride(1),
            zero.stride(0),
            zero.stride(1),
            zero_point_step.stride(0),
            zero_point_step.stride(1),
        ),
        GROUP_SIZE=group_size,
        BUCKET=bucket,
        BLOCK_M=block_m,
        DYNAMIC_M=dynamic_m,
    )


def _prototype(
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    scale_step: torch.Tensor,
    zero: torch.Tensor,
    zero_point_step: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    raise NotImplementedError


def _fake(bucket: int, x: torch.Tensor, ql: torch.Tensor, *args) -> torch.Tensor:
    return torch.empty((x.shape[0], ql.shape[0]), dtype=torch.bfloat16, device=x.device)


INT5_QUANTIZED_GEMM = QuantizedGemmFamily(
    "int5_quantized_gemm",
    SUPPORTED_BUCKETS,
    _prototype,
    _launch,
    _fake,
    _unsupported_reason,
)


__all__ = [
    "INT5_QUANTIZED_GEMM",
    "SUPPORTED_BUCKETS",
    "TILE_SPLIT_K_CANDIDATES",
    "int5_autotune_configs",
]
