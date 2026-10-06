# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)

"""Decode-sized (M <= 4) INT4 GEMM on ``CudaCoalescedInt4Tensor`` weights.

The kernels consume ``CudaCoalescedInt4Tensor`` storage directly:

* qdata: [N, K / 2], low nibble is even K and high nibble is odd K;
* scale/zero: [N, K / 32] uint8 codes;
* scale_step/zero_point_step: [N, K / 256] fp16;
* weight: (q - zero_code * zero_step) * (scale_code * scale_step).

W4A8 DP4A: each activation K32 block is quantized once to signed INT8 with an
FP32 scale, and the INT4 x INT8 dot products run on DP4A with the group
scale/zero correction applied in FP32. The output is BF16.

One op per row count (bucket) M in {1, 2, 3, 4},
``triton::int4_quantized_gemm_m{M}``, registered as ``INT4_QUANTIZED_GEMM``.
Each kernel skips rows at or above the runtime M, so a bucket also serves a
dynamic M up to its size.

Nothing is tuned offline. Each bucket's kernel is a ``triton.autotune`` over the
generic space (``quantized_gemm_utils.autotune_configs``): the implementations
(the generic row-blocked kernel, and explicit per-row-accumulator kernels with
the bucket's rows or one more) x rows per CTA x pipeline stages, with stage
counts above the main loop's trip count pruned. AOTInductor picks among the
rest at compile time on the target GPU. Under a dynamic M, an explicit kernel
branches to the one with exactly the runtime M rows.
Split-K comes from the device's SM count and reduces deterministically.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    launch_split_k_gemm,
    QuantizedGemmFamily,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import (
    _device_index,
    _dp4a_u8_s8,
    _sm_count,
    _warp_sum_f32,
    autotune_configs,
    check_contiguous,
    check_device,
    check_dtypes,
    check_k,
    check_rows,
    first_reason,
    prune_by_main_loop_trips,
    quantize_activations_q8,
    split_k_for,
)

_GROUP_SIZE = 32
_SUPER_BLOCK = 256
_TL_GROUP_SIZE = tl.constexpr(32)

SUPPORTED_BUCKETS = (1, 2, 3, 4)


@triton.jit
def _int4_w4a8_dp4a_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    scale_step,
    zero,
    zero_point_step,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    """C-shim-style W4A8: one warp owns one output row."""
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
    partial = tl.zeros((BLOCK_M, BLOCK_N * 32), dtype=tl.float32)
    groups_per_split: tl.constexpr = tl.cdiv(groups, SPLIT_K)
    first_group = split_id * groups_per_split
    last_group = tl.minimum(first_group + groups_per_split, groups)

    for group_offset in tl.range(0, groups_per_split, 32, num_stages=PIPELINE_STAGES):
        group = first_group + group_offset + lane
        group_mask = n_mask & (group < last_group)
        activation_mask = (offs_m[:, None] < M) & group_mask[None, :]
        qbase = qdata + offs_n * stride_qn + group * 16 * stride_qk
        qword_base = (offs_m[:, None] * groups + group[None, :]) * 8
        word = tl.arange(0, 4)
        packed = tl.load(
            qbase.to(tl.pointer_type(tl.uint32))[:, None] + word[None, :],
            mask=group_mask[:, None],
            other=0,
        )
        q_even = packed & 0x0F0F0F0F
        q_odd = (packed >> 4) & 0x0F0F0F0F
        a_even = tl.load(
            qwords + qword_base[:, :, None] + word[None, None, :],
            mask=activation_mask[:, :, None],
            other=0,
        )
        a_odd = tl.load(
            qwords + qword_base[:, :, None] + 4 + word[None, None, :],
            mask=activation_mask[:, :, None],
            other=0,
        )
        dot_words = _dp4a_u8_s8(
            q_even[None, :, :],
            a_even,
            tl.zeros((BLOCK_M, BLOCK_N * 32, 4), dtype=tl.int32),
        )
        dot_words = _dp4a_u8_s8(q_odd[None, :, :], a_odd, dot_words)
        dot = tl.sum(dot_words, axis=2)

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
        scale_code = tl.load(
            scale + offs_n * stride_sn + group * stride_sk,
            mask=group_mask,
            other=0,
        ).to(tl.float32)
        zero_code = tl.load(
            zero + offs_n * stride_zn + group * stride_zk,
            mask=group_mask,
            other=0,
        ).to(tl.float32)
        super_block = group // 8
        scale_step_value = tl.load(
            scale_step + offs_n * stride_ssn + super_block * stride_ssk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        zero_step_value = tl.load(
            zero_point_step + offs_n * stride_zsn + super_block * stride_zsk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        corrected = (
            dot.to(tl.float32) - activation_sum * (zero_code * zero_step_value)[None, :]
        )
        contribution = (
            corrected * activation_scale * (scale_code * scale_step_value)[None, :]
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
def _w4a8_row_contribution(
    qwords,
    x_scale,
    q_even,
    q_odd,
    ws,
    wz,
    group,
    activation_mask,
    row,
    groups: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Accumulate one activation row using shared weight metadata."""
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
        q_even,
        activation_even,
        tl.zeros((BLOCK_N * 32, 4), dtype=tl.int32),
    )
    dot_words = _dp4a_u8_s8(q_odd, activation_odd, dot_words)
    dot = tl.sum(dot_words, axis=1).to(tl.float32)
    activation_scale = tl.load(
        x_scale + row * groups + group,
        mask=activation_mask,
        other=0.0,
    ).to(tl.float32)
    ones = tl.full((BLOCK_N * 32, 4), 0x01010101, dtype=tl.uint32)
    activation_sum_words = _dp4a_u8_s8(
        ones,
        activation_even,
        tl.zeros((BLOCK_N * 32, 4), dtype=tl.int32),
    )
    activation_sum_words = _dp4a_u8_s8(ones, activation_odd, activation_sum_words)
    activation_sum = tl.sum(activation_sum_words, axis=1).to(tl.float32)
    return tl.where(
        activation_mask,
        (dot - activation_sum * wz) * activation_scale * ws,
        0.0,
    )


@triton.jit
def _w4a8_row_contribution_if_active(
    qwords,
    x_scale,
    q_even,
    q_odd,
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
    """A row the runtime M may not reach: skipped by a uniform branch when M is
    dynamic, masked when it is static."""
    if DYNAMIC_M:
        contribution = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
        if row < M:
            contribution = _w4a8_row_contribution(
                qwords,
                x_scale,
                q_even,
                q_odd,
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
        return _w4a8_row_contribution(
            qwords,
            x_scale,
            q_even,
            q_odd,
            ws,
            wz,
            group,
            group_mask & (row < M),
            row,
            groups,
            BLOCK_N,
        )


@triton.jit
def _int4_w4a8_dp4a_m1_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    scale_step,
    zero,
    zero_point_step,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    """M1 W4A8 DP4A with one explicit per-thread row accumulator."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    split_id = tl.program_id(2)
    thread = tl.arange(0, BLOCK_N * 32)
    warp = thread // 32
    lane = thread % 32
    offs_n = pid_n * BLOCK_N + warp
    n_mask = offs_n < N
    groups: tl.constexpr = K // _TL_GROUP_SIZE
    partial0 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    groups_per_split: tl.constexpr = tl.cdiv(groups, SPLIT_K)
    first_group = split_id * groups_per_split
    last_group = tl.minimum(first_group + groups_per_split, groups)
    row0 = pid_m * BLOCK_M

    for group_offset in tl.range(0, groups_per_split, 32, num_stages=PIPELINE_STAGES):
        group = first_group + group_offset + lane
        group_mask = n_mask & (group < last_group)
        qbase = qdata + offs_n * stride_qn + group * 16 * stride_qk
        word = tl.arange(0, 4)
        packed = tl.load(
            qbase.to(tl.pointer_type(tl.uint32))[:, None] + word[None, :],
            mask=group_mask[:, None],
            other=0,
        )
        q_even = packed & 0x0F0F0F0F
        q_odd = (packed >> 4) & 0x0F0F0F0F
        scale_code = tl.load(
            scale + offs_n * stride_sn + group * stride_sk,
            mask=group_mask,
            other=0,
        ).to(tl.float32)
        zero_code = tl.load(
            zero + offs_n * stride_zn + group * stride_zk,
            mask=group_mask,
            other=0,
        ).to(tl.float32)
        super_block = group // 8
        scale_step_value = tl.load(
            scale_step + offs_n * stride_ssn + super_block * stride_ssk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        zero_step_value = tl.load(
            zero_point_step + offs_n * stride_zsn + super_block * stride_zsk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        ws = scale_code * scale_step_value
        wz = zero_code * zero_step_value
        partial0 += _w4a8_row_contribution(
            qwords,
            x_scale,
            q_even,
            q_odd,
            ws,
            wz,
            group,
            group_mask & (row0 < M),
            row0,
            groups,
            BLOCK_N,
        )

    result0 = _warp_sum_f32(partial0)
    out_base = out + split_id * stride_os + offs_n * stride_on
    tl.store(
        out_base + row0 * stride_om,
        result0.to(tl.float32) if SPLIT_K > 1 else result0.to(tl.bfloat16),
        mask=n_mask & (lane == 0) & (row0 < M),
    )


@triton.jit
def _int4_w4a8_dp4a_m2_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    scale_step,
    zero,
    zero_point_step,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    """M2 W4A8 DP4A with two explicit per-thread row accumulators."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    split_id = tl.program_id(2)
    thread = tl.arange(0, BLOCK_N * 32)
    warp = thread // 32
    lane = thread % 32
    offs_n = pid_n * BLOCK_N + warp
    n_mask = offs_n < N
    groups: tl.constexpr = K // _TL_GROUP_SIZE
    partial0 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial1 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    groups_per_split: tl.constexpr = tl.cdiv(groups, SPLIT_K)
    first_group = split_id * groups_per_split
    last_group = tl.minimum(first_group + groups_per_split, groups)
    row0 = pid_m * BLOCK_M
    row1 = row0 + 1

    for group_offset in tl.range(0, groups_per_split, 32, num_stages=PIPELINE_STAGES):
        group = first_group + group_offset + lane
        group_mask = n_mask & (group < last_group)
        qbase = qdata + offs_n * stride_qn + group * 16 * stride_qk
        word = tl.arange(0, 4)
        packed = tl.load(
            qbase.to(tl.pointer_type(tl.uint32))[:, None] + word[None, :],
            mask=group_mask[:, None],
            other=0,
        )
        q_even = packed & 0x0F0F0F0F
        q_odd = (packed >> 4) & 0x0F0F0F0F
        scale_code = tl.load(
            scale + offs_n * stride_sn + group * stride_sk,
            mask=group_mask,
            other=0,
        ).to(tl.float32)
        zero_code = tl.load(
            zero + offs_n * stride_zn + group * stride_zk,
            mask=group_mask,
            other=0,
        ).to(tl.float32)
        super_block = group // 8
        scale_step_value = tl.load(
            scale_step + offs_n * stride_ssn + super_block * stride_ssk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        zero_step_value = tl.load(
            zero_point_step + offs_n * stride_zsn + super_block * stride_zsk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        ws = scale_code * scale_step_value
        wz = zero_code * zero_step_value
        partial0 += _w4a8_row_contribution(
            qwords,
            x_scale,
            q_even,
            q_odd,
            ws,
            wz,
            group,
            group_mask & (row0 < M),
            row0,
            groups,
            BLOCK_N,
        )
        partial1 += _w4a8_row_contribution_if_active(
            qwords,
            x_scale,
            q_even,
            q_odd,
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

    result0 = _warp_sum_f32(partial0)
    result1 = _warp_sum_f32(partial1)
    out_base = out + split_id * stride_os + offs_n * stride_on
    store_mask = n_mask & (lane == 0)
    tl.store(
        out_base + row0 * stride_om,
        result0.to(tl.float32) if SPLIT_K > 1 else result0.to(tl.bfloat16),
        mask=store_mask & (row0 < M),
    )
    tl.store(
        out_base + row1 * stride_om,
        result1.to(tl.float32) if SPLIT_K > 1 else result1.to(tl.bfloat16),
        mask=store_mask & (row1 < M),
    )


@triton.jit
def _int4_w4a8_dp4a_m3_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    scale_step,
    zero,
    zero_point_step,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    """M3 W4A8 DP4A with three explicit per-thread row accumulators."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    split_id = tl.program_id(2)
    thread = tl.arange(0, BLOCK_N * 32)
    warp = thread // 32
    lane = thread % 32
    offs_n = pid_n * BLOCK_N + warp
    n_mask = offs_n < N
    groups: tl.constexpr = K // _TL_GROUP_SIZE
    partial0 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial1 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial2 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    groups_per_split: tl.constexpr = tl.cdiv(groups, SPLIT_K)
    first_group = split_id * groups_per_split
    last_group = tl.minimum(first_group + groups_per_split, groups)
    row0 = pid_m * BLOCK_M
    row1 = row0 + 1
    row2 = row0 + 2

    for group_offset in tl.range(0, groups_per_split, 32, num_stages=PIPELINE_STAGES):
        group = first_group + group_offset + lane
        group_mask = n_mask & (group < last_group)
        qbase = qdata + offs_n * stride_qn + group * 16 * stride_qk
        word = tl.arange(0, 4)
        packed = tl.load(
            qbase.to(tl.pointer_type(tl.uint32))[:, None] + word[None, :],
            mask=group_mask[:, None],
            other=0,
        )
        q_even = packed & 0x0F0F0F0F
        q_odd = (packed >> 4) & 0x0F0F0F0F
        scale_code = tl.load(
            scale + offs_n * stride_sn + group * stride_sk,
            mask=group_mask,
            other=0,
        ).to(tl.float32)
        zero_code = tl.load(
            zero + offs_n * stride_zn + group * stride_zk,
            mask=group_mask,
            other=0,
        ).to(tl.float32)
        super_block = group // 8
        scale_step_value = tl.load(
            scale_step + offs_n * stride_ssn + super_block * stride_ssk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        zero_step_value = tl.load(
            zero_point_step + offs_n * stride_zsn + super_block * stride_zsk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        ws = scale_code * scale_step_value
        wz = zero_code * zero_step_value
        partial0 += _w4a8_row_contribution(
            qwords,
            x_scale,
            q_even,
            q_odd,
            ws,
            wz,
            group,
            group_mask & (row0 < M),
            row0,
            groups,
            BLOCK_N,
        )
        partial1 += _w4a8_row_contribution_if_active(
            qwords,
            x_scale,
            q_even,
            q_odd,
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
        partial2 += _w4a8_row_contribution_if_active(
            qwords,
            x_scale,
            q_even,
            q_odd,
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

    result0 = _warp_sum_f32(partial0)
    result1 = _warp_sum_f32(partial1)
    result2 = _warp_sum_f32(partial2)
    out_base = out + split_id * stride_os + offs_n * stride_on
    store_mask = n_mask & (lane == 0)
    tl.store(
        out_base + row0 * stride_om,
        result0.to(tl.float32) if SPLIT_K > 1 else result0.to(tl.bfloat16),
        mask=store_mask & (row0 < M),
    )
    tl.store(
        out_base + row1 * stride_om,
        result1.to(tl.float32) if SPLIT_K > 1 else result1.to(tl.bfloat16),
        mask=store_mask & (row1 < M),
    )
    tl.store(
        out_base + row2 * stride_om,
        result2.to(tl.float32) if SPLIT_K > 1 else result2.to(tl.bfloat16),
        mask=store_mask & (row2 < M),
    )


@triton.jit
def _int4_w4a8_dp4a_m4_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    scale_step,
    zero,
    zero_point_step,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    """M4 W4A8 DP4A with four explicit per-thread row accumulators."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    split_id = tl.program_id(2)
    thread = tl.arange(0, BLOCK_N * 32)
    warp = thread // 32
    lane = thread % 32
    offs_n = pid_n * BLOCK_N + warp
    n_mask = offs_n < N
    groups: tl.constexpr = K // _TL_GROUP_SIZE
    partial0 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial1 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial2 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    partial3 = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
    groups_per_split: tl.constexpr = tl.cdiv(groups, SPLIT_K)
    first_group = split_id * groups_per_split
    last_group = tl.minimum(first_group + groups_per_split, groups)
    row0 = pid_m * BLOCK_M
    row1 = row0 + 1
    row2 = row0 + 2
    row3 = row0 + 3

    for group_offset in tl.range(0, groups_per_split, 32, num_stages=PIPELINE_STAGES):
        group = first_group + group_offset + lane
        group_mask = n_mask & (group < last_group)
        qbase = qdata + offs_n * stride_qn + group * 16 * stride_qk
        word = tl.arange(0, 4)
        packed = tl.load(
            qbase.to(tl.pointer_type(tl.uint32))[:, None] + word[None, :],
            mask=group_mask[:, None],
            other=0,
        )
        q_even = packed & 0x0F0F0F0F
        q_odd = (packed >> 4) & 0x0F0F0F0F
        scale_code = tl.load(
            scale + offs_n * stride_sn + group * stride_sk,
            mask=group_mask,
            other=0,
        ).to(tl.float32)
        zero_code = tl.load(
            zero + offs_n * stride_zn + group * stride_zk,
            mask=group_mask,
            other=0,
        ).to(tl.float32)
        super_block = group // 8
        scale_step_value = tl.load(
            scale_step + offs_n * stride_ssn + super_block * stride_ssk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        zero_step_value = tl.load(
            zero_point_step + offs_n * stride_zsn + super_block * stride_zsk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        ws = scale_code * scale_step_value
        wz = zero_code * zero_step_value
        partial0 += _w4a8_row_contribution(
            qwords,
            x_scale,
            q_even,
            q_odd,
            ws,
            wz,
            group,
            group_mask & (row0 < M),
            row0,
            groups,
            BLOCK_N,
        )
        partial1 += _w4a8_row_contribution_if_active(
            qwords,
            x_scale,
            q_even,
            q_odd,
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
        partial2 += _w4a8_row_contribution_if_active(
            qwords,
            x_scale,
            q_even,
            q_odd,
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
        partial3 += _w4a8_row_contribution_if_active(
            qwords,
            x_scale,
            q_even,
            q_odd,
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
    result1 = _warp_sum_f32(partial1)
    result2 = _warp_sum_f32(partial2)
    result3 = _warp_sum_f32(partial3)
    out_base = out + split_id * stride_os + offs_n * stride_on
    store_mask = n_mask & (lane == 0)
    tl.store(
        out_base + row0 * stride_om,
        result0.to(tl.float32) if SPLIT_K > 1 else result0.to(tl.bfloat16),
        mask=store_mask & (row0 < M),
    )
    tl.store(
        out_base + row1 * stride_om,
        result1.to(tl.float32) if SPLIT_K > 1 else result1.to(tl.bfloat16),
        mask=store_mask & (row1 < M),
    )
    tl.store(
        out_base + row2 * stride_om,
        result2.to(tl.float32) if SPLIT_K > 1 else result2.to(tl.bfloat16),
        mask=store_mask & (row2 < M),
    )
    tl.store(
        out_base + row3 * stride_om,
        result3.to(tl.float32) if SPLIT_K > 1 else result3.to(tl.bfloat16),
        mask=store_mask & (row3 < M),
    )


@triton.jit
def _int4_w4a8_bucket_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    scale_step,
    zero,
    zero_point_step,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
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
    BUCKET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
    ROWS: tl.constexpr,
):
    """One bucket's W4A8 kernel. ROWS selects the implementation: 0 is the
    generic row-blocked kernel, R > 0 the explicit R-row kernel (R >= the
    bucket; rows at or above the runtime M are masked). Under a dynamic M the
    explicit kernel branches, uniformly, to the explicit kernel with exactly M
    rows."""
    if ROWS == 0:
        _int4_w4a8_dp4a_kernel(
            qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
            out, M, N, K, stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn,
            stride_ssk, stride_zn, stride_zk, stride_zsn, stride_zsk, stride_os,
            stride_om, stride_on, BLOCK_M, BLOCK_N, SPLIT_K, PIPELINE_STAGES, DYNAMIC_M,
        )
    elif DYNAMIC_M:
        _int4_w4a8_dp4a_exact_rows(
            qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
            out, M, N, K, stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn,
            stride_ssk, stride_zn, stride_zk, stride_zsn, stride_zsk, stride_os,
            stride_om, stride_on, BLOCK_M, BLOCK_N, SPLIT_K, PIPELINE_STAGES, DYNAMIC_M, ROWS,
        )
    else:
        _int4_w4a8_dp4a_rows(
            qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
            out, M, N, K, stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn,
            stride_ssk, stride_zn, stride_zk, stride_zsn, stride_zsk, stride_os,
            stride_om, stride_on, BLOCK_M, BLOCK_N, SPLIT_K, PIPELINE_STAGES, DYNAMIC_M, ROWS,
        )


@triton.jit
def _int4_w4a8_dp4a_rows(
    qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
    out, M, N: tl.constexpr, K: tl.constexpr, stride_qn: tl.constexpr,
    stride_qk: tl.constexpr, stride_sn: tl.constexpr, stride_sk: tl.constexpr,
    stride_ssn: tl.constexpr, stride_ssk: tl.constexpr, stride_zn: tl.constexpr,
    stride_zk: tl.constexpr, stride_zsn: tl.constexpr, stride_zsk: tl.constexpr,
    stride_os: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr, DYNAMIC_M: tl.constexpr, ROWS: tl.constexpr,
):
    """The explicit ROWS-row kernel."""
    if ROWS == 1:
        _int4_w4a8_dp4a_m1_kernel(
            qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
            out, M, N, K, stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn,
            stride_ssk, stride_zn, stride_zk, stride_zsn, stride_zsk, stride_os,
            stride_om, stride_on, BLOCK_M, BLOCK_N, SPLIT_K, PIPELINE_STAGES, DYNAMIC_M,
        )
    elif ROWS == 2:
        _int4_w4a8_dp4a_m2_kernel(
            qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
            out, M, N, K, stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn,
            stride_ssk, stride_zn, stride_zk, stride_zsn, stride_zsk, stride_os,
            stride_om, stride_on, BLOCK_M, BLOCK_N, SPLIT_K, PIPELINE_STAGES, DYNAMIC_M,
        )
    elif ROWS == 3:
        _int4_w4a8_dp4a_m3_kernel(
            qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
            out, M, N, K, stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn,
            stride_ssk, stride_zn, stride_zk, stride_zsn, stride_zsk, stride_os,
            stride_om, stride_on, BLOCK_M, BLOCK_N, SPLIT_K, PIPELINE_STAGES, DYNAMIC_M,
        )
    else:
        _int4_w4a8_dp4a_m4_kernel(
            qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
            out, M, N, K, stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn,
            stride_ssk, stride_zn, stride_zk, stride_zsn, stride_zsk, stride_os,
            stride_om, stride_on, BLOCK_M, BLOCK_N, SPLIT_K, PIPELINE_STAGES, DYNAMIC_M,
        )


@triton.jit
def _int4_w4a8_dp4a_exact_rows(
    qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
    out, M, N: tl.constexpr, K: tl.constexpr, stride_qn: tl.constexpr,
    stride_qk: tl.constexpr, stride_sn: tl.constexpr, stride_sk: tl.constexpr,
    stride_ssn: tl.constexpr, stride_ssk: tl.constexpr, stride_zn: tl.constexpr,
    stride_zk: tl.constexpr, stride_zsn: tl.constexpr, stride_zsk: tl.constexpr,
    stride_os: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr, DYNAMIC_M: tl.constexpr, ROWS: tl.constexpr,
):
    """Dynamic M: the explicit kernel with exactly M rows (M <= ROWS)."""
    if M == 1:
        _int4_w4a8_dp4a_m1_kernel(
            qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
            out, M, N, K, stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn,
            stride_ssk, stride_zn, stride_zk, stride_zsn, stride_zsk, stride_os,
            stride_om, stride_on, BLOCK_M, BLOCK_N, SPLIT_K, PIPELINE_STAGES, DYNAMIC_M,
        )
    elif ROWS >= 2 and M == 2:
        _int4_w4a8_dp4a_m2_kernel(
            qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
            out, M, N, K, stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn,
            stride_ssk, stride_zn, stride_zk, stride_zsn, stride_zsk, stride_os,
            stride_om, stride_on, BLOCK_M, BLOCK_N, SPLIT_K, PIPELINE_STAGES, DYNAMIC_M,
        )
    elif ROWS >= 3 and M == 3:
        _int4_w4a8_dp4a_m3_kernel(
            qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
            out, M, N, K, stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn,
            stride_ssk, stride_zn, stride_zk, stride_zsn, stride_zsk, stride_os,
            stride_om, stride_on, BLOCK_M, BLOCK_N, SPLIT_K, PIPELINE_STAGES, DYNAMIC_M,
        )
    else:
        _int4_w4a8_dp4a_rows(
            qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step,
            out, M, N, K, stride_qn, stride_qk, stride_sn, stride_sk, stride_ssn,
            stride_ssk, stride_zn, stride_zk, stride_zsn, stride_zsk, stride_os,
            stride_om, stride_on, BLOCK_M, BLOCK_N, SPLIT_K, PIPELINE_STAGES, DYNAMIC_M, ROWS,
        )


def _main_loop_trips(args) -> int:
    """Trips of every INT4 kernel's main loop: 32 K32 groups per trip over this
    split's share of the K / 32 groups."""
    groups_per_split = triton.cdiv(int(args["K"]) // _GROUP_SIZE, int(args["SPLIT_K"]))
    return triton.cdiv(groups_per_split, 32)


def int4_autotune_configs(bucket: int) -> list[triton.Config]:
    """The generic space over three implementations: the generic kernel, the
    explicit kernel with the bucket's rows, and the next-larger explicit
    kernel, whose extra row is masked but whose code can be faster."""
    rows = sorted({bucket, min(bucket + 1, max(SUPPORTED_BUCKETS))})
    return autotune_configs([{"ROWS": 0}] + [{"ROWS": r} for r in rows])


def _prune(configs, named_args, **kwargs):
    return prune_by_main_loop_trips(_main_loop_trips, configs, named_args, **kwargs)


# One autotuner per bucket, so each bucket keeps its own picks.
_BUCKET_KERNELS = {
    bucket: triton.autotune(
        configs=int4_autotune_configs(bucket),
        key=["N", "K", "SPLIT_K"],
        prune_configs_by={"early_config_prune": _prune},
    )(_int4_w4a8_bucket_kernel)
    for bucket in SUPPORTED_BUCKETS
}


def _unsupported_reason(
    bucket: int,
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    scale_step: torch.Tensor,
    zero: torch.Tensor,
    zero_point_step: torch.Tensor,
    group_size: int,
) -> Optional[str]:
    weights = (qdata, scale, scale_step, zero, zero_point_step)
    if x.dim() != 2 or qdata.dim() != 2:
        return "expects a rank-2 activation and qdata"
    M, K = x.shape
    N, K_half = qdata.shape
    reason = first_reason(
        check_dtypes(
            (
                ("activation", x, (torch.bfloat16,)),
                ("qdata", qdata, (torch.uint8, torch.int8)),
                ("scale codes", scale, (torch.uint8,)),
                ("zero codes", zero, (torch.uint8,)),
                ("scale_step", scale_step, (torch.float16,)),
                ("zero_point_step", zero_point_step, (torch.float16,)),
            )
        ),
        None if group_size == _GROUP_SIZE else f"group_size must be {_GROUP_SIZE}, got {group_size}",
        check_k(K, _SUPER_BLOCK),
        check_rows(M, bucket),
        check_contiguous((x, *weights)),
        check_device(x, weights),
    )
    if reason is not None:
        return reason
    if K_half * 2 != K:
        return f"qdata K/2 mismatch: x K={K}, qdata K/2={K_half}"
    if tuple(scale.shape) != (N, K // _GROUP_SIZE) or tuple(zero.shape) != (N, K // _GROUP_SIZE):
        return "scale/zero shape does not match [N, K/32]"
    if tuple(scale_step.shape) != (N, K // _SUPER_BLOCK) or tuple(zero_point_step.shape) != (
        N,
        K // _SUPER_BLOCK,
    ):
        return "step shape does not match [N, K/256]"
    return None


def _launch(
    bucket: int,
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    scale_step: torch.Tensor,
    zero: torch.Tensor,
    zero_point_step: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    M, K = x.shape
    N = qdata.shape[0]
    split_k = split_k_for(bucket, int(N), int(K), _sm_count(_device_index(x.device)))
    qwords, x_scale, x_sum = quantize_activations_q8(x)
    return launch_split_k_gemm(
        _BUCKET_KERNELS[bucket],
        bucket=bucket,
        m=M,
        n=N,
        device=x.device,
        split_k=split_k,
        # The generic implementation spans BLOCK_M rows with tl.arange; the
        # explicit-row ones read exactly the bucket's rows.
        block_m=triton.next_power_of_2(bucket),
        inputs=(qwords, x_scale, x_sum, qdata, scale, scale_step, zero, zero_point_step),
        shape_args=(
            M,
            N,
            K,
            qdata.stride(0),
            qdata.stride(1),
            scale.stride(0),
            scale.stride(1),
            scale_step.stride(0),
            scale_step.stride(1),
            zero.stride(0),
            zero.stride(1),
            zero_point_step.stride(0),
            zero_point_step.stride(1),
        ),
        BUCKET=bucket,
        BLOCK_M=triton.next_power_of_2(bucket),
        DYNAMIC_M=not isinstance(M, int),
    )


def _prototype(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    scale_step: torch.Tensor,
    zero: torch.Tensor,
    zero_point_step: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    raise NotImplementedError


def _fake(bucket: int, x: torch.Tensor, qdata: torch.Tensor, *args) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], qdata.shape[0]), dtype=torch.bfloat16, device=x.device
    )


INT4_QUANTIZED_GEMM = QuantizedGemmFamily(
    "int4_quantized_gemm",
    SUPPORTED_BUCKETS,
    _prototype,
    _launch,
    _fake,
    _unsupported_reason,
)


__all__ = ["INT4_QUANTIZED_GEMM", "SUPPORTED_BUCKETS", "int4_autotune_configs"]
