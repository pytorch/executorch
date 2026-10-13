# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)

"""Decode-sized (M <= 4) W8A8 GEMM for natural-order INT8 weights.

The weight contract is qdata [N, K] INT8 and scale/zero [N, K/group_size]
BF16/INT8, with ``w = (qdata - zero) * scale``. Activations are quantized to
signed INT8 in natural-order K32 blocks. Each lane owns one K32 block and each
warp owns one output n. Signed weight bytes are biased to unsigned for the
shared U8xS8 DP4A helper, then corrected in FP32 by ``(zero + 128) * x_sum``.
The output is BF16; split-K partials and all kernel accumulators are FP32.

Each M bucket has an independent autotuner over the generic row-blocked kernel,
its explicit row count, and the next larger explicit count where one exists.
Dynamic M dispatches explicit variants to exactly the active row count, and
every load/store is row masked.
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
)

_GROUP_SIZE = 32
_SUPER_BLOCK = 256
_TL_GROUP_SIZE = tl.constexpr(32)

SUPPORTED_BUCKETS = (1, 2, 3, 4)


@triton.jit
def _int8_w8a8_dp4a_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    zero,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
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
    """Generic row-blocked W8A8 kernel."""
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
        qbase = qdata + offs_n * stride_qn + group * 32 * stride_qk
        qword_base = (offs_m[:, None] * groups + group[None, :]) * 8
        word = tl.arange(0, 8)
        packed = tl.load(
            qbase.to(tl.pointer_type(tl.uint32))[:, None] + word[None, :],
            mask=group_mask[:, None],
            other=0,
        )
        q_unsigned = packed ^ 0x80808080
        activation = tl.load(
            qwords + qword_base[:, :, None] + word[None, None, :],
            mask=activation_mask[:, :, None],
            other=0,
        )
        dot_words = _dp4a_u8_s8(
            q_unsigned[None, :, :],
            activation,
            tl.zeros((BLOCK_M, BLOCK_N * 32, 8), dtype=tl.int32),
        )
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
        metadata_group = group * _TL_GROUP_SIZE // GROUP_SIZE
        weight_scale = tl.load(
            scale + offs_n * stride_sn + metadata_group * stride_sk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        zero_bias = (
            tl.load(
                zero + offs_n * stride_zn + metadata_group * stride_zk,
                mask=group_mask,
                other=0,
            ).to(tl.float32)
            + 128.0
        )
        contribution = (
            (dot - zero_bias[None, :] * activation_sum)
            * activation_scale
            * weight_scale[None, :]
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
def _w8a8_row_contribution(
    qwords,
    x_scale,
    x_sum,
    q_unsigned,
    weight_scale,
    zero_bias,
    group,
    activation_mask,
    row,
    groups: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Accumulate one activation row using shared weight data."""
    word = tl.arange(0, 8)
    qword_base = (row * groups + group) * 8
    activation = tl.load(
        qwords + qword_base[:, None] + word[None, :],
        mask=activation_mask[:, None],
        other=0,
    )
    dot_words = _dp4a_u8_s8(
        q_unsigned,
        activation,
        tl.zeros((BLOCK_N * 32, 8), dtype=tl.int32),
    )
    dot = tl.sum(dot_words, axis=1).to(tl.float32)
    activation_scale = tl.load(
        x_scale + row * groups + group,
        mask=activation_mask,
        other=0.0,
    ).to(tl.float32)
    activation_sum = tl.load(
        x_sum + row * groups + group,
        mask=activation_mask,
        other=0,
    ).to(tl.float32)
    return tl.where(
        activation_mask,
        (dot - zero_bias * activation_sum) * activation_scale * weight_scale,
        0.0,
    )


@triton.jit
def _w8a8_row_contribution_if_active(
    qwords,
    x_scale,
    x_sum,
    q_unsigned,
    weight_scale,
    zero_bias,
    group,
    group_mask,
    row,
    M,
    groups: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DYNAMIC_M: tl.constexpr,
):
    """Skip a dynamically inactive row with a warp-uniform branch."""
    if DYNAMIC_M:
        contribution = tl.zeros((BLOCK_N * 32,), dtype=tl.float32)
        if row < M:
            contribution = _w8a8_row_contribution(
                qwords,
                x_scale,
                x_sum,
                q_unsigned,
                weight_scale,
                zero_bias,
                group,
                group_mask,
                row,
                groups,
                BLOCK_N,
            )
        return contribution
    else:
        return _w8a8_row_contribution(
            qwords,
            x_scale,
            x_sum,
            q_unsigned,
            weight_scale,
            zero_bias,
            group,
            group_mask & (row < M),
            row,
            groups,
            BLOCK_N,
        )


@triton.jit
def _int8_w8a8_dp4a_explicit_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    zero,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
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
    """W8A8 DP4A with ROWS explicit per-thread accumulators."""
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
        qbase = qdata + offs_n * stride_qn + group * 32 * stride_qk
        word = tl.arange(0, 8)
        packed = tl.load(
            qbase.to(tl.pointer_type(tl.uint32))[:, None] + word[None, :],
            mask=group_mask[:, None],
            other=0,
        )
        q_unsigned = packed ^ 0x80808080
        metadata_group = group * _TL_GROUP_SIZE // GROUP_SIZE
        weight_scale = tl.load(
            scale + offs_n * stride_sn + metadata_group * stride_sk,
            mask=group_mask,
            other=0.0,
        ).to(tl.float32)
        zero_bias = (
            tl.load(
                zero + offs_n * stride_zn + metadata_group * stride_zk,
                mask=group_mask,
                other=0,
            ).to(tl.float32)
            + 128.0
        )
        partial0 += _w8a8_row_contribution(
            qwords,
            x_scale,
            x_sum,
            q_unsigned,
            weight_scale,
            zero_bias,
            group,
            group_mask & (row0 < M),
            row0,
            groups,
            BLOCK_N,
        )
        if ROWS >= 2:
            partial1 += _w8a8_row_contribution_if_active(
                qwords,
                x_scale,
                x_sum,
                q_unsigned,
                weight_scale,
                zero_bias,
                group,
                group_mask,
                row1,
                M,
                groups,
                BLOCK_N,
                DYNAMIC_M,
            )
        if ROWS >= 3:
            partial2 += _w8a8_row_contribution_if_active(
                qwords,
                x_scale,
                x_sum,
                q_unsigned,
                weight_scale,
                zero_bias,
                group,
                group_mask,
                row2,
                M,
                groups,
                BLOCK_N,
                DYNAMIC_M,
            )
        if ROWS >= 4:
            partial3 += _w8a8_row_contribution_if_active(
                qwords,
                x_scale,
                x_sum,
                q_unsigned,
                weight_scale,
                zero_bias,
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
def _int8_w8a8_dp4a_m1_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    zero,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
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
    """M1 W8A8 DP4A variant."""
    _int8_w8a8_dp4a_explicit_kernel(
        qwords,
        x_scale,
        x_sum,
        qdata,
        scale,
        zero,
        out,
        M,
        N,
        K,
        stride_qn,
        stride_qk,
        stride_sn,
        stride_sk,
        stride_zn,
        stride_zk,
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


@triton.jit
def _int8_w8a8_dp4a_m2_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    zero,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
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
    """M2 W8A8 DP4A variant."""
    _int8_w8a8_dp4a_explicit_kernel(
        qwords,
        x_scale,
        x_sum,
        qdata,
        scale,
        zero,
        out,
        M,
        N,
        K,
        stride_qn,
        stride_qk,
        stride_sn,
        stride_sk,
        stride_zn,
        stride_zk,
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


@triton.jit
def _int8_w8a8_dp4a_m3_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    zero,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
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
    """M3 W8A8 DP4A variant."""
    _int8_w8a8_dp4a_explicit_kernel(
        qwords,
        x_scale,
        x_sum,
        qdata,
        scale,
        zero,
        out,
        M,
        N,
        K,
        stride_qn,
        stride_qk,
        stride_sn,
        stride_sk,
        stride_zn,
        stride_zk,
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


@triton.jit
def _int8_w8a8_dp4a_m4_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    zero,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
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
    """M4 W8A8 DP4A variant."""
    _int8_w8a8_dp4a_explicit_kernel(
        qwords,
        x_scale,
        x_sum,
        qdata,
        scale,
        zero,
        out,
        M,
        N,
        K,
        stride_qn,
        stride_qk,
        stride_sn,
        stride_sk,
        stride_zn,
        stride_zk,
        stride_os,
        stride_om,
        stride_on,
        GROUP_SIZE,
        BLOCK_M,
        BLOCK_N,
        SPLIT_K,
        PIPELINE_STAGES,
        DYNAMIC_M,
        4,
    )


@triton.jit
def _int8_w8a8_bucket_kernel(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    zero,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
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
        _int8_w8a8_dp4a_kernel(
            qwords,
            x_scale,
            x_sum,
            qdata,
            scale,
            zero,
            out,
            M,
            N,
            K,
            stride_qn,
            stride_qk,
            stride_sn,
            stride_sk,
            stride_zn,
            stride_zk,
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
        _int8_w8a8_dp4a_exact_rows(
            qwords,
            x_scale,
            x_sum,
            qdata,
            scale,
            zero,
            out,
            M,
            N,
            K,
            stride_qn,
            stride_qk,
            stride_sn,
            stride_sk,
            stride_zn,
            stride_zk,
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
        _int8_w8a8_dp4a_rows(
            qwords,
            x_scale,
            x_sum,
            qdata,
            scale,
            zero,
            out,
            M,
            N,
            K,
            stride_qn,
            stride_qk,
            stride_sn,
            stride_sk,
            stride_zn,
            stride_zk,
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
def _int8_w8a8_dp4a_rows(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    zero,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
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
    if ROWS == 1:
        _int8_w8a8_dp4a_m1_kernel(
            qwords,
            x_scale,
            x_sum,
            qdata,
            scale,
            zero,
            out,
            M,
            N,
            K,
            stride_qn,
            stride_qk,
            stride_sn,
            stride_sk,
            stride_zn,
            stride_zk,
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
    elif ROWS == 2:
        _int8_w8a8_dp4a_m2_kernel(
            qwords,
            x_scale,
            x_sum,
            qdata,
            scale,
            zero,
            out,
            M,
            N,
            K,
            stride_qn,
            stride_qk,
            stride_sn,
            stride_sk,
            stride_zn,
            stride_zk,
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
    elif ROWS == 3:
        _int8_w8a8_dp4a_m3_kernel(
            qwords,
            x_scale,
            x_sum,
            qdata,
            scale,
            zero,
            out,
            M,
            N,
            K,
            stride_qn,
            stride_qk,
            stride_sn,
            stride_sk,
            stride_zn,
            stride_zk,
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
    else:
        _int8_w8a8_dp4a_m4_kernel(
            qwords,
            x_scale,
            x_sum,
            qdata,
            scale,
            zero,
            out,
            M,
            N,
            K,
            stride_qn,
            stride_qk,
            stride_sn,
            stride_sk,
            stride_zn,
            stride_zk,
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


@triton.jit
def _int8_w8a8_dp4a_exact_rows(
    qwords,
    x_scale,
    x_sum,
    qdata,
    scale,
    zero,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_zk: tl.constexpr,
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
    """Dynamic M: dispatch to the explicit implementation with exactly M rows."""
    if M == 1:
        _int8_w8a8_dp4a_m1_kernel(
            qwords,
            x_scale,
            x_sum,
            qdata,
            scale,
            zero,
            out,
            M,
            N,
            K,
            stride_qn,
            stride_qk,
            stride_sn,
            stride_sk,
            stride_zn,
            stride_zk,
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
    elif ROWS >= 2 and M == 2:
        _int8_w8a8_dp4a_m2_kernel(
            qwords,
            x_scale,
            x_sum,
            qdata,
            scale,
            zero,
            out,
            M,
            N,
            K,
            stride_qn,
            stride_qk,
            stride_sn,
            stride_sk,
            stride_zn,
            stride_zk,
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
    elif ROWS >= 3 and M == 3:
        _int8_w8a8_dp4a_m3_kernel(
            qwords,
            x_scale,
            x_sum,
            qdata,
            scale,
            zero,
            out,
            M,
            N,
            K,
            stride_qn,
            stride_qk,
            stride_sn,
            stride_sk,
            stride_zn,
            stride_zk,
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
    else:
        _int8_w8a8_dp4a_rows(
            qwords,
            x_scale,
            x_sum,
            qdata,
            scale,
            zero,
            out,
            M,
            N,
            K,
            stride_qn,
            stride_qk,
            stride_sn,
            stride_sk,
            stride_zn,
            stride_zk,
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


def int8_autotune_configs(bucket: int) -> list[triton.Config]:
    """Generic, bucket-row, and deduplicated next-bucket implementations."""
    rows = sorted({bucket, min(bucket + 1, max(SUPPORTED_BUCKETS))})
    return autotune_configs([{"ROWS": 0}] + [{"ROWS": row} for row in rows])


def _prune(configs, named_args, **kwargs):
    return prune_by_main_loop_trips(_main_loop_trips, configs, named_args, **kwargs)


_BUCKET_KERNELS = {
    bucket: triton.autotune(
        configs=int8_autotune_configs(bucket),
        key=["N", "K", "SPLIT_K", "DYNAMIC_M"],
        prune_configs_by={"early_config_prune": _prune},
    )(_int8_w8a8_bucket_kernel)
    for bucket in SUPPORTED_BUCKETS
}


def _unsupported_reason(
    bucket: int,
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
) -> Optional[str]:
    weights = (qdata, scale, zero)
    if any(tensor.dim() != 2 for tensor in (x, *weights)):
        return "expects rank-2 activation, qdata, scale, and zero tensors"

    M, K = x.shape
    N, qdata_k = qdata.shape
    weight_dims = (*qdata.shape, *scale.shape, *zero.shape)
    weight_shape_reason = (
        None
        if all(isinstance(dim, int) for dim in weight_dims)
        else "weight shapes must be static"
    )
    qdata_offset = qdata.storage_offset()
    alignment_reason = (
        None
        if isinstance(qdata_offset, int) and qdata_offset % 4 == 0
        else "qdata storage offset must be 4-byte aligned"
    )
    if (
        not isinstance(group_size, int)
        or group_size <= 0
        or group_size % _GROUP_SIZE != 0
        or (group_size & (group_size - 1)) != 0
    ):
        group_reason = (
            "group_size must be a positive power-of-two multiple of "
            f"{_GROUP_SIZE}, got {group_size}"
        )
    else:
        group_reason = None

    reason = first_reason(
        check_dtypes(
            (
                ("activation", x, (torch.bfloat16,)),
                ("qdata", qdata, (torch.int8,)),
                ("scale", scale, (torch.bfloat16,)),
                ("zero", zero, (torch.int8,)),
            )
        ),
        group_reason,
        check_k(K, _SUPER_BLOCK),
        weight_shape_reason,
        alignment_reason,
        check_rows(M, bucket),
        check_contiguous((x, *weights)),
        check_device(x, weights),
    )
    if reason is not None:
        return reason
    if qdata_k != K:
        return f"qdata K mismatch: x K={K}, qdata K={qdata_k}"
    if K % group_size != 0:
        return f"group_size={group_size} must divide K={K}"
    metadata_shape = (N, K // group_size)
    if tuple(scale.shape) != metadata_shape or tuple(zero.shape) != metadata_shape:
        return "scale/zero shape does not match [N, K/group_size]"
    return None


def _launch(
    bucket: int,
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
    **launch_params,
) -> torch.Tensor:
    """Runs the bucket kernel. Whether M is dynamic is read here, before
    split-K timing replaces ``x`` with a concrete tensor, so a dynamic-M call
    is timed on the dynamic path and cached apart from a static M."""
    return _launch_rows(
        bucket,
        x,
        qdata,
        scale,
        zero,
        group_size,
        not isinstance(x.shape[0], int),
        **launch_params,
    )


@autotune_launch_param("split_k", SPLIT_K_CANDIDATES)
def _launch_rows(
    bucket: int,
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
    dynamic_m: bool,
    *,
    split_k: int,
) -> torch.Tensor:
    M, K = x.shape
    N = qdata.shape[0]
    check_split_k(split_k, int(K))
    qwords, x_scale, x_sum = quantize_activations_q8(x, natural_order=True)
    block_m = triton.next_power_of_2(bucket)
    return launch_split_k_gemm(
        _BUCKET_KERNELS[bucket],
        bucket=bucket,
        m=M,
        n=N,
        device=x.device,
        split_k=split_k,
        block_m=block_m,
        inputs=(qwords, x_scale, x_sum, qdata, scale, zero),
        shape_args=(
            M,
            N,
            K,
            qdata.stride(0),
            qdata.stride(1),
            scale.stride(0),
            scale.stride(1),
            zero.stride(0),
            zero.stride(1),
        ),
        GROUP_SIZE=group_size,
        BUCKET=bucket,
        BLOCK_M=block_m,
        DYNAMIC_M=dynamic_m,
    )


def _prototype(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    raise NotImplementedError


def _fake(bucket: int, x: torch.Tensor, qdata: torch.Tensor, *args) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], qdata.shape[0]), dtype=torch.bfloat16, device=x.device
    )


INT8_QUANTIZED_GEMM = QuantizedGemmFamily(
    "int8_quantized_gemm",
    (1, 2, 3, 4),
    _prototype,
    _launch,
    _fake,
    _unsupported_reason,
)


__all__ = ["INT8_QUANTIZED_GEMM", "SUPPORTED_BUCKETS", "int8_autotune_configs"]
