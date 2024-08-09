/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "inttypes.h"
#include "stddef.h"
#include "xa_type_def.h"

/* For NNLIB APIs */
#include "xa_nnlib_kernels_api.h"

namespace impl {
namespace HiFi {
namespace kernels {

void memcpy(void* dst, const void* src, size_t num_bytes);

WORD32 matmul_asym8uxasym8u_asym8u(
    UWORD8* __restrict__ p_out, // output uint8 matrix
    const UWORD8* __restrict__ p_mat1, // weight uint8 matrix
    const UWORD8* __restrict__ p_vec1, // input uint8 matrix
    const WORD32* __restrict__ p_bias, // bias int32 vec
    WORD32 rows, // rows of p_mat1
    WORD32 cols1, // columns of p_mat1
    WORD32 row_stride1, // row stride of p_mat1
    WORD32 vec_count, // rows of p_mat2
    WORD32 vec_offset, // vec_offset of p_mat2.
    WORD32 out_offset, // out_offset, i.e., offset of next output element
    WORD32 out_stride, // out_stride, i.e., stride to go to next output row
    WORD32 mat1_zero_bias, // zero_point of p_mat1
    WORD32 vec1_zero_bias, // zero_point of p_vec1
    const WORD32* __restrict__ out_multiplier,
    const WORD32* __restrict__ out_shift,
    WORD32 out_zero_bias,
    bool per_channel_quantized = false); // per-channel quantized weight

template <typename T>
T quantize(const float x, float scale, int32_t zero_point);

template <typename T>
float dequantize(const T x, float scale, int32_t zero_point);

template <typename T>
void quantize(
    T* __restrict__ y,
    const float* __restrict__ x,
    float scale,
    int32_t zero_point,
    size_t size);

// Deuantize an int8_t/uint8_t/int16_t array to an fp32 array
template <typename T>
void dequantize(
    float* __restrict__ y,
    const T* __restrict__ x,
    float scale,
    int32_t zero_point,
    size_t size);

}; // namespace kernels
}; // namespace HiFi
}; // namespace impl
