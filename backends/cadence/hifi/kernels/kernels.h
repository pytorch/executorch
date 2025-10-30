/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/kernel/kernel_includes.h>
#include <stddef.h>
#include <xa_type_def.h>
/* For NNLIB APIs */
#include "xa_nnlib_kernels_api.h"

using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Result;

/* Potential NNLIB function/APIs */

extern "C" WORD32 xa_nn_broadcast_32_32(
    WORD32* __restrict__ p_out,
    const int* const out_shape,
    WORD32* __restrict__ p_in,
    const int* const in_shape,
    int num_dims);

extern "C" WORD32 xa_nn_concat_32_32(
    WORD32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const WORD32** pp_inps,
    const WORD32* const* pp_inps_shape,
    WORD32 num_out_dims,
    WORD32 num_inp,
    WORD32 num_inp_dims,
    WORD32 axis);

extern "C" WORD32 xa_nn_elm_add_broadcast_4D_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp1,
    const WORD32* const p_inp1_shape,
    const FLOAT32* __restrict__ p_inp2,
    const WORD32* const p_inp2_shape);

extern "C" void
xa_nn_elm_atan2_f32(FLOAT32* z, const FLOAT32* y, const FLOAT32* x, WORD32 N);

extern "C" WORD32 xa_nn_elm_clamp_f32xf32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const FLOAT32* __restrict__ p_inp,
    const FLOAT32* __restrict__ p_min,
    const FLOAT32* __restrict__ p_max,
    WORD32 num_elm);

extern "C" WORD32 xa_nn_elm_clamp_broadcast_4D_f32Xf32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp,
    const WORD32* const p_inp_shape,
    const FLOAT32* __restrict__ p_min,
    const WORD32* const p_min_shape,
    const FLOAT32* __restrict__ p_max,
    const WORD32* const p_max_shape);

extern "C" WORD32 xa_nn_elm_div_broadcast_4D_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp1,
    const WORD32* const p_inp1_shape,
    const FLOAT32* __restrict__ p_inp2,
    const WORD32* const p_inp2_shape);

extern "C" WORD32 xa_nn_elm_div_mode_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const FLOAT32* __restrict__ p_inp1,
    const FLOAT32* __restrict__ p_inp2,
    WORD32 num_elm,
    WORD32 mode);

extern "C" WORD32 xa_nn_elm_div_mode_broadcast_4D_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp1,
    const WORD32* const p_inp1_shape,
    const FLOAT32* __restrict__ p_inp2,
    const WORD32* const p_inp2_shape,
    WORD32 mode);

extern "C" WORD32 xa_nn_elm_greater_lesser_equal_f32xf32_f32(
    WORD8* __restrict__ p_out,
    const FLOAT32* __restrict__ p_inp1,
    const FLOAT32* __restrict__ p_inp2,
    WORD32 num_elm,
    WORD32 kernel_type);

extern "C" WORD32 xa_nn_elm_greater_lesser_equal_broadcast_4D_f32xf32_f32(
    WORD8* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp1,
    const WORD32* const p_inp1_shape,
    const FLOAT32* __restrict__ p_inp2,
    const WORD32* const p_inp2_shape,
    WORD32 kernel_type);

extern "C" WORD32 xa_nn_elm_fmod_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const FLOAT32* __restrict__ p_inp1,
    const FLOAT32* __restrict__ p_inp2,
    WORD32 num_elm);

extern "C" WORD32 xa_nn_elm_fmod_broadcast_4D_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp1,
    const WORD32* const p_inp1_shape,
    const FLOAT32* __restrict__ p_inp2,
    const WORD32* const p_inp2_shape);

extern "C" WORD32 xa_nn_elm_logicalxor_boolxbool_bool(
    WORD8* __restrict__ p_out,
    const WORD8* __restrict__ p_inp1,
    const WORD8* __restrict__ p_inp2,
    WORD32 num_elm);

extern "C" WORD32 xa_nn_elm_maximum_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const FLOAT32* __restrict__ p_inp1,
    const FLOAT32* __restrict__ p_inp2,
    WORD32 num_elm);

extern "C" WORD32 xa_nn_elm_maximum_broadcast_4D_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp1,
    const WORD32* const p_inp1_shape,
    const FLOAT32* __restrict__ p_inp2,
    const WORD32* const p_inp2_shape);

extern "C" WORD32 xa_nn_elm_minimum_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const FLOAT32* __restrict__ p_inp1,
    const FLOAT32* __restrict__ p_inp2,
    WORD32 num_elm);

extern "C" WORD32 xa_nn_elm_minimum_broadcast_4D_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp1,
    const WORD32* const p_inp1_shape,
    const FLOAT32* __restrict__ p_inp2,
    const WORD32* const p_inp2_shape);

extern "C" WORD32 xa_nn_elm_mul_broadcast_4D_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp1,
    const WORD32* const p_inp1_shape,
    const FLOAT32* __restrict__ p_inp2,
    const WORD32* const p_inp2_shape);

extern "C" void xa_nn_elm_pow_f32(
    FLOAT32* __restrict__ z,
    const FLOAT32* __restrict__ x,
    const FLOAT32* __restrict__ y,
    WORD32 N);

extern "C" WORD32 xa_nn_elm_remainder_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const FLOAT32* __restrict__ p_inp1,
    const FLOAT32* __restrict__ p_inp2,
    WORD32 num_elm);

extern "C" WORD32 xa_nn_elm_remainder_broadcast_4D_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp1,
    const WORD32* const p_inp1_shape,
    const FLOAT32* __restrict__ p_inp2,
    const WORD32* const p_inp2_shape);

extern "C" WORD32 xa_nn_elm_where_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const FLOAT32* __restrict__ p_inp1,
    const FLOAT32* __restrict__ p_inp2,
    const unsigned char* __restrict__ p_condition,
    WORD32 num_elm);

extern "C" WORD32 xa_nn_elm_where_broadcast_4D_f32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp1,
    const WORD32* const p_inp1_shape,
    const FLOAT32* __restrict__ p_inp2,
    const WORD32* const p_inp2_shape,
    const unsigned char* __restrict__ p_condition,
    const WORD32* const p_condition_shape);

extern "C" WORD32 xa_nn_reduce_mean_4D_f32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp,
    const WORD32* const p_inp_shape,
    const WORD32* __restrict__ p_axis,
    WORD32 num_out_dims,
    WORD32 num_inp_dims,
    WORD32 num_axis_dims,
    void* __restrict__ p_scratch_in);

extern "C" WORD32 xa_nn_transpose_32_32(
    WORD32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const WORD32* __restrict__ p_inp,
    const WORD32* const p_inp_shape,
    const WORD32* __restrict__ p_permute_vec,
    WORD32 num_out_dims,
    WORD32 num_inp_dims);

namespace impl {
namespace HiFi {
namespace kernels {

void* allocate_temp_memory(KernelRuntimeContext& ctx, size_t size);

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

WORD32 xa_nn_matmul_asym8uxasym8u_asym8u(
    UWORD8* __restrict__ p_out,
    const UWORD8* __restrict__ p_mat1,
    const UWORD8* __restrict__ p_mat2,
    const WORD32* __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_stride,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride,
    WORD32 mat1_zero_bias,
    WORD32 vec1_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias);

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

} // namespace kernels
} // namespace HiFi
} // namespace impl
