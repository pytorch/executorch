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

extern "C" void
xa_nn_elm_atan2_f32(FLOAT32* z, const FLOAT32* y, const FLOAT32* x, WORD32 N);

extern "C" WORD32 xa_nn_elm_clamp_broadcast_4D_f32Xf32xf32_f32(
    FLOAT32* __restrict__ p_out,
    const WORD32* const p_out_shape,
    const FLOAT32* __restrict__ p_inp,
    const WORD32* const p_inp_shape,
    const FLOAT32* __restrict__ p_min,
    const WORD32* const p_min_shape,
    const FLOAT32* __restrict__ p_max,
    const WORD32* const p_max_shape);

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

extern "C" WORD32 xa_nn_im2row_quantized(
    const WORD8* __restrict__ data_im,
    const WORD32 in_zero_point,
    /* input parameters*/
    const WORD32 channels,
    const WORD32 height,
    const WORD32 width,
    /* output parameters */
    const WORD32 out_height,
    const WORD32 out_width,
    /* convolution parameters */
    const WORD32 kernel_h,
    const WORD32 kernel_w,
    const WORD32 pad_h,
    const WORD32 pad_w,
    const WORD32 stride_h,
    const WORD32 stride_w,
    const WORD32 dilation_h,
    const WORD32 dilation_w,
    WORD8* __restrict__ data_col,
    WORD32 channels_last);

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

namespace impl {
namespace HiFi {
namespace kernels {

void* allocate_temp_memory(KernelRuntimeContext& ctx, size_t size);

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
