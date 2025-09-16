/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inttypes.h"
#include "stddef.h"

namespace impl {
namespace generic {
namespace kernels {

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

template <typename IT, typename OT>
OT requantize(
    const IT in,
    float in_scale,
    int32_t in_zero_point,
    float inv_out_scale,
    int32_t out_zero_point);

template <typename IT, typename OT>
void requantize(
    OT* __restrict__ out,
    const IT* __restrict__ in,
    float in_scale,
    int32_t in_zero_point,
    float inv_out_scale,
    int32_t out_zero_point,
    size_t size);

} // namespace kernels
} // namespace generic
} // namespace impl
