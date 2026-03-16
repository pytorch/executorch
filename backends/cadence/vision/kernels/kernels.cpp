/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/vision/kernels/kernels.h>
#include <math.h>
#include <algorithm>
#include <cstring>
#include <limits>

namespace impl {
namespace vision {
namespace kernels {

void* allocate_temp_memory(KernelRuntimeContext& ctx, size_t size) {
  Result<void*> temp_mem_res = ctx.allocate_temp(size);
  return temp_mem_res.ok() ? temp_mem_res.get() : nullptr;
}

// Quantize a fp32 value to an int8_t/uint8_t value
template <typename T>
T quantize(const float x, float scale, int32_t zero_point) {
  constexpr float kMinValue = static_cast<float>(std::numeric_limits<T>::min());
  constexpr float kMaxValue = static_cast<float>(std::numeric_limits<T>::max());
  float tmp = roundf(x * scale + zero_point);
  return std::max(std::min(tmp, kMaxValue), kMinValue);
}

// Quantize an fp32 array to an int8_t/uint8_t array
template <typename T>
void quantize(
    T* __restrict__ y,
    const float* __restrict__ x,
    float inv_scale,
    int32_t zero_point,
    size_t size) {
  for (size_t i = 0; i < size; ++i) {
    y[i] = quantize<T>(x[i], inv_scale, zero_point);
  }
}

// Dequantize an int8_t/uint8_t value to an fp32 value
template <typename T>
float dequantize(const T x, float scale, int32_t zero_point) {
  return scale * (x - zero_point);
}

// Dequantize an int8_t/uint8_t/int16_t array to an fp32 array
template <typename T>
void dequantize(
    float* __restrict__ y,
    const T* __restrict__ x,
    float scale,
    int32_t zero_point,
    size_t size) {
  for (size_t i = 0; i < size; ++i) {
    y[i] = dequantize<T>(x[i], scale, zero_point);
  }
}

// Requantize the int8_t/uint8_t in value to a uint8_t/int8_t out value.
// The scale and zero_point for requantization are in the args.
template <typename IT, typename OT>
OT requantize(
    const IT in,
    float in_scale,
    int32_t in_zero_point,
    float inv_out_scale,
    int32_t out_zero_point) {
  float dequant = dequantize<IT>(in, in_scale, in_zero_point);
  return quantize<OT>(dequant, inv_out_scale, out_zero_point);
}

// Requantize the int8_t/uint8_t in array to a uint8_t/int8_t out array.
// The scale and zero_point for requantization are in the args.
template <typename IT, typename OT>
void requantize(
    OT* __restrict__ out,
    const IT* __restrict__ in,
    float in_scale,
    int32_t in_zero_point,
    float inv_out_scale,
    int32_t out_zero_point,
    size_t size) {
  for (size_t i = 0; i < size; ++i) {
    out[i] = requantize<IT, OT>(
        in[i], in_scale, in_zero_point, inv_out_scale, out_zero_point);
  }
}

// explicit template instantiation

#define typed_quantize_val(dtype) \
  template dtype quantize(const float x, float inv_scale, int32_t zero_point);
typed_quantize_val(int8_t);
typed_quantize_val(uint8_t);
typed_quantize_val(int16_t);
typed_quantize_val(uint16_t);
typed_quantize_val(int32_t);
#undef typed_quantize_val

#define typed_quantize_vec(dtype)  \
  template void quantize(          \
      dtype* __restrict__ y,       \
      const float* __restrict__ x, \
      float inv_scale,             \
      int32_t zero_point,          \
      size_t size);
typed_quantize_vec(int8_t);
typed_quantize_vec(uint8_t);
typed_quantize_vec(int16_t);
typed_quantize_vec(uint16_t);
typed_quantize_vec(int32_t);
#undef typed_quantize_vec

#define typed_dequantize_val(dtype) \
  template float dequantize(const dtype x, float scale, int32_t zero_point);
typed_dequantize_val(int8_t);
typed_dequantize_val(uint8_t);
typed_dequantize_val(int16_t);
typed_dequantize_val(uint16_t);
typed_dequantize_val(int32_t);
#undef typed_dequantize_val

#define typed_dequantize_vec(dtype) \
  template void dequantize(         \
      float* __restrict__ y,        \
      const dtype* __restrict__ x,  \
      float scale,                  \
      int32_t zero_point,           \
      size_t size);
typed_dequantize_vec(int8_t);
typed_dequantize_vec(uint8_t);
typed_dequantize_vec(int16_t);
typed_dequantize_vec(uint16_t);
typed_dequantize_vec(int32_t);
#undef typed_dequantize_vec

#define typed_requantize_val(itype, otype) \
  template otype requantize(               \
      const itype in,                      \
      float in_scale,                      \
      int32_t in_zero_point,               \
      float inv_out_scale,                 \
      int32_t out_zero_point);
typed_requantize_val(int8_t, int8_t);
typed_requantize_val(int8_t, uint8_t);
typed_requantize_val(int8_t, int16_t);
typed_requantize_val(int8_t, uint16_t);
typed_requantize_val(uint8_t, int8_t);
typed_requantize_val(uint8_t, uint8_t);
typed_requantize_val(uint8_t, int16_t);
typed_requantize_val(uint8_t, uint16_t);
typed_requantize_val(int16_t, int8_t);
typed_requantize_val(int16_t, uint8_t);
typed_requantize_val(int16_t, int16_t);
typed_requantize_val(int16_t, uint16_t);
typed_requantize_val(uint16_t, int8_t);
typed_requantize_val(uint16_t, uint8_t);
typed_requantize_val(uint16_t, int16_t);
typed_requantize_val(uint16_t, uint16_t);
#undef typed_requantize_val

#define typed_requantize_vec(itype, otype) \
  template void requantize(                \
      otype* __restrict__ out,             \
      const itype* __restrict__ in,        \
      float in_scale,                      \
      int32_t in_zero_point,               \
      float inv_out_scale,                 \
      int32_t out_zero_point,              \
      size_t size);
typed_requantize_vec(int8_t, int8_t);
typed_requantize_vec(int8_t, uint8_t);
typed_requantize_vec(int8_t, int16_t);
typed_requantize_vec(int8_t, uint16_t);
typed_requantize_vec(uint8_t, int8_t);
typed_requantize_vec(uint8_t, uint8_t);
typed_requantize_vec(uint8_t, int16_t);
typed_requantize_vec(uint8_t, uint16_t);
typed_requantize_vec(int16_t, int8_t);
typed_requantize_vec(int16_t, uint8_t);
typed_requantize_vec(int16_t, int16_t);
typed_requantize_vec(int16_t, uint16_t);
typed_requantize_vec(uint16_t, int8_t);
typed_requantize_vec(uint16_t, uint8_t);
typed_requantize_vec(uint16_t, int16_t);
typed_requantize_vec(uint16_t, uint16_t);
#undef typed_requantize_vec

}; // namespace kernels
}; // namespace vision
}; // namespace impl
