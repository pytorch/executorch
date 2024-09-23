/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <math.h>
#include <algorithm>
#include <cstring>
#include <numeric>

namespace impl {
namespace reference {
namespace kernels {

// Quantize a fp32 value to an int8_t/uint8_t value
template <typename T>
__attribute__((always_inline)) T
quantize(const float x, float scale, int32_t zero_point) {
  constexpr float min_val = std::numeric_limits<T>::min();
  constexpr float max_val = std::numeric_limits<T>::max();
  float tmp = roundf(x * scale + zero_point);
  return std::max(std::min(tmp, max_val), min_val);
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
__attribute__((always_inline)) float
dequantize(const T x, float scale, int32_t zero_point) {
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

// explicit template instantiation

#define typed_quantize_val(dtype)                         \
  template __attribute__((always_inline)) dtype quantize( \
      const float x, float inv_scale, int32_t zero_point);
typed_quantize_val(int8_t);
typed_quantize_val(uint8_t);
typed_quantize_val(int16_t);
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
typed_quantize_vec(int32_t);
#undef typed_quantize_vec

#define typed_dequantize_val(dtype)                         \
  template __attribute__((always_inline)) float dequantize( \
      const dtype x, float scale, int32_t zero_point);
typed_dequantize_val(int8_t);
typed_dequantize_val(uint8_t);
typed_dequantize_val(int16_t);
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
typed_dequantize_vec(int32_t);
#undef typed_dequantize_vec

}; // namespace kernels
}; // namespace reference
}; // namespace impl
