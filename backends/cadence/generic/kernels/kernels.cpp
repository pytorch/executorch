/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace impl {
namespace generic {
namespace kernels {

// Quantize a fp32 value to an int8_t/uint8_t value
template <typename T>
T quantize(const float x, float scale, int32_t zero_point) {
  // constexpr float min_val = std::numeric_limits<T>::min();
  // constexpr float max_val = std::numeric_limits<T>::max();
  // float tmp = roundf(x * scale + zero_point);
  // return std::max(std::min(tmp, max_val), min_val);
  // Match Executorch CPU kernel implementation at
  // https://fburl.com/code/fxizw6u6
  int64_t qvalue;
  qvalue = static_cast<int64_t>(zero_point + std::nearbyint(scale * x));

  qvalue = std::max<int64_t>(qvalue, std::numeric_limits<T>::min());
  qvalue = std::min<int64_t>(qvalue, std::numeric_limits<T>::max());
  return static_cast<T>(qvalue);
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

} // namespace kernels
} // namespace generic
} // namespace impl
