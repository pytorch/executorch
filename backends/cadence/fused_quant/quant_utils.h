/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/assert.h>

namespace cadence {
namespace fused_quant {

struct QParams {
  const float* scales;
  const int64_t* zero_points;
  int32_t quant_min;
  int32_t quant_max;
  int64_t num_channels;
  int64_t axis_stride;

  float scale_at(int64_t i) const {
    return scales[channel_idx(i)];
  }

  int32_t zero_point_at(int64_t i) const {
    return static_cast<int32_t>(zero_points[channel_idx(i)]);
  }

 private:
  int64_t channel_idx(int64_t i) const {
    if (num_channels == 1) {
      return 0;
    }
    return (i / axis_stride) % num_channels;
  }
};

inline QParams extract_qparams(
    const executorch::aten::optional<executorch::aten::Tensor>& scale_tensor,
    const executorch::aten::optional<executorch::aten::Tensor>& zp_tensor,
    int64_t quant_min,
    int64_t quant_max,
    executorch::aten::optional<int64_t> axis,
    const executorch::aten::Tensor& data_tensor) {
  const auto& scale = scale_tensor.value();
  const auto& zp = zp_tensor.value();

  int64_t num_channels = scale.numel();
  int64_t axis_stride = 1;
  if (axis.has_value()) {
    for (int64_t d = axis.value() + 1; d < data_tensor.dim(); ++d) {
      axis_stride *= data_tensor.size(d);
    }
  }

  return {
      scale.const_data_ptr<float>(),
      zp.const_data_ptr<int64_t>(),
      static_cast<int32_t>(quant_min),
      static_cast<int32_t>(quant_max),
      num_channels,
      axis_stride,
  };
}

template <typename T>
inline float dequantize(T val, float scale, int32_t zero_point) {
  return (static_cast<float>(val) - static_cast<float>(zero_point)) * scale;
}

template <typename T>
inline T quantize(
    float val,
    float scale,
    int32_t zero_point,
    int32_t qmin,
    int32_t qmax) {
  int32_t quantized =
      static_cast<int32_t>(std::round(val / scale)) + zero_point;
  quantized = std::max(qmin, std::min(qmax, quantized));
  return static_cast<T>(quantized);
}

template <typename T>
inline void
dequantize_buffer(const T* src, float* dst, int64_t numel, const QParams& qp) {
  for (int64_t i = 0; i < numel; ++i) {
    dst[i] = dequantize(src[i], qp.scale_at(i), qp.zero_point_at(i));
  }
}

template <typename T>
inline void
quantize_buffer(const float* src, T* dst, int64_t numel, const QParams& qp) {
  for (int64_t i = 0; i < numel; ++i) {
    dst[i] = quantize<T>(
        src[i],
        qp.scale_at(i),
        qp.zero_point_at(i),
        qp.quant_min,
        qp.quant_max);
  }
}

// Dispatch on ScalarType, binding `scalar_t` for the body.
#define FUSED_QUANT_DTYPE_SWITCH(scalar_type, scalar_t, ...)                   \
  switch (scalar_type) {                                                       \
    case executorch::aten::ScalarType::Byte: {                                 \
      using scalar_t = uint8_t;                                                \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case executorch::aten::ScalarType::Char: {                                 \
      using scalar_t = int8_t;                                                 \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case executorch::aten::ScalarType::Short: {                                \
      using scalar_t = int16_t;                                                \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case executorch::aten::ScalarType::Int: {                                  \
      using scalar_t = int32_t;                                                \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      ET_CHECK_MSG(                                                            \
          false, "Unsupported dtype: %hhd", static_cast<int8_t>(scalar_type)); \
  }

} // namespace fused_quant
} // namespace cadence
