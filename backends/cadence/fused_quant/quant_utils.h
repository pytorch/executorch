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

// Upper bound on tensor rank for affine block indexing. Reference quant kernels
// operate on small ranks (linear rank 2, conv rank 4); 8 leaves headroom.
static constexpr int kMaxAffineDim = 8;

// Affine quantization params. Scale/zero_point are either a singleton
// (per-tensor) or a full-rank tensor whose shape encodes the affine block
// layout: ``block_size[d] = data.size(d) / scale.size(d)``. This single
// representation covers per-tensor, per-channel, per-group, and blockwise. The
// scale element for a data element at flat index ``i`` is found by decomposing
// ``i`` into per-dim coordinates, mapping each to its block (``coord /
// block_size[d]``), and re-linearizing through the scale strides.
struct QParams {
  const float* scales;
  const int64_t* zero_points;
  int32_t quant_min;
  int32_t quant_max;
  bool per_tensor;
  int64_t ndim;
  int64_t data_strides[kMaxAffineDim];
  int64_t scale_strides[kMaxAffineDim];
  int64_t block_size[kMaxAffineDim];

  float scale_at(int64_t i) const {
    return scales[scale_idx(i)];
  }

  int32_t zero_point_at(int64_t i) const {
    return static_cast<int32_t>(zero_points[scale_idx(i)]);
  }

 private:
  int64_t scale_idx(int64_t i) const {
    if (per_tensor) {
      return 0;
    }
    int64_t idx = 0;
    int64_t rem = i;
    for (int64_t d = 0; d < ndim; ++d) {
      const int64_t coord = rem / data_strides[d];
      rem -= coord * data_strides[d];
      idx += (coord / block_size[d]) * scale_strides[d];
    }
    return idx;
  }
};

inline QParams extract_qparams(
    const std::optional<executorch::aten::Tensor>& scale_tensor,
    const std::optional<executorch::aten::Tensor>& zp_tensor,
    int64_t quant_min,
    int64_t quant_max,
    const executorch::aten::Tensor& data_tensor) {
  const auto& scale = scale_tensor.value();
  const auto& zp = zp_tensor.value();

  QParams qp{};
  qp.scales = scale.const_data_ptr<float>();
  qp.zero_points = zp.const_data_ptr<int64_t>();
  qp.quant_min = static_cast<int32_t>(quant_min);
  qp.quant_max = static_cast<int32_t>(quant_max);

  // A singleton scale broadcasts across the whole tensor (per-tensor); no block
  // layout to derive, and the scale rank need not match the data rank.
  if (scale.numel() == 1) {
    qp.per_tensor = true;
    return qp;
  }

  const int64_t ndim = data_tensor.dim();
  ET_CHECK_MSG(
      scale.dim() == ndim,
      "per-channel/group scale must be full-rank (rank %d) to match data rank %d",
      static_cast<int>(scale.dim()),
      static_cast<int>(ndim));
  ET_CHECK_MSG(
      ndim <= kMaxAffineDim,
      "tensor rank %d exceeds kMaxAffineDim %d",
      static_cast<int>(ndim),
      static_cast<int>(kMaxAffineDim));

  qp.per_tensor = false;
  qp.ndim = ndim;
  int64_t data_stride = 1;
  int64_t scale_stride = 1;
  for (int64_t d = ndim - 1; d >= 0; --d) {
    qp.data_strides[d] = data_stride;
    qp.scale_strides[d] = scale_stride;
    qp.block_size[d] = data_tensor.size(d) / scale.size(d);
    data_stride *= data_tensor.size(d);
    scale_stride *= scale.size(d);
  }
  return qp;
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
