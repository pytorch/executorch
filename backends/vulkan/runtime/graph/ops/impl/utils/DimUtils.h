/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

namespace vkcompute {

/*
 * Maps a semantic dimension name to an integer that
 * corresponds to its innermost ordering in a 4D tensor in
 * NCHW format. Width is the innermost dimension, so it
 * corresponds to 1, height is the next innermost, so it
 * corresponds to 2, and so on.
 */
enum Dim4DType : uint32_t {
  DIM4D_WIDTH = 1u,
  DIM4D_HEIGHT = 2u,
  DIM4D_CHANNEL = 3u,
  DIM4D_BATCH = 4u,
};

inline Dim4DType normalize_to_dim4d(const vTensor& v_in, int32_t dim) {
  return static_cast<Dim4DType>(v_in.dim() - dim);
}

/*
 * Semantic dimension names for a 1D tensor
 */
struct Dim1D {
  static constexpr uint32_t Length = 1u;
};

/*
 * Semantic dimension names for a 2D Convolution kernel.
 */
struct DimConv2DKernel {
  static constexpr uint32_t Width = 1u;
  static constexpr uint32_t Height = 2u;
  static constexpr uint32_t InChannels = 3u;
  static constexpr uint32_t OutChannels = 4u;
};

/*
 * The same as the above, except for a 2D Transposed Convolution kernel.
 */
struct DimTConv2DKernel {
  static constexpr uint32_t Width = 1u;
  static constexpr uint32_t Height = 2u;
  static constexpr uint32_t OutChannels = 3u;
  static constexpr uint32_t InChannels = 4u;
};

/*
 * The functions below safely return the size of the dimension at the N-th
 * innermost index. If the dimensionality of the size array is not sufficient
 * then 1 will be returned. The structs above are intended to be used with
 * these functions.
 */
template <uint32_t N>
uint32_t dim_at(const std::vector<int64_t>& sizes) {
  const uint32_t dims = sizes.size();
  return dims < N ? 1 : api::utils::safe_downcast<uint32_t>(sizes[dims - N]);
}

inline uint32_t dim_at(const std::vector<int64_t>& sizes, Dim4DType dim4d) {
  const uint32_t dims = sizes.size();
  return dims < dim4d
      ? 1
      : api::utils::safe_downcast<uint32_t>(sizes[dims - dim4d]);
}

template <uint32_t N>
uint32_t dim_at(const vTensor& v_in) {
  return dim_at<N>(v_in.sizes());
}

inline uint32_t dim_at(const vTensor& v_in, Dim4DType dim4d) {
  return dim_at(v_in.sizes(), dim4d);
}

inline std::ostream& operator<<(std::ostream& os, Dim4DType dim4d) {
  switch (dim4d) {
    case DIM4D_WIDTH:
      os << "DIM4D_WIDTH";
      break;
    case DIM4D_HEIGHT:
      os << "DIM4D_HEIGHT";
      break;
    case DIM4D_CHANNEL:
      os << "DIM4d_CHANNEL";
      break;
    case DIM4D_BATCH:
      os << "DIM4D_BATCH";
      break;
    default:
      os << "DimUnknown";
      break;
  }
  return os;
}
} // namespace vkcompute
