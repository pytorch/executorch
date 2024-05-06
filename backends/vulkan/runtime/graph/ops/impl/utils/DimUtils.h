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

// A canonical way to represent dimensions as enum. Motivation behind a
// canonical enum is that in the user tensor, it is using a "big-endian"-ish
// mechanism to reference a dimension in a nchw-tensor, leading to tensor of
// different dimension have different mapping from dim to the underlying texture
// dimension. For instasnce, for a 2d (height x width) tensors, dim 0 refers to
// height and dim 1 refers to width; for a 4d (batch x channel x height x width)
// tensor, dim 0 refers to batch and dim 1 refers to channel. Using this
// canonical enum allows us to bring clarity in code.

enum NchwDim : uint32_t {
  DimWidth = 1u,
  DimHeight = 2u,
  DimChannel = 3u,
  DimBatch = 4u,
};

// Convert a dim provided by user into canonical enum.
inline NchwDim normalize_to_nchw_dim(const vTensor& v_in, int32_t dim) {
  return static_cast<NchwDim>(v_in.dim() - dim);
}

/*
 * Maps a semantic dimension name to an integer that
 * corresponds to its innermost ordering in a 4D tensor in
 * NCHW format. Width is the innermost dimension, so it
 * corresponds to 1, height is the next innermost, so it
 * corresponds to 2, and so on.
 */
struct Dim4D {
  static constexpr uint32_t Width = DimWidth;
  static constexpr uint32_t Height = DimHeight;
  static constexpr uint32_t Channel = DimChannel;
  static constexpr uint32_t Batch = DimBatch;
};

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

inline uint32_t dim_at(const std::vector<int64_t>& sizes, NchwDim nchw_dim) {
  const uint32_t dims = sizes.size();
  return dims < nchw_dim
      ? 1
      : api::utils::safe_downcast<uint32_t>(sizes[dims - nchw_dim]);
}

template <uint32_t N>
uint32_t dim_at(const vTensor& v_in) {
  return dim_at<N>(v_in.sizes());
}

inline uint32_t dim_at(const vTensor& v_in, NchwDim nchw_dim) {
  return dim_at(v_in.sizes(), nchw_dim);
}

inline std::ostream& operator<<(std::ostream& os, NchwDim nchw_dim) {
  switch (nchw_dim) {
    case DimWidth:
      os << "DimWidth";
      break;
    case DimHeight:
      os << "DimHeight";
      break;
    case DimChannel:
      os << "DimChannel";
      break;
    case DimBatch:
      os << "DimBatch";
      break;
    default:
      os << "DimUnknown";
      break;
  }
  return os;
}

} // namespace vkcompute
