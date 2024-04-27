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
struct Dim4D {
  static constexpr uint32_t Width = 1u;
  static constexpr uint32_t Height = 2u;
  static constexpr uint32_t Channel = 3u;
  static constexpr uint32_t Batch = 4u;
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

template <uint32_t N>
uint32_t dim_at(const vTensor& v_in) {
  return dim_at<N>(v_in.sizes());
}

// A canonical way to represent dimensions as enum. Intended to use the same
// value as Dim4D for potential future refactoring.

enum NchwDim {
  DimWidth = 1,
  DimHeight = 2,
  DimChannel = 3,
  DimBatch = 4,
};

/* This function return a NchwDim
 * given a Tensor and a user provided dim. The reason for this normalization is
 * that in the user tensor coordinate, it is using a "big-endian" mechanism when
 * referring to a nchw dimension, in that dim=0 refers to the batch dimension in
 * a 4d tensor but dim=0 reference to height in a 2d tensor. Despite in a common
 * texture representation of channel packing, a 2d tensor has exactly the same
 * layout as a 4d with the batch and channel size equals to 1. This function
 * returns a canonical dimension to simplify dimension reasoning in the code.
 *
 */

inline NchwDim normalize_to_nchw_dim(const vTensor& v_in, int32_t dim) {
  return static_cast<NchwDim>(v_in.dim() - dim);
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
