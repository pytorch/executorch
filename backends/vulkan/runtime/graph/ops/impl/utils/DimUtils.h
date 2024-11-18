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
 * Maps a semantic dimension name to an integer that corresponds to its
 * innermost ordering in a 4D tensor in NCHW format. In a way, it is the
 * "negative index" associated with a dim. For instance: in a NCHW tensor, Width
 * is the innermost dimension, so it corresponds to 1, height is the next
 * innermost, so it corresponds to 2, and so on.
 */
enum DimIndex : int32_t {
  DIM_LAST = -1,
  DIM_2ND_LAST = -2,
  DIM_3RD_LAST = -3,
  DIM_4TH_LAST = -4,
};

constexpr DimIndex kWidth4D = DimIndex::DIM_LAST;
constexpr DimIndex kHeight4D = DimIndex::DIM_2ND_LAST;
constexpr DimIndex kChannel4D = DimIndex::DIM_3RD_LAST;
constexpr DimIndex kBatch4D = DimIndex::DIM_4TH_LAST;

inline DimIndex normalize_to_dim_index(const api::vTensor& v_in, int32_t dim) {
  return dim < 0 ? static_cast<DimIndex>(dim)
                 : static_cast<DimIndex>(dim - v_in.dim());
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

inline int32_t dim_at(const std::vector<int64_t>& sizes, DimIndex dim_index) {
  const uint32_t dims = sizes.size();
  // Recall that dim_index is a negative index.
  return dims < -dim_index
      ? 1
      : utils::safe_downcast<int32_t>(sizes[dims + dim_index]);
}

template <DimIndex DI>
int32_t dim_at(const std::vector<int64_t>& sizes) {
  return dim_at(sizes, DI);
}

template <DimIndex DI>
int32_t dim_at(const api::vTensor& v_in) {
  return dim_at(v_in.sizes(), DI);
}

inline int32_t dim_at(const api::vTensor& v_in, DimIndex dim_index) {
  return dim_at(v_in.sizes(), dim_index);
}

inline std::ostream& operator<<(std::ostream& os, DimIndex dim_index) {
  switch (dim_index) {
    case kWidth4D:
      os << "kWidth4D";
      break;
    case kHeight4D:
      os << "kHeight4D";
      break;
    case kChannel4D:
      os << "kChannel4D";
      break;
    case kBatch4D:
      os << "kBatch4D";
      break;
    default:
      os << "kDim4DUnknown";
      break;
  }
  return os;
}
} // namespace vkcompute
