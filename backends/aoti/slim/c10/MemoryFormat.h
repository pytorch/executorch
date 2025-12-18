/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/platform/assert.h>

#include <cstdint>
#include <ostream>
#include <vector>

// Memory format is not the property of a Tensor. It is the way to tell an
// operator how the result should be organized in memory and nothing more. That
// means memory format should never be used as return value for any tensor state
// interrogation functions (internally and externally).
//
// Possible options are:
//  Preserve:
//    If any of the input tensors is in channels_last format, operator output
//    should be in channels_last format
//
//  Contiguous:
//    Regardless of input tensors format, the output should be contiguous
//    Tensor.
//
//  ChannelsLast:
//    Regardless of input tensors format, the output should be in channels_last
//    format.

namespace c10 {

using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::IntArrayRef;

enum class MemoryFormat : int8_t {
  Contiguous,
  Preserve,
  ChannelsLast,
  ChannelsLast3d,
  NumOptions
};

// If you are seeing this, it means that this call site was not checked if
// the memory format could be preserved, and it was switched to old default
// behaviour of contiguous
#define LEGACY_CONTIGUOUS_MEMORY_FORMAT ::c10::get_contiguous_memory_format()

inline MemoryFormat get_contiguous_memory_format() {
  return MemoryFormat::Contiguous;
}

inline std::ostream& operator<<(
    std::ostream& stream,
    MemoryFormat memory_format) {
  switch (memory_format) {
    case MemoryFormat::Preserve:
      return stream << "Preserve";
    case MemoryFormat::Contiguous:
      return stream << "Contiguous";
    case MemoryFormat::ChannelsLast:
      return stream << "ChannelsLast";
    case MemoryFormat::ChannelsLast3d:
      return stream << "ChannelsLast3d";
    default:
      ET_CHECK_MSG(
          false,
          "Unknown memory format %d",
          static_cast<int>(memory_format));
  }
}

// Note: Hardcoded the channel last stride indices here to get better
// performance
template <typename T>
inline std::vector<T> get_channels_last_strides_2d(ArrayRef<T> sizes) {
  std::vector<T> strides(sizes.size());
  switch (sizes.size()) {
    case 4:
      strides[1] = 1;
      strides[3] = sizes[1];
      strides[2] = strides[3] * sizes[3];
      strides[0] = strides[2] * sizes[2];
      return strides;
    case 3:
      strides[0] = 1;
      strides[2] = sizes[0];
      strides[1] = strides[2] * sizes[2];
      return strides;
    default:
      ET_CHECK_MSG(
          false,
          "ChannelsLast2d doesn't support size %zu",
          static_cast<size_t>(sizes.size()));
  }
}

inline std::vector<int64_t> get_channels_last_strides_2d(IntArrayRef sizes) {
  return get_channels_last_strides_2d<int64_t>(sizes);
}

template <typename T>
std::vector<T> get_channels_last_strides_3d(ArrayRef<T> sizes) {
  std::vector<T> strides(sizes.size());
  switch (sizes.size()) {
    case 5:
      strides[1] = 1;
      strides[4] = sizes[1];
      strides[3] = strides[4] * sizes[4];
      strides[2] = strides[3] * sizes[3];
      strides[0] = strides[2] * sizes[2];
      return strides;
    case 4:
      strides[0] = 1;
      strides[3] = sizes[0];
      strides[2] = strides[3] * sizes[3];
      strides[1] = strides[2] * sizes[2];
      return strides;
    default:
      ET_CHECK_MSG(
          false,
          "ChannelsLast3d doesn't support size %zu",
          static_cast<size_t>(sizes.size()));
  }
}

inline std::vector<int64_t> get_channels_last_strides_3d(IntArrayRef sizes) {
  return get_channels_last_strides_3d<int64_t>(sizes);
}

// NOTE:
// Below are Helper functions for is_channels_last_strides_xd.
// 1. Please do not combine these helper functions, each helper function handles
// exactly one case of sizes + memory_format, by doing this, the strides indices
// will be a constant array and we can access it using constant index number,
// the compiler will fully unroll the loop on strides indices to gain a better
// performance.
// 2. No error check in helper function, caller ensures the correctness of the
// input
// 3. All helper functions have similar comments, only 1st helper function is
// commented here.
template <typename T>
inline bool is_channels_last_strides_2d_s4(
    const ArrayRef<T> sizes,
    const ArrayRef<T> strides) {
  T min = 0;
  // special case for trivial C dimension. default to NCHW
  if (strides[1] == 0) {
    return false;
  }
  // loop strides indices
  for (auto& d : {1, 3, 2, 0}) {
    if (sizes[d] == 0) {
      return false;
    }
    if (strides[d] < min) {
      return false;
    }
    // Fallback to NCHW as default layout for ambiguous cases
    // This is the flaw of implicit memory_format from strides.
    // N111 tensor with identical strides for size 1 dimension;
    // Two cases could lead us here:
    // a. N111 contiguous Tensor ([N,1,1,1]@[1,1,1,1])
    // b. N11W contiguous Tensor sliced on the W-dimension.
    // ([N,1,1,1]@[W,W,W,W])
    if (d == 0 && min == strides[1]) {
      return false;
    }
    // This is necessary to:
    // 1. distinguish the memory_format of N1H1;
    //     [H, 1, 1, 1] channels_last stride
    //     [H, H, 1, 1] contiguous stride
    // 2. permutation of 1C1W:
    //     [1, C, 1, H]@[HC, H, H, 1] transpose(1, 3)
    //     [1, H, 1, C]@[HC, 1, H, H] shouldn't be identified as channels_last
    min = strides[d];
    if (sizes[d] > 1) {
      min *= sizes[d];
    }
  }
  return true;
}

template <typename T>
inline bool is_channels_last_strides_3d_s5(
    const ArrayRef<T> sizes,
    const ArrayRef<T> strides) {
  T min = 0;
  if (strides[1] == 0) {
    return false;
  }
  for (auto& d : {1, 4, 3, 2, 0}) {
    if (sizes[d] == 0) {
      return false;
    }
    if (strides[d] < min) {
      return false;
    }
    if (d == 0 && min == strides[1]) {
      return false;
    }
    min = strides[d];
    if (sizes[d] > 1) {
      min *= sizes[d];
    }
  }
  return true;
}

template <typename T>
inline bool is_channels_last_strides_2d(
    const ArrayRef<T> sizes,
    const ArrayRef<T> strides) {
  switch (sizes.size()) {
    case 4:
      return is_channels_last_strides_2d_s4(sizes, strides);
      // NOLINTNEXTLINE(bugprone-branch-clone)
    case 3:
      // TODO dim == 3 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

template <typename T>
inline bool is_channels_last_strides_3d(
    const ArrayRef<T> sizes,
    const ArrayRef<T> strides) {
  switch (sizes.size()) {
    case 5:
      return is_channels_last_strides_3d_s5(sizes, strides);
      // NOLINTNEXTLINE(bugprone-branch-clone)
    case 4:
      // TODO dim == 4 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

inline bool is_channels_last_strides_2d(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  return is_channels_last_strides_2d<int64_t>(sizes, strides);
}

inline bool is_channels_last_strides_3d(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  return is_channels_last_strides_3d<int64_t>(sizes, strides);
}

} // namespace c10
