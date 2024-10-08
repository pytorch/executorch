/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ostream>

namespace vkcompute {

// Convenience constexpr to attach semantic names to WHCN dimension index
namespace WHCN {

constexpr int32_t kWidthDim = 0;
constexpr int32_t kHeightDim = 1;
constexpr int32_t kChannelsDim = 2;

} // namespace WHCN

namespace utils {

//
// GPU Storage Options
//

/**
 * The enum below is used to describe what type of GPU memory will be used to
 * store a particular tensor's data.
 *
 * BUFFER means that a SSBO (Shader Storage Buffer Object) will be used.
 * TEXTURE_3D means that a 3-dimensional image texture will be used.
 * TEXTURE_2D means that a 2-dimensional image texture will be used.
 *
 * UNKNOWN is not expected to be used.
 */
enum class StorageType : uint8_t {
  BUFFER,
  TEXTURE_3D,
  TEXTURE_2D,
};

static constexpr StorageType kBuffer = StorageType::BUFFER;
static constexpr StorageType kTexture3D = StorageType::TEXTURE_3D;
static constexpr StorageType kTexture2D = StorageType::TEXTURE_2D;

/*
 * A tensor's memory layout is defined in one of two ways:
 *
 * 1. If it's a buffer backed tensor, the memory layout is defined by its
 *    `dim_order`, and by extension its `strides`.
 * 2. If it's a texture backed tensor, the memory layout is defined by the
 *    combination of its `axis_map` and its `packed_dim`.
 *
 * Providing explicit memory layout metadata upon tensor construction is not
 * very convenient from an API perspective, so the `GPUMemoryLayout` serves as
 * an abstraction that is used to determine how to initialize a tensor's layout
 * metadata based on the developer's intent. A `GPUMemoryLayout` is provided to
 * the constructor of `vTensor`, which will use it to determine how to set its
 * `dim_order` if it's a buffer backed tensor, or how to set its `axis_map` and
 * `packed_dim` if it's a texture backed tensor.
 *
 * Note that GPUMemoryLayout is not stored as a tensor property, as it does not
 * have any meaning after the vTensor is constructed. After construction,
 * methods such as `virtual_transpose()` may be used to modify the tensor's
 * layout metadata that cannot be represented by any `GPUMemoryLayout` entry.
 * Nonetheless, a "best guess" of the closest memory layout can be produced via
 * the `estimate_memory_layout()` API of `vTensor`.
 *
 * Currently, only 3 memory layouts are provided, but more will be added in the
 * future that will enable different functionality such as minimizing texture
 * memory footprint.
 */
enum class GPUMemoryLayout : uint8_t {
  /*
   * The below memory layouts will produce a `vTensor` with the following
   * properties:
   *
   * 1. For buffer backed tensors, the `dim_order` will be the same as a
   *    contiguous dim order, but with the specified dim last in the dim order.
   * 2. For texture backed tensors, the packed dim will be the specified dim.
   *    The axis map will be `{0, 1, 2, 2}`.
   */
  TENSOR_WIDTH_PACKED = 0u,
  TENSOR_HEIGHT_PACKED = 1u,
  TENSOR_CHANNELS_PACKED = 2u,
};

static constexpr GPUMemoryLayout kWidthPacked =
    GPUMemoryLayout::TENSOR_WIDTH_PACKED;

static constexpr GPUMemoryLayout kHeightPacked =
    GPUMemoryLayout::TENSOR_HEIGHT_PACKED;

static constexpr GPUMemoryLayout kChannelsPacked =
    GPUMemoryLayout::TENSOR_CHANNELS_PACKED;

template <typename T>
T to_packed_dim(const GPUMemoryLayout layout) {
  switch (layout) {
    case kWidthPacked:
      return 0;
    case kHeightPacked:
      return 1;
    case kChannelsPacked:
      return 2;
  };
  // Should be unreachable
  return 0;
}

inline std::ostream& operator<<(
    std::ostream& os,
    const GPUMemoryLayout layout) {
  switch (layout) {
    case kWidthPacked:
      os << "TENSOR_WIDTH_PACKED";
      break;
    case kHeightPacked:
      os << "TENSOR_HEIGHT_PACKED";
      break;
    case kChannelsPacked:
      os << "TENSOR_CHANNELS_PACKED";
      break;
  }
  return os;
}

} // namespace utils
} // namespace vkcompute
