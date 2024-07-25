/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace vkcompute {
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
 * The enum below is used to describe how tensor data is laid out when stored in
 * GPU memory; specifically, it indicates how tensor data is packed along a
 * texel (i.e. a vector of 4 scalar values).
 *
 * Each enum entry indicates which tensor dimension is packed along a texel, and
 * it's value is set to the index of that dimension in WHCN dimension order. For
 * instance, the width dimension corresponds to index 0, so the
 * TENSOR_WIDTH_PACKED enum entry is set to 0.
 *
 * When interpreted as an integer, the enum value can be used as a dim index
 * representing the packed dimension. This is used in shaders to resolve tensor
 * indexing calculations.
 */
enum class GPUMemoryLayout : uint8_t {
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

/*
 * Given a GPUMemoryLayout, return an offset that can be used to determine the
 * index of the dimension that is packed along texels, assuming NCHW dimension
 * order. The index of the packed dimension will be ndim - offset.
 */
template <typename T>
T to_packed_dim_nchw_offset(const GPUMemoryLayout layout) {
  return static_cast<T>(layout) + 1;
}

} // namespace utils
} // namespace vkcompute
