/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY bugprone-branch-clone

#include <cstddef>
#include <cstdint>

#include <executorch/backends/vulkan/runtime/api/vk_api.h>

#include <executorch/backends/vulkan/runtime/api/Exception.h>

#ifdef USE_VULKAN_FP16_INFERENCE
#define VK_FORMAT_FLOAT4 VK_FORMAT_R16G16B16A16_SFLOAT
#else
#define VK_FORMAT_FLOAT4 VK_FORMAT_R32G32B32A32_SFLOAT
#endif /* USE_VULKAN_FP16_INFERENCE */

#define VK_FORALL_SCALAR_TYPES(_)                  \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, Byte)        \
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, Char)         \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, Int)     \
  _(bool, VK_FORMAT_R8G8B8A8_SINT, Bool)           \
  _(uint16_t, VK_FORMAT_R16G16B16A16_SFLOAT, Half) \
  _(float, VK_FORMAT_FLOAT4, Float)                \
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, QInt8)        \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, QUInt8)      \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, QInt32)

namespace vkcompute {
namespace api {

//
// Scalar Types
//

enum class ScalarType : int8_t {
#define DEFINE_ENUM_VAL_(ctype, vkformat, name) name,
  VK_FORALL_SCALAR_TYPES(DEFINE_ENUM_VAL_)
#undef DEFINE_ENUM_VAL_
      Undefined,
  NumOptions
};

#define DEFINE_CONSTANT(ctype, vkformat, name) \
  constexpr ScalarType k##name = ScalarType::name;

VK_FORALL_SCALAR_TYPES(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

/*
 * Given a `ScalarType`, return the corresponding `VkFormat` that should be used
 * for image texture storage. The `ScalarType` to `VkFormat` mapping is dictated
 * by the `VK_FORALL_SCALAR_TYPE` macro in `api/Types.h`
 */
inline VkFormat to_vkformat(const ScalarType t) {
#define CASE_VK_FORMAT(ctype, vkformat, name) \
  case ScalarType::name:                      \
    return vkformat;

  switch (t) {
    VK_FORALL_SCALAR_TYPES(CASE_VK_FORMAT)
    default:
      VK_THROW("Unknown ScalarType: ", t);
  }
#undef CASE_VK_FORMAT
}

/*
 * Given a `VkFormat`, return the `ScalarType` that best represents the data
 * type of invidivual elements in an image texture of the `VkFormat`. Note that
 * this mapping is different from the `to_vkformat()` function, since different
 * `ScalarType`s may use the same `VkFormat`.
 */
inline ScalarType element_scalartype(const VkFormat vkformat) {
  switch (vkformat) {
    case VK_FORMAT_R8G8B8A8_SINT:
      return kChar;
    case VK_FORMAT_R8G8B8A8_UINT:
      return kByte;
    case VK_FORMAT_R32G32B32A32_SINT:
      return kInt;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return kFloat;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return kHalf;
    default:
      VK_THROW("No corresponding scalar type for unknown VkFormat: ", vkformat);
  }
}

/*
 * Given a ScalarType, return `sizeof(ctype)` where ctype is the C type
 * corresponding to the ScalarType. The C type to ScalarType mapping is dictated
 * by the VK_FORALL_SCALAR_TYPE macro in api/Types.h
 */
inline size_t element_size(const ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, vkformat, name) \
  case ScalarType::name:                             \
    return sizeof(ctype);

  switch (t) {
    VK_FORALL_SCALAR_TYPES(CASE_ELEMENTSIZE_CASE)
    default:
      VK_THROW("Unknown ScalarType: ", t);
  }
#undef CASE_ELEMENTSIZE_CASE
}

inline const char* to_string(const ScalarType t) {
#define CASE_TO_STRING(ctype, vkformat, name) \
  case ScalarType::name:                      \
    return #name;

  switch (t) {
    VK_FORALL_SCALAR_TYPES(CASE_TO_STRING)
    default:
      return "UNKNOWN_SCALAR_TYPE";
  }
#undef CASE_TO_STRING
}

inline std::ostream& operator<<(std::ostream& os, const ScalarType dtype) {
  return os << to_string(dtype);
}

//
// Map ScalarTypes to C++ types
//

template <ScalarType N>
struct ScalarTypeToCType;

#define SPECIALIZE_ScalarTypeToCType(ctype, vkformat, scalar_type)      \
  template <>                                                           \
  struct ScalarTypeToCType<::vkcompute::api::ScalarType::scalar_type> { \
    using type = ctype;                                                 \
  };

VK_FORALL_SCALAR_TYPES(SPECIALIZE_ScalarTypeToCType)

#undef SPECIALIZE_ScalarTypeToCPPType

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
 * Given a GPUMemoryLayout, return the index of the dimension that is packed
 * along texels, assuming WHCN dimension order.
 */
template <typename T>
T to_packed_dim_whcn_idx(const GPUMemoryLayout layout) {
  return static_cast<T>(layout);
}

/*
 * Given a GPUMemoryLayout, return an offset that can be used to determine the
 * index of the dimension that is packed along texels, assuming NCHW dimension
 * order. The index of the packed dimension will be ndim - offset.
 */
template <typename T>
T to_packed_dim_nchw_offset(const GPUMemoryLayout layout) {
  return static_cast<T>(layout) + 1;
}

} // namespace api
} // namespace vkcompute
