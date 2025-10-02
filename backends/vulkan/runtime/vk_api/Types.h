/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY bugprone-branch-clone

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <executorch/backends/vulkan/runtime/vk_api/Exception.h>

#include <cstddef>
#include <cstdint>

// X11 headers via volk define Bool, so we need to undef it
#if defined(__linux__)
#undef Bool
#endif

#ifdef USE_VULKAN_FP16_INFERENCE
#define VK_FORMAT_FLOAT4 VK_FORMAT_R16G16B16A16_SFLOAT
#else
#define VK_FORMAT_FLOAT4 VK_FORMAT_R32G32B32A32_SFLOAT
#endif /* USE_VULKAN_FP16_INFERENCE */

#define VK_FORALL_SCALAR_TYPES(_)                  \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, Byte)        \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, Bool)        \
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, Char)         \
  _(uint16_t, VK_FORMAT_R16G16B16A16_SFLOAT, Half) \
  _(uint16_t, VK_FORMAT_R16G16B16A16_UINT, UInt16) \
  _(int16_t, VK_FORMAT_R16G16B16A16_SINT, Short)   \
  _(uint32_t, VK_FORMAT_R32G32B32A32_UINT, UInt)   \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, Int)     \
  _(uint64_t, VK_FORMAT_R64G64B64A64_UINT, UInt64) \
  _(int64_t, VK_FORMAT_R64G64B64A64_SINT, Long)    \
  _(float, VK_FORMAT_FLOAT4, Float)                \
  _(double, VK_FORMAT_R64G64B64A64_SFLOAT, Double) \
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, QInt8)        \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, QUInt8)      \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, QInt32)  \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, Int8x4)

namespace vkcompute {
namespace vkapi {

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
    case VK_FORMAT_R64G64B64A64_SFLOAT:
      return kDouble;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return kFloat;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return kHalf;
    case VK_FORMAT_R8G8B8A8_SINT:
      return kChar;
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_UNORM:
      return kByte;
    case VK_FORMAT_R16G16B16A16_SINT:
      return kShort;
    case VK_FORMAT_R16G16B16A16_UINT:
      return kUInt16;
    case VK_FORMAT_R32G32B32A32_SINT:
      return kInt;
    case VK_FORMAT_R32G32B32A32_UINT:
      return kUInt;
    case VK_FORMAT_R64G64B64A64_SINT:
      return kLong;
    case VK_FORMAT_R64G64B64A64_UINT:
      return kUInt64;
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

#define SPECIALIZE_ScalarTypeToCType(ctype, vkformat, scalar_type)        \
  template <>                                                             \
  struct ScalarTypeToCType<::vkcompute::vkapi::ScalarType::scalar_type> { \
    using type = ctype;                                                   \
  };

VK_FORALL_SCALAR_TYPES(SPECIALIZE_ScalarTypeToCType)

#undef SPECIALIZE_ScalarTypeToCPPType

} // namespace vkapi
} // namespace vkcompute
