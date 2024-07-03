/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-security-vulnerable-memcpy

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>

#include <cstring>

namespace vkcompute {

template <typename T>
void memcpy_to_mapping_impl(
    const void* src,
    vkapi::MemoryMap& dst_mapping,
    const size_t nbytes) {
  T* data_ptr = dst_mapping.template data<T>();
  memcpy(data_ptr, reinterpret_cast<const T*>(src), nbytes);
}

template <typename T>
void memcpy_from_mapping_impl(
    vkapi::MemoryMap& src_mapping,
    void* dst,
    const size_t nbytes) {
  T* data_ptr = src_mapping.template data<T>();
  memcpy(reinterpret_cast<T*>(dst), data_ptr, nbytes);
}

void memcpy_to_mapping(
    const void* src,
    vkapi::MemoryMap& dst_mapping,
    const size_t nbytes,
    const vkapi::ScalarType dtype) {
#define DTYPE_CASE(ctype, vkformat, name)                    \
  case vkapi::ScalarType::name:                              \
    memcpy_to_mapping_impl<ctype>(src, dst_mapping, nbytes); \
    break;

  switch (dtype) {
    VK_FORALL_SCALAR_TYPES(DTYPE_CASE)
    default:
      VK_THROW("Unrecognized dtype!");
  }
#undef DTYPE_CASE
}

void memcpy_from_mapping(
    vkapi::MemoryMap& src_mapping,
    void* dst,
    const size_t nbytes,
    const vkapi::ScalarType dtype) {
#define DTYPE_CASE(ctype, vkformat, name)                      \
  case vkapi::ScalarType::name:                                \
    memcpy_from_mapping_impl<ctype>(src_mapping, dst, nbytes); \
    break;

  switch (dtype) {
    VK_FORALL_SCALAR_TYPES(DTYPE_CASE)
    default:
      VK_THROW("Unrecognized dtype!");
  }
#undef DTYPE_CASE
}

void copy_ptr_to_staging(
    const void* src,
    api::StorageBuffer& staging,
    const size_t nbytes) {
  vkapi::MemoryMap mapping(staging.buffer(), vkapi::MemoryAccessType::WRITE);
  mapping.invalidate();
  memcpy_to_mapping(src, mapping, nbytes, staging.dtype());
}

void copy_staging_to_ptr(
    api::StorageBuffer& staging,
    void* dst,
    const size_t nbytes) {
  vkapi::MemoryMap mapping(staging.buffer(), vkapi::MemoryAccessType::READ);
  mapping.invalidate();
  memcpy_from_mapping(mapping, dst, nbytes, staging.dtype());
}

void set_staging_zeros(api::StorageBuffer& staging, const size_t nbytes) {
  vkapi::MemoryMap mapping(staging.buffer(), vkapi::MemoryAccessType::WRITE);
  uint8_t* data_ptr = mapping.template data<uint8_t>();
  memset(data_ptr, 0, staging.nbytes());
}

vkapi::ShaderInfo get_nchw_to_tensor_shader(const api::vTensor& v_dst) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);

  kernel_name = "nchw_to_tensor";
  add_dtype_suffix(kernel_name, v_dst);
  add_storage_type_suffix(kernel_name, v_dst);

  return VK_KERNEL_FROM_STR(kernel_name);
}

vkapi::ShaderInfo get_tensor_to_nchw_shader(const api::vTensor& v_src) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);

  kernel_name = "tensor_to_nchw";
  add_dtype_suffix(kernel_name, v_src);
  add_storage_type_suffix(kernel_name, v_src);

  return VK_KERNEL_FROM_STR(kernel_name);
}

} // namespace vkcompute
