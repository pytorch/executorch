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
    api::MemoryMap& dst_mapping,
    const size_t nbytes) {
  T* data_ptr = dst_mapping.template data<T>();
  memcpy(data_ptr, reinterpret_cast<const T*>(src), nbytes);
}

template <typename T>
void memcpy_from_mapping_impl(
    api::MemoryMap& src_mapping,
    void* dst,
    const size_t nbytes) {
  T* data_ptr = src_mapping.template data<T>();
  memcpy(reinterpret_cast<T*>(dst), data_ptr, nbytes);
}

void memcpy_to_mapping(
    const void* src,
    api::MemoryMap& dst_mapping,
    const size_t nbytes,
    const api::ScalarType dtype) {
#define DTYPE_CASE(ctype, vkformat, name)                    \
  case api::ScalarType::name:                                \
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
    api::MemoryMap& src_mapping,
    void* dst,
    const size_t nbytes,
    const api::ScalarType dtype) {
#define DTYPE_CASE(ctype, vkformat, name)                      \
  case api::ScalarType::name:                                  \
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
  api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
  mapping.invalidate();
  memcpy_to_mapping(src, mapping, nbytes, staging.dtype());
}

void copy_staging_to_ptr(
    api::StorageBuffer& staging,
    void* dst,
    const size_t nbytes) {
  api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
  mapping.invalidate();
  memcpy_from_mapping(mapping, dst, nbytes, staging.dtype());
}

api::ShaderInfo get_nchw_to_image_shader(const vTensor& v_dst) {
  if (v_dst.is_quantized()) {
    VK_THROW("Quantized Tensors are currently not supported!");
  }

  std::stringstream kernel_name;

  switch (v_dst.storage_type()) {
    case api::kTexture3D:
      kernel_name << "nchw_to_image3d";
      break;
    case api::kTexture2D:
      kernel_name << "nchw_to_image2d";
      break;
    default:
      VK_THROW("No kernel available!");
  }

  apply_memory_layout_suffix(kernel_name, v_dst);
  apply_dtype_suffix(kernel_name, v_dst);

  return VK_KERNEL_FROM_STR(kernel_name.str());
}

api::ShaderInfo get_image_to_nchw_shader(const vTensor& v_src) {
  if (v_src.is_quantized()) {
    VK_THROW("Quantized Tensors are currently not supported!");
  }

  std::stringstream kernel_name;

  switch (v_src.storage_type()) {
    case api::kTexture3D:
      kernel_name << "image3d_to_nchw";
      break;
    case api::kTexture2D:
      kernel_name << "image2d_to_nchw";
      break;
    default:
      VK_THROW("No kernel available!");
  }

  apply_memory_layout_suffix(kernel_name, v_src);
  apply_dtype_suffix(kernel_name, v_src);

  return VK_KERNEL_FROM_STR(kernel_name.str());
}

} // namespace vkcompute
