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

namespace vkcompute {

bool is_bitw8(vkapi::ScalarType dtype) {
  return dtype == vkapi::kByte || dtype == vkapi::kChar ||
      dtype == vkapi::kQInt8 || dtype == vkapi::kQUInt8;
}

vkapi::ShaderInfo get_nchw_to_tensor_shader(
    ComputeGraph& graph,
    const ValueRef dst,
    bool int8_buffer_enabled,
    bool push_constant_variant) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);

  const vkapi::ScalarType dst_dtype = graph.dtype_of(dst);
  const utils::StorageType dst_storage_type = graph.storage_type_of(dst);

  if (is_bitw8(dst_dtype) && dst_storage_type != utils::kBuffer &&
      !int8_buffer_enabled) {
    kernel_name = "nchw_to_bitw8_image_nobitw8buffer";
    if (!push_constant_variant) {
      kernel_name += "_no_pc";
    }
    add_storage_type_suffix(kernel_name, dst_storage_type);
    add_dtype_suffix(kernel_name, dst_dtype);
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  if (dst_storage_type == utils::kBuffer) {
    kernel_name = "nchw_to_buffer";
    add_dtype_suffix(kernel_name, dst_dtype);
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  kernel_name = "nchw_to_image";
  if (!push_constant_variant) {
    kernel_name += "_no_pc";
  }
  add_storage_type_suffix(kernel_name, dst_storage_type);
  add_dtype_suffix(kernel_name, dst_dtype);

  return VK_KERNEL_FROM_STR(kernel_name);
}

vkapi::ShaderInfo get_tensor_to_nchw_shader(
    ComputeGraph& graph,
    const ValueRef src,
    bool int8_buffer_enabled,
    bool push_constant_variant) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);

  const vkapi::ScalarType src_dtype = graph.dtype_of(src);
  const utils::StorageType src_storage_type = graph.storage_type_of(src);

  if (is_bitw8(src_dtype) && src_storage_type != utils::kBuffer &&
      !int8_buffer_enabled) {
    kernel_name = "bitw8_image_to_nchw_nobitw8buffer";
    if (!push_constant_variant) {
      kernel_name += "_no_pc";
    }
    add_storage_type_suffix(kernel_name, src_storage_type);
    add_dtype_suffix(kernel_name, src_dtype);
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  if (src_storage_type == utils::kBuffer) {
    kernel_name = "buffer_to_nchw";
    add_dtype_suffix(kernel_name, src_dtype);
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  kernel_name = "image_to_nchw";
  if (!push_constant_variant) {
    kernel_name += "_no_pc";
  }
  add_storage_type_suffix(kernel_name, src_storage_type);
  add_dtype_suffix(kernel_name, src_dtype);

  return VK_KERNEL_FROM_STR(kernel_name);
}

} // namespace vkcompute
