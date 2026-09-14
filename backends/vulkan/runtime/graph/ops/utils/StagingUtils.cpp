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
    const vkapi::ScalarType staging_dtype,
    bool int8_buffer_enabled) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);

  const vkapi::ScalarType dst_dtype = graph.dtype_of(dst);
  const utils::StorageType dst_storage_type = graph.storage_type_of(dst);

  if (is_bitw8(dst_dtype) && dst_storage_type != utils::kBuffer &&
      !int8_buffer_enabled) {
    kernel_name = "nchw_to_bitw8_image_nobitw8buffer";
    add_storage_type_suffix(kernel_name, dst_storage_type);
    add_dtype_suffix(kernel_name, dst_dtype);
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  if (dst_storage_type == utils::kBuffer) {
    kernel_name = "nchw_to_buffer";
    add_dtype_suffix(kernel_name, dst_dtype);
    add_dtype_suffix(kernel_name, staging_dtype);
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  kernel_name = "nchw_to_image";
  add_storage_type_suffix(kernel_name, dst_storage_type);
  add_dtype_suffix(kernel_name, dst_dtype);
  add_dtype_suffix(kernel_name, staging_dtype);

  return VK_KERNEL_FROM_STR(kernel_name);
}

vkapi::ShaderInfo get_tensor_to_nchw_shader(
    ComputeGraph& graph,
    const ValueRef src,
    const vkapi::ScalarType staging_dtype,
    bool int8_buffer_enabled) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);

  const vkapi::ScalarType src_dtype = graph.dtype_of(src);
  const utils::StorageType src_storage_type = graph.storage_type_of(src);

  if (is_bitw8(src_dtype) && src_storage_type != utils::kBuffer &&
      !int8_buffer_enabled) {
    kernel_name = "bitw8_image_to_nchw_nobitw8buffer";
    add_storage_type_suffix(kernel_name, src_storage_type);
    add_dtype_suffix(kernel_name, src_dtype);
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  if (src_storage_type == utils::kBuffer) {
    kernel_name = "buffer_to_nchw";
    add_dtype_suffix(kernel_name, src_dtype);
    add_dtype_suffix(kernel_name, staging_dtype);
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  // On a discrete GPU the staging buffer is read back over PCIe, so the
  // output-centric (coalesced-write) variant is a large win. On integrated GPUs
  // the staging buffer is not PCIe-backed, and the coalesced variant's
  // redundant texture fetches make it a net loss -- so default to the
  // texel-centric variant there.
  //
  // Gate on device type, not has_unified_memory(): a discrete GPU with
  // Resizable BAR exposes a DEVICE_LOCAL | HOST_VISIBLE memory type, so
  // has_unified_memory() returns true for it and would wrongly route it to the
  // texel-centric path (measured ~10x end-to-end regression on an RTX 4080).
  const bool coalesced_writes =
      !graph.context()->adapter_ptr()->is_integrated_gpu();
  kernel_name = coalesced_writes ? "image_to_nchw_coalesced" : "image_to_nchw";
  add_storage_type_suffix(kernel_name, src_storage_type);
  add_dtype_suffix(kernel_name, src_dtype);
  add_dtype_suffix(kernel_name, staging_dtype);

  return VK_KERNEL_FROM_STR(kernel_name);
}

} // namespace vkcompute
