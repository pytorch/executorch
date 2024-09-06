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

vkapi::ShaderInfo get_nchw_to_tensor_shader(
    const api::vTensor& v_dst,
    const bool int8_buffer_enabled) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);

  if (v_dst.dtype() == vkapi::kChar &&
      v_dst.storage_type() == utils::kTexture3D && !int8_buffer_enabled) {
    return VK_KERNEL(nchw_to_int8_image_noint8);
  }

  if (v_dst.storage_type() == utils::kBuffer) {
    kernel_name = "nchw_to_buffer";
    add_dtype_suffix(kernel_name, v_dst);
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  kernel_name = "nchw_to_image";
  add_dtype_suffix(kernel_name, v_dst);
  add_storage_type_suffix(kernel_name, v_dst);

  return VK_KERNEL_FROM_STR(kernel_name);
}

vkapi::ShaderInfo get_tensor_to_nchw_shader(
    const api::vTensor& v_src,
    bool int8_buffer_enabled) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);

  if (v_src.dtype() == vkapi::kChar &&
      v_src.storage_type() == utils::kTexture3D && !int8_buffer_enabled) {
    return VK_KERNEL(int8_image_to_nchw_noint8);
  }

  if (v_src.storage_type() == utils::kBuffer) {
    kernel_name = "buffer_to_nchw";
    add_dtype_suffix(kernel_name, v_src);
    return VK_KERNEL_FROM_STR(kernel_name);
  }

  kernel_name = "image_to_nchw";
  add_dtype_suffix(kernel_name, v_src);
  add_storage_type_suffix(kernel_name, v_src);

  return VK_KERNEL_FROM_STR(kernel_name);
}

} // namespace vkcompute
