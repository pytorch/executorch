/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Conv2dIm2Col.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// Push constants are uploaded in 16-byte chunks (one ivec4 each) to comply
// with the per-entry size limit. Layout matches conv2d_im2col.glsl:
//   { ivec4 kernel_stride, ivec4 padding_dil, ivec4 dims }

void add_conv2d_im2col_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef im2col_out,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t padding_h,
    const int32_t padding_w,
    const int32_t dilation_h,
    const int32_t dilation_w,
    const int32_t Cin_padded,
    const int32_t H_out,
    const int32_t W_out) {
  const utils::StorageType out_storage = graph.storage_type_of(im2col_out);
  VK_CHECK_COND(
      out_storage == utils::kBuffer || out_storage == utils::kTexture2D ||
      out_storage == utils::kTexture3D);

  std::string kernel_name = "conv2d_im2col";
  add_storage_type_suffix(kernel_name, out_storage);
  add_dtype_suffix(kernel_name, graph.dtype_of(im2col_out));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  const int32_t M = H_out * W_out;
  // K_total is laid out so that 4-tiles share a kernel position; since
  // Cin_padded is a multiple of 4, K_total is also a multiple of 4.
  const int32_t K_total = kernel_h * kernel_w * Cin_padded;
  const int32_t K4_total = K_total / 4;

  const utils::ivec4 kernel_stride{kernel_h, kernel_w, stride_h, stride_w};
  const utils::ivec4 padding_dil{padding_h, padding_w, dilation_h, dilation_w};
  const utils::ivec4 dims{Cin_padded, W_out, H_out, K4_total};

  // Global wg: one thread per (k4, m) vec4 in the output.
  const utils::uvec3 global_wg_size{
      utils::safe_downcast<uint32_t>(K4_total),
      utils::safe_downcast<uint32_t>(M),
      1u};
  const utils::uvec3 local_wg_size{16u, 4u, 1u};

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      shader,
      global_wg_size,
      local_wg_size,
      // Inputs and Outputs
      {{im2col_out, vkapi::kWrite}, {in, vkapi::kRead}},
      // UBOs
      {graph.sizes_ubo(in)},
      // Push constants (3 × ivec4 = 48 bytes, split per 16-byte limit)
      {PushConstantDataInfo(&kernel_stride, sizeof(kernel_stride)),
       PushConstantDataInfo(&padding_dil, sizeof(padding_dil)),
       PushConstantDataInfo(&dims, sizeof(dims))},
      // Specialization constants
      {},
      // Resize args
      {},
      // Resizing logic
      nullptr));
}

} // namespace vkcompute
