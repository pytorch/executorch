/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_warp_blend_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef img0 = args.at(1).refs.at(0);
  graph->virtual_resize(out, graph->sizes_of(img0));
}

void add_warp_blend_node(
    ComputeGraph& graph,
    const ValueRef img0,
    const ValueRef img1,
    const ValueRef flow,
    const ValueRef mask,
    const ValueRef out) {
  // The shader hard-codes channels-packed texture indexing for every tensor.
  VK_CHECK_COND(
      graph.is_standard_channels_packed_texture_tensor(img0) &&
          graph.is_standard_channels_packed_texture_tensor(img1) &&
          graph.is_standard_channels_packed_texture_tensor(flow) &&
          graph.is_standard_channels_packed_texture_tensor(mask) &&
          graph.is_standard_channels_packed_texture_tensor(out),
      "Vulkan warp_blend requires all tensors to be channels-packed textures");

  // The shader binds all tensors with a single DTYPE selected via
  // dtype_of(out), so a mixed-precision graph would bind a buffer as the wrong
  // type and silently miscompute.
  VK_CHECK_COND(
      graph.dtype_of(img0) == graph.dtype_of(out) &&
          graph.dtype_of(img1) == graph.dtype_of(out) &&
          graph.dtype_of(flow) == graph.dtype_of(out) &&
          graph.dtype_of(mask) == graph.dtype_of(out),
      "Vulkan warp_blend requires all tensors to share dtype");

  // The shader reads flow and mask from z-slice n for output batch n: flow
  // channels 0-3 and mask channel 0, with no broadcasting.
  const std::vector<int64_t> out_sizes = graph.sizes_of(out);
  const std::vector<int64_t> flow_sizes = graph.sizes_of(flow);
  const std::vector<int64_t> mask_sizes = graph.sizes_of(mask);
  VK_CHECK_COND(
      flow_sizes.at(0) == out_sizes.at(0) && flow_sizes.at(1) == 4 &&
          mask_sizes.at(0) == out_sizes.at(0) && mask_sizes.at(1) == 1,
      "Vulkan warp_blend requires flow [N, 4, H, W] and mask [N, 1, H, W]");

  // flow and mask are upsampled in the shader from their own spatial size, so
  // they may be smaller than out, but both are read with flow's metadata.
  VK_CHECK_COND(
      flow_sizes.at(2) == mask_sizes.at(2) &&
          flow_sizes.at(3) == mask_sizes.at(3),
      "Vulkan warp_blend requires flow and mask to share spatial size");

  // Image samples are clamped to the output extent, so a smaller image would
  // be read past its own extent.
  VK_CHECK_COND(
      graph.sizes_of(img0) == out_sizes && graph.sizes_of(img1) == out_sizes,
      "Vulkan warp_blend requires img0, img1 and out to share sizes");

  std::string kernel_name("warp_blend");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{img0, img1, flow, mask}, vkapi::kRead}},
      // Shader params buffers
      {graph.meta_ubo(out), graph.meta_ubo(flow)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_warp_blend_node));
}

void warp_blend(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // (img0, img1, flow, mask, out)
  return add_warp_blend_node(
      graph, args[0], args[1], args[2], args[3], args[4]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.warp_blend.default, warp_blend);
}

} // namespace vkcompute
