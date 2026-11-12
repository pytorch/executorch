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

void resize_warp_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  graph->virtual_resize(out, graph->sizes_of(in));
}

void add_warp_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef flow,
    const ValueRef out) {
  // The shader hard-codes channels-packed texture indexing for in/flow/out.
  VK_CHECK_COND(
      graph.is_standard_channels_packed_texture_tensor(in) &&
          graph.is_standard_channels_packed_texture_tensor(flow) &&
          graph.is_standard_channels_packed_texture_tensor(out),
      "Vulkan warp requires in/flow/out to be channels-packed textures");

  VK_CHECK_COND(
      graph.dtype_of(in) == graph.dtype_of(out) &&
          graph.dtype_of(flow) == graph.dtype_of(out),
      "Vulkan warp requires in/flow/out to share dtype");

  // The shader reads in at the output's own z-slice.
  const std::vector<int64_t> out_sizes = graph.sizes_of(out);
  VK_CHECK_COND(
      graph.sizes_of(in) == out_sizes,
      "Vulkan warp requires in and out to share sizes");

  // The shader reads flow channels 0-1 at the output pixel from z-slice n for
  // output batch n, unclamped and with no broadcasting.
  const std::vector<int64_t> flow_sizes = graph.sizes_of(flow);
  VK_CHECK_COND(
      flow_sizes.at(0) == out_sizes.at(0) && flow_sizes.at(1) == 2 &&
          flow_sizes.at(2) == out_sizes.at(2) &&
          flow_sizes.at(3) == out_sizes.at(3),
      "Vulkan warp requires flow to be [N, 2, H, W] for in/out [N, C, H, W]");

  std::string kernel_name("warp");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in, flow}, vkapi::kRead}},
      // Shader params buffers
      {graph.meta_ubo(out), graph.meta_ubo(in)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_warp_node));
}

void warp(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // (input, flow, out)
  return add_warp_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.warp.default, warp);
}

} // namespace vkcompute
