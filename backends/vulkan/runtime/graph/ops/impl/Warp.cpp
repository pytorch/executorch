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
  // out has the same shape as in: [N, C, H, W].
  graph->virtual_resize(out, graph->sizes_of(in));
}

void add_warp_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef flow,
    const ValueRef out) {
  // The shader hard-codes channels-packed texture indexing for in/flow/out and
  // reads `flow` as a 2-channel field (one z-slice per batch). The partitioner
  // gates this via `inputs_storage` in op_registry.py::register_warp, but assert
  // here too to avoid a silent miscompute on a bypass path.
  VK_CHECK_COND(
      graph.is_standard_channels_packed_texture_tensor(in) &&
          graph.is_standard_channels_packed_texture_tensor(flow) &&
          graph.is_standard_channels_packed_texture_tensor(out),
      "Vulkan warp requires in/flow/out to be channels-packed textures");

  VK_CHECK_COND(
      graph.dtype_of(in) == graph.dtype_of(out) &&
          graph.dtype_of(flow) == graph.dtype_of(out),
      "Vulkan warp requires in/flow/out to share dtype");

  // The shader reads t_in with the output's own slice index pos.z, which only
  // lines up when in and out have the same shape. (Its sample coords are
  // separately clamped to inp.sizes, so only the slice index matters here.)
  VK_CHECK_COND(
      graph.sizes_of(in) == graph.sizes_of(out),
      "Vulkan warp requires in and out to share sizes");

  // Unlike the t_in sample, the flow fetch is NOT clamped: the shader iterates
  // the output extent and reads t_flow straight at (pos.x, pos.y), so a flow
  // smaller than out reads past its extent. warp_blend upsamples flow in-shader
  // and so allows a smaller one; warp has no such path and requires a match.
  VK_CHECK_COND(
      graph.sizes_of(flow).at(2) == graph.sizes_of(out).at(2) &&
          graph.sizes_of(flow).at(3) == graph.sizes_of(out).at(3),
      "Vulkan warp requires flow to match the spatial size of in/out");

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
      // Shader params buffers. in and out share spatial size; the shader reads
      // sizes/limits from both metadatas.
      {graph.meta_ubo(out), graph.meta_ubo(in)},
      // Push Constants
      {},
      // Specialization Constants — output hashed layout for packed_dim.
      {graph.hashed_layout_of(out)},
      // Resize Args
      {},
      // Resizing Logic
      resize_warp_node));
}

void warp(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // Argument order matches custom_ops_lib.py::warp: (input, flow, out)
  return add_warp_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.warp.default, warp);
}

} // namespace vkcompute
