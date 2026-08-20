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

void resize_pixel_shuffle_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const ValueRef upscale_factor_ref = resize_args.at(0);

  const int64_t r = graph->extract_scalar<int64_t>(upscale_factor_ref);

  std::vector<int64_t> in_sizes = graph->sizes_of(in);
  const int64_t ndim = static_cast<int64_t>(in_sizes.size());
  VK_CHECK_COND(ndim >= 3);

  std::vector<int64_t> out_sizes = in_sizes;
  out_sizes.at(ndim - 3) = in_sizes.at(ndim - 3) / (r * r);
  out_sizes.at(ndim - 2) = in_sizes.at(ndim - 2) * r;
  out_sizes.at(ndim - 1) = in_sizes.at(ndim - 1) * r;

  graph->virtual_resize(out, out_sizes);
}

void add_pixel_shuffle_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef upscale_factor_ref,
    const ValueRef out) {
  const int64_t r = graph.extract_scalar<int64_t>(upscale_factor_ref);
  VK_CHECK_COND(r >= 1);

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const int64_t ndim = static_cast<int64_t>(in_sizes.size());
  VK_CHECK_COND(ndim >= 3);
  VK_CHECK_COND(in_sizes.at(ndim - 3) % (r * r) == 0);

  std::string kernel_name = "pixel_shuffle";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList ubos = {graph.meta_ubo(out), graph.meta_ubo(in)};

  vkapi::SpecVarList spec_constants = {
      graph.hashed_layout_of(out),
      graph.hashed_layout_of(in),
      static_cast<int32_t>(r)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      ubos,
      // Push Constants
      {},
      // Specialization Constants
      spec_constants,
      // Resize Args
      {upscale_factor_ref},
      // Resizing Logic
      resize_pixel_shuffle_node));
}

void pixel_shuffle(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef in = args[0];
  const ValueRef upscale_factor_ref = args[1];
  const ValueRef out = args[2];
  add_pixel_shuffle_node(graph, in, upscale_factor_ref, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.pixel_shuffle.default, pixel_shuffle);
}

} // namespace vkcompute
