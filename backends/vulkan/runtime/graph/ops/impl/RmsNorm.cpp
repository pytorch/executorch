/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_rms_norm_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  graph->virtual_resize(out, graph->sizes_of(in));
}

utils::uvec3 rms_norm_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef in = args.at(1).refs.at(0);
  const auto& sizes = graph->sizes_of(in);
  const int64_t hidden = sizes.back();
  const int64_t numel = graph->numel_of(in);
  const uint32_t num_rows = utils::safe_downcast<uint32_t>(numel / hidden);
  return {1u, num_rows, 1u};
}

utils::uvec3 rms_norm_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;
  return {64u, 1u, 1u};
}

void add_rms_norm_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef eps,
    const ValueRef out) {
  ValueRef arg_weight = prepack_standard_like(graph, weight_data, in);

  float epsilon = graph.extract_scalar<float>(eps);

  const bool is_buffer = graph.is_buffer_storage(in);

  std::string kernel_name("rms_norm");
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  if (!is_buffer) {
    VK_CHECK_COND(check_same_packed_dim(graph, in, out));
    VK_CHECK_COND(
        graph.packed_dim_of(in) == WHCN::kWidthDim,
        "RmsNorm texture path requires width-packed input");
  }

  vkapi::ParamsBindList param_ubos = {graph.meta_ubo(out), graph.meta_ubo(in)};
  vkapi::SpecVarList spec_constants = {graph.hashed_layout_of(in)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      rms_norm_global_wg_size,
      rms_norm_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in, arg_weight}, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      {PushConstantDataInfo(&epsilon, sizeof(epsilon))},
      // Specialization Constants
      spec_constants,
      // Resize Args
      {},
      // Resizing Logic
      resize_rms_norm_node));
}

void rms_norm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // et_vk.rms_norm(input, weight, epsilon) -> output
  return add_rms_norm_node(graph, args[0], args[1], args[2], args[3]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.rms_norm.default, rms_norm);
}

} // namespace vkcompute
