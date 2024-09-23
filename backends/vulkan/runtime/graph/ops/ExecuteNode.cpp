/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/BindingUtils.h>

namespace vkcompute {

ExecuteNode::ExecuteNode(
    ComputeGraph& graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const utils::uvec3& local_workgroup_size,
    const std::vector<ArgGroup>& args,
    const vkapi::ParamsBindList& params,
    const vkapi::SpecVarList& spec_vars,
    const ResizeFunction& resize_fn,
    const std::vector<ValueRef>& resize_args)
    : shader_(shader),
      global_workgroup_size_(global_workgroup_size),
      local_workgroup_size_(local_workgroup_size),
      args_(args),
      params_(params),
      spec_vars_(spec_vars),
      resize_fn_(resize_fn),
      resize_args_(resize_args) {
  graph.update_descriptor_counts(shader, /*execute = */ true);
}

ExecuteNode::ExecuteNode(
    const ResizeFunction& resize_fn,
    const std::vector<ValueRef>& resize_args)
    : shader_(),
      global_workgroup_size_({0u, 0u, 0u}),
      local_workgroup_size_({0u, 0u, 0u}),
      args_(),
      params_(),
      spec_vars_(),
      resize_fn_(resize_fn),
      resize_args_(resize_args) {}

void ExecuteNode::encode(ComputeGraph* graph) {
  if (!shader_) {
    return;
  }
  api::Context* const context = graph->context();
  vkapi::PipelineBarrier pipeline_barrier{};

  std::unique_lock<std::mutex> cmd_lock = context->dispatch_lock();

  context->report_shader_dispatch_start(
      shader_.kernel_name,
      global_workgroup_size_,
      local_workgroup_size_,
      node_id_);

  vkapi::DescriptorSet descriptor_set =
      context->get_descriptor_set(shader_, local_workgroup_size_, spec_vars_);

  uint32_t idx = 0;
  idx = bind_values_to_descriptor_set(
      graph, args_, pipeline_barrier, descriptor_set, idx);

  bind_params_to_descriptor_set(params_, descriptor_set, idx);

  context->register_shader_dispatch(
      descriptor_set, pipeline_barrier, shader_, global_workgroup_size_);

  context->report_shader_dispatch_end();
}

} // namespace vkcompute
