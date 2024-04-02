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
#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

namespace vkcompute {

PrepackNode::PrepackNode(
    ComputeGraph& graph,
    const api::ShaderInfo& shader,
    const api::utils::uvec3& global_workgroup_size,
    const api::utils::uvec3& local_workgroup_size,
    const ValueRef tref,
    const ValueRef packed,
    const std::vector<std::shared_ptr<api::UniformParamsBuffer>>& params)
    : shader_(shader),
      global_workgroup_size_(global_workgroup_size),
      local_workgroup_size_(local_workgroup_size),
      tref_(tref),
      packed_(packed),
      params_(params) {
  graph.update_descriptor_counts(shader, /*execute = */ false);
}

void PrepackNode::encode(ComputeGraph* graph) {
  api::Context* const context = graph->context();
  api::PipelineBarrier pipeline_barrier{};

  TensorRef tref = graph->get_val(tref_).toTensorRef();
  vTensor packed = graph->get_val(packed_).toTensor();

  size_t numel = api::utils::multiply_integers(tref.sizes);
  api::StorageBuffer staging(graph->context(), tref.dtype, numel);
  size_t nbytes = numel * api::element_size(tref.dtype);
  copy_ptr_to_staging(tref.data, staging, nbytes);

  std::unique_lock<std::mutex> cmd_lock = context->dispatch_lock();

  api::DescriptorSet descriptor_set =
      context->get_descriptor_set(shader_, local_workgroup_size_);

  uint32_t idx = 0;
  bind_tensor_to_descriptor_set(
      packed,
      pipeline_barrier,
      api::MemoryAccessType::WRITE,
      descriptor_set,
      idx++);
  bind_staging_to_descriptor_set(staging, descriptor_set, idx++);
  bind_params_to_descriptor_set(params_, descriptor_set, idx);

  context->register_shader_dispatch(
      descriptor_set, pipeline_barrier, shader_, global_workgroup_size_);
}

} // namespace vkcompute
