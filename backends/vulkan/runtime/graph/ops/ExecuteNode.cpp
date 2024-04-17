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
    const api::ShaderInfo& shader,
    const api::utils::uvec3& global_workgroup_size,
    const api::utils::uvec3& local_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<std::shared_ptr<api::UniformParamsBuffer>>& params,
    const ResizeFunction& resize_fn,
    const std::vector<ValueRef>& resize_args)
    : shader_(shader),
      global_workgroup_size_(global_workgroup_size),
      local_workgroup_size_(local_workgroup_size),
      args_(args),
      params_(params),
      resize_fn_(resize_fn),
      resize_args_(resize_args) {
  graph.update_descriptor_counts(shader, /*execute = */ true);
}

ExecuteNode::ExecuteNode(
      ComputeGraph& graph,
      const ArgGroup& src,
      const ArgGroup& dst,
      const api::utils::uvec3& copy_range,
      const api::utils::uvec3& src_offset,
      const api::utils::uvec3& dst_offset)
  :
    src_(src), dst_(dst), copy_range_(copy_range),
    src_offset_(src_offset), dst_offset_(dst_offset) {
  // TODO: Update descriptor counts in graph.
}


void ExecuteNode::encode_shader(ComputeGraph* graph) {
  api::Context* const context = graph->context();
  api::PipelineBarrier pipeline_barrier{};

  std::unique_lock<std::mutex> cmd_lock = context->dispatch_lock();

  api::DescriptorSet descriptor_set =
      context->get_descriptor_set(shader_, *local_workgroup_size_);

  uint32_t idx = 0;
  idx = bind_values_to_descriptor_set(
      graph, args_, pipeline_barrier, descriptor_set, idx);
  bind_params_to_descriptor_set(params_, descriptor_set, idx);

  context->register_shader_dispatch(
      descriptor_set, pipeline_barrier, shader_, *global_workgroup_size_);
}

void ExecuteNode::encode_copy(ComputeGraph* graph) {
  api::Context* const context = graph->context();
  api::PipelineBarrier pipeline_barrier{};

  vTensorPtr src_v_t = graph->get_tensor(src_->refs[0]);
  api::VulkanImage& src_image = src_v_t->image(
      pipeline_barrier,
      api::PipelineStage::COMPUTE, api::MemoryAccessType::READ);

  vTensorPtr dst_v_t = graph->get_tensor(dst_->refs[0]);
  api::VulkanImage& dst_image = dst_v_t->image(
      pipeline_barrier,
      api::PipelineStage::COMPUTE, api::MemoryAccessType::WRITE);

  context->register_copy(
      pipeline_barrier,
      src_image,
      dst_image,
      *copy_range_,
      *src_offset_,
      *dst_offset_);
}

void ExecuteNode::encode(ComputeGraph* graph) {
  if (shader_.src_code.size > 0) {
    return encode_shader(graph);
  } else {
    return encode_copy(graph);
  }
}

} // namespace vkcompute
