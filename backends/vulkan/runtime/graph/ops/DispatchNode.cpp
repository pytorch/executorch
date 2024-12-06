/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/DispatchNode.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/BindingUtils.h>

namespace vkcompute {

uint32_t PushConstantDataInfo::write(
    void* dst,
    const uint32_t dst_offset,
    const uint32_t max_dst_size) const {
  if (tensorUniformData != nullptr) {
    return tensorUniformData->write_attribute(
        dst, dst_offset, max_dst_size, payload_.attr);
  }

  VK_CHECK_COND(
      (dst_offset + payload_.dataSize) <= max_dst_size,
      "Attempting to write push constant data outside data boundary.");
  memcpy((uint8_t*)dst + dst_offset, payload_.data, payload_.dataSize);
  return payload_.dataSize;
}

DispatchNode::DispatchNode(
    ComputeGraph& graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const utils::uvec3& local_workgroup_size,
    const std::vector<ArgGroup>& args,
    const vkapi::ParamsBindList& params,
    const vkapi::SpecVarList& spec_vars,
    const ResizeFunction& resize_fn,
    const std::vector<ValueRef>& resize_args,
    const std::vector<PushConstantDataInfo>& push_constants)
    : ExecuteNode(resize_fn, resize_args, args, shader.kernel_name),
      shader_(shader),
      global_workgroup_size_(global_workgroup_size),
      local_workgroup_size_(local_workgroup_size),
      params_(params),
      spec_vars_(spec_vars),
      push_constants_(push_constants) {
  graph.update_descriptor_counts(shader, /*execute = */ true);
}

void DispatchNode::encode(ComputeGraph* graph) {
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

  uint8_t push_constants_data[128];
  uint32_t push_constants_offset = 0;

  for (const auto& push_constant : push_constants_) {
    push_constants_offset +=
        push_constant.write(push_constants_data, push_constants_offset, 128);
  }
  context->register_shader_dispatch(
      descriptor_set,
      pipeline_barrier,
      shader_,
      global_workgroup_size_,
      push_constants_data,
      push_constants_offset);

  context->report_shader_dispatch_end();
}

} // namespace vkcompute
