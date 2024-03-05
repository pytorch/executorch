/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/Utils.h>

namespace at {
namespace native {
namespace vulkan {

void ExecuteNode::encode(ComputeGraph* graph) {
  api::Context* const context = graph->context();
  api::PipelineBarrier pipeline_barrier{};

  std::unique_lock<std::mutex> cmd_lock = context->dispatch_lock();

  api::DescriptorSet descriptor_set =
      context->get_descriptor_set(shader_, local_workgroup_size_);

  uint32_t idx = 0;
  idx = bind_values_to_descriptor_set(
      graph,
      outputs_,
      pipeline_barrier,
      api::MemoryAccessType::WRITE,
      descriptor_set,
      idx);
  idx = bind_values_to_descriptor_set(
      graph,
      inputs_,
      pipeline_barrier,
      api::MemoryAccessType::READ,
      descriptor_set,
      idx);
  descriptor_set.bind(idx, params_.buffer());

  context->register_shader_dispatch(
      descriptor_set, pipeline_barrier, shader_, global_workgroup_size_);
}

} // namespace vulkan
} // namespace native
} // namespace at
