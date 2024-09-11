/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/utils/BindingUtils.h>

namespace vkcompute {

void bind_tensor_to_descriptor_set(
    api::vTensor& tensor,
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::MemoryAccessType accessType,
    vkapi::DescriptorSet& descriptor_set,
    const uint32_t idx) {
  if (tensor.buffer()) {
    vkapi::VulkanBuffer& buffer = tensor.buffer(
        pipeline_barrier, vkapi::PipelineStage::COMPUTE, accessType);
    descriptor_set.bind(idx, buffer);
  } else {
    vkapi::VulkanImage& image = tensor.image(
        pipeline_barrier, vkapi::PipelineStage::COMPUTE, accessType);
    descriptor_set.bind(idx, image);
  }
}

uint32_t bind_values_to_descriptor_set(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    vkapi::PipelineBarrier& pipeline_barrier,
    vkapi::DescriptorSet& descriptor_set,
    const uint32_t base_idx) {
  uint32_t idx = base_idx;
  for (auto& arg : args) {
    for (auto& ref : arg.refs) {
      if (graph->val_is_tensor(ref)) {
        bind_tensor_to_descriptor_set(
            *(graph->get_tensor(ref)),
            pipeline_barrier,
            arg.access,
            descriptor_set,
            idx++);
      } else if (graph->val_is_staging(ref)) {
        bind_staging_to_descriptor_set(
            *(graph->get_staging(ref)), descriptor_set, idx++);
      } else {
        VK_THROW("Unsupported type: ", graph->get_val_type(ref));
      }
    }
  }
  return idx;
}

uint32_t bind_params_to_descriptor_set(
    const vkapi::ParamsBindList& params,
    vkapi::DescriptorSet& descriptor_set,
    const uint32_t base_idx) {
  uint32_t idx = base_idx;
  for (auto& param : params.bind_infos) {
    descriptor_set.bind(idx++, param);
  }
  return idx;
}

void bind_staging_to_descriptor_set(
    api::StagingBuffer& staging,
    vkapi::DescriptorSet& descriptor_set,
    const uint32_t idx) {
  descriptor_set.bind(idx, staging.buffer());
}

} // namespace vkcompute
