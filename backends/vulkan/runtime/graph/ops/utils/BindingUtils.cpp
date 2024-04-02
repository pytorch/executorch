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
    vTensor& tensor,
    api::PipelineBarrier& pipeline_barrier,
    const api::MemoryAccessType accessType,
    api::DescriptorSet& descriptor_set,
    const uint32_t idx) {
  if (tensor.buffer()) {
    api::VulkanBuffer& buffer = tensor.buffer(
        pipeline_barrier, api::PipelineStage::COMPUTE, accessType);
    descriptor_set.bind(idx, buffer);
  } else {
    api::VulkanImage& image =
        tensor.image(pipeline_barrier, api::PipelineStage::COMPUTE, accessType);
    descriptor_set.bind(idx, image);
  }
}

uint32_t bind_values_to_descriptor_set(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    api::PipelineBarrier& pipeline_barrier,
    api::DescriptorSet& descriptor_set,
    const uint32_t base_idx) {
  uint32_t idx = base_idx;
  for (auto& arg : args) {
    for (auto& ref : arg.refs) {
      Value& val = graph->get_val(ref);
      if (val.isTensor()) {
        bind_tensor_to_descriptor_set(
            val.toTensor(),
            pipeline_barrier,
            arg.access,
            descriptor_set,
            idx++);
      } else if (val.isStaging()) {
        bind_staging_to_descriptor_set(val.toStaging(), descriptor_set, idx++);
      } else {
        VK_THROW("Unsupported type: ", val.type());
      }
    }
  }
  return idx;
}

uint32_t bind_params_to_descriptor_set(
    std::vector<std::shared_ptr<api::UniformParamsBuffer>>& params,
    api::DescriptorSet& descriptor_set,
    const uint32_t base_idx) {
  uint32_t idx = base_idx;
  for (auto& param : params) {
    descriptor_set.bind(idx++, param->buffer());
  }
  return idx;
}

void bind_staging_to_descriptor_set(
    api::StorageBuffer& staging,
    api::DescriptorSet& descriptor_set,
    const uint32_t idx) {
  descriptor_set.bind(idx, staging.buffer());
}

} // namespace vkcompute
