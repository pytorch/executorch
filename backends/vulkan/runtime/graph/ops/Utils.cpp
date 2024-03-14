/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/Utils.h>

namespace at {
namespace native {
namespace vulkan {

api::utils::ivec4 get_size_as_ivec4(const vTensor& t) {
  return api::utils::make_ivec4(
      {dim_at<Dim4D::Width>(t),
       dim_at<Dim4D::Height>(t),
       dim_at<Dim4D::Channel>(t),
       dim_at<Dim4D::Batch>(t)});
}

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
    const std::vector<ValueRef>& args,
    api::PipelineBarrier& pipeline_barrier,
    const api::MemoryAccessType accessType,
    api::DescriptorSet& descriptor_set,
    const uint32_t base_idx) {
  uint32_t idx = base_idx;
  for (auto& arg : args) {
    Value& val = graph->get_val(arg);
    if (val.isTensor()) {
      vTensor& tensor = val.toTensor();
      bind_tensor_to_descriptor_set(
          tensor, pipeline_barrier, accessType, descriptor_set, idx++);
    } else {
      VK_THROW("Unsupported type: ", val.type());
    }
  }
  return idx;
}

} // namespace vulkan
} // namespace native
} // namespace at
