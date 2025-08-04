/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/utils/BindingUtils.h>

namespace vkcompute {

uint32_t bind_values_to_descriptor_set(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    vkapi::PipelineBarrier& pipeline_barrier,
    vkapi::DescriptorSet& descriptor_set,
    const uint32_t base_idx) {
  uint32_t idx = base_idx;
  for (auto& arg : args) {
    for (auto& ref : arg.refs) {
      graph->bind_value_to_descriptor_set(
          ref, pipeline_barrier, arg.access, descriptor_set, idx++);
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
