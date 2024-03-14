/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace at {
namespace native {
namespace vulkan {

#define DECLARE_OP_FN(function) \
  ValueRef function(ComputeGraph& graph, const std::vector<ValueRef>& args);

api::utils::ivec4 get_size_as_ivec4(const vTensor& t);

void bind_tensor_to_descriptor_set(
    vTensor& tensor,
    api::PipelineBarrier& pipeline_barrier,
    const api::MemoryAccessType accessType,
    api::DescriptorSet& descriptor_set,
    const uint32_t idx);

uint32_t bind_values_to_descriptor_set(
    ComputeGraph* graph,
    const std::vector<ValueRef>& args,
    api::PipelineBarrier& pipeline_barrier,
    const api::MemoryAccessType accessType,
    api::DescriptorSet& descriptor_set,
    const uint32_t base_idx);

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
