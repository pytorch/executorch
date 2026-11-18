/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <vector>

#include <executorch/backends/vulkan/runtime/api/containers/StagingBuffer.h>
#include <executorch/backends/vulkan/runtime/vk_api/Descriptor.h>
#include <executorch/backends/vulkan/runtime/vk_api/Pipeline.h>
#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

//
// For objects in the graph
//

uint32_t bind_values_to_descriptor_set(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    vkapi::PipelineBarrier& pipeline_barrier,
    vkapi::DescriptorSet& descriptor_set,
    const uint32_t base_idx);

//
// For objects NOT in the graph
//

uint32_t bind_params_to_descriptor_set(
    const vkapi::ParamsBindList& params,
    vkapi::DescriptorSet& descriptor_set,
    const uint32_t base_idx);

void bind_staging_to_descriptor_set(
    api::StagingBuffer& staging,
    vkapi::DescriptorSet& descriptor_set,
    const uint32_t idx);

} // namespace vkcompute
