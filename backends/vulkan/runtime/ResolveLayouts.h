/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <unordered_map>

#include <executorch/backends/vulkan/serialization/schema_generated.h>

namespace vkcompute {

class ComputeGraph;

void resolve_memory_layouts(
    const vkgraph::VkGraph* flatbuffer,
    ComputeGraph* compute_graph,
    std::unordered_map<uint32_t, vkgraph::VkMemoryLayout>&
        memory_layout_overrides);

} // namespace vkcompute
