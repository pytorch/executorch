/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <vector>

namespace vkcompute {

void add_permute_node(
    ComputeGraph& graph,
    ValueRef in,
    const std::vector<int64_t>& permute_dims,
    ValueRef out);

} // namespace vkcompute
