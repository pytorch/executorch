/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

void add_split_with_sizes_node(
    ComputeGraph& graph,
    const ValueRef input,
    const std::vector<int64_t>& split_sizes,
    const int64_t dim,
    const ValueRef out_list_ref);

} // namespace vkcompute
