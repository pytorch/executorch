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

namespace vkcompute {

void add_reduce_per_row_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef keepdim_ref,
    const ValueRef output,
    const std::string& op_name);

} // namespace vkcompute
