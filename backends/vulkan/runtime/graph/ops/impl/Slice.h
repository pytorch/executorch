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

void add_slice_view_node(
    ComputeGraph& graph,
    ValueRef in_ref,
    ValueRef dim_ref,
    ValueRef opt_start_ref,
    ValueRef opt_end_ref,
    ValueRef opt_step_ref,
    ValueRef out_ref);

} // namespace vkcompute
