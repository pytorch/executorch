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

void add_prepack_int8x4_buffer_node(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const ValueRef tensor);

void add_staging_to_int8x4_buffer_node(
    ComputeGraph& graph,
    const ValueRef in_staging,
    const ValueRef tensor);

void add_int8x4_buffer_to_staging_node(
    ComputeGraph& graph,
    const ValueRef tensor,
    const ValueRef staging_data);

} // namespace vkcompute
