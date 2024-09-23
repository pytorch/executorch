/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <cstring>

namespace vkcompute {

void add_staging_to_tensor_node(
    ComputeGraph& graph,
    const ValueRef in_staging,
    const ValueRef out_tensor);

void add_tensor_to_staging_node(
    ComputeGraph& graph,
    const ValueRef in_tensor,
    const ValueRef out_staging);

ValueRef prepack_if_tensor_ref(
    ComputeGraph& graph,
    const ValueRef v,
    const utils::GPUMemoryLayout layout);

ValueRef prepack_buffer_if_tensor_ref(
    ComputeGraph& graph,
    const ValueRef v,
    const utils::GPUMemoryLayout layout);

ValueRef prepack_if_tensor_ref(ComputeGraph& graph, const ValueRef v);

ValueRef prepack_buffer_if_tensor_ref(ComputeGraph& graph, const ValueRef v);

} // namespace vkcompute
