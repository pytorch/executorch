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

ValueRef prepack_biases(
    ComputeGraph& graph,
    const ValueRef vref,
    const ValueRef weight,
    const bool transposed,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout memory_layout);

void check_conv_args(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out);

} // namespace vkcompute
