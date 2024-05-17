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

void add_matmul_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef out,
    const ValueRef mat2_is_transposed);

} // namespace vkcompute
