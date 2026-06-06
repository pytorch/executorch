/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

namespace vkcompute {

// Helpers shared by the linear/matmul tiled and coopmat dispatch paths.

// Prepack a floating-point weight tensor for the linear/matmul kernels.
// Source layout: [N, K] (is_transposed=true) or [K, N] (is_transposed=false),
// optionally batched. Output layout: 4OC x 4IC blocked, packed as
// kWidthPacked. When force_buffer is true, the packed tensor uses buffer
// storage (required by the coopmat shader); otherwise texture2d is used
// when it fits within max_texture2d_dim.
ValueRef prepack_fp_linear_weight(
    ComputeGraph& graph,
    const ValueRef weight_data,
    bool is_transposed,
    int64_t B,
    bool force_buffer = false);

// Resize logic for linear-shaped output: takes M from the input's penultimate
// dim, N from resize_args[0], optional batch from input's leading dim.
void resize_linear_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

// Resize logic for matmul (mat1 @ mat2): output rows from mat1, cols from
// mat2, batch dims propagated from mat1.
void resize_matmul_tiled_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

} // namespace vkcompute
