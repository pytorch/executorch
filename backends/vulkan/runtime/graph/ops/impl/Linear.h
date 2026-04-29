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

ValueRef prepack_fp_linear_weight(
    ComputeGraph& graph,
    const ValueRef weight_data,
    bool is_transposed,
    int64_t B,
    bool force_buffer = false);

void add_linear_tiled_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    bool has_bias,
    const ValueRef out,
    int32_t weight_B = 1,
    float alpha = 1.0f,
    float beta = 1.0f);

void add_linear_coopmat_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    bool has_bias,
    const ValueRef out,
    int32_t weight_B = 1);

} // namespace vkcompute
