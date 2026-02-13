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

//
// Binary operations for int8x4 tensors
//

void add_q8ta_binary_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input_a,
    const ValueRef packed_int8_input_b,
    const ValueRef input_a_scale,
    const ValueRef input_a_zp,
    const ValueRef input_b_scale,
    const ValueRef input_b_zp,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef alpha,
    const ValueRef packed_int8_output,
    const std::string& op_name);

} // namespace vkcompute
