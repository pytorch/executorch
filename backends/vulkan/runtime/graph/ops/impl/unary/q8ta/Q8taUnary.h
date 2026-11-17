/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/unary/UnaryOp.h>

namespace vkcompute {

//
// Unary operations for int8x4 tensors
//

void add_q8ta_unary_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef packed_int8_output,
    const UnaryOpType op_type);

} // namespace vkcompute
