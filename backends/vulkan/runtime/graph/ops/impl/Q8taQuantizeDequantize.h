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
// Generic Quantize, Dequantize (memory layout agnostic)
//

void add_q8ta_quantize_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef packed_int8_output);

void add_q8ta_dequantize_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef fp_output);

} // namespace vkcompute
