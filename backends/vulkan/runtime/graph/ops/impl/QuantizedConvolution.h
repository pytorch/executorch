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
// Quantize and dequantize functions for conv2d that can be reused by other
// operations
//

/**
 * Add a dispatch node to quantize a floating-point input tensor to a packed
 * int8 tensor for use in quantized operations.
 */
void add_quantize_and_pack_q8ta_conv2d_input_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef packed_int8_input);

/**
 * Add a dispatch node to unpack and dequantize a packed int8 output tensor back
 * to a floating-point tensor.
 */
void add_unpack_and_dequantize_q8ta_conv2d_output_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_output,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef fp_output);

} // namespace vkcompute
