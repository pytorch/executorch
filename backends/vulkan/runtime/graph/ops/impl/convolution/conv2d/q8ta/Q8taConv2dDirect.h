/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>

namespace vkcompute {

ValueRef prepack_quantized_conv2d_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef weight_data,
    const ValueRef input,
    const ValueRef output,
    const ValueRef groups,
    const ValueRef kernel_size);

void add_q8ta_conv2d_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef packed_int8_input_im2col,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const uint32_t activation_type,
    const ValueRef packed_int8_output);

void q8ta_conv2d_general_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

} // namespace vkcompute
