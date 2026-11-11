/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/QuantizationConfig.h>

namespace vkcompute {

//
// General utils
//

bool is_gemv(ComputeGraph* graph, const ValueRef& fp_input);

//
// Quantize, Dequantize for Linear/Matmul
//

void add_quantize_and_pack_4h4w_node(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const ValueRef fp_input,
    const ValueRef packed_input_scale,
    const ValueRef packed_input_zp,
    const ValueRef input_scale_data,
    const ValueRef input_zp_data,
    const ValueRef packed_int_input,
    const ValueRef group_size);

void add_quantize_and_pack_4h4w_with_group_sums_node(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const ValueRef fp_input,
    const ValueRef int_input_sums,
    const ValueRef packed_input_scales,
    const ValueRef packed_input_zps,
    const ValueRef packed_int_input,
    const ValueRef group_size);

//
// Quantize, Dequantize for Convolution
//

void add_quantize_and_pack_4w4c_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef packed_int8_input);

void add_unpack_4w4c_and_dequantize_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_output,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef fp_output);

} // namespace vkcompute
