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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>

namespace vkcompute {

inline constexpr int64_t kQ8taConv2dIm2ColScratchBudgetBytes =
    16 * 1024 * 1024;

struct Q8taConv2dStreamPlan final {
  int64_t aligned_out_width;
  int64_t rows_per_tile;
  int64_t num_tiles;
  int64_t scratch_bytes;
  bool feasible;
};

Q8taConv2dStreamPlan make_q8ta_conv2d_stream_plan(
    int64_t batch,
    int64_t flattened_kernel_size,
    int64_t out_height,
    int64_t out_width,
    int64_t scratch_budget_bytes);

enum class ActivationType : uint32_t {
  kNone = 0,
  kRelu = 1,
};

enum class Q8taConv2dPwMode : uint8_t {
  kStandalone,
  kIm2Col,
  kStreamingIm2Col,
};

enum class Q8taIm2ColMode : uint8_t {
  kFull,
  kStreaming,
};

enum class Q8taConv2dPwKernel : uint8_t {
  kAuto,
  kFallback,
};

ActivationType activation_type_from_string(const std::string& activation);

bool q8ta_conv2d_check_packed_dim_info(const api::PackedDimInfo& info);

bool q8ta_conv2d_check_4w4c_packed_dim_info(const api::PackedDimInfo& info);

ValueRef prepack_quantized_conv2d_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef weight_data,
    const ValueRef input,
    const ValueRef output,
    const ValueRef groups,
    const ValueRef kernel_size);

ValueRef prepack_quantized_conv2d_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef weight_data,
    const ValueRef input,
    const ValueRef output,
    const ValueRef groups,
    const ValueRef kernel_size);

ValueRef prepack_quantized_conv2d_dw_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef weight_data,
    const ValueRef kernel_size);

void add_q8ta_conv2d_dw_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
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

void add_conv2d_dw_q8ta_q8csw_q8to_4w4c_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
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
    const ValueRef packed_int8_output);

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

void add_q8ta_conv2d_pw_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const uint32_t activation_type,
    const ValueRef packed_int8_output,
    const int32_t groups = 1,
    const Q8taConv2dPwMode mode = Q8taConv2dPwMode::kStandalone,
    const Q8taConv2dPwKernel kernel = Q8taConv2dPwKernel::kAuto,
    const ValueRef conv_input = kDummyValueRef,
    const ValueRef kernel_size = kDummyValueRef,
    const ValueRef stride = kDummyValueRef,
    const ValueRef padding = kDummyValueRef,
    const ValueRef dilation = kDummyValueRef,
    const int32_t stream_row_offset = 0);

std::vector<int64_t> calculate_q8ta_im2col_sizes(
    ComputeGraph* graph,
    const ValueRef& input,
    const ValueRef& output,
    const ValueRef& kernel_size,
    const ValueRef& groups);

void add_q8ta_im2col_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef packed_int8_output,
    const ValueRef packed_int8_im2col,
    const int32_t zp,
    Q8taIm2ColMode mode,
    const int32_t stream_row_offset);

void q8ta_conv2d_im2col(ComputeGraph& graph, const std::vector<ValueRef>& args);

void q8ta_conv2d_im2col_with_kernel(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    Q8taConv2dPwKernel kernel);

void q8ta_conv2d_general(ComputeGraph& graph, const std::vector<ValueRef>& args);

// Transposed convolution

void q8ta_conv2d_transposed(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

} // namespace vkcompute
