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

void add_q8ta_conv2d_pw_node(
    ComputeGraph& graph,
    const bool use_unsigned_dot,
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
    const ValueRef conv_input = kDummyValueRef,
    const ValueRef kernel_size = kDummyValueRef,
    const ValueRef stride = kDummyValueRef,
    const ValueRef padding = kDummyValueRef,
    const ValueRef dilation = kDummyValueRef,
    const bool is_im2col = false,
    const ValueRef stream_row_offset_ref = kDummyValueRef,
    const ValueRef max_im2col_rows_ref = kDummyValueRef);

constexpr int64_t kMaxUnsignedDotAccumulatorBytes = 33025;

bool can_use_unsigned_pw_dot(
    const vkapi::Adapter& adapter,
    int64_t k_per_group);

void q8ta_conv2d_pw_impl(
    ComputeGraph& graph,
    bool use_unsigned_dot,
    const std::vector<ValueRef>& args);

void q8ta_conv2d_pw(ComputeGraph& graph, const std::vector<ValueRef>& args);

} // namespace vkcompute
