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

inline constexpr int64_t kQ8taConv2dIm2ColScratchBudgetBytes = 16 * 1024 * 1024;
inline constexpr int64_t kQ8taConv2dMaxRowsPerTile = 65535;

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

// max_buffer_bytes is Adapter::max_buffer_numel(), which returns
// maxStorageBufferRange in bytes (not elements) — directly comparable with
// the byte-denominated scratch budget.
Q8taConv2dStreamPlan make_q8ta_conv2d_stream_plan_for_device(
    int64_t batch,
    int64_t flattened_kernel_size,
    int64_t out_height,
    int64_t out_width,
    uint64_t max_buffer_bytes);

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
    const ValueRef stream_row_offset_ref,
    const ValueRef max_im2col_rows_ref = kDummyValueRef);

void q8ta_conv2d_im2col(ComputeGraph& graph, const std::vector<ValueRef>& args);

void q8ta_conv2d_im2col_impl(
    ComputeGraph& graph,
    bool use_unsigned_dot,
    const std::vector<ValueRef>& args);

} // namespace vkcompute
