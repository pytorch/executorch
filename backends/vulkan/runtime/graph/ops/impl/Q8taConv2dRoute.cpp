/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taConv2dRoute.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taConv2d.h>
#include <executorch/backends/vulkan/runtime/utils/VecUtils.h>

#include <limits>

namespace vkcompute {

namespace {

// Computes the per-group im2col kernel columns (in-channels x kernel height
// x kernel width, aligned up to 4) for the stream-plan sizing. Returns false
// (leaving the output untouched) when any dimension is non-positive or an
// intermediate product would overflow int64; callers fail closed to the
// direct path.
bool calculate_aligned_kernel_size(
    const Q8taConv2dRouteParams& params,
    int64_t& aligned_kernel_size) {
  if (params.in_channels_per_group <= 0 || params.kernel_height <= 0 ||
      params.kernel_width <= 0 ||
      params.in_channels_per_group >
          std::numeric_limits<int64_t>::max() / params.kernel_height) {
    return false;
  }
  const int64_t channel_kernel_height =
      params.in_channels_per_group * params.kernel_height;
  if (channel_kernel_height >
      std::numeric_limits<int64_t>::max() / params.kernel_width) {
    return false;
  }
  const int64_t unaligned_kernel_size =
      channel_kernel_height * params.kernel_width;
  if (unaligned_kernel_size > std::numeric_limits<int64_t>::max() - 3) {
    return false;
  }
  aligned_kernel_size = utils::align_up_4(unaligned_kernel_size);
  return true;
}

} // namespace

bool should_use_q8ta_conv2d_im2col(const Q8taConv2dRouteParams& params) {
  if (params.batch <= 0 || params.groups <= 0 ||
      params.in_channels_per_group <= 0 || params.out_channels <= 0 ||
      params.kernel_height <= 0 || params.kernel_width <= 0 ||
      params.out_height <= 0 || params.out_width <= 0 ||
      params.out_height >
          std::numeric_limits<int64_t>::max() / params.out_width) {
    return false;
  }
  int64_t flattened_kernel_size;
  if (!calculate_aligned_kernel_size(params, flattened_kernel_size)) {
    return false;
  }
  const bool im2col_eligible = params.in_channels_per_group % 4 == 0;
  if (!im2col_eligible) {
    return false;
  }
  // Grouped im2col partitions output blocks per group in the PW GEMM
  // (group_idx = oc_block / OC4_per_group), so each group must own whole
  // packed-4 output blocks; otherwise one block straddles two groups and
  // reads the wrong group's weights.
  if (params.groups > 1 &&
      (params.out_channels % params.groups != 0 ||
       params.out_channels / params.groups % 4 != 0)) {
    return false;
  }

  const int64_t spatial_out = params.out_height * params.out_width;
  if (params.batch > 1) {
    constexpr int64_t kMinFlattenedKernelSize = 1024;
    constexpr int64_t kMaxSpatialOutput = 64;
    // Size the probe plan with the same budget the consumer uses, so
    // num_tiles == 1 here means a single tile at execution too.
    const Q8taConv2dStreamPlan full_plan =
        make_q8ta_conv2d_stream_plan_for_device(
            params.batch,
            flattened_kernel_size,
            params.out_height,
            params.out_width,
            params.max_buffer_bytes);
    // Device-independent fast path: a large kernel over a tiny output makes
    // the im2col materialization negligible next to the GEMM, and a single
    // scratch tile means no streaming overhead, so this wins on every
    // device without needing vendor-specific tuning.
    const bool use_single_tile_batched_im2col = params.groups == 1 &&
        flattened_kernel_size >= kMinFlattenedKernelSize &&
        spatial_out <= kMaxSpatialOutput && full_plan.feasible &&
        full_plan.num_tiles == 1;
    if (use_single_tile_batched_im2col) {
      return true;
    }

    if (!params.is_mali) {
      return false;
    }

    // Mali: route all eligible batched convolutions through bounded
    // streaming im2col. The remaining guards are correctness bounds, not
    // perf cliffs.
    if (!params.supports_int8_dot_product ||
        // Conservative superset of the dispatch-time unsigned-path check
        // (which compares the unaligned weight K): reject the aligned
        // per-group K above the accumulator bound on every int8-dot path,
        // failing closed for signed-path shapes near the bound as well.
        flattened_kernel_size > kMaxUnsignedDotAccumulatorBytes) {
      return false;
    }
    int64_t plan_kernel_size = flattened_kernel_size;
    if (params.groups > 1) {
      if (plan_kernel_size >
          std::numeric_limits<int64_t>::max() / params.groups) {
        return false;
      }
      plan_kernel_size *= params.groups;
    }
    const Q8taConv2dStreamPlan device_plan =
        make_q8ta_conv2d_stream_plan_for_device(
            params.batch,
            plan_kernel_size,
            params.out_height,
            params.out_width,
            params.max_buffer_bytes);
    return device_plan.feasible;
  }

  if (params.is_mali) {
    return true;
  }

  // Single-batch heuristic: im2col pays off when wide channels
  // amortize the materialization over GEMM work, or when a small output
  // keeps the materialized buffer cheap. Anything else stays direct.
  return params.groups == 1 &&
      (params.in_channels_per_group >= 32 || spatial_out <= 4096);
}

} // namespace vkcompute
