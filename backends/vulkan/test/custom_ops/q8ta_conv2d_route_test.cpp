// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taConv2dRoute.h>

#include <gtest/gtest.h>

#include <limits>

namespace vkcompute {
namespace {

constexpr uint64_t kLargeDeviceBuffer = 1ULL << 32;

Q8taConv2dRouteParams make_mali_grouped_params() {
  return {
      /*is_mali=*/true,
      /*supports_int8_dot_product=*/true,
      kLargeDeviceBuffer,
      /*batch=*/60,
      /*groups=*/2,
      /*in_channels_per_group=*/32,
      /*out_channels=*/64,
      /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*out_height=*/128,
      /*out_width=*/128,
  };
}

TEST(Q8taConv2dRouteTest, RoutesBatchedRegularMaliConvolutionToIm2Col) {
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col({
      /*is_mali=*/true,
      /*supports_int8_dot_product=*/true,
      kLargeDeviceBuffer,
      /*batch=*/60,
      /*groups=*/1,
      /*in_channels_per_group=*/64,
      /*out_channels=*/128,
      /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*out_height=*/20,
      /*out_width=*/26,
  }));

  EXPECT_TRUE(should_use_q8ta_conv2d_im2col({
      /*is_mali=*/true,
      /*supports_int8_dot_product=*/true,
      kLargeDeviceBuffer,
      /*batch=*/60,
      /*groups=*/1,
      /*in_channels_per_group=*/128,
      /*out_channels=*/256,
      /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*out_height=*/10,
      /*out_width=*/13,
  }));
}

TEST(Q8taConv2dRouteTest, RoutesMeasuredSceneXGroupedConvolutionsToIm2Col) {
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(make_mali_grouped_params()));

  Q8taConv2dRouteParams params = make_mali_grouped_params();
  params.groups = 4;
  params.in_channels_per_group = 32;
  params.out_channels = 128;
  params.kernel_height = 5;
  params.kernel_width = 5;
  params.out_height = 64;
  params.out_width = 64;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));

  params = make_mali_grouped_params();
  params.out_height = 64;
  params.out_width = 64;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));

  params = make_mali_grouped_params();
  params.groups = 3;
  params.in_channels_per_group = 32;
  params.out_channels = 96;
  params.kernel_height = 4;
  params.kernel_width = 4;
  params.out_height = 64;
  params.out_width = 64;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
}

TEST(Q8taConv2dRouteTest, RoutesPreviouslyOutOfEnvelopeShapesToIm2Col) {
  Q8taConv2dRouteParams params = make_mali_grouped_params();
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));

  params.is_mali = false;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.is_mali = true;
  params.supports_int8_dot_product = false;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.supports_int8_dot_product = true;

  // No group-count gate: regular and wide grouped shapes route alike, as
  // long as each group owns whole packed-4 output blocks.
  params.groups = 1;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.groups = 5;
  params.out_channels = 80;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.groups = 2;
  params.out_channels = 64;

  // No kernel squareness/size gate.
  params.kernel_width = 5;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.kernel_height = 2;
  params.kernel_width = 2;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.kernel_height = 6;
  params.kernel_width = 6;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.kernel_height = 3;
  params.kernel_width = 3;

  // Packing alignment is still required.
  params.in_channels_per_group = 31;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.in_channels_per_group = 34;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.in_channels_per_group = 32;

  // Output channels indivisible by groups fail closed.
  params.out_channels = 63;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.out_channels = 64;
  // No spatial window gates.
  params.out_height = 63;
  params.out_width = 64;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.out_height = 128;
  params.out_width = 129;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
}

TEST(Q8taConv2dRouteTest, RejectsMisalignedGroupedOutputChannels) {
  // The PW GEMM derives group_idx = oc_block / OC4_per_group, so a group
  // with a non-multiple-of-4 channel count would straddle packed blocks.
  Q8taConv2dRouteParams params = make_mali_grouped_params();
  params.groups = 3;
  params.out_channels = 66;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.groups = 5;
  params.out_channels = 65;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.out_channels = 80;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
}

TEST(Q8taConv2dRouteTest, RejectsMisalignedGroupedSingleBatchMali) {
  // The single-batch Mali branch dispatches the same grouped PW shader, so
  // the packed-4 output-block requirement binds there too.
  Q8taConv2dRouteParams params = make_mali_grouped_params();
  params.batch = 1;
  params.groups = 2;
  params.out_channels = 66;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.out_channels = 64;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
}

TEST(Q8taConv2dRouteTest, RoutesSingleTileScratchToIm2ColOffTileToDirect) {
  // The legacy full-scratch envelope (up to 32MiB) narrowed to a single
  // 16MiB streaming tile. With K=1152 and 8x8 output, batch 227 needs
  // 15.96MiB in one tile (routes im2col) while batch 228 needs 16.03MiB in
  // two tiles (stays direct on non-Mali, though both fit the old budget).
  auto make_params = [](int64_t batch) {
    return Q8taConv2dRouteParams{
        /*is_mali=*/false,
        /*supports_int8_dot_product=*/true,
        kLargeDeviceBuffer,
        batch,
        /*groups=*/1,
        /*in_channels_per_group=*/128,
        /*out_channels=*/128,
        /*kernel_height=*/3,
        /*kernel_width=*/3,
        /*out_height=*/8,
        /*out_width=*/8,
    };
  };
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(make_params(227)));
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(make_params(228)));
}

TEST(Q8taConv2dRouteTest, RespectsMaliDeviceBufferLimitForGroupedShapes) {
  Q8taConv2dRouteParams params = make_mali_grouped_params();
  params.groups = 4;
  params.out_channels = 128;
  params.kernel_height = 5;
  params.kernel_width = 5;
  params.out_height = 64;
  params.out_width = 64;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));

  // One im2col row is 3200 kernel columns x 64 aligned width bytes.
  constexpr uint64_t kBytesPerRow = 3200 * 64;
  params.max_buffer_bytes = kBytesPerRow;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.max_buffer_bytes = kBytesPerRow - 1;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
}

TEST(Q8taConv2dRouteTest, RejectsOverflowingMaliGroupedGeometry) {
  Q8taConv2dRouteParams params = make_mali_grouped_params();
  params.groups = 4;
  params.out_channels = 128;
  // max()/36 + 2 is 0 (mod 4), so it passes the alignment gate and reaches
  // the grouped overflow guard: align(C*3*3) > max()/4 with groups == 4.
  params.in_channels_per_group = std::numeric_limits<int64_t>::max() / 36 + 2;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));

  params = make_mali_grouped_params();
  params.out_height = std::numeric_limits<int64_t>::max();
  params.out_width = 2;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
}

TEST(Q8taConv2dRouteTest, RejectsIneligibleBatchedMaliConvolutions) {
  Q8taConv2dRouteParams params = {
      /*is_mali=*/true,
      /*supports_int8_dot_product=*/true,
      kLargeDeviceBuffer,
      /*batch=*/60,
      /*groups=*/1,
      /*in_channels_per_group=*/64,
      /*out_channels=*/128,
      /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*out_height=*/20,
      /*out_width=*/26,
  };

  // Grouped and pointwise batched shapes route alike on Mali.
  params.groups = 2;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.groups = 1;
  params.kernel_height = 1;
  params.kernel_width = 1;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.kernel_height = 3;
  params.kernel_width = 3;

  // Packing alignment is still required.
  params.in_channels_per_group = 62;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.in_channels_per_group = 64;

  // Kernels past the unsigned-dot accumulator bound stay direct.
  params.in_channels_per_group = 4096;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
}

TEST(Q8taConv2dRouteTest, RoutesSmallAndLargeBatchedShapesToIm2Col) {
  Q8taConv2dRouteParams params = {
      /*is_mali=*/true,
      /*supports_int8_dot_product=*/true,
      kLargeDeviceBuffer,
      /*batch=*/60,
      /*groups=*/1,
      /*in_channels_per_group=*/32,
      /*out_channels=*/64,
      /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*out_height=*/20,
      /*out_width=*/26,
  };
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));

  // No kernel-size gate.
  params.in_channels_per_group = 64;
  params.kernel_height = 2;
  params.kernel_width = 2;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.in_channels_per_group = 32;
  params.kernel_height = 3;
  params.kernel_width = 3;

  params.supports_int8_dot_product = false;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.supports_int8_dot_product = true;
  params.in_channels_per_group = 30;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.in_channels_per_group = 32;

  // No output-channel or spatial window gates.
  params.out_channels = 63;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.out_channels = 64;
  params.out_height = 8;
  params.out_width = 16;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.out_height = 8;
  params.out_width = 15;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.out_height = 32;
  params.out_width = 32;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.out_height = 1;
  params.out_width = 521;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.out_height = 20;
  params.out_width = 26;

  // No pointwise exclusion.
  params.in_channels_per_group = 256;
  params.kernel_height = 1;
  params.kernel_width = 1;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
}

TEST(Q8taConv2dRouteTest, RespectsMaliDeviceBufferLimit) {
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col({
      /*is_mali=*/true,
      /*supports_int8_dot_product=*/true,
      /*max_buffer_bytes=*/15 * 1024,
      /*batch=*/60,
      /*groups=*/1,
      /*in_channels_per_group=*/64,
      /*out_channels=*/128,
      /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*out_height=*/20,
      /*out_width=*/26,
  }));

  // One im2col row is 576 kernel columns x 28 aligned width bytes; the
  // device plan is feasible down to exactly that budget.
  constexpr uint64_t kBytesPerRow = 576 * 28;
  Q8taConv2dRouteParams params = {
      /*is_mali=*/true,
      /*supports_int8_dot_product=*/true,
      /*max_buffer_bytes=*/kBytesPerRow,
      /*batch=*/60,
      /*groups=*/1,
      /*in_channels_per_group=*/64,
      /*out_channels=*/128,
      /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*out_height=*/20,
      /*out_width=*/26,
  };
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
  params.max_buffer_bytes = kBytesPerRow - 1;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
}

TEST(Q8taConv2dRouteTest, PreservesNonMaliBatchedHeuristic) {
  Q8taConv2dRouteParams params = {
      /*is_mali=*/false,
      /*supports_int8_dot_product=*/true,
      kLargeDeviceBuffer,
      /*batch=*/60,
      /*groups=*/1,
      /*in_channels_per_group=*/128,
      /*out_channels=*/256,
      /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*out_height=*/10,
      /*out_width=*/13,
  };
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));

  params.out_height = 8;
  params.out_width = 8;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));

  params.is_mali = true;
  params.supports_int8_dot_product = false;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));

  params.is_mali = false;
  params.max_buffer_bytes = 1024;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
}

TEST(Q8taConv2dRouteTest, LegacyBatchedRoutePrecedesMaliExtensionGates) {
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col({
      /*is_mali=*/true,
      /*supports_int8_dot_product=*/false,
      kLargeDeviceBuffer,
      /*batch=*/2,
      /*groups=*/1,
      /*in_channels_per_group=*/1024,
      /*out_channels=*/32,
      /*kernel_height=*/1,
      /*kernel_width=*/1,
      /*out_height=*/8,
      /*out_width=*/8,
  }));
}

TEST(Q8taConv2dRouteTest, PreservesSingleBatchPolicy) {
  Q8taConv2dRouteParams params = {
      /*is_mali=*/true,
      /*supports_int8_dot_product=*/true,
      kLargeDeviceBuffer,
      /*batch=*/1,
      /*groups=*/4,
      /*in_channels_per_group=*/8,
      /*out_channels=*/32,
      /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*out_height=*/128,
      /*out_width=*/128,
  };
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));

  params.is_mali = false;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.groups = 1;
  params.in_channels_per_group = 32;
  EXPECT_TRUE(should_use_q8ta_conv2d_im2col(params));
}

TEST(Q8taConv2dRouteTest, RejectsInvalidAndOverflowingGeometry) {
  Q8taConv2dRouteParams params = {
      /*is_mali=*/true,
      /*supports_int8_dot_product=*/true,
      kLargeDeviceBuffer,
      /*batch=*/60,
      /*groups=*/1,
      /*in_channels_per_group=*/64,
      /*out_channels=*/128,
      /*kernel_height=*/3,
      /*kernel_width=*/3,
      /*out_height=*/20,
      /*out_width=*/26,
  };

  params.groups = 0;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.groups = 1;
  params.batch = 0;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.batch = 60;
  params.out_channels = 0;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
  params.out_channels = 128;
  params.out_height = std::numeric_limits<int64_t>::max();
  params.out_width = 2;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));

  params.out_height = 20;
  params.out_width = 26;
  params.in_channels_per_group = std::numeric_limits<int64_t>::max();
  params.kernel_height = 2;
  params.kernel_width = 1;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));

  params.in_channels_per_group = 4;
  params.kernel_height = std::numeric_limits<int64_t>::max() / 4;
  params.kernel_width = 2;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));

  params.in_channels_per_group = std::numeric_limits<int64_t>::max() - 2;
  params.kernel_height = 1;
  params.kernel_width = 1;
  EXPECT_FALSE(should_use_q8ta_conv2d_im2col(params));
}

} // namespace
} // namespace vkcompute
