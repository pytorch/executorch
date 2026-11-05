// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/vk_api/DispatchGrid.h>

#include <gtest/gtest.h>

namespace vkcompute {
namespace {

TEST(BenchmarkDispatchGrid, PreservesArbitraryExplicitLocalDimensions) {
  const LocalWorkGroup lwg(10, 10, 1);
  EXPECT_EQ(static_cast<utils::uvec3>(lwg), (utils::uvec3{10, 10, 1}));
  EXPECT_EQ(lwg.nthreads(), 100);
  EXPECT_NO_THROW(lwg.validate({1024, 1024, 64}, 128));
  EXPECT_ANY_THROW(lwg.validate({8, 1024, 64}, 128));
  EXPECT_ANY_THROW(lwg.validate({1024, 1024, 64}, 64));
  EXPECT_ANY_THROW(LocalWorkGroup(0, 10, 1).validate({1024, 1024, 64}, 128));
  EXPECT_ANY_THROW(LocalWorkGroup(10, 10, 0).validate({1024, 1024, 64}, 128));
}

TEST(BenchmarkDispatchGrid, NonPowerOfTwoProductsRespectInvocationLimit) {
  const LocalWorkGroup lwg(3, 3, 3);
  EXPECT_EQ(lwg.nthreads(), 27);
  EXPECT_NO_THROW(lwg.validate({1024, 1024, 64}, 27));
  EXPECT_NO_THROW(lwg.validate({1024, 1024, 64}, 28));
  EXPECT_ANY_THROW(lwg.validate({1024, 1024, 64}, 26));
}

TEST(BenchmarkDispatchGrid, RejectsOverflowingLocalInvocationCounts) {
  const LocalWorkGroup lwg(65536, 65536, 1);
  EXPECT_ANY_THROW(lwg.validate({65536, 65536, 1}, 1024));
  EXPECT_ANY_THROW(lwg.nthreads());
}

TEST(BenchmarkDispatchGrid, AutomaticLocalShapesKeepPowerOfTwoFitting) {
  LocalWorkGroup lwg(kSquareLwg, 64);
  EXPECT_EQ(static_cast<utils::uvec3>(lwg), (utils::uvec3{8, 8, 1}));
  lwg.fit_to_global(GlobalWorkGrid({2048, 1, 1}, kTextureExtentsWorkGrid));
  EXPECT_EQ(static_cast<utils::uvec3>(lwg), (utils::uvec3{64, 1, 1}));
  EXPECT_EQ(lwg.nthreads(), 64);
  LocalWorkGroup fixed(10, 10, 1);
  fixed.fit_to_global(GlobalWorkGrid({2048, 1, 1}, kTextureExtentsWorkGrid));
  EXPECT_EQ(static_cast<utils::uvec3>(fixed), (utils::uvec3{10, 10, 1}));
}

TEST(BenchmarkDispatchGrid, LinearWrappingSupportsExplicitLocalDimensions) {
  GlobalWorkGrid gwg({1000, 1, 1}, kLinearWorkGrid);
  const LocalWorkGroup lwg(10, 10, 1);
  gwg.wrap_linear_dispatch({8, 8, 8}, lwg);
  EXPECT_EQ(gwg.extents(), (utils::uvec3{40, 30, 1}));
  EXPECT_EQ(gwg.required_lwg_size(), lwg);
  EXPECT_NO_THROW(gwg.validate(lwg, {8, 8, 8}));
  EXPECT_ANY_THROW(gwg.validate(LocalWorkGroup(8, 8, 1), {8, 8, 8}));
}

} // namespace
} // namespace vkcompute
