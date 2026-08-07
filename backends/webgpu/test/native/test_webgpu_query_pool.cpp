/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUQueryPool.h>

#include <gtest/gtest.h>

namespace executorch::backends::webgpu {
namespace {

TEST(WebGPUQueryResultState, StartsInvalidAtGenerationZero) {
  detail::WebGPUQueryResultState state;
  EXPECT_FALSE(state.results_valid());
  EXPECT_EQ(state.result_generation(), 0u);
}

TEST(WebGPUQueryResultState, InvalidationPreservesMonotonicGeneration) {
  detail::WebGPUQueryResultState state;
  state.complete();
  EXPECT_TRUE(state.results_valid());
  EXPECT_EQ(state.result_generation(), 1u);

  state.invalidate();
  EXPECT_FALSE(state.results_valid());
  EXPECT_EQ(state.result_generation(), 1u);

  state.complete();
  EXPECT_TRUE(state.results_valid());
  EXPECT_EQ(state.result_generation(), 2u);
}

TEST(WebGPUQueryResultState, ExtractionPredicateFailsClosed) {
  const uint64_t ticks[] = {10, 20};

  EXPECT_TRUE(detail::query_result_extraction_succeeded(
      0, WGPUWaitStatus_TimedOut, WGPUMapAsyncStatus_Error, nullptr));
  EXPECT_FALSE(detail::query_result_extraction_succeeded(
      1,
      WGPUWaitStatus_TimedOut,
      WGPUMapAsyncStatus_Success,
      ticks));
  EXPECT_FALSE(detail::query_result_extraction_succeeded(
      1,
      WGPUWaitStatus_Success,
      WGPUMapAsyncStatus_Error,
      ticks));
  EXPECT_FALSE(detail::query_result_extraction_succeeded(
      1,
      WGPUWaitStatus_Success,
      WGPUMapAsyncStatus_Success,
      nullptr));
  EXPECT_TRUE(detail::query_result_extraction_succeeded(
      1,
      WGPUWaitStatus_Success,
      WGPUMapAsyncStatus_Success,
      ticks));
}

} // namespace
} // namespace executorch::backends::webgpu
