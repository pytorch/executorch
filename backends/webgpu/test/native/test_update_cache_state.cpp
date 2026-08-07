/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/ops/update_cache/UpdateCacheState.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace executorch::backends::webgpu;

namespace {

LiveUpdateCacheInputs valid_inputs() {
  return {
      {1, 512, 2, 4},
      {1, 1024, 2, 4},
      4,
      4,
      0,
      64,
      65535,
  };
}

TEST(UpdateCacheState, ComputesCompleteStateBeforeOneCommit) {
  int commits = 0;
  LiveUpdateCacheState committed = {};

  refresh_live_update_cache_state(
      valid_inputs(), [&](const LiveUpdateCacheState& state) {
        ++commits;
        committed = state;
      });

  EXPECT_EQ(commits, 1);
  EXPECT_EQ(committed.params.numel, 4096u);
  EXPECT_EQ(committed.params.dst_offset, 0u);
  EXPECT_EQ(committed.params.cache_numel, 8192u);
  EXPECT_EQ(committed.workgroup_count_x, 64u);
}

TEST(UpdateCacheState, RejectsEveryInvalidInputBeforeCommit) {
  std::vector<LiveUpdateCacheInputs> invalid;
  auto input = valid_inputs();
  input.value_dims = {1, 1, 2};
  invalid.push_back(input);
  input = valid_inputs();
  input.value_dims[1] = 0;
  invalid.push_back(input);
  input = valid_inputs();
  input.value_dims[0] = 2;
  invalid.push_back(input);
  input = valid_inputs();
  input.cache_dims[2] = 3;
  invalid.push_back(input);
  input = valid_inputs();
  input.start_pos = -1;
  invalid.push_back(input);
  input = valid_inputs();
  input.start_pos = 513;
  invalid.push_back(input);
  input = valid_inputs();
  input.max_workgroups_per_dimension = 63;
  invalid.push_back(input);
  input = valid_inputs();
  input.workgroup_size = 0;
  invalid.push_back(input);
  input = valid_inputs();
  input.value_dims = {1, 1, 65536, 65536};
  input.cache_dims = input.value_dims;
  invalid.push_back(input);

  for (const auto& candidate : invalid) {
    int commits = 0;
    EXPECT_THROW(
        refresh_live_update_cache_state(
            candidate, [&](const LiveUpdateCacheState&) { ++commits; }),
        std::runtime_error);
    EXPECT_EQ(commits, 0);
  }
}

TEST(UpdateCacheState, RejectsStartPositionMultiplicationOverflow) {
  LiveUpdateCacheInputs input = {
      {1, 1, 1, 3},
      {1, 2, 1, 3},
      4,
      4,
      INT64_C(6148914691236517206),
      64,
      65535,
  };
  int commits = 0;

  EXPECT_THROW(
      refresh_live_update_cache_state(
          input, [&](const LiveUpdateCacheState&) { ++commits; }),
      std::runtime_error);
  EXPECT_EQ(commits, 0);
}

TEST(UpdateCacheState, RejectsNumelOverflowBeforeCommit) {
  auto input = valid_inputs();
  input.value_dims = {
      1, std::numeric_limits<int64_t>::max(), 2, 2};
  input.cache_dims = input.value_dims;
  int commits = 0;

  EXPECT_THROW(
      refresh_live_update_cache_state(
          input, [&](const LiveUpdateCacheState&) { ++commits; }),
      std::runtime_error);
  EXPECT_EQ(commits, 0);
}

} // namespace
