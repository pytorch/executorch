/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/batching/step_plan.h>

#include <gtest/gtest.h>

#include <vector>

namespace cb = ::executorch::backends::cuda::batching;
using cb::StepMethod;

namespace {

struct Expected {
  int offset;
  int length;
  StepMethod method;
};

void expect_plan(
    int total,
    int max_step,
    int min_prefill,
    const std::vector<Expected>& expected) {
  const auto slices = cb::plan_slices(total, max_step, min_prefill);
  ASSERT_EQ(slices.size(), expected.size());
  int covered = 0;
  for (size_t i = 0; i < slices.size(); ++i) {
    EXPECT_EQ(slices[i].offset, expected[i].offset) << i;
    EXPECT_EQ(slices[i].length, expected[i].length) << i;
    EXPECT_EQ(slices[i].method, expected[i].method) << i;
    // In order and without gaps: a slice attends what its predecessors wrote.
    EXPECT_EQ(slices[i].offset, covered) << i;
    covered += slices[i].length;
  }
  EXPECT_EQ(covered, total);
}

constexpr auto D = StepMethod::Decode;
constexpr auto P = StepMethod::Prefill;

} // namespace

TEST(StepPlanTest, OneTokenRunsDecode) {
  expect_plan(1, 8, 2, {{0, 1, D}});
}

TEST(StepPlanTest, TwoTokensRunPrefill) {
  expect_plan(2, 8, 2, {{0, 2, P}});
}

TEST(StepPlanTest, UpToTheWidestStepIsOneForward) {
  expect_plan(8, 8, 2, {{0, 8, P}});
}

TEST(StepPlanTest, WiderBatchesSliceAndALoneTailTokenRunsDecode) {
  expect_plan(9, 8, 2, {{0, 8, P}, {8, 1, D}});
  expect_plan(10, 8, 2, {{0, 8, P}, {8, 2, P}});
  expect_plan(16, 8, 2, {{0, 8, P}, {8, 8, P}});
}

TEST(StepPlanTest, ShortSlicesBelowThePrefillBoundRunAsDecodes) {
  // A prefill exported from five tokens up: two to four run token by token.
  expect_plan(4, 8, 5, {{0, 1, D}, {1, 1, D}, {2, 1, D}, {3, 1, D}});
  expect_plan(5, 8, 5, {{0, 5, P}});
  expect_plan(11, 8, 5, {{0, 8, P}, {8, 1, D}, {9, 1, D}, {10, 1, D}});
}

TEST(StepPlanTest, EmptyBatchPlansNothing) {
  expect_plan(0, 8, 2, {});
}
