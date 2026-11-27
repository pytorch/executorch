/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/batching/step_plan.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

#include <vector>

namespace cb = ::executorch::backends::cuda::batching;

namespace {

struct Expected {
  int offset;
  int length;
  const char* method;
  int width;
};

// forward_{1,2,4,8} and forward_others up to 32 tokens.
const std::vector<cb::ForwardMethod> kDense = {
    {"forward_1", 1, true},
    {"forward_2", 2, true},
    {"forward_4", 4, true},
    {"forward_8", 8, true},
    {"forward_others", 32, false},
};

void expect_plan(
    int total,
    const std::vector<cb::ForwardMethod>& methods,
    const std::vector<Expected>& expected) {
  const auto slices = cb::plan_slices(total, methods);
  ASSERT_EQ(slices.size(), expected.size());
  int covered = 0;
  for (size_t i = 0; i < slices.size(); ++i) {
    EXPECT_EQ(slices[i].offset, expected[i].offset) << i;
    EXPECT_EQ(slices[i].length, expected[i].length) << i;
    EXPECT_EQ(methods[slices[i].method].name, expected[i].method) << i;
    EXPECT_EQ(slices[i].width, expected[i].width) << i;
    // In order and without gaps: a slice attends what its predecessors wrote.
    EXPECT_EQ(slices[i].offset, covered) << i;
    covered += slices[i].length;
  }
  EXPECT_EQ(covered, total);
}

} // namespace

TEST(StepPlanTest, ExactWidthsRunTheirStaticMethod) {
  expect_plan(1, kDense, {{0, 1, "forward_1", 1}});
  expect_plan(2, kDense, {{0, 2, "forward_2", 2}});
  expect_plan(4, kDense, {{0, 4, "forward_4", 4}});
  expect_plan(8, kDense, {{0, 8, "forward_8", 8}});
}

TEST(StepPlanTest, WidthsBetweenStaticMethodsPadToTheNextOne) {
  expect_plan(3, kDense, {{0, 3, "forward_4", 4}});
  expect_plan(5, kDense, {{0, 5, "forward_8", 8}});
  expect_plan(7, kDense, {{0, 7, "forward_8", 8}});
}

TEST(StepPlanTest, WiderThanEveryStaticMethodRunsDynamicUnpadded) {
  expect_plan(9, kDense, {{0, 9, "forward_others", 9}});
  expect_plan(32, kDense, {{0, 32, "forward_others", 32}});
}

TEST(StepPlanTest, WiderThanTheWidestMethodSlicesAndRoutesTheRest) {
  expect_plan(
      33, kDense, {{0, 32, "forward_others", 32}, {32, 1, "forward_1", 1}});
  expect_plan(
      38, kDense, {{0, 32, "forward_others", 32}, {32, 6, "forward_8", 8}});
  expect_plan(
      75,
      kDense,
      {{0, 32, "forward_others", 32},
       {32, 32, "forward_others", 32},
       {64, 11, "forward_others", 11}});
}

TEST(StepPlanTest, SparseStaticMethodsPadAcrossGaps) {
  const std::vector<cb::ForwardMethod> sparse = {
      {"forward_1", 1, true},
      {"forward_4", 4, true},
      {"forward_others", 32, false},
  };
  expect_plan(2, sparse, {{0, 2, "forward_4", 4}});
  expect_plan(4, sparse, {{0, 4, "forward_4", 4}});
  expect_plan(5, sparse, {{0, 5, "forward_others", 5}});
}

TEST(StepPlanTest, StaticOnlyProgramsSliceAtTheirWidest) {
  const std::vector<cb::ForwardMethod> static_only = {
      {"forward_1", 1, true},
      {"forward_8", 8, true},
  };
  expect_plan(
      11, static_only, {{0, 8, "forward_8", 8}, {8, 3, "forward_8", 8}});
}

TEST(StepPlanTest, EmptyBatchPlansNothing) {
  expect_plan(0, kDense, {});
  EXPECT_TRUE(cb::plan_slices(4, {}).empty());
}

TEST(StepPlanTest, NonPositiveWidestMethodIsRejected) {
  // A zero-wide slice never advances, so planning would never return.
  const std::vector<cb::ForwardMethod> zero = {{"forward_0", 0, true}};
  const std::vector<cb::ForwardMethod> negative = {
      {"forward_others", -1, false}};
  ET_EXPECT_DEATH(cb::plan_slices(1, zero), "widest method");
  ET_EXPECT_DEATH(cb::plan_slices(1, negative), "widest method");
}
