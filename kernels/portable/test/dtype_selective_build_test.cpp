/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using torch::executor::ScalarTypeToCppType;

TEST(DtypeSelectiveBuildTest, UnknownOp) {
  ET_EXPECT_DEATH(
      ET_SWITCH_TWO_TYPES(
          Float,
          Int,
          exec_aten::ScalarType::Float,
          ctx,
          "unknown.out",
          // @lint-ignore CLANGTIDY clang-diagnostic-unused-local-typedef
          CTYPE_OUT,
          [&] { return true; }),
      "");
}

TEST(DtypeSelectiveBuildTest, OpWithoutDtype) {
  ET_EXPECT_DEATH(
      ET_SWITCH_TWO_TYPES(
          Float,
          Int,
          exec_aten::ScalarType::Int,
          ctx,
          "add.out",
          // @lint-ignore CLANGTIDY clang-diagnostic-unused-local-typedef
          CTYPE_OUT,
          [&] { return true; }),
      "");
}

TEST(DtypeSelectiveBuildTest, OpWithDtype) {
  ASSERT_EQ(
      ET_SWITCH_TWO_TYPES(
          Float,
          Int,
          exec_aten::ScalarType::Float,
          ctx,
          "add.out",
          // @lint-ignore CLANGTIDY clang-diagnostic-unused-local-typedef
          CTYPE_OUT,
          [&] { return true; }),
      true);
}
