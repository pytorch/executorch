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
using executorch::aten::ScalarType;
using torch::executor::ScalarTypeToCppType;

TEST(DtypeSelectiveBuildTest, UnknownOp) {
  // Create a minimal context for error handling in ET_SWITCH
  struct {
    [[noreturn]] void fail(torch::executor::Error /* error */) {
      ET_CHECK_MSG(false, "Unsupported dtype");
    }
  } ctx;
  ET_EXPECT_DEATH(
      ET_SWITCH_TWO_TYPES(
          Float,
          Int,
          executorch::aten::ScalarType::Float,
          ctx,
          "unknown.out",
          // @lint-ignore CLANGTIDY clang-diagnostic-unused-local-typedef
          CTYPE_OUT,
          [&] { return true; }),
      "");
}

TEST(DtypeSelectiveBuildTest, OpWithoutDtype) {
  // Create a minimal context for error handling in ET_SWITCH
  struct {
    [[noreturn]] void fail(torch::executor::Error /* error */) {
      ET_CHECK_MSG(false, "Unsupported dtype");
    }
  } ctx;
  ET_EXPECT_DEATH(
      ET_SWITCH_TWO_TYPES(
          Float,
          Int,
          executorch::aten::ScalarType::Int,
          ctx,
          "add.out",
          // @lint-ignore CLANGTIDY clang-diagnostic-unused-local-typedef
          CTYPE_OUT,
          [&] { return true; }),
      "");
}

TEST(DtypeSelectiveBuildTest, OpWithDtype) {
  // Create a minimal context for error handling in ET_SWITCH
  struct {
    [[noreturn]] void fail(torch::executor::Error /* error */) {
      ET_CHECK_MSG(false, "Unsupported dtype");
    }
  } ctx;
  ASSERT_EQ(
      ET_SWITCH_TWO_TYPES(
          Float,
          Int,
          executorch::aten::ScalarType::Float,
          ctx,
          "add.out",
          // @lint-ignore CLANGTIDY clang-diagnostic-unused-local-typedef
          CTYPE_OUT,
          [&] { return true; }),
      true);
}
