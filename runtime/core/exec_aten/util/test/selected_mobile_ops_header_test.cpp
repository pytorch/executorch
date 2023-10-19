/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using torch::executor::ScalarTypeToCppType;

TEST(SelectedMobileOpsHeaderTest, UnknownOp) {
  ET_EXPECT_DEATH(
      ET_SWITCH_TWO_TYPES(
          Float,
          Int,
          exec_aten::ScalarType::Float,
          ctx,
          "unknown_op",
          CTYPE_OUT,
          [&] { return true; }),
      "");
}

TEST(SelectedMobileOpsHeaderTest, OpWithoutDtype) {
  ET_EXPECT_DEATH(
      ET_SWITCH_TWO_TYPES(
          Float,
          Int,
          exec_aten::ScalarType::Double,
          ctx,
          "test_op",
          CTYPE_OUT,
          [&] { return true; }),
      "");
}

TEST(SelectedMobileOpsHeaderTest, OpWithDtype) {
  ASSERT_EQ(
      ET_SWITCH_TWO_TYPES(
          Float,
          Int,
          exec_aten::ScalarType::Float,
          ctx,
          "test_op",
          CTYPE_OUT,
          [&] { return true; }),
      true);
  ASSERT_EQ(
      ET_SWITCH_TWO_TYPES(
          Float,
          Int,
          exec_aten::ScalarType::Int,
          ctx,
          "test_op",
          CTYPE_OUT,
          [&] { return true; }),
      true);
}
