/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <typeinfo>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/test/evalue_lean_type_name.h>
#include <gtest/gtest.h>

// Distinct names keep lean and ATen EValue members from merging.
TEST(EValueLeanAndAtenTest, AreDistinctTypes) {
  EXPECT_STRNE(
      evalue_lean_and_aten_test::lean_evalue_type_name(),
      typeid(executorch::runtime::EValue).name());
}
