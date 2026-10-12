/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <sstream>

#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/evalue_util/test/print_evalue_lean_helper.h>
#include <executorch/runtime/core/evalue.h>
#include <gtest/gtest.h>

// One edge-items setting on a stream must apply to lean and ATen EValues alike.
TEST(PrintEvalueLeanAndAtenTest, EdgeItemsApplyToBothModes) {
  std::array<double, 7> list = {-3.0, -2.2, -1, 0, 3.3, 4.0, 5.5};
  executorch::aten::ArrayRef<double> double_ref(list.data(), list.size());
  executorch::runtime::EValue aten_value(&double_ref);

  std::ostringstream lean_os;
  lean_os << executorch::extension::evalue_edge_items(1);
  print_evalue_lean_and_aten_test::print_lean_double_list(lean_os);
  EXPECT_STREQ(lean_os.str().c_str(), "(len=7)[-3., ..., 5.5]");

  std::ostringstream aten_os;
  aten_os << executorch::extension::evalue_edge_items(1) << aten_value;
  EXPECT_STREQ(aten_os.str().c_str(), "(len=7)[-3., ..., 5.5]");
}
