/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/evalue_util/test/print_evalue_lean_helper.h>

#include <array>

#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/runtime/core/evalue.h>

namespace print_evalue_lean_and_aten_test {

void print_lean_double_list(std::ostream& os) {
  std::array<double, 7> list = {-3.0, -2.2, -1, 0, 3.3, 4.0, 5.5};
  executorch::aten::ArrayRef<double> double_ref(list.data(), list.size());
  executorch::runtime::EValue value(&double_ref);
  os << value;
}

} // namespace print_evalue_lean_and_aten_test
