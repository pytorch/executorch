/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <iterator>

#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>

// Generated from cadence_ops_lib's selected_operators.yaml; see
// backends/cadence/CMakeLists.txt.
#include "cadence_ops_lib_selected_ops.h"

namespace {

using ::executorch::runtime::get_registered_kernels;
using ::executorch::runtime::registry_has_op_function;

TEST(CadenceOpRegistryTest, EverySelectedOpIsRegistered) {
  ::executorch::runtime::runtime_init();
  for (const char* op : kSelectedOps) {
    EXPECT_TRUE(registry_has_op_function(op)) << op;
  }
  std::printf(
      "Checked %d selected ops against %d registered kernels.\n",
      static_cast<int>(std::size(kSelectedOps)),
      static_cast<int>(get_registered_kernels().size()));
}

} // namespace
