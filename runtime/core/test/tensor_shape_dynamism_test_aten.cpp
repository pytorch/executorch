/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/tensor_shape_dynamism.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::TensorShapeDynamism;

TEST(TensorShapeDynamismTest, CanBuildInATenMode) {
  // Demonstrate that aten mode can include the header and see the enum. If this
  // builds, the test passes.

#ifndef USE_ATEN_LIB
#error "This test should only be built in aten mode"
#endif

  EXPECT_NE(TensorShapeDynamism::STATIC, TensorShapeDynamism::DYNAMIC_BOUND);
}
