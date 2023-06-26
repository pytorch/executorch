// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/core/kernel_types/TensorShapeDynamism.h>

#include <gtest/gtest.h>

using namespace ::testing;

using torch::executor::TensorShapeDynamism;

TEST(TensorShapeDynamismTest, CanBuildInATenMode) {
  // Demonstrate that aten mode can include the header and see the enum. If this
  // builds, the test passes.

#ifndef USE_ATEN_LIB
#error "This test should only be built in aten mode"
#endif

  EXPECT_NE(TensorShapeDynamism::STATIC, TensorShapeDynamism::DYNAMIC_BOUND);
}
