/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 */
#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <executorch/backends/metal/core/MetalKernel.h>

using executorch::backends::metal_v2::MetalKernel;

namespace {

class MetalKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!MTLCreateSystemDefaultDevice()) GTEST_SKIP();
  }
};

TEST_F(MetalKernelTest, ConstructWithNilPipeline) {
  // Tests can construct a kernel wrapper with nil pipeline + a name.
  MetalKernel k(nil, "stub_kernel");
  EXPECT_STREQ(k.name(), "stub_kernel");
  EXPECT_EQ(k.pipeline(), nil);
}

TEST_F(MetalKernelTest, MaxThreadsPerThreadgroupSafeOnNilPipeline) {
  // Nil pipeline → safe behavior expected (no crash). Returned value
  // is implementation-defined for nil pipeline.
  MetalKernel k(nil, "stub");
  (void)k.maxThreadsPerThreadgroup();
}

}  // namespace
