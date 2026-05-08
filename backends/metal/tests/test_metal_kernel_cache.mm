/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 */
#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <executorch/backends/metal/core/MetalKernelCache.h>
#include <executorch/backends/metal/core/MetalKernel.h>
#include <memory>
#include <string>

using executorch::backends::metal_v2::MetalKernelCache;
using executorch::backends::metal_v2::MetalKernel;

namespace {
class MetalKernelCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!MTLCreateSystemDefaultDevice()) GTEST_SKIP();
    MetalKernelCache::shared().resetForTesting();
  }
  void TearDown() override { MetalKernelCache::shared().resetForTesting(); }
};
TEST_F(MetalKernelCacheTest, FindOnEmptyReturnsNull) {
  EXPECT_EQ(MetalKernelCache::shared().find("nope"), nullptr);
}

TEST_F(MetalKernelCacheTest, FindOrInsertCachesResult) {
  int n = 0;
  auto factory = [&]() -> std::unique_ptr<MetalKernel> {
    ++n;
    return std::unique_ptr<MetalKernel>(new MetalKernel(nil, "stub"));
  };
  auto* k1 = MetalKernelCache::shared().findOrInsert("tk_cached", factory);
  auto* k2 = MetalKernelCache::shared().findOrInsert("tk_cached", factory);
  EXPECT_EQ(k1, k2);
  EXPECT_EQ(n, 1);
}
TEST_F(MetalKernelCacheTest, ResetForTestingClears) {
  MetalKernelCache::shared().findOrInsert("tk_clear",
      []() { return std::unique_ptr<MetalKernel>(new MetalKernel(nil, "stub")); });
  ASSERT_NE(MetalKernelCache::shared().find("tk_clear"), nullptr);
  MetalKernelCache::shared().resetForTesting();
  EXPECT_EQ(MetalKernelCache::shared().find("tk_clear"), nullptr);
}

TEST_F(MetalKernelCacheTest, FindLibraryOnEmptyReturnsNil) {
  EXPECT_EQ(MetalKernelCache::shared().findLibrary("nolib"), nil);
}

TEST_F(MetalKernelCacheTest, FindPsoOnEmptyReturnsNil) {
  EXPECT_EQ(MetalKernelCache::shared().findPso("nopso"), nil);
}

// --- Library sub-store roundtrip --------------------------------------

TEST_F(MetalKernelCacheTest, InsertAndFindLibrary) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_NE(device, nil);
  // Build a trivial library to insert.
  NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                   "kernel void k(device float* o [[buffer(0)]]) {}";
  NSError* err = nil;
  id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
  ASSERT_NE(lib, nil) << "failed to build trivial MTLLibrary";
  // R: API contract — cache retains its own copy; we release our +1
  // from -newLibraryWithSource: ourselves.
  MetalKernelCache::shared().insertLibrary("test_lib_key", lib);
  id<MTLLibrary> found = MetalKernelCache::shared().findLibrary("test_lib_key");
  EXPECT_EQ(found, lib);
  [lib release];
}

TEST_F(MetalKernelCacheTest, InsertAndFindPso) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_NE(device, nil);
  NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                   "kernel void k(device float* o [[buffer(0)]]) {}";
  NSError* err = nil;
  id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
  ASSERT_NE(lib, nil);
  id<MTLFunction> f = [lib newFunctionWithName:@"k"];
  ASSERT_NE(f, nil);
  id<MTLComputePipelineState> pso =
      [device newComputePipelineStateWithFunction:f error:&err];
  ASSERT_NE(pso, nil);
  // R: API contract — cache retains its own copy; we release our +1
  // from -newComputePipelineStateWithFunction: ourselves.
  MetalKernelCache::shared().insertPso("test_pso_key", pso);
  id<MTLComputePipelineState> found =
      MetalKernelCache::shared().findPso("test_pso_key");
  EXPECT_EQ(found, pso);
  [pso release];
  [f release];
  [lib release];
}

}  // namespace
