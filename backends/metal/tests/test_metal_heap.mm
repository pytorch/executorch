/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 */
#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <executorch/backends/metal/core/MetalHeap.h>

using executorch::backends::metal_v2::MetalHeap;

namespace {
class MetalHeapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) GTEST_SKIP();
    heap_ = std::make_unique<MetalHeap>(device_, 4 * 1024 * 1024);
  }
  void TearDown() override { heap_.reset(); }
  id<MTLDevice> device_ = nil;
  std::unique_ptr<MetalHeap> heap_;
};

TEST_F(MetalHeapTest, TotalSizeReportsCapacity) {
  EXPECT_GE(heap_->totalSize(), 4u * 1024 * 1024);
}

TEST_F(MetalHeapTest, AllocBufferSucceedsForSmall) {
  if (!heap_->nativeHeap()) {
    FAIL() << "heap creation failed";
  }
  id<MTLBuffer> b = heap_->allocBuffer(1024);
  EXPECT_NE(b, nil) << "allocBuffer(1024) on a 4 MiB Placement heap returned nil";
}

TEST_F(MetalHeapTest, AllocBufferReturnsNilOnExhaustion) {
  // Allocate way more than the heap can hold.
  id<MTLBuffer> b = heap_->allocBuffer(1024 * 1024 * 1024);  // 1 GiB
  EXPECT_EQ(b, nil);
}

TEST_F(MetalHeapTest, NativeHeapAccessor) {
  if (@available(macOS 15.0, iOS 18.0, *)) {
    EXPECT_NE(heap_->nativeHeap(), nil);
  }
}

}  // namespace
