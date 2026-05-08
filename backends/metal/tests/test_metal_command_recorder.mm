/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 */
#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/backends/metal/core/MetalCommandRecorder.h>
#include <executorch/backends/metal/core/MetalAllocator.h>
#include <vector>

using executorch::backends::metal_v2::MetalStream;
using executorch::backends::metal_v2::MetalCommandRecorder;

namespace {

class MetalCommandRecorderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!MTLCreateSystemDefaultDevice()) GTEST_SKIP() << "no Metal device";
    stream_ = std::make_unique<MetalStream>();
  }
  void TearDown() override { stream_.reset(); }
  std::unique_ptr<MetalStream> stream_;
};

TEST_F(MetalCommandRecorderTest, BoundBufferCountStartsZero) {
  EXPECT_EQ(stream_->recorder().boundBufferCountForTesting(), 0u);
}

TEST_F(MetalCommandRecorderTest, DeclareSideDoorBindsAddsToBindList) {
  void* p = stream_->allocator().alloc(1024);
  ASSERT_NE(p, nullptr);
  auto bind = stream_->allocator().bufferForPtr(p, 1024);
  ASSERT_NE(bind.mtl, nil);
  size_t before = stream_->recorder().boundBufferCountForTesting();
  id<MTLBuffer> bufs[1] = { bind.mtl };
  stream_->recorder().declareSideDoorBinds(bufs, 1);
  EXPECT_EQ(stream_->recorder().boundBufferCountForTesting(), before + 1);
  EXPECT_TRUE(stream_->recorder().isBoundForTesting(bind.mtl));
  stream_->allocator().free(p);
}

TEST_F(MetalCommandRecorderTest, DeclareSideDoorBindsDedupes) {
  void* p = stream_->allocator().alloc(1024);
  ASSERT_NE(p, nullptr);
  auto bind = stream_->allocator().bufferForPtr(p, 1024);
  ASSERT_NE(bind.mtl, nil);
  size_t before = stream_->recorder().boundBufferCountForTesting();
  id<MTLBuffer> bufs[3] = { bind.mtl, bind.mtl, bind.mtl };
  stream_->recorder().declareSideDoorBinds(bufs, 3);
  // Dedup: only one new entry despite 3 calls.
  EXPECT_EQ(stream_->recorder().boundBufferCountForTesting(), before + 1);
  stream_->allocator().free(p);
}

TEST_F(MetalCommandRecorderTest, DeclareSideDoorBindsZeroCountIsSafe) {
  size_t before = stream_->recorder().boundBufferCountForTesting();
  stream_->recorder().declareSideDoorBinds(nullptr, 0);
  EXPECT_EQ(stream_->recorder().boundBufferCountForTesting(), before);
}

TEST_F(MetalCommandRecorderTest, SetFlushIntervalDoesNotCrash) {
  stream_->recorder().setFlushInterval(10);
  stream_->recorder().setFlushInterval(0);  // disable auto-flush
  stream_->recorder().setFlushInterval(1);
}

// recordBind dedup: same buffer registered twice in one CB → counted once.
TEST_F(MetalCommandRecorderTest, RecordBindDedup) {
  void* a = stream_->allocator().alloc(1024);
  void* b = stream_->allocator().alloc(1024);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  auto ba = stream_->allocator().bufferForPtr(a, 1024);
  auto bb = stream_->allocator().bufferForPtr(b, 1024);
  ASSERT_NE(ba.mtl, nil);
  ASSERT_NE(bb.mtl, nil);

  size_t before = stream_->recorder().boundBufferCountForTesting();
  // Register a, b, a again — should be 2 unique entries.
  id<MTLBuffer> binds[3] = { ba.mtl, bb.mtl, ba.mtl };
  stream_->recorder().declareSideDoorBinds(binds, 3);
  EXPECT_EQ(stream_->recorder().boundBufferCountForTesting(), before + 2);

  stream_->allocator().free(a);
  stream_->allocator().free(b);
}

}  // namespace
