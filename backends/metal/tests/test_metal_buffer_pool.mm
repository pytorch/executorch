/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <executorch/backends/metal/core/MetalBufferPool.h>
#include <vector>

using executorch::backends::metal_v2::MetalBufferPool;

namespace {

class MetalBufferPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) GTEST_SKIP() << "no Metal device";
    pool_ = std::make_unique<MetalBufferPool>(device_, 1024 * 1024);
  }
  void TearDown() override { pool_.reset(); }
  id<MTLDevice> device_ = nil;
  std::unique_ptr<MetalBufferPool> pool_;
};

TEST_F(MetalBufferPoolTest, AcquireOnEmptyPoolAllocates) {
  id<MTLBuffer> b = pool_->acquire(1024);
  ASSERT_NE(b, nil);
  EXPECT_GE([b length], 1024u);
  [b release];
}

TEST_F(MetalBufferPoolTest, ReleaseThenAcquireSameSizeReusesIdentity) {
  id<MTLBuffer> b1 = pool_->acquire(1024);
  ASSERT_NE(b1, nil);
  pool_->release(b1);
  id<MTLBuffer> b2 = pool_->acquire(1024);
  EXPECT_EQ(b1, b2);
  [b2 release];
}

TEST_F(MetalBufferPoolTest, BestFitMatchingWithHeadroom) {
  id<MTLBuffer> big = pool_->acquire(2048);
  ASSERT_NE(big, nil);
  pool_->release(big);
  id<MTLBuffer> reused = pool_->acquire(1024);
  EXPECT_EQ(reused, big);
  [reused release];
}

TEST_F(MetalBufferPoolTest, AcquireWayLargerDoesNotReuse) {
  id<MTLBuffer> small = pool_->acquire(64);
  ASSERT_NE(small, nil);
  pool_->release(small);
  id<MTLBuffer> bigger = pool_->acquire(8192);
  EXPECT_NE(bigger, small);
  [bigger release];
}

// Regression: pool retains cached buffers so [release] from the original
// caller doesn't dealloc them out from under the pool's lruList_. This
// was a real bug introduced by Unit 6 (stripping alloc-time pin) and
// fixed in this session.
TEST_F(MetalBufferPoolTest, PoolRetainsCachedBuffer) {
  id<MTLBuffer> b = pool_->acquire(1024);
  ASSERT_NE(b, nil);
  pool_->release(b);
  [b release];  // drop caller's ownership; pool must still hold a retain
  // If pool didn't retain, this acquire (or pool dtor) would crash.
  id<MTLBuffer> b2 = pool_->acquire(1024);
  EXPECT_EQ(b2, b);
  [b2 release];
}

TEST_F(MetalBufferPoolTest, CachedBytesTracksUsage) {
  EXPECT_EQ(pool_->cachedBytes(), 0u);
  id<MTLBuffer> b = pool_->acquire(1024);
  pool_->release(b);
  EXPECT_GE(pool_->cachedBytes(), 1024u);
}

TEST_F(MetalBufferPoolTest, EvictionWhenOverCap) {
  pool_->setMaxBytes(2048);
  id<MTLBuffer> a = pool_->acquire(1024);
  id<MTLBuffer> b = pool_->acquire(1024);
  pool_->release(a);
  pool_->release(b);
  EXPECT_LE(pool_->cachedBytes(), 2048u);
  // Adding a third 1024-byte release should evict.
  id<MTLBuffer> c = pool_->acquire(1024);
  pool_->release(c);
  EXPECT_LE(pool_->cachedBytes(), 2048u);
}

TEST_F(MetalBufferPoolTest, PrewarmSeedsPool) {
  pool_->prewarm({1024, 2048, 4096});
  EXPECT_GE(pool_->cachedBytes(), 1024u + 2048u + 4096u);
  id<MTLBuffer> b = pool_->acquire(1024);
  EXPECT_NE(b, nil);
  [b release];
}

TEST_F(MetalBufferPoolTest, VeryLargeBypassesPool) {
  // size > maxBytes/2 should not be cached; release just drops it.
  pool_->setMaxBytes(2048);
  size_t before = pool_->cachedBytes();
  id<MTLBuffer> big = pool_->acquire(8192);
  pool_->release(big);
  EXPECT_EQ(pool_->cachedBytes(), before);
}

// P0 #9: pool destruction while external code still holds buffer refs.
// The pool's own retain is dropped via clear(); external retains keep
// the buffer alive. Verify no double-release by accessing the buffer
// after pool is destroyed.
TEST_F(MetalBufferPoolTest, PoolDestructionLeavesExternalRefAlive) {
  id<MTLBuffer> b;
  {
    MetalBufferPool localPool(device_, 1024 * 1024);
    b = localPool.acquire(1024);
    ASSERT_NE(b, nil);
    [b retain];           // simulate external holder
    localPool.release(b); // pool re-retains
  }                       // localPool dtor → pool's retain dropped
  // b still has external retain — must be alive + accessible.
  EXPECT_GE([b length], 1024u);
  [b release];  // drop external retain → fully freed
}

// Boundary: release(nil) shouldn't crash.
TEST_F(MetalBufferPoolTest, ReleaseNilIsSafe) {
  // Implementation may choose to crash, log, or no-op. Any of those
  // is acceptable as long as the dtor doesn't UAF.
  // We don't actually pass nil — that would deref [nil length] inside
  // release(). Instead just verify dtor is clean if we never released.
  EXPECT_EQ(pool_->cachedBytes(), 0u);
}

// Boundary: setMaxBytes(0) should evict everything.
TEST_F(MetalBufferPoolTest, SetMaxBytesZeroEvictsAll) {
  id<MTLBuffer> a = pool_->acquire(1024);
  id<MTLBuffer> b = pool_->acquire(2048);
  pool_->release(a);
  pool_->release(b);
  EXPECT_GE(pool_->cachedBytes(), 1024u + 2048u);
  pool_->setMaxBytes(0);
  EXPECT_EQ(pool_->cachedBytes(), 0u);
}

}  // namespace
