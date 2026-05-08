/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 */
#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <executorch/backends/metal/core/HazardTracker.h>
using executorch::backends::metal_v2::HazardTracker;
namespace {
class HazardTrackerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) GTEST_SKIP() << "no Metal device";
    bufA_ = [device_ newBufferWithLength:1024 options:MTLResourceStorageModeShared];
    bufB_ = [device_ newBufferWithLength:1024 options:MTLResourceStorageModeShared];
  }
  void TearDown() override { [bufA_ release]; [bufB_ release]; }
  id<MTLDevice> device_ = nil;
  id<MTLBuffer> bufA_ = nil;
  id<MTLBuffer> bufB_ = nil;
};
TEST_F(HazardTrackerTest, FirstDispatchNoBarrier) {
  HazardTracker h;
  h.trackInput(bufA_, 0, 1024);
  h.trackOutput(bufB_, 0, 1024);
  EXPECT_FALSE(h.needsBarrierForPending());
  h.commitPending(false);
  EXPECT_EQ(h.barrierStats().inserted, 0u);
  EXPECT_EQ(h.barrierStats().skipped, 1u);
}
TEST_F(HazardTrackerTest, RAWHazardNeedsBarrier) {
  HazardTracker h;
  h.trackOutput(bufA_, 0, 1024);
  h.commitPending(false);
  h.trackInput(bufA_, 0, 1024);
  EXPECT_TRUE(h.needsBarrierForPending());
  h.commitPending(true);
  EXPECT_EQ(h.barrierStats().inserted, 1u);
  EXPECT_EQ(h.barrierStats().skipped, 1u);
}
TEST_F(HazardTrackerTest, WAWHazardNeedsBarrier) {
  HazardTracker h;
  h.trackOutput(bufA_, 0, 1024);
  h.commitPending(false);
  h.trackOutput(bufA_, 0, 1024);
  EXPECT_TRUE(h.needsBarrierForPending());
}
TEST_F(HazardTrackerTest, WARDetectsBarrier) {
  // dispatch[0] reads bufA. dispatch[1] writes bufA. On MTL4 concurrent
  // dispatch the write could land before the read sampled the value;
  // a WAR barrier is required.
  HazardTracker h;
  h.trackInput(bufA_, 0, 1024);
  h.commitPending(false);  // dispatch[0]: read-only, no prior writers, skip
  h.trackOutput(bufA_, 0, 1024);
  EXPECT_TRUE(h.needsBarrierForPending())
      << "WAR (write after read on overlapping range) must insert a barrier";
  h.commitPending(true);
  EXPECT_EQ(h.barrierStats().inserted, 1u);
  EXPECT_EQ(h.barrierStats().skipped, 1u);
}
TEST_F(HazardTrackerTest, WARDisjointRangesNoBarrier) {
  // Read [0,512), then write [512,1024) — disjoint, no WAR.
  HazardTracker h;
  h.trackInput(bufA_, 0, 512);
  h.commitPending(false);
  h.trackOutput(bufA_, 512, 1024);
  EXPECT_FALSE(h.needsBarrierForPending());
}
TEST_F(HazardTrackerTest, IndependentDispatchesNoBarrier) {
  HazardTracker h;
  h.trackOutput(bufA_, 0, 1024);
  h.commitPending(false);
  h.trackInput(bufB_, 0, 1024);
  h.trackOutput(bufB_, 0, 1024);
  EXPECT_FALSE(h.needsBarrierForPending());
}
TEST_F(HazardTrackerTest, SubRangeNonOverlapNoBarrier) {
  HazardTracker h;
  h.trackOutput(bufA_, 0, 512);
  h.commitPending(false);
  h.trackInput(bufA_, 512, 1024);
  EXPECT_FALSE(h.needsBarrierForPending());
}
TEST_F(HazardTrackerTest, SubRangeOverlapNeedsBarrier) {
  HazardTracker h;
  h.trackOutput(bufA_, 0, 512);
  h.commitPending(false);
  h.trackInput(bufA_, 256, 768);
  EXPECT_TRUE(h.needsBarrierForPending());
}
TEST_F(HazardTrackerTest, ResetDropsWriters) {
  HazardTracker h;
  h.trackOutput(bufA_, 0, 1024);
  h.commitPending(false);
  h.reset();
  h.trackInput(bufA_, 0, 1024);
  EXPECT_FALSE(h.needsBarrierForPending());
}
TEST_F(HazardTrackerTest, ExternalWriteCreatesHazard) {
  HazardTracker h;
  h.notifyExternalWrite(bufA_, 0, 1024);
  h.trackInput(bufA_, 0, 1024);
  EXPECT_TRUE(h.needsBarrierForPending());
}

// P0 #10: empty range (lo == hi). Should be a no-op (zero-byte write).
TEST_F(HazardTrackerTest, EmptyRangeIsNoOp) {
  HazardTracker h;
  h.trackOutput(bufA_, 100, 100);  // empty
  h.commitPending(false);
  // Empty writer shouldn't conflict with anything.
  h.trackInput(bufA_, 0, 1024);
  EXPECT_FALSE(h.needsBarrierForPending());
}

// P0 #12: writers merging on adjacency. Insert two adjacent ranges,
// then insert a third that overlaps the merged region.
TEST_F(HazardTrackerTest, WritersMergingAdjacentRanges) {
  HazardTracker h;
  h.trackOutput(bufA_, 0, 100);
  h.commitPending(false);
  h.trackOutput(bufA_, 100, 200);  // adjacent — should merge to [0, 200)
  h.commitPending(false);
  // Now read [50, 150) which spans the (formerly two, now merged) range.
  h.trackInput(bufA_, 50, 150);
  EXPECT_TRUE(h.needsBarrierForPending()) << "merged writer should still cover [50,150)";
}

// P0 #13: many distinct parents in writers map.
TEST_F(HazardTrackerTest, MultiParentTracking) {
  HazardTracker h;
  // Create 10 distinct buffers and write to each.
  std::vector<id<MTLBuffer>> bufs;
  for (int i = 0; i < 10; ++i) {
    bufs.push_back([device_ newBufferWithLength:1024 options:MTLResourceStorageModeShared]);
    h.trackOutput(bufs.back(), 0, 1024);
  }
  h.commitPending(false);
  // Reading from buffer 7 should hazard.
  h.trackInput(bufs[7], 0, 512);
  EXPECT_TRUE(h.needsBarrierForPending());
  for (auto b : bufs) [b release];
}
}  // namespace
