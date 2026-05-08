/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// test_residency_manager — unit tests for the thin MTLResidencySet wrapper.
//
// Observable behavior we can test without introspecting the Metal driver:
//   - construction succeeds on supported OS (isEnabled() returns true)
//   - add/remove/commit don't crash on null or unsupported-OS no-op paths
//   - commit is a no-op on the "nothing changed since last commit" path
//     (dirty-bit optimization); add/remove flips dirty back on
//   - nativeSet() returns the underlying MTLResidencySet pointer when
//     enabled and nil otherwise
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <gtest/gtest.h>
#include <span>

#include <executorch/backends/metal/core/ResidencyManager.h>

using executorch::backends::metal_v2::ResidencyManager;

namespace {

class ResidencyManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device_, nullptr) << "no Metal device available";
    buffer_ = [device_ newBufferWithLength:4096
                                   options:MTLResourceStorageModeShared];
    ASSERT_NE(buffer_, nil);
  }
  void TearDown() override {
  }
  // Typedef so std::span<...>(...) parses cleanly in Obj-C++ (the parser
  // gets confused by `id<MTLBuffer>` immediately followed by `(`).
  using Buf = id<MTLBuffer>;
  id<MTLDevice> device_ = nil;
  id<MTLBuffer> buffer_ = nil;
};

// Construction never throws; isEnabled reflects the OS/build gate.
TEST_F(ResidencyManagerTest, CtorDoesNotThrow) {
  ResidencyManager r(device_);
  // isEnabled() depends on macOS/iOS version AND ET_RESIDENCY_SET_AVAILABLE;
  // either value is acceptable.
  (void)r.isEnabled();
}

// add/remove are safe to call with null and on disabled managers.
TEST_F(ResidencyManagerTest, AddRemoveNullSafe) {
  ResidencyManager r(device_);
  r.add(nil);    // must not crash
  r.remove(nil); // must not crash
}

// commit() is safe to call before any add/remove.
TEST_F(ResidencyManagerTest, CommitOnEmptyIsSafe) {
  ResidencyManager r(device_);
  r.commit();  // no-op; must not crash
  r.commit();  // dirty-bit path: still no crash
}

// commit after add then commit without changes: second commit is a
// dirty-bit no-op. We can't observe the driver, but we can at least
// verify the calls don't throw and that the sequence runs to completion.
TEST_F(ResidencyManagerTest, CommitIdempotentWhenClean) {
  ResidencyManager r(device_);
  r.add(buffer_);
  r.commit();

  // Now nothing has changed; commit() should early-return (dirty==false).
  r.commit();
  r.commit();
  // Should still be safe to add more and commit.
  r.remove(buffer_);
  r.commit();
}

// nativeSet() reflects whether the underlying Metal set was created.
TEST_F(ResidencyManagerTest, NativeSetMatchesIsEnabled) {
  ResidencyManager r(device_);
  if (r.isEnabled()) {
    if (@available(macOS 15.0, iOS 18.0, *)) {
      EXPECT_NE(r.nativeSet(), nil);
    }
  } else {
    if (@available(macOS 15.0, iOS 18.0, *)) {
      EXPECT_EQ(r.nativeSet(), nil);
    }
  }
}

// Mixed add/remove across many buffers.
TEST_F(ResidencyManagerTest, AddRemoveBatch) {
  ResidencyManager r(device_);
  constexpr int N = 16;
  id<MTLBuffer> bufs[N];
  for (int i = 0; i < N; ++i) {
    bufs[i] = [device_ newBufferWithLength:1024
                                   options:MTLResourceStorageModeShared];
    r.add(bufs[i]);
  }
  r.commit();
  for (int i = 0; i < N; ++i) {
    r.remove(bufs[i]);
  }
  r.commit();
  (void)0;
}

//===----------------------------------------------------------------------===//
// Refcounted pin/unpin (refcounted pin surface).
//===----------------------------------------------------------------------===//

// pin then unpin: 0→1→0; refcountForTesting reflects state.
TEST_F(ResidencyManagerTest, PinUnpinRefcountBasic) {
  ResidencyManager r(device_);
  if (!r.isEnabled()) {
    GTEST_SKIP() << "ResidencySet not available on this OS";
  }
  EXPECT_EQ(r.refcountForTesting(buffer_), 0);
  r.pin(buffer_);
  EXPECT_EQ(r.refcountForTesting(buffer_), 1);
  r.unpin(buffer_);
  EXPECT_EQ(r.refcountForTesting(buffer_), 0);
}

// Two pins then one unpin: refcount 0→1→2→1; buffer still in set.
TEST_F(ResidencyManagerTest, PinPinUnpinKeepsInSet) {
  ResidencyManager r(device_);
  if (!r.isEnabled()) {
    GTEST_SKIP();
  }
  r.pin(buffer_);
  r.pin(buffer_);
  EXPECT_EQ(r.refcountForTesting(buffer_), 2);
  r.unpin(buffer_);
  EXPECT_EQ(r.refcountForTesting(buffer_), 1);
  // Final unpin drops to zero and erases.
  r.unpin(buffer_);
  EXPECT_EQ(r.refcountForTesting(buffer_), 0);
}

// pinBatch / unpinBatch behave like pin/unpin per element.
TEST_F(ResidencyManagerTest, PinBatchUnpinBatch) {
  ResidencyManager r(device_);
  if (!r.isEnabled()) {
    GTEST_SKIP();
  }
  constexpr int N = 8;
  id<MTLBuffer> bufs[N];
  for (int i = 0; i < N; ++i) {
    bufs[i] = [device_ newBufferWithLength:512
                                   options:MTLResourceStorageModeShared];
  }
  r.pinBatch(bufs, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(r.refcountForTesting(bufs[i]), 1) << "i=" << i;
  }
  r.unpinBatch(bufs, N);
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(r.refcountForTesting(bufs[i]), 0) << "i=" << i;
  }
}

// Independent buffers don't interfere.
TEST_F(ResidencyManagerTest, PinIndependentBuffers) {
  ResidencyManager r(device_);
  if (!r.isEnabled()) {
    GTEST_SKIP();
  }
  id<MTLBuffer> a =
      [device_ newBufferWithLength:512 options:MTLResourceStorageModeShared];
  id<MTLBuffer> b =
      [device_ newBufferWithLength:512 options:MTLResourceStorageModeShared];
  r.pin(a);
  EXPECT_EQ(r.refcountForTesting(a), 1);
  EXPECT_EQ(r.refcountForTesting(b), 0);
  r.pin(b);
  EXPECT_EQ(r.refcountForTesting(a), 1);
  EXPECT_EQ(r.refcountForTesting(b), 1);
  r.unpin(a);
  EXPECT_EQ(r.refcountForTesting(a), 0);
  EXPECT_EQ(r.refcountForTesting(b), 1);
  r.unpin(b);
  EXPECT_EQ(r.refcountForTesting(b), 0);
}

// Legacy add/remove route through pin/unpin (Unit 2 contract).
TEST_F(ResidencyManagerTest, AddRemoveRoutesToPinUnpin) {
  ResidencyManager r(device_);
  if (!r.isEnabled()) {
    GTEST_SKIP();
  }
  r.add(buffer_);
  EXPECT_EQ(r.refcountForTesting(buffer_), 1);
  r.remove(buffer_);
  EXPECT_EQ(r.refcountForTesting(buffer_), 0);
}

// refcountForTesting on never-pinned buffer returns 0; nil-safe.
TEST_F(ResidencyManagerTest, RefcountForTestingNilOrUnknown) {
  ResidencyManager r(device_);
  if (!r.isEnabled()) {
    GTEST_SKIP();
  }
  EXPECT_EQ(r.refcountForTesting(nil), 0);
  EXPECT_EQ(r.refcountForTesting(buffer_), 0);
}

//===----------------------------------------------------------------------===//
// pinHeap / unpinHeap.
//===----------------------------------------------------------------------===//

TEST_F(ResidencyManagerTest, PinHeapUnpinHeapNullSafe) {
  ResidencyManager r(device_);
  // Should not crash on nil regardless of enabled state.
  r.pinHeap(nil);
  r.unpinHeap(nil);
}

TEST_F(ResidencyManagerTest, PinHeapBasic) {
  ResidencyManager r(device_);
  if (!r.isEnabled()) {
    GTEST_SKIP();
  }
  if (@available(macOS 15.0, iOS 18.0, *)) {
    MTLHeapDescriptor* desc = [[MTLHeapDescriptor alloc] init];
    desc.size = 64 * 1024;
    desc.storageMode = MTLStorageModeShared;
    id<MTLHeap> heap = [device_ newHeapWithDescriptor:desc];
    [desc release];
    if (!heap) {
      GTEST_SKIP() << "newHeapWithDescriptor failed";
    }
    r.pinHeap(heap);    // must not crash; also flips dirty_
    r.commit();         // flush
    r.unpinHeap(heap);  // must not crash
    r.commit();
    [heap release];
  }
}

//===----------------------------------------------------------------------===//
// nudgeResidency: re-issues requestResidency. No observable state change
// other than not crashing.
//===----------------------------------------------------------------------===//

TEST_F(ResidencyManagerTest, NudgeResidencyDoesNotCrash) {
  ResidencyManager r(device_);
  r.nudgeResidency();  // disabled or enabled; both should be safe
  r.pin(buffer_);
  r.commit();
  r.nudgeResidency();
  r.unpin(buffer_);
}

//===----------------------------------------------------------------------===//
// addQueueResidency / removeQueueResidency symmetry.
//===----------------------------------------------------------------------===//

TEST_F(ResidencyManagerTest, AddRemoveQueueResidency) {
  ResidencyManager r(device_);
  id<MTLCommandQueue> queue = [device_ newCommandQueue];
  ASSERT_NE(queue, nil);
  // Disabled-path: both calls are safe no-ops.
  // Enabled-path: addResidencySet:/removeResidencySet: succeed without throw.
  r.addQueueResidency(queue);
  r.removeQueueResidency(queue);
  // Idempotent w.r.t. duplicate add (Apple docs allow it).
  r.addQueueResidency(queue);
  r.addQueueResidency(queue);
  r.removeQueueResidency(queue);
  [queue release];
}

//===----------------------------------------------------------------------===//
// stats() counters reflect activity.
//===----------------------------------------------------------------------===//

TEST_F(ResidencyManagerTest, StatsCountersIncrement) {
  ResidencyManager r(device_);
  if (!r.isEnabled()) {
    GTEST_SKIP();
  }
  auto before = r.stats();
  r.pin(buffer_);
  r.unpin(buffer_);
  auto after = r.stats();
  EXPECT_GE(after.pin_calls, before.pin_calls + 1);
  EXPECT_GE(after.unpin_calls, before.unpin_calls + 1);
  EXPECT_GE(after.mu_acquired, before.mu_acquired + 2);
}

TEST_F(ResidencyManagerTest, StatsCountersBatched) {
  ResidencyManager r(device_);
  if (!r.isEnabled()) {
    GTEST_SKIP();
  }
  constexpr int N = 4;
  id<MTLBuffer> bufs[N];
  for (int i = 0; i < N; ++i) {
    bufs[i] = [device_ newBufferWithLength:128
                                   options:MTLResourceStorageModeShared];
  }
  auto before = r.stats();
  r.pinBatch(bufs, N);
  r.unpinBatch(bufs, N);
  auto after = r.stats();
  // pinBatch + unpinBatch each contribute N to pin_calls/unpin_calls.
  EXPECT_GE(after.pin_calls, before.pin_calls + N);
  EXPECT_GE(after.unpin_calls, before.unpin_calls + N);
  // Two mutex acquisitions (one per batched call), regardless of N.
  EXPECT_GE(after.mu_acquired, before.mu_acquired + 2);
}

}  // namespace
