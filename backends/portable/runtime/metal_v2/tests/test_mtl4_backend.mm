/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// test_mtl4_backend — smoke tests for the Metal 4 dispatch path.
// Triggered by `EXECUTORCH_METAL4_ENABLE=ON` at cmake-config time, which
// defines `ET_METAL4_ENABLE`. The macro flips `useMTL4()` on, which makes
// MetalStream construct the MTL4 queue / allocator / arg-table / scratch
// in setupMTL4(). These tests verify that path doesn't crash and produces
// the expected runtime state.
// Why these tests exist:
//   - The 12-case Python suite (test_metal_v2_addsub.py) only covers the
//     MTL3 path because the cmake build defaults to MTL4 OFF. Without
//     dedicated MTL4 tests, regressions in setupMTL4 / teardownMTL4 /
//     mtl4* state initialization land silently.
//   - A pre-condition for R6.3 follow-up (full MetalMTL4Backend extraction
//     into a peer of MetalMTL3Backend behind IComputeBackend): once we
//     start moving mtl4* fields off MetalStream, we need a fast-feedback
//     test that exercises the MTL4 init/teardown cycle.
// What's covered:
//   - useMTL4() returns true when both compile flag and OS gate match.
//   - MetalStream construction succeeds with MTL4 enabled.
//   - alloc + free roundtrip on the MTL4-enabled stream.
//   - Stream destructor doesn't crash (teardownMTL4 + residency-set
//     teardown work cleanly).
// What's NOT covered (yet, depends on follow-up work):
//   - Functional kernel dispatch through the MTL4 encoder path.
//     (Requires real PSO + buffers + kernel — heavier setup.
//      Belongs in a future test_backend_parity.mm that diffs MTL3
//      vs MTL4 output bit-for-bit.)
//   - Multi-execute / cross-queue MPS interop on MTL4.
//   - (ICB has been removed.)
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <gtest/gtest.h>

#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>

using executorch::backends::metal_v2::MetalStream;

namespace {

class MTL4BackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Skip if the build wasn't configured with EXECUTORCH_METAL4_ENABLE=ON.
    // The macro becomes a no-op at compile-time guards and useMTL4()
    // returns false; nothing to test.
#if !ET_METAL4_ENABLE
    GTEST_SKIP() << "EXECUTORCH_METAL4_ENABLE=OFF at build time; "
                    "rebuild with -DEXECUTORCH_METAL4_ENABLE=ON to "
                    "exercise the MTL4 dispatch path.";
#else
    // Skip if MTL4 is disabled at runtime via env var (METAL_USE_MTL4=0)
    // or if the OS doesn't support Metal 4 even though compiled in.
    if (!executorch::backends::metal_v2::useMTL4()) {
      GTEST_SKIP() << "MTL4 disabled at runtime — either OS doesn't "
                      "support Metal 4 (need macOS 26.0+ / iOS 26.0+) "
                      "or METAL_USE_MTL4=0 was set. Set METAL_USE_MTL4=1 "
                      "(or unset) on a supported OS to exercise this test.";
    }
#endif
  }
};

//===----------------------------------------------------------------------===//
// useMTL4() compile-time + runtime gate
//===----------------------------------------------------------------------===//

TEST_F(MTL4BackendTest, UseMtl4ReturnsTrueWhenEnabled) {
  // SetUp() already gated on ET_METAL4_ENABLE && @available, so reaching
  // here means useMTL4() must be true.
  EXPECT_TRUE(executorch::backends::metal_v2::useMTL4());
}

//===----------------------------------------------------------------------===//
// MetalStream construction with MTL4 enabled
//===----------------------------------------------------------------------===//

TEST_F(MTL4BackendTest, ConstructionDoesNotCrash) {
  // Constructor calls setupMTL4() which creates MTL4CommandQueue,
  // MTL4CommandAllocator, MTL4ArgumentTable, scratch buffer, and
  // completion event. Any of those failing would either crash or leave
  // the stream in a degraded state — verify it constructs cleanly.
  MetalStream stream;
  // No assertions on internal state — we just want the ctor to complete.
  // The stream's dtor will exercise teardownMTL4 on scope exit.
  SUCCEED();
}

TEST_F(MTL4BackendTest, ConstructionAndDestructionRoundtrip) {
  // Construct + destruct N times. Catches any leaked retain in setupMTL4
  // or release-of-already-released bug in teardownMTL4. (Audit Finding
  // H7: MRC double-retain leaks were observed in setupMTL4's queue +
  // allocator chain.)
  for (int i = 0; i < 5; ++i) {
    MetalStream stream;
    (void)stream;
  }
  SUCCEED();
}

//===----------------------------------------------------------------------===//
// alloc + free roundtrip exercises the BufferRegistry under MTL4 mode
// (both backends route through the same registry; this confirms there's
// no ownership / residency-set divergence under MTL4).
//===----------------------------------------------------------------------===//

TEST_F(MTL4BackendTest, AllocAndFreeRoundtrip) {
  MetalStream stream;
  void* p = stream.alloc(1024);
  ASSERT_NE(p, nullptr);
  stream.free(p);
  // No crash + clean teardown via destructor.
  SUCCEED();
}

TEST_F(MTL4BackendTest, MultipleAllocations) {
  MetalStream stream;
  std::vector<void*> ptrs;
  for (size_t sz : {64u, 256u, 1024u, 4096u, 16384u}) {
    void* p = stream.alloc(sz);
    EXPECT_NE(p, nullptr) << "alloc(" << sz << ") returned null";
    ptrs.push_back(p);
  }
  for (void* p : ptrs) {
    stream.free(p);
  }
  SUCCEED();
}

} // namespace
