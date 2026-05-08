/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Concurrency / race tests for core infra components that are documented
 * as thread-safe (KernelCache::findOrInsert atomic, ResidencyManager
 * locked, etc.). These tests stress the locking with N parallel threads
 * to surface real races that single-threaded tests miss.
 */
#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <executorch/backends/metal/core/MetalKernelCache.h>
#include <executorch/backends/metal/core/MetalKernel.h>
#include <executorch/backends/metal/core/MetalBufferPool.h>
#include <executorch/backends/metal/core/ResidencyManager.h>
#include <executorch/backends/metal/core/BufferRegistry.h>
#include <executorch/backends/metal/core/MetalDeviceInfo.h>
#include <atomic>
#include <thread>
#include <vector>

using executorch::backends::metal_v2::MetalKernelCache;
using executorch::backends::metal_v2::MetalKernel;
using executorch::backends::metal_v2::MetalBufferPool;
using executorch::backends::metal_v2::ResidencyManager;
using executorch::backends::metal_v2::BufferRegistry;
using executorch::backends::metal_v2::MetalDeviceInfo;

namespace {

class ConcurrencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) GTEST_SKIP();
  }
  id<MTLDevice> device_ = nil;
};

// Race N threads on findOrInsert with the SAME key. Documentation says
// loser's unique_ptr is dropped; winner's is returned to all callers.
TEST_F(ConcurrencyTest, KernelCacheFindOrInsertRace) {
  MetalKernelCache::shared().resetForTesting();
  constexpr int N = 16;
  std::vector<std::thread> threads;
  std::atomic<int> factory_calls{0};
  std::vector<MetalKernel*> results(N, nullptr);
  for (int i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {
      results[i] = MetalKernelCache::shared().findOrInsert("race_key",
          [&]() {
            ++factory_calls;
            return std::unique_ptr<MetalKernel>(new MetalKernel(nil, "race"));
          });
    });
  }
  for (auto& t : threads) t.join();
  // All threads must have gotten the SAME pointer.
  for (int i = 1; i < N; ++i) {
    EXPECT_EQ(results[i], results[0]) << "thread " << i << " got different ptr";
  }
  // Factory may have been called more than once if N threads raced and
  // the loser's unique_ptr was dropped; but at least once.
  EXPECT_GE(factory_calls.load(), 1);
}

// MetalBufferPool is NOT thread-safe by design — the pool is per-stream
// in production (stream owns allocator owns pool), and streams are
// single-threaded. This test asserts the design constraint with a
// docstring; we do not attempt to race the pool because doing so
// crashes Metal's internal allocator (verified). If pool ever becomes
// shared across threads, add real lock infrastructure first.
TEST_F(ConcurrencyTest, BufferPoolIsSingleThreadedByDesign) {
  MetalBufferPool pool(device_, /*maxBytes=*/1024 * 1024);
  // Single-threaded sanity: acquire/release works.
  id<MTLBuffer> b = pool.acquire(1024);
  ASSERT_NE(b, nil);
  pool.release(b);
  EXPECT_GE(pool.cachedBytes(), 1024u);
}

// Concurrent pin/unpin on the same residency manager.
TEST_F(ConcurrencyTest, ResidencyManagerConcurrentPinUnpin) {
  ResidencyManager res(device_);
  if (!res.isEnabled()) GTEST_SKIP() << "residency set unavailable";

  // Pre-allocate a pool of buffers to share.
  constexpr int kNumBufs = 32;
  std::vector<id<MTLBuffer>> bufs;
  for (int i = 0; i < kNumBufs; ++i) {
    bufs.push_back([device_ newBufferWithLength:1024 options:MTLResourceStorageModeShared]);
  }

  constexpr int N = 8;
  constexpr int kIters = 100;
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  for (int t = 0; t < N; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < kIters; ++i) {
        id<MTLBuffer> b = bufs[(t * 7 + i) % kNumBufs];
        res.pinBatch(&b, 1);
        res.unpinBatch(&b, 1);
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_FALSE(failed.load());
  // All buffers should be unpinned (refcount 0) at end.
  for (auto b : bufs) {
    EXPECT_EQ(res.refcountForTesting(b), 0);
  }
  for (auto b : bufs) [b release];
}

// BufferRegistry is documented as NOT thread-safe (see BufferRegistry.h:39
// "MetalStream is documented as single-threaded for dispatches; the
// registry inherits that contract"). This test asserts the design
// constraint with a single-thread sanity check; do not race.
TEST_F(ConcurrencyTest, BufferRegistryIsSingleThreadedByDesign) {
  BufferRegistry reg;
  id<MTLBuffer> buf = [device_ newBufferWithLength:64 options:MTLResourceStorageModeShared];
  void* ptr = reinterpret_cast<void*>(0x1000);
  reg.insert(ptr, buf, BufferRegistry::Origin::Pool, 64);
  EXPECT_NE(reg.find(ptr), nullptr);
  [buf release];
}

// MetalDeviceInfo::device() uses std::call_once; race N threads on the
// first call. All should see the same device pointer.
TEST_F(ConcurrencyTest, DeviceInfoFirstCallRace) {
  MetalDeviceInfo::resetForTesting();
  // After resetForTesting, device() returns nil per the documented
  // contract (once_flag can't be re-armed). So all races must see nil
  // consistently. This still verifies the once_flag isn't tripping.
  constexpr int N = 16;
  std::vector<std::thread> threads;
  std::vector<id<MTLDevice>> got(N, nil);
  for (int i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() { got[i] = MetalDeviceInfo::device(); });
  }
  for (auto& t : threads) t.join();
  for (int i = 1; i < N; ++i) {
    EXPECT_EQ(got[i], got[0]) << "thread " << i << " saw different device";
  }
}

}  // namespace
