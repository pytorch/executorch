/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/XNNWeightsCache.h>
#include <executorch/backends/xnnpack/runtime/XNNWeightsCacheManager.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

using executorch::backends::xnnpack::XNNWeightsCacheManager;
using executorch::backends::xnnpack::delegate::XNNWeightsCache;
using executorch::runtime::Error;

class XNNWeightsCacheManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Log calls will abort if PAL is not initialized.
    executorch::runtime::runtime_init();
    manager_ = std::make_unique<XNNWeightsCacheManager>();
  }

  std::unique_ptr<XNNWeightsCacheManager> manager_;
};

// --- Core dedup semantics ---

TEST_F(XNNWeightsCacheManagerTest, SamePathReturnsSameInstance) {
  auto a = manager_->get_or_create("/tmp/test_cache_same.bin");
  auto b = manager_->get_or_create("/tmp/test_cache_same.bin");
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(a.get().get(), b.get().get())
      << "same path must return the same shared instance";
}

TEST_F(XNNWeightsCacheManagerTest, DifferentPathsReturnDifferentInstances) {
  auto a = manager_->get_or_create("/tmp/test_cache_a.bin");
  auto b = manager_->get_or_create("/tmp/test_cache_b.bin");
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  EXPECT_NE(a.get().get(), b.get().get())
      << "different paths must return independent instances";
}

TEST_F(XNNWeightsCacheManagerTest, EmptyPathSharedAcrossCallers) {
  auto a = manager_->get_or_create("");
  auto b = manager_->get_or_create("");
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  ASSERT_NE(a.get(), nullptr);
  ASSERT_NE(b.get(), nullptr);
  // Empty-path sharing keeps XNNPACK's name-based dedup working
  // across PTEs (otherwise each init re-packs every weight).
  EXPECT_EQ(a.get().get(), b.get().get());
  EXPECT_EQ(manager_->live_count(), 0u)
      << "empty-path sharing is kept off the path-keyed map";
}

TEST_F(XNNWeightsCacheManagerTest, EmptyPathRecreatedAfterAllRefsDrop) {
  XNNWeightsCache* first_addr = nullptr;
  {
    auto a = manager_->get_or_create("");
    ASSERT_TRUE(a.ok());
    first_addr = a.get().get();
  }
  // All shared_ptrs dropped → weak_ptr expires → next call gets a
  // fresh instance. Verifies the empty-path cache is not pinned for
  // the manager's lifetime.
  auto b = manager_->get_or_create("");
  ASSERT_TRUE(b.ok());
  EXPECT_NE(b.get().get(), first_addr);
}

TEST_F(XNNWeightsCacheManagerTest, EmptyPathDoesNotShareWithMmapPath) {
  auto empty = manager_->get_or_create("");
  auto mmap = manager_->get_or_create("/tmp/test_cache_isolation.bin");
  ASSERT_TRUE(empty.ok());
  ASSERT_TRUE(mmap.ok());
  // Empty-path cache stays separate from any mmap-path cache —
  // mmap-path caller's fd/flock state must never leak into a
  // heap-only caller's instance.
  EXPECT_NE(empty.get().get(), mmap.get().get());
  EXPECT_EQ(manager_->live_count(), 1u)
      << "only the mmap-path call registers in the path-keyed map";
}

// --- weak_ptr cleanup ---

TEST_F(XNNWeightsCacheManagerTest, ExpiredEntryDoesNotLeak) {
  {
    auto a = manager_->get_or_create("/tmp/test_cache_expire.bin");
    ASSERT_TRUE(a.ok());
    EXPECT_EQ(manager_->live_count(), 1u);
  }
  // shared_ptr dropped → weak_ptr in map is now expired. Live count
  // reports 0 even though the dead entry is still in the map.
  EXPECT_EQ(manager_->live_count(), 0u);
}

TEST_F(XNNWeightsCacheManagerTest, ExpiredEntryRecreatedOnNextCall) {
  void* first_addr = nullptr;
  {
    auto a = manager_->get_or_create("/tmp/test_cache_recreate.bin");
    ASSERT_TRUE(a.ok());
    first_addr = a.get().get();
  }
  // Address re-use is allowed but not required; the only guarantee is
  // that we get a usable instance, not a dangling shared_ptr.
  auto b = manager_->get_or_create("/tmp/test_cache_recreate.bin");
  ASSERT_TRUE(b.ok());
  ASSERT_NE(b.get(), nullptr);
  // Live count should be 1 again — the stale entry was erased and
  // replaced.
  EXPECT_EQ(manager_->live_count(), 1u);
  // Quiet the unused-variable warning when ABI prevents address reuse.
  (void)first_addr;
}

// --- Concurrent same-path returns the same instance ---

TEST_F(XNNWeightsCacheManagerTest, ConcurrentSamePathSameInstance) {
  constexpr int kThreads = 16;
  std::vector<std::shared_ptr<XNNWeightsCache>> results(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  std::atomic<int> ready{0};
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([this, &results, &ready, i] {
      // Spin to maximize the chance of true concurrent entry into
      // get_or_create.
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (ready.load(std::memory_order_acquire) < kThreads) {
        std::this_thread::yield();
      }
      auto r = manager_->get_or_create("/tmp/test_cache_race.bin");
      ASSERT_TRUE(r.ok());
      results[i] = r.get();
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  // All N threads must hold the exact same instance pointer.
  for (int i = 1; i < kThreads; ++i) {
    EXPECT_EQ(results[0].get(), results[i].get())
        << "thread " << i << " got a different instance";
  }
  EXPECT_EQ(manager_->live_count(), 1u);
}

TEST_F(XNNWeightsCacheManagerTest, ConcurrentDifferentPathsIndependent) {
  // Different paths must not block each other beyond the brief
  // meta_mutex_ window. We can't easily measure wall-clock parallelism
  // in a unit test, but we CAN assert each thread gets its own
  // instance with no collisions.
  constexpr int kThreads = 8;
  std::vector<std::shared_ptr<XNNWeightsCache>> results(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([this, &results, i] {
      std::string path = "/tmp/test_cache_diff_" + std::to_string(i) + ".bin";
      auto r = manager_->get_or_create(path);
      ASSERT_TRUE(r.ok());
      results[i] = r.get();
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (int i = 0; i < kThreads; ++i) {
    for (int j = i + 1; j < kThreads; ++j) {
      EXPECT_NE(results[i].get(), results[j].get());
    }
  }
  EXPECT_EQ(manager_->live_count(), kThreads);
}

// --- save_all walks live caches ---

TEST_F(XNNWeightsCacheManagerTest, SaveAllNoLiveInstancesIsOk) {
  EXPECT_EQ(manager_->save_all(), Error::Ok);
}

TEST_F(XNNWeightsCacheManagerTest, SaveAllWalksLiveCaches) {
  auto a = manager_->get_or_create("/tmp/test_cache_save_a.bin");
  auto b = manager_->get_or_create("/tmp/test_cache_save_b.bin");
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(manager_->live_count(), 2u);
  // Both caches are still live (held by a/b shared_ptrs above). Neither
  // has been through initialize_for_runtime, so save_packed_index
  // short-circuits on fd<0 and returns Ok.
  EXPECT_EQ(manager_->save_all(), Error::Ok);
}

TEST_F(XNNWeightsCacheManagerTest, SaveAllSkipsExpiredEntries) {
  {
    auto a = manager_->get_or_create("/tmp/test_cache_save_expired.bin");
    ASSERT_TRUE(a.ok());
  }
  // The entry's weak_ptr is now expired. save_all must not crash on
  // the dead entry; opportunistically erases it.
  EXPECT_EQ(manager_->save_all(), Error::Ok);
  EXPECT_EQ(manager_->live_count(), 0u);
}

// --- Path is set on the instance before publishing ---

TEST_F(XNNWeightsCacheManagerTest, NonEmptyPathRegistersInMap) {
  auto a = manager_->get_or_create("/tmp/test_cache_register.bin");
  ASSERT_TRUE(a.ok());
  EXPECT_EQ(manager_->live_count(), 1u);
}
