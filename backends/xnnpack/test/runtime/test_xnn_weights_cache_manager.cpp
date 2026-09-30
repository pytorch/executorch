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

#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
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

  void TearDown() override {
    for (const auto& path : temp_paths_) {
      std::remove(path.c_str());
    }
  }

  // Unique per test and per process. A leftover file from an earlier run
  // flips initialize_for_runtime between the load and fresh-create branches,
  // and two concurrent runs would race on the same path.
  std::string TempPath(const char* tag) {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    auto path = std::string(::testing::TempDir()) + "xnnwc_" + info->name() +
        "_" + tag + "_" + std::to_string(static_cast<long long>(::getpid())) +
        ".bin";
    std::remove(path.c_str());
    temp_paths_.push_back(path);
    return path;
  }

  std::unique_ptr<XNNWeightsCacheManager> manager_;
  std::vector<std::string> temp_paths_;
};

// --- Core dedup semantics ---

TEST_F(XNNWeightsCacheManagerTest, SamePathReturnsSameInstance) {
  auto a = manager_->get_or_create(TempPath("same"));
  auto b = manager_->get_or_create(TempPath("same"));
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(a.get().get(), b.get().get())
      << "same path must return the same shared instance";
}

TEST_F(XNNWeightsCacheManagerTest, DifferentPathsReturnDifferentInstances) {
  auto a = manager_->get_or_create(TempPath("a"));
  auto b = manager_->get_or_create(TempPath("b"));
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
  auto mmap = manager_->get_or_create(TempPath("isolation"));
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
    auto a = manager_->get_or_create(TempPath("expire"));
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
    auto a = manager_->get_or_create(TempPath("recreate"));
    ASSERT_TRUE(a.ok());
    first_addr = a.get().get();
  }
  // Address re-use is allowed but not required; the only guarantee is
  // that we get a usable instance, not a dangling shared_ptr.
  auto b = manager_->get_or_create(TempPath("recreate"));
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
  // Resolve the path up front: TempPath() appends to temp_paths_, which is
  // not safe to call from the racing threads.
  const std::string race_path = TempPath("race");
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([this, &results, &ready, &race_path, i] {
      // Spin to maximize the chance of true concurrent entry into
      // get_or_create.
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (ready.load(std::memory_order_acquire) < kThreads) {
        std::this_thread::yield();
      }
      auto r = manager_->get_or_create(race_path);
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
  std::vector<std::string> paths;
  paths.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    paths.push_back(TempPath(("diff_" + std::to_string(i)).c_str()));
  }
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([this, &results, &paths, i] {
      auto r = manager_->get_or_create(paths[i]);
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
  auto a = manager_->get_or_create(TempPath("save_a"));
  auto b = manager_->get_or_create(TempPath("save_b"));
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
    auto a = manager_->get_or_create(TempPath("save_expired"));
    ASSERT_TRUE(a.ok());
  }
  // The entry's weak_ptr is now expired. save_all must not crash on
  // the dead entry; opportunistically erases it.
  EXPECT_EQ(manager_->save_all(), Error::Ok);
  EXPECT_EQ(manager_->live_count(), 0u);
}

// --- Path is set on the instance before publishing ---

TEST_F(XNNWeightsCacheManagerTest, NonEmptyPathRegistersInMap) {
  auto a = manager_->get_or_create(TempPath("register"));
  ASSERT_TRUE(a.ok());
  EXPECT_EQ(manager_->live_count(), 1u);
}

// --- Packed-cache telemetry (host-visible fallback reporting) ---

TEST_F(XNNWeightsCacheManagerTest, StatsDisabledWhenNoCacheEverUsed) {
  const auto stats = manager_->aggregate_stats();
  EXPECT_EQ(
      stats.state,
      executorch::backends::xnnpack::delegate::PackedCacheState::Disabled);
  EXPECT_EQ(stats.last_errno, 0);
  EXPECT_EQ(stats.file_bytes, 0);
  EXPECT_EQ(stats.heap_bytes, 0);
  EXPECT_EQ(stats.mapped_bytes, 0);
}

TEST_F(XNNWeightsCacheManagerTest, StatsReportOpenFailureWithErrno) {
  // A path whose parent directory does not exist: open(O_RDWR|O_CREAT) fails
  // with ENOENT, which is the same branch a full disk takes with ENOSPC.
  auto cache = manager_->get_or_create("/nonexistent_dir_xnnwc/cache.bin");
  ASSERT_TRUE(cache.ok());
  {
    std::lock_guard<std::mutex> lock(cache.get()->mutex());
    ASSERT_EQ(cache.get()->initialize_for_runtime(nullptr, nullptr), Error::Ok)
        << "a fallback must stay non-fatal";
  }

  const auto report = manager_->report();
  EXPECT_EQ(
      report.aggregate.state,
      executorch::backends::xnnpack::delegate::PackedCacheState::HeapFallback)
      << "an unusable path must be reported as a heap fallback, not silently";

  // failure/errno are deliberately absent from the aggregate: they belong to
  // one cache. dominant_fallback names which one, even though this cache
  // never allocated (open failed before any pack).
  ASSERT_GE(report.dominant_fallback, 0);
  const auto& dominant =
      report.per_cache[static_cast<size_t>(report.dominant_fallback)].stats;
  EXPECT_EQ(
      dominant.failure,
      executorch::backends::xnnpack::delegate::PackedCacheFailure::OpenFailed);
  EXPECT_NE(dominant.last_errno, 0)
      << "errno is what distinguishes ENOSPC from a path problem";
  EXPECT_EQ(
      report.aggregate.failure,
      executorch::backends::xnnpack::delegate::PackedCacheFailure::None)
      << "the aggregate must not adopt one cache's failure";
}

TEST_F(XNNWeightsCacheManagerTest, HeapReasonIsGlobalArgmaxNotPerCache) {
  // Two caches whose local dominant reasons disagree with the global one.
  // Cache A: a failed grow plus a smaller unnamed pack. Cache B: unnamed only,
  // larger in total than A's grow contribution. Summing per cache would report
  // A's reason; summing per reason reports UnnamedConstant, which is correct.
  auto a = manager_->get_or_create(TempPath("argmax_a"));
  auto b = manager_->get_or_create(TempPath("argmax_b"));
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());

  const auto unnamed_pack = [](XNNWeightsCache* c, size_t n) {
    auto* provider = c->get();
    int dummy = 0;
    xnn_weights_cache_look_up_key key{};
    key.kernel = &dummy;
    key.bias = nullptr;
    provider->look_up(provider->context, &key);
    ASSERT_NE(provider->reserve_space(provider->context, n), nullptr);
  };

  {
    std::lock_guard<std::mutex> lock(a.get()->mutex());
    ASSERT_EQ(a.get()->initialize_for_runtime(nullptr, nullptr), Error::Ok);
    unnamed_pack(a.get().get(), 8192);
  }
  {
    std::lock_guard<std::mutex> lock(b.get()->mutex());
    ASSERT_EQ(b.get()->initialize_for_runtime(nullptr, nullptr), Error::Ok);
    unnamed_pack(b.get().get(), 16384);
    unnamed_pack(b.get().get(), 16384);
  }

  const auto report = manager_->report();
  EXPECT_EQ(
      report.aggregate.heap_reason,
      executorch::backends::xnnpack::delegate::PackedCacheHeapReason::
          UnnamedConstant);
  const auto unnamed_idx =
      static_cast<size_t>(executorch::backends::xnnpack::delegate::
                              PackedCacheHeapReason::UnnamedConstant);
  EXPECT_EQ(
      report.aggregate.heap_bytes_by_reason[unnamed_idx],
      report.aggregate.heap_bytes)
      << "per-reason totals must sum to the same heap_bytes";
}

TEST_F(XNNWeightsCacheManagerTest, PerCacheIsSortedByPathForStableIndices) {
  // dominant_fallback is an index into per_cache, so the order must not
  // depend on unordered_map iteration.
  auto z = manager_->get_or_create(TempPath("zzz"));
  auto a = manager_->get_or_create(TempPath("aaa"));
  ASSERT_TRUE(z.ok());
  ASSERT_TRUE(a.ok());

  const auto report = manager_->report();
  ASSERT_GE(report.per_cache.size(), 2u);
  for (size_t i = 1; i < report.per_cache.size(); ++i) {
    EXPECT_LE(report.per_cache[i - 1].path, report.per_cache[i].path);
  }
}

TEST_F(XNNWeightsCacheManagerTest, HeapFallbackWinsOverFileBackedInAggregate) {
  auto bad = manager_->get_or_create("/nonexistent_dir_xnnwc/cache.bin");
  auto good = manager_->get_or_create(TempPath("stats_ok"));
  ASSERT_TRUE(bad.ok());
  ASSERT_TRUE(good.ok());
  {
    std::lock_guard<std::mutex> lock(good.get()->mutex());
    ASSERT_EQ(good.get()->initialize_for_runtime(nullptr, nullptr), Error::Ok);
  }
  {
    std::lock_guard<std::mutex> lock(bad.get()->mutex());
    ASSERT_EQ(bad.get()->initialize_for_runtime(nullptr, nullptr), Error::Ok);
  }

  EXPECT_EQ(
      manager_->aggregate_stats().state,
      executorch::backends::xnnpack::delegate::PackedCacheState::HeapFallback)
      << "if any live cache fell back, the process is carrying that memory";
}

// A binary "did the file open" flag is not enough: a cache can load
// successfully and still serve most of its packed bytes from heap. These
// cover the byte accounting that distinguishes the two.

TEST_F(XNNWeightsCacheManagerTest, MappedBytesCountedWhenFileBacked) {
  auto cache = manager_->get_or_create(TempPath("bytes_mapped"));
  ASSERT_TRUE(cache.ok());
  {
    std::lock_guard<std::mutex> lock(cache.get()->mutex());
    ASSERT_EQ(cache.get()->initialize_for_runtime(nullptr, nullptr), Error::Ok);
    auto* provider = cache.get()->get();
    ASSERT_NE(provider->reserve_space(provider->context, 4096), nullptr);
  }

  const auto stats = manager_->aggregate_stats();
  EXPECT_GT(stats.mapped_bytes, 0);
  EXPECT_EQ(stats.heap_bytes, 0) << "the healthy case must report zero heap";
}

TEST_F(XNNWeightsCacheManagerTest, HeapBytesAttributedToUnnamedConstant) {
  auto cache = manager_->get_or_create(TempPath("bytes_unnamed"));
  ASSERT_TRUE(cache.ok());
  {
    std::lock_guard<std::mutex> lock(cache.get()->mutex());
    ASSERT_EQ(cache.get()->initialize_for_runtime(nullptr, nullptr), Error::Ok);
    auto* provider = cache.get()->get();
    // A look_up whose kernel pointer was never named marks the next
    // reserve_space as an unnamed constant, which routes to heap.
    int dummy = 0;
    xnn_weights_cache_look_up_key key{};
    key.kernel = &dummy;
    key.bias = nullptr;
    provider->look_up(provider->context, &key);
    ASSERT_NE(provider->reserve_space(provider->context, 4096), nullptr);
  }

  const auto stats = manager_->aggregate_stats();
  EXPECT_GT(stats.heap_bytes, 0) << "heap bytes must be counted, not hidden";
  EXPECT_EQ(
      stats.heap_reason,
      executorch::backends::xnnpack::delegate::PackedCacheHeapReason::
          UnnamedConstant);
}

TEST_F(XNNWeightsCacheManagerTest, FileBackedStateDoesNotImplyZeroHeap) {
  // The case a state flag alone reports as healthy: the file opened fine, so
  // state is FileBacked, yet packed bytes still went to heap. Only the byte
  // split makes that visible.
  auto cache = manager_->get_or_create(TempPath("bytes_split"));
  ASSERT_TRUE(cache.ok());
  {
    std::lock_guard<std::mutex> lock(cache.get()->mutex());
    ASSERT_EQ(cache.get()->initialize_for_runtime(nullptr, nullptr), Error::Ok);
    auto* provider = cache.get()->get();
    int dummy = 0;
    xnn_weights_cache_look_up_key key{};
    key.kernel = &dummy;
    key.bias = nullptr;
    provider->look_up(provider->context, &key);
    ASSERT_NE(provider->reserve_space(provider->context, 4096), nullptr);
  }

  const auto stats = manager_->aggregate_stats();
  EXPECT_EQ(
      stats.state,
      executorch::backends::xnnpack::delegate::PackedCacheState::FileBacked)
      << "the file opened, so state alone looks healthy";
  EXPECT_GT(stats.heap_bytes, 0)
      << "but heap bytes are non-zero — this is what state alone hides";
}

TEST_F(XNNWeightsCacheManagerTest, AggregateStatsTakesNoInstanceLock) {
  // aggregate_stats() must not wait on XNNWeightsCache::mutex(): that mutex is
  // held across all of xnn_create_runtime, so a telemetry read that blocked on
  // it would stall inference for the length of a model compile.
  auto cache = manager_->get_or_create(TempPath("nolock"));
  ASSERT_TRUE(cache.ok());
  std::lock_guard<std::mutex> held(cache.get()->mutex());
  const auto stats = manager_->aggregate_stats(); // must not deadlock
  EXPECT_EQ(stats.heap_bytes, 0);
}

TEST_F(XNNWeightsCacheManagerTest, EmptyPathHeapIsNotCountedAsFallback) {
  // The shared heap-only instance handed to callers that never configured a
  // path. Its heap use is intended, so it must not inflate heap_bytes for a
  // model in the same process that did opt in.
  auto opted_out = manager_->get_or_create("");
  ASSERT_TRUE(opted_out.ok());
  {
    std::lock_guard<std::mutex> lock(opted_out.get()->mutex());
    ASSERT_EQ(
        opted_out.get()->initialize_for_runtime(nullptr, nullptr), Error::Ok);
    auto* provider = opted_out.get()->get();
    int dummy = 0;
    xnn_weights_cache_look_up_key key{};
    key.kernel = &dummy;
    key.bias = nullptr;
    provider->look_up(provider->context, &key);
    ASSERT_NE(provider->reserve_space(provider->context, 4096), nullptr);
  }

  const auto stats = manager_->aggregate_stats();
  EXPECT_EQ(stats.heap_bytes, 0)
      << "a cache that never opted into file backing is not a fallback";
}

TEST_F(XNNWeightsCacheManagerTest, OptedInHeapStillCountedAlongsideOptedOut) {
  // Both kinds live at once: only the opted-in instance's heap bytes count.
  auto opted_out = manager_->get_or_create("");
  auto opted_in = manager_->get_or_create(TempPath("mixed"));
  ASSERT_TRUE(opted_out.ok());
  ASSERT_TRUE(opted_in.ok());
  for (auto* cache : {opted_out.get().get(), opted_in.get().get()}) {
    std::lock_guard<std::mutex> lock(cache->mutex());
    ASSERT_EQ(cache->initialize_for_runtime(nullptr, nullptr), Error::Ok);
    auto* provider = cache->get();
    int dummy = 0;
    xnn_weights_cache_look_up_key key{};
    key.kernel = &dummy;
    key.bias = nullptr;
    provider->look_up(provider->context, &key);
    ASSERT_NE(provider->reserve_space(provider->context, 8192), nullptr);
  }

  const auto stats = manager_->aggregate_stats();
  EXPECT_GT(stats.heap_bytes, 0) << "the opted-in instance's heap must count";
  EXPECT_EQ(
      stats.heap_reason,
      executorch::backends::xnnpack::delegate::PackedCacheHeapReason::
          UnnamedConstant)
      << "and NotOptedIn must never win the argmax";
}
