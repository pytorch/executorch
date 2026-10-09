/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/extension/cuda/runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__linux__) && !defined(EXECUTORCH_USE_HIP)
#include <dlfcn.h>
#endif

#include <executorch/backends/cuda/runtime/cuda_allocator.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

using executorch::backends::cuda::CudaAllocator;
using executorch::backends::cuda::CudaGraphPhase;
using executorch::backends::cuda::CudaGraphState;
using executorch::runtime::Error;
using executorch::runtime::etensor::DeviceIndex;

#if defined(__linux__) && !defined(EXECUTORCH_USE_HIP)
// Force rounding independently of how the GPU packs its allocations.
namespace {
alignas(4096) thread_local char fake_storage[16384];
thread_local void* fake_block = fake_storage + 256;
thread_local int fake_malloc_remaining = 0;
thread_local void* freed_fake_block = nullptr;
thread_local size_t fake_malloc_size = 0;
thread_local int fake_free_count = 0;
} // namespace

// Weak, so the real ones win if a static CUDA runtime is linked in. The linker
// can still leave them out, and then a stand-in has nothing to call.
extern "C" __attribute__((weak)) cudaError_t
cudaMalloc(void** ptr, size_t size) {
  if (fake_malloc_remaining > 0) {
    --fake_malloc_remaining;
    *ptr = fake_block;
    fake_malloc_size = size;
    return cudaSuccess;
  }
  static const auto real = reinterpret_cast<cudaError_t (*)(void**, size_t)>(
      dlsym(RTLD_NEXT, "cudaMalloc"));
  if (real == nullptr) {
    ADD_FAILURE() << "the test replaced cudaMalloc but found no real one";
    return cudaErrorUnknown;
  }
  return real(ptr, size);
}

extern "C" __attribute__((weak)) cudaError_t cudaFree(void* ptr) {
  const auto address = reinterpret_cast<uintptr_t>(ptr);
  const auto begin = reinterpret_cast<uintptr_t>(fake_storage);
  if (address >= begin && address < begin + sizeof(fake_storage)) {
    freed_fake_block = ptr;
    ++fake_free_count;
    return cudaSuccess;
  }
  static const auto real =
      reinterpret_cast<cudaError_t (*)(void*)>(dlsym(RTLD_NEXT, "cudaFree"));
  if (real == nullptr) {
    ADD_FAILURE() << "the test replaced cudaFree but found no real one";
    return cudaErrorUnknown;
  }
  return real(ptr);
}
#endif

class CudaAllocatorTest : public testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();

    cudaError_t err = cudaGetDeviceCount(&device_count_);
    if (err != cudaSuccess || device_count_ == 0) {
      // A job that requires a device without memory pools requires a device:
      // skipping here would leave the fallback untested while the job passes.
      if (std::getenv("EXECUTORCH_CUDA_TEST_REQUIRE_NO_MEMORY_POOLS") !=
          nullptr) {
        FAIL() << "this job expects a CUDA device, but none is available";
      }
      GTEST_SKIP() << "CUDA not available";
    }
  }

  // The pool is meant to stay warm, so without this a test that measured
  // reserved bytes would see whatever an earlier one left behind, and the order
  // would matter.
  void TearDown() override {
    if (device_count_ > 0) {
      CudaAllocator::release_cached_memory(-1);
    }
  }

  // One past the last valid device ordinal, so switching to it always fails.
  // Only the tests that need such an ordinal call this, so the fit check lives
  // here rather than in SetUp, where it would also skip the device-0 tests.
  DeviceIndex missing_device() const {
    return static_cast<DeviceIndex>(device_count_);
  }

  // missing_device() has to stay a valid-but-absent ordinal. DeviceIndex is
  // int8_t, so on a host with more than 127 visible GPUs the count wraps to a
  // negative index (which the >= -1 argument check rejects for a different
  // reason) or, at 256, back onto real device 0.
  bool missing_device_fits() const {
    return device_count_ <= std::numeric_limits<DeviceIndex>::max();
  }

  int device_count_ = 0;
};

// A distinct backend class would bring back a second allocator singleton.
static_assert(
    std::is_same_v<CudaAllocator, executorch::extension::cuda::CudaAllocator>);

TEST(CudaAllocatorCompatibilityTest, OnlyTheSingletonCanBeConstructed) {
  EXPECT_FALSE(std::is_default_constructible_v<CudaAllocator>);
  EXPECT_FALSE(std::is_copy_constructible_v<CudaAllocator>);
  EXPECT_FALSE(std::is_move_constructible_v<CudaAllocator>);
  EXPECT_FALSE(std::is_copy_assignable_v<CudaAllocator>);
  EXPECT_FALSE(std::is_move_assignable_v<CudaAllocator>);
}

TEST_F(CudaAllocatorTest, CopyRoundtrip) {
  CudaAllocator& a = CudaAllocator::instance();
  constexpr size_t N = 1024;
  auto res = a.allocate(N, 0);
  ASSERT_TRUE(res.ok());
  void* dptr = res.get();

  std::vector<uint8_t> h_src(N, 42), h_dst(N, 0);
  ASSERT_EQ(a.copy_host_to_device(dptr, h_src.data(), N, 0), Error::Ok);
  EXPECT_EQ(a.copy_device_to_host(h_dst.data(), dptr, N, 0), Error::Ok);
  EXPECT_EQ(h_src, h_dst);

  a.deallocate(dptr, 0);
}

TEST_F(CudaAllocatorTest, CopyRoundtripWithCallerStream) {
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(device, 0) << "test assumes single GPU device 0";
  // TODO: validate caller stream device matches index once CallerStreamGuard
  // exposes device. For now assert single-GPU case.
  cudaStream_t s;
  ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
  {
    executorch::extension::cuda::CallerStreamGuard g(s);

    CudaAllocator& a = CudaAllocator::instance();
    auto res = a.allocate(256, 0);
    ASSERT_TRUE(res.ok());
    void* d = res.get();
    std::vector<uint8_t> h_src(256, 5), h_dst(256, 0);
    ASSERT_EQ(a.copy_host_to_device(d, h_src.data(), 256, 0), Error::Ok);
    EXPECT_EQ(a.copy_device_to_host(h_dst.data(), d, 256, 0), Error::Ok);
    EXPECT_EQ(h_src, h_dst);
    EXPECT_EQ(cudaStreamSynchronize(s), cudaSuccess);

    a.deallocate(d, 0);
  }
  ASSERT_EQ(cudaStreamDestroy(s), cudaSuccess);
}

TEST_F(CudaAllocatorTest, CopyHostToDeviceNullDstReturnsInvalidArgument) {
  CudaAllocator& a = CudaAllocator::instance();
  // null dst should fail gracefully not CHECK abort
  std::vector<uint8_t> h(8, 1);
  Error e = a.copy_host_to_device(nullptr, h.data(), 8, 0);
  EXPECT_EQ(e, Error::InvalidArgument)
      << "expected InvalidArgument for null dst, got "
      << static_cast<uint32_t>(e);
}

TEST_F(CudaAllocatorTest, CopyHostToDeviceNullSrcReturnsInvalidArgument) {
  CudaAllocator& a = CudaAllocator::instance();
  void* dummy_dst = reinterpret_cast<void*>(0x1);
  Error e = a.copy_host_to_device(dummy_dst, nullptr, 8, 0);
  EXPECT_EQ(e, Error::InvalidArgument)
      << "expected InvalidArgument for null src, got "
      << static_cast<uint32_t>(e);
}

TEST_F(CudaAllocatorTest, CopyDeviceToHostNullDstReturnsInvalidArgument) {
  CudaAllocator& a = CudaAllocator::instance();
  void* dummy_src = reinterpret_cast<void*>(0x1);
  Error e = a.copy_device_to_host(nullptr, dummy_src, 8, 0);
  EXPECT_EQ(e, Error::InvalidArgument)
      << "expected InvalidArgument for null dst, got "
      << static_cast<uint32_t>(e);
}

TEST_F(CudaAllocatorTest, CopyDeviceToHostNullSrcReturnsInvalidArgument) {
  CudaAllocator& a = CudaAllocator::instance();
  std::vector<uint8_t> h(8, 1);
  // null src should fail gracefully not CHECK abort
  Error e = a.copy_device_to_host(h.data(), nullptr, 8, 0);
  EXPECT_EQ(e, Error::InvalidArgument)
      << "expected InvalidArgument for null src, got "
      << static_cast<uint32_t>(e);
}

TEST_F(CudaAllocatorTest, AllocateOnMissingDeviceFails) {
  if (!missing_device_fits()) {
    GTEST_SKIP() << "device count " << device_count_
                 << " leaves no absent ordinal in DeviceIndex";
  }
  CudaAllocator& a = CudaAllocator::instance();
  auto res = a.allocate(1024, missing_device());
  ASSERT_FALSE(res.ok()) << "allocate must not report success for device "
                         << static_cast<int>(missing_device())
                         << ", which does not exist";
  EXPECT_EQ(res.error(), Error::Internal);
}

TEST_F(CudaAllocatorTest, LargeAlignmentsWithLiveBlocksRoundtrip) {
  (void)cudaGetLastError();
  CudaAllocator& a = CudaAllocator::instance();
  for (int cycle = 0; cycle < 64; ++cycle) {
    std::vector<void*> live;
    for (size_t alignment : {512, 4096}) {
      for (size_t nbytes : {1, 256, 512, 1000, 2048, 8192, 65536}) {
        SCOPED_TRACE(
            testing::Message() << cycle << ": " << alignment << ", " << nbytes);
        auto res = a.allocate(nbytes, 0, alignment);
        EXPECT_TRUE(res.ok());
        if (!res.ok()) {
          for (void* ptr : live) {
            a.deallocate(ptr, 0);
          }
          return;
        }
        void* ptr = res.get();
        live.push_back(ptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, 0u);
        std::vector<uint8_t> src(nbytes, cycle + live.size()), dst(nbytes, 0);
        EXPECT_EQ(a.copy_host_to_device(ptr, src.data(), nbytes, 0), Error::Ok);
        EXPECT_EQ(a.copy_device_to_host(dst.data(), ptr, nbytes, 0), Error::Ok);
        EXPECT_EQ(src, dst);
      }
    }
    for (void* ptr : live) {
      a.deallocate(ptr, 0);
      EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }
  }
}

TEST_F(CudaAllocatorTest, ConcurrentAlignedAllocations) {
  std::vector<std::thread> threads;
  threads.reserve(4);
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([] {
      CudaAllocator& a = CudaAllocator::instance();
      for (int cycle = 0; cycle < 32; ++cycle) {
        for (size_t alignment : {256, 4096}) {
          auto res = a.allocate(1000, 0, alignment);
          ASSERT_TRUE(res.ok());
          EXPECT_EQ(reinterpret_cast<uintptr_t>(res.get()) % alignment, 0u);
          a.deallocate(res.get(), 0);
          EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

// A device call would fail with Internal instead of the size validation error.
TEST_F(CudaAllocatorTest, PaddingOverflowFailsBeforeAnyCudaCall) {
  if (!missing_device_fits()) {
    GTEST_SKIP() << "device count " << device_count_
                 << " leaves no absent ordinal in DeviceIndex";
  }
  CudaAllocator& a = CudaAllocator::instance();
  for (size_t alignment : {512, 4096}) {
    const size_t largest = std::numeric_limits<size_t>::max() - (alignment - 1);
    auto overflow = a.allocate(largest + 1, missing_device(), alignment);
    ASSERT_FALSE(overflow.ok());
    EXPECT_EQ(overflow.error(), Error::InvalidArgument);

    auto fits = a.allocate(largest, missing_device(), alignment);
    ASSERT_FALSE(fits.ok());
    EXPECT_EQ(fits.error(), Error::Internal);
  }
}

#if defined(__linux__) && !defined(EXECUTORCH_USE_HIP)
class CudaAllocatorStandInTest : public CudaAllocatorTest {
 protected:
  void SetUp() override {
    CudaAllocatorTest::SetUp();
    if (IsSkipped()) {
      return;
    }
    auto& a = CudaAllocator::instance();
    fake_block = fake_storage + 256;
    fake_malloc_remaining = 1;
    freed_fake_block = nullptr;
    auto probe = a.allocate(1024, 0);
    const bool malloc_replaced = fake_malloc_remaining == 0;
    fake_malloc_remaining = 0;
    if (probe.ok()) {
      a.deallocate(probe.get(), 0);
    }
    if (!malloc_replaced || freed_fake_block != fake_block) {
      GTEST_SKIP()
          << "cudaMalloc and cudaFree cannot be replaced in this build";
    }
    fake_malloc_size = 0;
    fake_free_count = 0;
  }

  void TearDown() override {
    fake_malloc_remaining = 0;
    CudaAllocatorTest::TearDown();
  }
};

// A free inside allocate() would wait for all work queued on the device.
TEST_F(CudaAllocatorStandInTest, AllocateMakesOneCallAndNeverFrees) {
  auto& a = CudaAllocator::instance();
  for (size_t alignment : {64, 256, 512, 4096}) {
    for (size_t offset : {0, 256}) {
      SCOPED_TRACE(testing::Message() << alignment << ", " << offset);
      fake_block = fake_storage + offset;
      fake_malloc_remaining = 1;
      fake_malloc_size = 0;
      fake_free_count = 0;
      auto res = a.allocate(1024, 0, alignment);
      ASSERT_TRUE(res.ok());
      EXPECT_EQ(fake_malloc_remaining, 0);
      EXPECT_EQ(fake_free_count, 0);
      EXPECT_EQ(
          fake_malloc_size, alignment > 256 ? 1024 + alignment - 1 : 1024);
      EXPECT_EQ(reinterpret_cast<uintptr_t>(res.get()) % alignment, 0u);
      a.deallocate(res.get(), 0);
      EXPECT_EQ(freed_fake_block, fake_block);
      EXPECT_EQ(fake_free_count, 1);
    }
  }
}

// cudaMalloc promises 256 bytes, so a block that misses a smaller alignment
// is a broken runtime, not something padding should hide.
TEST_F(CudaAllocatorStandInTest, MisalignedBlockAtSmallAlignmentIsRefused) {
  auto& a = CudaAllocator::instance();
  fake_block = fake_storage + 128;
  fake_malloc_remaining = 1;
  fake_malloc_size = 0;
  fake_free_count = 0;
  auto res = a.allocate(1024, 0, 256);
  ASSERT_FALSE(res.ok());
  EXPECT_EQ(res.error(), Error::NotSupported);
  EXPECT_EQ(fake_malloc_size, 1024u);
  EXPECT_EQ(fake_free_count, 1);
  EXPECT_EQ(freed_fake_block, fake_block);
}

TEST_F(CudaAllocatorStandInTest, RoundedAllocationFreesOriginalPointer) {
  auto& a = CudaAllocator::instance();
  for (size_t alignment : {512, 4096}) {
    for (int cycle = 0; cycle < 64; ++cycle) {
      fake_block = fake_storage + 256;
      fake_malloc_remaining = 1;
      fake_free_count = 0;
      auto res = a.allocate(1024, 0, alignment);
      ASSERT_TRUE(res.ok());
      EXPECT_EQ(res.get(), fake_storage + alignment);
      a.deallocate(res.get(), 0);
      EXPECT_EQ(freed_fake_block, fake_block);
      EXPECT_EQ(fake_free_count, 1);
    }

    // A second live rounded block keeps the lock-free path from skipping the
    // lookup, so a stale entry for the reused address would be found.
    fake_block = fake_storage + alignment + 256;
    fake_malloc_remaining = 1;
    auto live = a.allocate(1024, 0, alignment);
    ASSERT_TRUE(live.ok());
    fake_block = fake_storage + alignment;
    fake_malloc_remaining = 1;
    auto reused = a.allocate(1024, 0, alignment);
    ASSERT_TRUE(reused.ok());
    EXPECT_EQ(reused.get(), fake_block);
    a.deallocate(reused.get(), 0);
    EXPECT_EQ(freed_fake_block, fake_block);
    a.deallocate(live.get(), 0);
    EXPECT_EQ(freed_fake_block, fake_storage + alignment + 256);
  }
}
#endif

TEST_F(CudaAllocatorTest, CopyHostToDeviceOnMissingDeviceFails) {
  if (!missing_device_fits()) {
    GTEST_SKIP() << "device count " << device_count_
                 << " leaves no absent ordinal in DeviceIndex";
  }
  CudaAllocator& a = CudaAllocator::instance();
  constexpr size_t N = 64;
  auto res = a.allocate(N, 0);
  ASSERT_TRUE(res.ok());
  void* dptr = res.get();

  std::vector<uint8_t> h(N, 7);
  EXPECT_EQ(
      a.copy_host_to_device(dptr, h.data(), N, missing_device()),
      Error::Internal);

  a.deallocate(dptr, 0);
}

TEST_F(CudaAllocatorTest, CopyDeviceToHostOnMissingDeviceFails) {
  if (!missing_device_fits()) {
    GTEST_SKIP() << "device count " << device_count_
                 << " leaves no absent ordinal in DeviceIndex";
  }
  CudaAllocator& a = CudaAllocator::instance();
  constexpr size_t N = 64;
  auto res = a.allocate(N, 0);
  ASSERT_TRUE(res.ok());
  void* dptr = res.get();

  std::vector<uint8_t> h(N, 0);
  EXPECT_EQ(
      a.copy_device_to_host(h.data(), dptr, N, missing_device()),
      Error::Internal);

  a.deallocate(dptr, 0);
}

// The pool attributes these exercise have no HIP equivalent in the
// compatibility header, and the allocator's pool code is compiled out on ROCm
// for the same reason, so there is nothing to test there.
#if !defined(EXECUTORCH_USE_HIP)

namespace {
uint64_t reserved_bytes(cudaMemPool_t pool) {
  uint64_t reserved = 0;
  EXPECT_EQ(
      cudaMemPoolGetAttribute(
          pool, cudaMemPoolAttrReservedMemCurrent, &reserved),
      cudaSuccess);
  return reserved;
}

bool device_supports_memory_pools(int device) {
  int value = 0;
  return cudaDeviceGetAttribute(
             &value, cudaDevAttrMemoryPoolsSupported, device) == cudaSuccess &&
      value != 0;
}
} // namespace

// The fallback tests only mean something on a device without memory pools,
// and skip elsewhere. A job whose runner is meant to have no pools sets this
// variable, so a runner that gains them fails instead of skipping and leaving
// the fallback untested while the job stays green.
class CudaAllocatorNoPoolTest : public CudaAllocatorTest {
 protected:
  void SetUp() override {
    CudaAllocatorTest::SetUp();
    if (IsSkipped() || !device_supports_memory_pools(0)) {
      return;
    }
    if (std::getenv("EXECUTORCH_CUDA_TEST_REQUIRE_NO_MEMORY_POOLS") !=
        nullptr) {
      FAIL() << "this job expects a device without memory pools, but device 0 "
                "supports them, so the fallback would go untested";
    }
    GTEST_SKIP() << "device 0 supports memory pools; covered by the pool tests";
  }
};

// The pool tests below only mean something on a device with memory pools.
// Data-center GPUs in TCC mode on Windows have none, and there the allocator
// takes the synchronous path the CudaAllocatorNoPoolTest cases check.
class CudaAllocatorPoolTest : public CudaAllocatorTest {
 protected:
  void SetUp() override {
    CudaAllocatorTest::SetUp();
    if (!IsSkipped() && !device_supports_memory_pools(0)) {
      GTEST_SKIP() << "device 0 does not support memory pools";
    }
  }
};

// Works on every device: with memory pools through the pool, without them
// through cudaMalloc. This is what failed on a device without pools, where
// cudaMallocAsync returned cudaErrorNotSupported and nothing fell back.
TEST_F(CudaAllocatorTest, AllocateAsyncRoundtrip) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  constexpr size_t kBytes = 1u << 20;
  auto res = CudaAllocator::allocate_async(kBytes, 0, stream);
  ASSERT_TRUE(res.ok()) << "allocate_async failed on device 0";
  std::vector<uint8_t> h_src(kBytes, 9), h_dst(kBytes, 0);
  ASSERT_EQ(
      cudaMemcpyAsync(
          res.get(), h_src.data(), kBytes, cudaMemcpyHostToDevice, stream),
      cudaSuccess);
  ASSERT_EQ(
      cudaMemcpyAsync(
          h_dst.data(), res.get(), kBytes, cudaMemcpyDeviceToHost, stream),
      cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(h_src, h_dst);

  CudaAllocator::deallocate_async(res.get(), 0, stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Without memory pools the allocation comes from cudaMalloc, and the matching
// free has to release it with cudaFree: cudaFreeAsync on that memory fails and
// leaves the block allocated, and a free skipped by mistake leaks it.
TEST_F(CudaAllocatorNoPoolTest, FallsBackWithoutMemoryPools) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  auto res = CudaAllocator::allocate_async(4096, -1, stream);
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(cudaMemsetAsync(res.get(), 0, 4096, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  CudaAllocator::deallocate_async(res.get(), -1, stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, res.get()), cudaSuccess);
  EXPECT_NE(attributes.type, cudaMemoryTypeDevice)
      << "the fallback allocation was not released by deallocate_async";

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// The fallback must not run inside a graph capture. In relaxed mode cudaMalloc
// and cudaFree succeed there, so a block freed during capture would be handed
// out again while every replay of the graph still writes to it. Refusing keeps
// the loud failure the stream-ordered call gave before.
TEST_F(CudaAllocatorNoPoolTest, FallbackRefusesWhileCapturing) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed),
      cudaSuccess);
  auto backend = CudaAllocator::allocate_async(4096, 0, stream);
  auto scratch = CudaAllocator::allocate_stream_ordered(4096, 0, stream);
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
  EXPECT_FALSE(backend.ok()) << "allocate_async fell back inside a capture";
  EXPECT_FALSE(scratch.ok())
      << "allocate_stream_ordered fell back inside a capture";
  if (graph != nullptr) {
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// The other half: a block freed while a capture records work that uses it
// must stay allocated, or the next allocation can get the same memory while
// every replay still writes to it. Both free paths, through a replay.
TEST_F(CudaAllocatorNoPoolTest, FallbackKeepsBlocksFreedWhileCapturing) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  constexpr size_t kBytes = 4096;
  auto backend = CudaAllocator::allocate_async(kBytes, 0, stream);
  auto scratch = CudaAllocator::allocate_stream_ordered(kBytes, 0, stream);
  ASSERT_TRUE(backend.ok());
  ASSERT_TRUE(scratch.ok());

  ASSERT_EQ(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed),
      cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(backend.get(), 0x5a, kBytes, stream), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(scratch.get(), 0x5a, kBytes, stream), cudaSuccess);
  CudaAllocator::deallocate_async(backend.get(), 0, stream);
  EXPECT_EQ(
      CudaAllocator::deallocate_stream_ordered(scratch.get(), 0, stream),
      Error::NotSupported)
      << "a block kept during capture must be reported, not freed silently";
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
  cudaGraphExec_t graph_exec = nullptr;
  ASSERT_EQ(
      cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
      cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  for (void* block : {backend.get(), scratch.get()}) {
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, block), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice)
        << "a block freed during capture was released while the graph uses it";
    std::vector<uint8_t> host(kBytes, 0);
    ASSERT_EQ(
        cudaMemcpy(host.data(), block, kBytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(host, std::vector<uint8_t>(kBytes, 0x5a));
  }

  ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  CudaAllocator::deallocate_async(backend.get(), 0, stream);
  EXPECT_EQ(
      CudaAllocator::deallocate_stream_ordered(scratch.get(), 0, stream),
      Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// A method that asks for a CUDA graph must run without one on such a device:
// its allocations would hit the capture refusal above and abort the process.
TEST_F(CudaAllocatorNoPoolTest, CudaGraphStaysOffWithoutMemoryPools) {
  CudaGraphState state;
  EXPECT_FALSE(state.start_warmup());
  EXPECT_EQ(state.phase, CudaGraphPhase::Disabled);
}

// The scratch path keeps working outside a capture on every device.
TEST_F(CudaAllocatorTest, StreamOrderedScratchRoundtrip) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  auto res = CudaAllocator::allocate_stream_ordered(4096, 0, stream);
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(cudaMemsetAsync(res.get(), 0, 4096, stream), cudaSuccess);
  EXPECT_EQ(
      CudaAllocator::deallocate_stream_ordered(res.get(), 0, stream),
      Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// With memory pools a method that asks for a CUDA graph gets one.
TEST_F(CudaAllocatorPoolTest, CudaGraphStartsWithMemoryPools) {
  CudaGraphState state;
  EXPECT_TRUE(state.start_warmup());
  EXPECT_EQ(state.phase, CudaGraphPhase::Warmup);
  EXPECT_GT(state.warmup_remaining, 0);
}

// The delegate allocates from a pool it owns, so its retained memory must not
// land in the device default pool that other users of the async allocator
// share.
// The retention threshold is the whole point of owning a pool: at the default
// of zero the driver empties it on every synchronize. Nothing else in this
// suite notices a smaller value, so it is asserted directly.
TEST_F(CudaAllocatorPoolTest, PoolRetainsMemoryWithoutLimit) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  auto res = CudaAllocator::allocate_async(8u << 20, 0, stream);
  ASSERT_TRUE(res.ok());

  cudaMemPool_t pool = CudaAllocator::pool_for_device(0);
  ASSERT_NE(pool, nullptr);

  uint64_t threshold = 0;
  ASSERT_EQ(
      cudaMemPoolGetAttribute(
          pool, cudaMemPoolAttrReleaseThreshold, &threshold),
      cudaSuccess);
  EXPECT_EQ(threshold, UINT64_MAX)
      << "the pool must hold on to freed memory rather than return it";

  CudaAllocator::deallocate_async(res.get(), 0, stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(CudaAllocatorPoolTest, AllocatesFromItsOwnPool) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  constexpr size_t kBytes = 8u << 20;
  auto res = CudaAllocator::allocate_async(kBytes, 0, stream);
  ASSERT_TRUE(res.ok());

  cudaMemPool_t owned = CudaAllocator::pool_for_device(0);
  ASSERT_NE(owned, nullptr) << "the allocator should have created its own pool";
  cudaMemPool_t default_pool = nullptr;
  ASSERT_EQ(cudaDeviceGetMemPool(&default_pool, 0), cudaSuccess);
  EXPECT_NE(owned, default_pool) << "the pool must not be the device default";

  // The live block is reserved in the owned pool, which is what identifies it
  // as the pool actually serving this allocation.
  EXPECT_GE(reserved_bytes(owned), kBytes);

  CudaAllocator::deallocate_async(res.get(), 0, stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Freed memory is kept so repeated allocation stays cheap, which means a plain
// free no longer shrinks the pool. Without an explicit release a long lived
// process would hold that memory after every program was gone.
TEST_F(CudaAllocatorPoolTest, ReleaseCachedMemoryReturnsPoolMemory) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  constexpr size_t kBytes = 8u << 20;
  auto res = CudaAllocator::allocate_async(kBytes, 0, stream);
  ASSERT_TRUE(res.ok());
  CudaAllocator::deallocate_async(res.get(), 0, stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  cudaMemPool_t owned = CudaAllocator::pool_for_device(0);
  ASSERT_NE(owned, nullptr);
  // Freed and synchronized, and still held, which is the point of the change.
  ASSERT_GT(reserved_bytes(owned), 0u)
      << "the pool should hold the freed block for reuse";

  CudaAllocator::release_cached_memory(0);

  EXPECT_EQ(reserved_bytes(owned), 0u)
      << "released memory should go back to the driver";

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Releasing must not disturb allocations that are still in use.
TEST_F(CudaAllocatorPoolTest, ReleaseCachedMemoryKeepsLiveAllocations) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  // Large enough that the two blocks land in separate driver reservations. At a
  // few megabytes they share one, so nothing can be released while either is
  // live.
  constexpr size_t kBytes = 64u << 20;
  auto live = CudaAllocator::allocate_async(kBytes, 0, stream);
  ASSERT_TRUE(live.ok());
  auto temp = CudaAllocator::allocate_async(kBytes, 0, stream);
  ASSERT_TRUE(temp.ok());
  CudaAllocator::deallocate_async(temp.get(), 0, stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  cudaMemPool_t owned = CudaAllocator::pool_for_device(0);
  ASSERT_NE(owned, nullptr);
  const uint64_t before = reserved_bytes(owned);

  CudaAllocator::release_cached_memory(0);

  // The freed block goes back and the live one stays reserved, so the pool
  // gives up only what is not in use.
  const uint64_t after = reserved_bytes(owned);
  EXPECT_LT(after, before) << "the freed block should have been released";
  EXPECT_GE(after, kBytes) << "the live block must still be reserved";

  EXPECT_EQ(cudaMemsetAsync(live.get(), 0, kBytes, stream), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  CudaAllocator::deallocate_async(live.get(), 0, stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// A negative index releases every device this backend has allocated on, not the
// current one. This runner has a single GPU, so the two cannot be told apart
// here; what it pins is that the sentinel is resolved rather than passed to the
// driver.
TEST_F(CudaAllocatorPoolTest, ReleaseCachedMemoryAcceptsTheAllDevicesSentinel) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  constexpr size_t kBytes = 8u << 20;
  auto res = CudaAllocator::allocate_async(kBytes, 0, stream);
  ASSERT_TRUE(res.ok());
  CudaAllocator::deallocate_async(res.get(), 0, stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  cudaMemPool_t owned = CudaAllocator::pool_for_device(-1);
  ASSERT_NE(owned, nullptr) << "the sentinel should resolve to this device";
  ASSERT_GT(reserved_bytes(owned), 0u)
      << "the pool should hold the freed block for reuse";

  CudaAllocator::release_cached_memory(-1);

  EXPECT_EQ(reserved_bytes(owned), 0u);

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Memory allocated during a graph capture goes to the device graph pool, which
// the pool trim cannot reach, so releasing has to trim that too. Without the
// graph trim this is the only new test that fails.
TEST_F(CudaAllocatorPoolTest, ReleaseCachedMemoryReturnsGraphMemory) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  constexpr size_t kBytes = 64u << 20;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  ASSERT_EQ(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed),
      cudaSuccess);
  auto captured = CudaAllocator::allocate_async(kBytes, 0, stream);
  ASSERT_TRUE(captured.ok());
  CudaAllocator::deallocate_async(captured.get(), 0, stream);
  ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
  ASSERT_EQ(
      cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
      cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  size_t reserved = 0;
  ASSERT_EQ(
      cudaDeviceGetGraphMemAttribute(
          0, cudaGraphMemAttrReservedMemCurrent, &reserved),
      cudaSuccess);
  ASSERT_GT(reserved, 0u) << "the capture should have reserved graph memory";

  ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);

  CudaAllocator::release_cached_memory(0);

  ASSERT_EQ(
      cudaDeviceGetGraphMemAttribute(
          0, cudaGraphMemAttrReservedMemCurrent, &reserved),
      cudaSuccess);
  EXPECT_EQ(reserved, 0u) << "graph memory should go back to the driver";

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

#endif // !EXECUTORCH_USE_HIP
