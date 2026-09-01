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
#include <limits>
#include <vector>

#include <executorch/backends/cuda/runtime/cuda_allocator.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

using executorch::backends::cuda::CudaAllocator;
using executorch::runtime::Error;
using executorch::runtime::etensor::DeviceIndex;

class CudaAllocatorTest : public testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();

    cudaError_t err = cudaGetDeviceCount(&device_count_);
    if (err != cudaSuccess || device_count_ == 0) {
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
} // namespace

// The delegate allocates from a pool it owns, so its retained memory must not
// land in the device default pool that other users of the async allocator
// share.
// The retention threshold is the whole point of owning a pool: at the default
// of zero the driver empties it on every synchronize. Nothing else in this
// suite notices a smaller value, so it is asserted directly.
TEST_F(CudaAllocatorTest, PoolRetainsMemoryWithoutLimit) {
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

  ASSERT_EQ(CudaAllocator::deallocate_async(res.get(), 0, stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(CudaAllocatorTest, AllocatesFromItsOwnPool) {
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
TEST_F(CudaAllocatorTest, ReleaseCachedMemoryReturnsPoolMemory) {
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
TEST_F(CudaAllocatorTest, ReleaseCachedMemoryKeepsLiveAllocations) {
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
TEST_F(CudaAllocatorTest, ReleaseCachedMemoryAcceptsTheAllDevicesSentinel) {
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
TEST_F(CudaAllocatorTest, ReleaseCachedMemoryReturnsGraphMemory) {
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
