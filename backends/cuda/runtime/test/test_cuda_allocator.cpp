/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include <executorch/backends/cuda/runtime/cuda_allocator.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

using executorch::backends::cuda::CudaAllocator;
using executorch::runtime::Error;

class CudaAllocatorTest : public testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }
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
