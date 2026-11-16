/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The delegate selects a stream and the shims read it back, so both have to
// share one current-stream state. A Windows build without
// EXECUTORCH_BUILD_SHARED is where two copies could appear: this program links
// the static backend, and the shims sit in their own DLL.

#include <gtest/gtest.h>

#include <cstdlib>
#include <thread>

#include <executorch/backends/cuda/runtime/shims/cuda_guard.h>
#include <executorch/runtime/platform/platform.h>

using executorch::backends::cuda::aoti_torch_get_current_cuda_stream;
using executorch::backends::cuda::clearCurrentCUDAStream;
using executorch::backends::cuda::setCurrentCUDAStream;
using executorch::runtime::Error;

class CudaCurrentStreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      if (std::getenv("EXECUTORCH_CUDA_TEST_REQUIRE_NO_MEMORY_POOLS") !=
          nullptr) {
        FAIL() << "this job expects a CUDA device, but none is available";
      }
      GTEST_SKIP() << "CUDA not available";
    }
  }
};

TEST_F(CudaCurrentStreamTest, ShimsSeeTheStreamTheBackendSelects) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(setCurrentCUDAStream(stream, 0), Error::Ok);
  void* seen = nullptr;
  EXPECT_EQ(aoti_torch_get_current_cuda_stream(0, &seen), Error::Ok);
  EXPECT_EQ(seen, static_cast<void*>(stream));

  clearCurrentCUDAStream(0);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// The delegate runs on cudaStreamPerThread when the caller selects no stream,
// so a thread that has selected none uses it too.
TEST_F(CudaCurrentStreamTest, ShimsUseThePerThreadStreamWhenNoneIsSelected) {
  void* seen = nullptr;
  std::thread([&seen] {
    EXPECT_EQ(aoti_torch_get_current_cuda_stream(0, &seen), Error::Ok);
  }).join();
  EXPECT_EQ(seen, static_cast<void*>(cudaStreamPerThread));
}
