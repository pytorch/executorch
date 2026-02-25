/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>

using namespace executorch::backends::cuda;
using namespace executorch::runtime;

// TODO(gasoonjia): Multiple device tests were not included due to test
// environment limitations. These tests should be added in the future when
// multi-GPU test environments are available,

class CUDAStreamGuardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();

    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available or no CUDA devices found";
    }
    device_count_ = device_count;

    ASSERT_EQ(cudaGetDevice(&original_device_), cudaSuccess);

    ASSERT_EQ(cudaStreamCreate(&test_stream1_), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&test_stream2_), cudaSuccess);
  }

  void TearDown() override {
    if (test_stream1_) {
      ASSERT_EQ(cudaStreamDestroy(test_stream1_), cudaSuccess);
    }
    if (test_stream2_) {
      ASSERT_EQ(cudaStreamDestroy(test_stream2_), cudaSuccess);
    }

    if (device_count_ > 0) {
      ASSERT_EQ(cudaSetDevice(original_device_), cudaSuccess);
    }
  }

  int device_count_ = 0;
  int original_device_ = 0;
  cudaStream_t test_stream1_ = nullptr;
  cudaStream_t test_stream2_ = nullptr;
};

TEST_F(CUDAStreamGuardTest, BasicStreamSwitching) {
  auto guard_result = CUDAStreamGuard::create(test_stream1_, 0);
  ASSERT_TRUE(guard_result.ok());
  CUDAStreamGuard guard = std::move(guard_result.get());

  EXPECT_EQ(guard.stream(), test_stream1_);
  EXPECT_EQ(guard.device_index(), 0);

  auto current_stream_result = getCurrentCUDAStream(0);
  ASSERT_TRUE(current_stream_result.ok());
  EXPECT_EQ(current_stream_result.get(), test_stream1_);

  int current_device;
  ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  EXPECT_EQ(current_device, 0);
}

TEST_F(CUDAStreamGuardTest, StreamSwitchingOnSameDevice) {
  Error err = setCurrentCUDAStream(test_stream1_, 0);
  ASSERT_EQ(err, Error::Ok);

  auto current_stream_result = getCurrentCUDAStream(0);
  ASSERT_TRUE(current_stream_result.ok());
  EXPECT_EQ(current_stream_result.get(), test_stream1_);

  {
    auto guard_result = CUDAStreamGuard::create(test_stream2_, 0);
    ASSERT_TRUE(guard_result.ok());
    CUDAStreamGuard guard = std::move(guard_result.get());

    auto new_stream_result = getCurrentCUDAStream(0);
    ASSERT_TRUE(new_stream_result.ok());
    EXPECT_EQ(new_stream_result.get(), test_stream2_);
    EXPECT_EQ(guard.stream(), test_stream2_);
  }

  auto restored_stream_result = getCurrentCUDAStream(0);
  ASSERT_TRUE(restored_stream_result.ok());
  EXPECT_EQ(restored_stream_result.get(), test_stream1_);
}

TEST_F(CUDAStreamGuardTest, NestedStreamGuards) {
  cudaStream_t initial_stream;
  ASSERT_EQ(cudaStreamCreate(&initial_stream), cudaSuccess);

  Error err = setCurrentCUDAStream(initial_stream, 0);
  ASSERT_EQ(err, Error::Ok);

  {
    auto guard1_result = CUDAStreamGuard::create(test_stream1_, 0);
    ASSERT_TRUE(guard1_result.ok());
    CUDAStreamGuard guard1 = std::move(guard1_result.get());

    auto stream_result = getCurrentCUDAStream(0);
    ASSERT_TRUE(stream_result.ok());
    EXPECT_EQ(stream_result.get(), test_stream1_);

    {
      auto guard2_result = CUDAStreamGuard::create(test_stream2_, 0);
      ASSERT_TRUE(guard2_result.ok());
      CUDAStreamGuard guard2 = std::move(guard2_result.get());

      auto stream_result2 = getCurrentCUDAStream(0);
      ASSERT_TRUE(stream_result2.ok());
      EXPECT_EQ(stream_result2.get(), test_stream2_);
    }

    auto stream_result3 = getCurrentCUDAStream(0);
    ASSERT_TRUE(stream_result3.ok());
    EXPECT_EQ(stream_result3.get(), test_stream1_);
  }

  auto final_stream_result = getCurrentCUDAStream(0);
  ASSERT_TRUE(final_stream_result.ok());
  EXPECT_EQ(final_stream_result.get(), initial_stream);

  ASSERT_EQ(cudaStreamDestroy(initial_stream), cudaSuccess);
}

TEST_F(CUDAStreamGuardTest, SameStreamNoChange) {
  Error err = setCurrentCUDAStream(test_stream1_, 0);
  ASSERT_EQ(err, Error::Ok);

  {
    auto guard_result = CUDAStreamGuard::create(test_stream1_, 0);
    ASSERT_TRUE(guard_result.ok());
    CUDAStreamGuard guard = std::move(guard_result.get());

    auto stream_result = getCurrentCUDAStream(0);
    ASSERT_TRUE(stream_result.ok());
    EXPECT_EQ(stream_result.get(), test_stream1_);
    EXPECT_EQ(guard.stream(), test_stream1_);
  }

  auto final_stream_result = getCurrentCUDAStream(0);
  ASSERT_TRUE(final_stream_result.ok());
  EXPECT_EQ(final_stream_result.get(), test_stream1_);
}

TEST_F(CUDAStreamGuardTest, StreamAccessor) {
  auto guard_result = CUDAStreamGuard::create(test_stream1_, 0);
  ASSERT_TRUE(guard_result.ok());
  CUDAStreamGuard guard = std::move(guard_result.get());

  EXPECT_EQ(guard.stream(), test_stream1_);
  EXPECT_EQ(guard.device_index(), 0);
}

TEST_F(CUDAStreamGuardTest, SetStreamMethod) {
  auto guard_result = CUDAStreamGuard::create(test_stream1_, 0);
  ASSERT_TRUE(guard_result.ok());
  CUDAStreamGuard guard = std::move(guard_result.get());

  EXPECT_EQ(guard.stream(), test_stream1_);

  Error err = guard.set_stream(test_stream2_, 0);
  EXPECT_EQ(err, Error::Ok);

  EXPECT_EQ(guard.stream(), test_stream2_);

  auto current_stream_result = getCurrentCUDAStream(0);
  ASSERT_TRUE(current_stream_result.ok());
  EXPECT_EQ(current_stream_result.get(), test_stream2_);
}

TEST_F(CUDAStreamGuardTest, MoveConstructor) {
  auto guard1_result = CUDAStreamGuard::create(test_stream1_, 0);
  ASSERT_TRUE(guard1_result.ok());
  CUDAStreamGuard guard1 = std::move(guard1_result.get());

  EXPECT_EQ(guard1.stream(), test_stream1_);
  EXPECT_EQ(guard1.device_index(), 0);

  CUDAStreamGuard guard2 = std::move(guard1);

  EXPECT_EQ(guard2.stream(), test_stream1_);
  EXPECT_EQ(guard2.device_index(), 0);

  auto current_stream_result = getCurrentCUDAStream(0);
  ASSERT_TRUE(current_stream_result.ok());
  EXPECT_EQ(current_stream_result.get(), test_stream1_);
}

TEST_F(CUDAStreamGuardTest, MoveConstructorRestoresOnlyOnce) {
  cudaStream_t initial_stream;
  ASSERT_EQ(cudaStreamCreate(&initial_stream), cudaSuccess);

  Error err = setCurrentCUDAStream(initial_stream, 0);
  ASSERT_EQ(err, Error::Ok);

  {
    auto guard1_result = CUDAStreamGuard::create(test_stream1_, 0);
    ASSERT_TRUE(guard1_result.ok());
    CUDAStreamGuard guard1 = std::move(guard1_result.get());

    { CUDAStreamGuard guard2 = std::move(guard1); }

    auto stream_result = getCurrentCUDAStream(0);
    ASSERT_TRUE(stream_result.ok());
    EXPECT_EQ(stream_result.get(), initial_stream);
  }

  auto final_stream_result = getCurrentCUDAStream(0);
  ASSERT_TRUE(final_stream_result.ok());
  EXPECT_EQ(final_stream_result.get(), initial_stream);

  ASSERT_EQ(cudaStreamDestroy(initial_stream), cudaSuccess);
}

TEST_F(CUDAStreamGuardTest, InvalidDeviceIndex) {
  auto guard_result = CUDAStreamGuard::create(test_stream1_, 999);
  EXPECT_FALSE(guard_result.ok());
}

TEST_F(CUDAStreamGuardTest, NegativeDeviceIndex) {
  auto guard_result = CUDAStreamGuard::create(test_stream1_, -2);
  EXPECT_FALSE(guard_result.ok());
}

TEST_F(CUDAStreamGuardTest, CopyConstructorDeleted) {
  static_assert(
      !std::is_copy_constructible_v<CUDAStreamGuard>,
      "CUDAStreamGuard should not be copy constructible");
}

TEST_F(CUDAStreamGuardTest, CopyAssignmentDeleted) {
  static_assert(
      !std::is_copy_assignable_v<CUDAStreamGuard>,
      "CUDAStreamGuard should not be copy assignable");
}

TEST_F(CUDAStreamGuardTest, MoveAssignmentDeleted) {
  static_assert(
      !std::is_move_assignable_v<CUDAStreamGuard>,
      "CUDAStreamGuard should not be move assignable");
}

TEST_F(CUDAStreamGuardTest, NullStreamPointer) {
  auto guard_result = CUDAStreamGuard::create(nullptr, 0);
  ASSERT_TRUE(guard_result.ok());
  CUDAStreamGuard guard = std::move(guard_result.get());

  EXPECT_EQ(guard.stream(), nullptr);

  auto current_stream_result = getCurrentCUDAStream(0);
  ASSERT_TRUE(current_stream_result.ok());
}
