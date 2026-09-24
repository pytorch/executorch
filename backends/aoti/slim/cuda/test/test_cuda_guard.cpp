/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>

using namespace executorch::backends::cuda;
using namespace executorch::runtime;

// TODO(gasoonjia): Multiple device tests were not included due to test
// environment limitations. These tests should be added in the future when
// multi-GPU test environments are available,

class CUDAGuardTest : public ::testing::Test {
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
  }

  void TearDown() override {
    if (device_count_ > 0) {
      ASSERT_EQ(cudaSetDevice(original_device_), cudaSuccess);
    }
  }

  int device_count_ = 0;
  int original_device_ = 0;
};

TEST_F(CUDAGuardTest, BasicDeviceSwitching) {
  int current_device;
  ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);

  {
    auto guard_result = CUDAGuard::create(0);
    ASSERT_TRUE(guard_result.ok());
    CUDAGuard guard = std::move(guard_result.get());

    int device_after_guard;
    ASSERT_EQ(cudaGetDevice(&device_after_guard), cudaSuccess);
    EXPECT_EQ(device_after_guard, 0);
    EXPECT_EQ(guard.current_device(), 0);
    EXPECT_EQ(guard.original_device(), current_device);
  }

  int device_after_destruction;
  ASSERT_EQ(cudaGetDevice(&device_after_destruction), cudaSuccess);
  EXPECT_EQ(device_after_destruction, current_device);
}

TEST_F(CUDAGuardTest, SameDeviceNoSwitching) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  {
    auto guard_result = CUDAGuard::create(0);
    ASSERT_TRUE(guard_result.ok());
    CUDAGuard guard = std::move(guard_result.get());

    int current_device;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, 0);
    EXPECT_EQ(guard.current_device(), 0);
    EXPECT_EQ(guard.original_device(), 0);
  }

  int final_device;
  ASSERT_EQ(cudaGetDevice(&final_device), cudaSuccess);
  EXPECT_EQ(final_device, 0);
}

TEST_F(CUDAGuardTest, InvalidDeviceIndex) {
  auto guard_result = CUDAGuard::create(999);
  EXPECT_FALSE(guard_result.ok());
}

TEST_F(CUDAGuardTest, NegativeDeviceIndex) {
  auto guard_result = CUDAGuard::create(-2);
  EXPECT_FALSE(guard_result.ok());
}

// Compile-time type-trait checks. These do not need a CUDA device, so they
// live outside the CUDAGuardTest fixture (whose SetUp() calls GTEST_SKIP
// when no CUDA device is available).
TEST_F(CUDAGuardTest, MovingLeavesOneRestorer) {
  int entry_device = -1;
  ASSERT_EQ(cudaGetDevice(&entry_device), cudaSuccess);
  int other_device = entry_device == 0 ? 1 : 0;
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (other_device >= device_count) {
    GTEST_SKIP() << "needs a second device to tell one restore from two";
  }

  {
    auto created = CUDAGuard::create(other_device);
    ASSERT_TRUE(created.ok());
    {
      CUDAGuard moved(std::move(created.get()));
      int inside = -1;
      ASSERT_EQ(cudaGetDevice(&inside), cudaSuccess);
      EXPECT_EQ(inside, other_device);
    }
    // The guard that took over the borrow has gone, so the entry device is
    // back.
    int after_move_target = -1;
    ASSERT_EQ(cudaGetDevice(&after_move_target), cudaSuccess);
    EXPECT_EQ(after_move_target, entry_device);
    // Move somewhere else, so a second restore would be visible rather than
    // landing on the value that is already current.
    ASSERT_EQ(cudaSetDevice(other_device), cudaSuccess);
  }
  // The moved-from guard has now gone too. It must not have restored again.
  int after_moved_from = -1;
  ASSERT_EQ(cudaGetDevice(&after_moved_from), cudaSuccess);
  EXPECT_EQ(after_moved_from, other_device)
      << "the moved-from guard restored as well, so the device was put back twice";
  ASSERT_EQ(cudaSetDevice(entry_device), cudaSuccess);
}

TEST(CUDAGuardCompileTimeTest, CopyConstructorDeleted) {
  static_assert(
      !std::is_copy_constructible_v<CUDAGuard>,
      "CUDAGuard should not be copy constructible");
}

TEST(CUDAGuardCompileTimeTest, CopyAssignmentDeleted) {
  static_assert(
      !std::is_copy_assignable_v<CUDAGuard>,
      "CUDAGuard should not be copy assignable");
}

TEST(CUDAGuardCompileTimeTest, MoveAssignmentDeleted) {
  static_assert(
      !std::is_move_assignable_v<CUDAGuard>,
      "CUDAGuard should not be move assignable");
}
