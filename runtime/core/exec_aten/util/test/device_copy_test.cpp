/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/test/mock_cuda_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <array>

using executorch::aten::Device;
using executorch::aten::DeviceType;
using executorch::runtime::Error;
using executorch::runtime::internal::copy_between_devices;
using executorch::runtime::testing::MockCudaAllocator;

// Copying into memory a program planned on an accelerator has to go through that device's allocator.
// A plain memcpy into a device pointer is undefined, and it crashed a program exported to keep its
// activations on the device, which is the case these tests cover.
class DeviceCopyTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
    // Registered once for the whole suite, because the registry refuses a second allocator for a
    // device type and holds the pointer for the lifetime of the process.
    static MockCudaAllocator allocator;
    allocator_ = &allocator;
    executorch::runtime::register_device_allocator(allocator_);
  }

  void SetUp() override {
    allocator_->reset();
  }

  static MockCudaAllocator* allocator_;
  static constexpr Device kHost{DeviceType::CPU};
  static constexpr Device kDevice{DeviceType::CUDA, 0};
  std::array<uint8_t, 8> source_{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<uint8_t, 8> destination_{};
};

MockCudaAllocator* DeviceCopyTest::allocator_ = nullptr;

TEST_F(DeviceCopyTest, HostToHostDoesNotReachTheAllocator) {
  ASSERT_EQ(
      copy_between_devices(
          destination_.data(), kHost, source_.data(), kHost, source_.size()),
      Error::Ok);
  EXPECT_EQ(destination_, source_);
  // A host copy must not depend on a device being registered at all.
  EXPECT_EQ(allocator_->h2d_count_, 0);
  EXPECT_EQ(allocator_->d2h_count_, 0);
  EXPECT_EQ(allocator_->d2d_count_, 0);
}

TEST_F(DeviceCopyTest, HostToDeviceUsesTheAllocator) {
  ASSERT_EQ(
      copy_between_devices(
          destination_.data(), kDevice, source_.data(), kHost, source_.size()),
      Error::Ok);
  EXPECT_EQ(allocator_->h2d_count_, 1);
  EXPECT_EQ(allocator_->last_h2d_size_, source_.size());
  EXPECT_EQ(destination_, source_);
}

TEST_F(DeviceCopyTest, DeviceToHostUsesTheAllocator) {
  ASSERT_EQ(
      copy_between_devices(
          destination_.data(), kHost, source_.data(), kDevice, source_.size()),
      Error::Ok);
  EXPECT_EQ(allocator_->d2h_count_, 1);
  EXPECT_EQ(destination_, source_);
}

TEST_F(DeviceCopyTest, DeviceToDeviceUsesTheAllocator) {
  // The case a program with device activations actually needs: the caller hands over device memory
  // and the runtime fills the device buffer the memory plan reserved, so neither end is host memory.
  ASSERT_EQ(
      copy_between_devices(
          destination_.data(), kDevice, source_.data(), kDevice, source_.size()),
      Error::Ok);
  EXPECT_EQ(allocator_->d2d_count_, 1);
  EXPECT_EQ(allocator_->h2d_count_, 0);
  EXPECT_EQ(destination_, source_);
}

TEST_F(DeviceCopyTest, ZeroBytesIsAccepted) {
  // A tensor with a zero-sized dimension reaches here with nothing to copy, and that is not an
  // error. It is worth pinning because the device path is easy to write in a way that rejects it.
  ASSERT_EQ(
      copy_between_devices(destination_.data(), kDevice, source_.data(), kHost, 0),
      Error::Ok);
  EXPECT_EQ(allocator_->h2d_count_, 1);
  EXPECT_EQ(allocator_->last_h2d_size_, 0u);
}
