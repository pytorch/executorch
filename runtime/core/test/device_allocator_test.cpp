/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/device_allocator.h>

#include <gtest/gtest.h>

#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;
using executorch::runtime::DeviceAllocator;
using executorch::runtime::DeviceAllocatorRegistry;
using executorch::runtime::Error;
using executorch::runtime::get_device_allocator;
using executorch::runtime::register_device_allocator;
using executorch::runtime::Result;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;
using executorch::runtime::etensor::kNumDeviceTypes;

/**
 * A mock DeviceAllocator implementation for testing purposes.
 * Tracks calls to verify the registry dispatches correctly.
 */
class MockDeviceAllocator : public DeviceAllocator {
 public:
  explicit MockDeviceAllocator(DeviceType type) : type_(type) {}

  Result<void*> allocate(
      size_t nbytes,
      DeviceIndex index,
      size_t alignment = DeviceAllocator::kDefaultAlignment) override {
    last_allocate_size_ = nbytes;
    last_allocate_index_ = index;
    last_allocate_alignment_ = alignment;
    allocate_call_count_++;
    return &dummy_buffer_;
  }

  void deallocate(void* ptr, DeviceIndex index) override {
    last_deallocate_ptr_ = ptr;
    last_deallocate_index_ = index;
    deallocate_call_count_++;
  }

  Error copy_host_to_device(
      void* dst,
      const void* src,
      size_t nbytes,
      DeviceIndex index) override {
    last_h2d_dst_ = dst;
    last_h2d_src_ = src;
    last_h2d_size_ = nbytes;
    last_h2d_index_ = index;
    copy_h2d_call_count_++;
    return Error::Ok;
  }

  Error copy_device_to_host(
      void* dst,
      const void* src,
      size_t nbytes,
      DeviceIndex index) override {
    last_d2h_dst_ = dst;
    last_d2h_src_ = src;
    last_d2h_size_ = nbytes;
    last_d2h_index_ = index;
    copy_d2h_call_count_++;
    return Error::Ok;
  }

  DeviceType device_type() const override {
    return type_;
  }

  // Reset all tracking state so tests can run against a clean baseline.
  void reset_counters() {
    last_allocate_size_ = 0;
    last_allocate_index_ = -1;
    last_allocate_alignment_ = 0;
    allocate_call_count_ = 0;

    last_deallocate_ptr_ = nullptr;
    last_deallocate_index_ = -1;
    deallocate_call_count_ = 0;

    last_h2d_dst_ = nullptr;
    last_h2d_src_ = nullptr;
    last_h2d_size_ = 0;
    last_h2d_index_ = -1;
    copy_h2d_call_count_ = 0;

    last_d2h_dst_ = nullptr;
    last_d2h_src_ = nullptr;
    last_d2h_size_ = 0;
    last_d2h_index_ = -1;
    copy_d2h_call_count_ = 0;
  }

  // Tracking variables for verification
  size_t last_allocate_size_ = 0;
  DeviceIndex last_allocate_index_ = -1;
  size_t last_allocate_alignment_ = 0;
  int allocate_call_count_ = 0;

  void* last_deallocate_ptr_ = nullptr;
  DeviceIndex last_deallocate_index_ = -1;
  int deallocate_call_count_ = 0;

  void* last_h2d_dst_ = nullptr;
  const void* last_h2d_src_ = nullptr;
  size_t last_h2d_size_ = 0;
  DeviceIndex last_h2d_index_ = -1;
  int copy_h2d_call_count_ = 0;

  void* last_d2h_dst_ = nullptr;
  const void* last_d2h_src_ = nullptr;
  size_t last_d2h_size_ = 0;
  DeviceIndex last_d2h_index_ = -1;
  int copy_d2h_call_count_ = 0;

 private:
  DeviceType type_;
  uint8_t dummy_buffer_[64] = {};
};

/**
 * Test fixture that owns a single MockDeviceAllocator with static lifetime
 * and registers it in DeviceAllocatorRegistry exactly once for the whole
 * test suite. Every test in this fixture exercises the same registered
 * allocator instance via get_device_allocator(), which mirrors how real
 * code is expected to use the registry (one allocator per device type,
 * registered during static initialization). Per-test isolation is provided
 * by reset_counters() in SetUp().
 */
class DeviceAllocatorTest : public ::testing::Test {
 protected:
  static MockDeviceAllocator& cuda_allocator() {
    static MockDeviceAllocator allocator(DeviceType::CUDA);
    return allocator;
  }

  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
    if (get_device_allocator(DeviceType::CUDA) == nullptr) {
      register_device_allocator(&cuda_allocator());
    }
  }

  void SetUp() override {
    cuda_allocator().reset_counters();
  }
};

TEST_F(DeviceAllocatorTest, RegisteredAllocatorReportsCorrectDeviceType) {
  DeviceAllocator* alloc = get_device_allocator(DeviceType::CUDA);
  ASSERT_NE(alloc, nullptr);
  EXPECT_EQ(alloc, &cuda_allocator());
  EXPECT_EQ(alloc->device_type(), DeviceType::CUDA);
}

TEST_F(DeviceAllocatorTest, AllocateAndDeallocate) {
  DeviceAllocator* alloc = get_device_allocator(DeviceType::CUDA);
  ASSERT_NE(alloc, nullptr);

  Result<void*> result = alloc->allocate(/*nbytes=*/512, /*index=*/0);
  EXPECT_TRUE(result.ok());
  void* ptr = result.get();
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(cuda_allocator().allocate_call_count_, 1);
  EXPECT_EQ(cuda_allocator().last_allocate_size_, 512);
  EXPECT_EQ(cuda_allocator().last_allocate_index_, 0);

  alloc->deallocate(ptr, /*index=*/0);
  EXPECT_EQ(cuda_allocator().deallocate_call_count_, 1);
  EXPECT_EQ(cuda_allocator().last_deallocate_ptr_, ptr);
  EXPECT_EQ(cuda_allocator().last_deallocate_index_, 0);
}

TEST_F(DeviceAllocatorTest, CopyHostToDevice) {
  DeviceAllocator* alloc = get_device_allocator(DeviceType::CUDA);
  ASSERT_NE(alloc, nullptr);

  uint8_t host_data[64] = {1, 2, 3, 4};
  uint8_t device_data[64] = {};

  Error err = alloc->copy_host_to_device(
      device_data, host_data, sizeof(host_data), /*index=*/0);

  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(cuda_allocator().copy_h2d_call_count_, 1);
  EXPECT_EQ(cuda_allocator().last_h2d_dst_, device_data);
  EXPECT_EQ(cuda_allocator().last_h2d_src_, host_data);
  EXPECT_EQ(cuda_allocator().last_h2d_size_, sizeof(host_data));
  EXPECT_EQ(cuda_allocator().last_h2d_index_, 0);
}

TEST_F(DeviceAllocatorTest, CopyDeviceToHost) {
  DeviceAllocator* alloc = get_device_allocator(DeviceType::CUDA);
  ASSERT_NE(alloc, nullptr);

  uint8_t device_data[64] = {5, 6, 7, 8};
  uint8_t host_data[64] = {};

  Error err = alloc->copy_device_to_host(
      host_data, device_data, sizeof(device_data), /*index=*/1);

  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(cuda_allocator().copy_d2h_call_count_, 1);
  EXPECT_EQ(cuda_allocator().last_d2h_dst_, host_data);
  EXPECT_EQ(cuda_allocator().last_d2h_src_, device_data);
  EXPECT_EQ(cuda_allocator().last_d2h_size_, sizeof(device_data));
  EXPECT_EQ(cuda_allocator().last_d2h_index_, 1);
}

TEST_F(DeviceAllocatorTest, RegistryGetUnregisteredReturnsNullptr) {
  // Getting an allocator for an unregistered device type should return nullptr.
  // The fixture only registers a CUDA allocator, so CPU must remain unset.
  DeviceAllocator* alloc = get_device_allocator(DeviceType::CPU);
  EXPECT_EQ(alloc, nullptr);
}

TEST_F(DeviceAllocatorTest, RegistrySingletonInstance) {
  // Verify that instance() returns the same object each time.
  DeviceAllocatorRegistry& instance1 = DeviceAllocatorRegistry::instance();
  DeviceAllocatorRegistry& instance2 = DeviceAllocatorRegistry::instance();

  EXPECT_EQ(&instance1, &instance2);
}

// EXPECT_DEATH requires gtest death-test support, which is unavailable on
// platforms without fork() (e.g. iOS).  Skip on those platforms.
#if GTEST_HAS_DEATH_TEST
TEST_F(DeviceAllocatorTest, RegisteringSameDeviceTypeTwiceAborts) {
  // The fixture has already registered cuda_allocator() for CUDA; attempting
  // to register a second allocator for the same device type must abort.
  MockDeviceAllocator another_allocator(DeviceType::CUDA);
  EXPECT_DEATH(
      register_device_allocator(&another_allocator),
      "Allocator already registered");
}
#endif // GTEST_HAS_DEATH_TEST
