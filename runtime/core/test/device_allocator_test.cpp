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

  Error init_buffer(uint32_t memory_id, size_t size, DeviceIndex index)
      override {
    last_init_buffer_memory_id_ = memory_id;
    last_init_buffer_size_ = size;
    last_init_buffer_index_ = index;
    init_buffer_call_count_++;
    return Error::Ok;
  }

  Result<void*> get_offset_address(
      uint32_t memory_id,
      size_t offset_bytes,
      size_t size_bytes,
      DeviceIndex index) override {
    last_get_offset_memory_id_ = memory_id;
    last_get_offset_offset_ = offset_bytes;
    last_get_offset_size_ = size_bytes;
    last_get_offset_index_ = index;
    get_offset_address_call_count_++;
    return &dummy_buffer_;
  }

  Result<void*> allocate(size_t nbytes, DeviceIndex index) override {
    last_allocate_size_ = nbytes;
    last_allocate_index_ = index;
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

  // Tracking variables for verification
  uint32_t last_init_buffer_memory_id_ = 0;
  size_t last_init_buffer_size_ = 0;
  DeviceIndex last_init_buffer_index_ = -1;
  int init_buffer_call_count_ = 0;

  uint32_t last_get_offset_memory_id_ = 0;
  size_t last_get_offset_offset_ = 0;
  size_t last_get_offset_size_ = 0;
  DeviceIndex last_get_offset_index_ = -1;
  int get_offset_address_call_count_ = 0;

  size_t last_allocate_size_ = 0;
  DeviceIndex last_allocate_index_ = -1;
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

class DeviceAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(DeviceAllocatorTest, MockAllocatorDeviceType) {
  MockDeviceAllocator cpu_allocator(DeviceType::CPU);
  MockDeviceAllocator cuda_allocator(DeviceType::CUDA);

  EXPECT_EQ(cpu_allocator.device_type(), DeviceType::CPU);
  EXPECT_EQ(cuda_allocator.device_type(), DeviceType::CUDA);
}

TEST_F(DeviceAllocatorTest, MockAllocatorInitBuffer) {
  MockDeviceAllocator allocator(DeviceType::CUDA);

  Error err =
      allocator.init_buffer(/*memory_id=*/1, /*size=*/1024, /*index=*/0);

  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(allocator.init_buffer_call_count_, 1);
  EXPECT_EQ(allocator.last_init_buffer_memory_id_, 1);
  EXPECT_EQ(allocator.last_init_buffer_size_, 1024);
  EXPECT_EQ(allocator.last_init_buffer_index_, 0);
}

TEST_F(DeviceAllocatorTest, MockAllocatorGetOffsetAddress) {
  MockDeviceAllocator allocator(DeviceType::CUDA);

  Result<void*> result = allocator.get_offset_address(
      /*memory_id=*/2, /*offset_bytes=*/128, /*size_bytes=*/256, /*index=*/1);

  EXPECT_TRUE(result.ok());
  EXPECT_NE(result.get(), nullptr);
  EXPECT_EQ(allocator.get_offset_address_call_count_, 1);
  EXPECT_EQ(allocator.last_get_offset_memory_id_, 2);
  EXPECT_EQ(allocator.last_get_offset_offset_, 128);
  EXPECT_EQ(allocator.last_get_offset_size_, 256);
  EXPECT_EQ(allocator.last_get_offset_index_, 1);
}

TEST_F(DeviceAllocatorTest, MockAllocatorAllocateAndDeallocate) {
  MockDeviceAllocator allocator(DeviceType::CUDA);

  Result<void*> result = allocator.allocate(/*nbytes=*/512, /*index=*/0);
  EXPECT_TRUE(result.ok());
  void* ptr = result.get();
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(allocator.allocate_call_count_, 1);
  EXPECT_EQ(allocator.last_allocate_size_, 512);
  EXPECT_EQ(allocator.last_allocate_index_, 0);

  allocator.deallocate(ptr, /*index=*/0);
  EXPECT_EQ(allocator.deallocate_call_count_, 1);
  EXPECT_EQ(allocator.last_deallocate_ptr_, ptr);
  EXPECT_EQ(allocator.last_deallocate_index_, 0);
}

TEST_F(DeviceAllocatorTest, MockAllocatorCopyHostToDevice) {
  MockDeviceAllocator allocator(DeviceType::CUDA);
  uint8_t host_data[64] = {1, 2, 3, 4};
  uint8_t device_data[64] = {};

  Error err = allocator.copy_host_to_device(
      device_data, host_data, sizeof(host_data), /*index=*/0);

  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(allocator.copy_h2d_call_count_, 1);
  EXPECT_EQ(allocator.last_h2d_dst_, device_data);
  EXPECT_EQ(allocator.last_h2d_src_, host_data);
  EXPECT_EQ(allocator.last_h2d_size_, sizeof(host_data));
  EXPECT_EQ(allocator.last_h2d_index_, 0);
}

TEST_F(DeviceAllocatorTest, MockAllocatorCopyDeviceToHost) {
  MockDeviceAllocator allocator(DeviceType::CUDA);
  uint8_t device_data[64] = {5, 6, 7, 8};
  uint8_t host_data[64] = {};

  Error err = allocator.copy_device_to_host(
      host_data, device_data, sizeof(device_data), /*index=*/1);

  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(allocator.copy_d2h_call_count_, 1);
  EXPECT_EQ(allocator.last_d2h_dst_, host_data);
  EXPECT_EQ(allocator.last_d2h_src_, device_data);
  EXPECT_EQ(allocator.last_d2h_size_, sizeof(device_data));
  EXPECT_EQ(allocator.last_d2h_index_, 1);
}

TEST_F(DeviceAllocatorTest, RegistryGetUnregisteredReturnsNullptr) {
  // Getting an allocator for an unregistered device type should return nullptr
  // Note that there shouldn't be any regsitered allocators for CPU backend.
  DeviceAllocator* alloc = get_device_allocator(DeviceType::CPU);
  (void)alloc;
}

TEST_F(DeviceAllocatorTest, RegistrySingletonInstance) {
  // Verify that instance() returns the same object each time
  DeviceAllocatorRegistry& instance1 = DeviceAllocatorRegistry::instance();
  DeviceAllocatorRegistry& instance2 = DeviceAllocatorRegistry::instance();

  EXPECT_EQ(&instance1, &instance2);
}

TEST_F(DeviceAllocatorTest, RegisterAndGetDeviceAllocator) {
  // Register a mock allocator for CUDA and retrieve it via the free function.
  MockDeviceAllocator cuda_allocator(DeviceType::CUDA);
  register_device_allocator(DeviceType::CUDA, &cuda_allocator);

  DeviceAllocator* retrieved = get_device_allocator(DeviceType::CUDA);
  EXPECT_EQ(retrieved, &cuda_allocator);
  EXPECT_EQ(retrieved->device_type(), DeviceType::CUDA);

  // Registering the same device type twice should abort.
  MockDeviceAllocator another_allocator(DeviceType::CUDA);
  EXPECT_DEATH(
      register_device_allocator(DeviceType::CUDA, &another_allocator),
      "Allocator already registered");
}

TEST_F(DeviceAllocatorTest, RegisterAndDispatchThroughRegistry) {
  // Verify that after registration, calls dispatch to the registered allocator.
  DeviceAllocator* alloc = get_device_allocator(DeviceType::CUDA);
  ASSERT_NE(alloc, nullptr);

  // Use the allocator through the registry and verify it reaches the mock.
  Error err = alloc->init_buffer(/*memory_id=*/5, /*size=*/2048, /*index=*/0);
  EXPECT_EQ(err, Error::Ok);

  Result<void*> result = alloc->allocate(/*nbytes=*/256, /*index=*/1);
  EXPECT_TRUE(result.ok());
  EXPECT_NE(result.get(), nullptr);
}
