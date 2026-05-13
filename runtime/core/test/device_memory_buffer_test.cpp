/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/device_memory_buffer.h>

#include <gtest/gtest.h>

#include <executorch/runtime/platform/runtime.h>

using executorch::runtime::DeviceAllocator;
using executorch::runtime::DeviceMemoryBuffer;
using executorch::runtime::Error;
using executorch::runtime::get_device_allocator;
using executorch::runtime::register_device_allocator;
using executorch::runtime::Result;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;

/**
 * A mock DeviceAllocator for testing DeviceMemoryBuffer.
 * Returns pointers into a local buffer and tracks call counts.
 */
class MockAllocator : public DeviceAllocator {
 public:
  explicit MockAllocator(DeviceType type) : type_(type) {}

  Result<void*> allocate(
      size_t nbytes,
      DeviceIndex index,
      size_t alignment = DeviceAllocator::kDefaultAlignment) override {
    allocate_count_++;
    last_allocate_size_ = nbytes;
    last_allocate_alignment_ = alignment;
    return static_cast<void*>(buffer_);
  }

  void deallocate(void* ptr, DeviceIndex index) override {
    deallocate_count_++;
    last_deallocate_ptr_ = ptr;
  }

  Error copy_host_to_device(
      void* dst,
      const void* src,
      size_t nbytes,
      DeviceIndex index) override {
    return Error::Ok;
  }

  Error copy_device_to_host(
      void* dst,
      const void* src,
      size_t nbytes,
      DeviceIndex index) override {
    return Error::Ok;
  }

  DeviceType device_type() const override {
    return type_;
  }

  int allocate_count_ = 0;
  int deallocate_count_ = 0;
  size_t last_allocate_size_ = 0;
  size_t last_allocate_alignment_ = 0;
  void* last_deallocate_ptr_ = nullptr;
  uint8_t buffer_[256] = {};

 private:
  DeviceType type_;
};

// Global mock registered once before all tests run.
static MockAllocator g_mock_cuda(DeviceType::CUDA);

class DeviceMemoryBufferTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
    register_device_allocator(&g_mock_cuda);
  }

  void SetUp() override {
    // Reset counters before each test.
    g_mock_cuda.allocate_count_ = 0;
    g_mock_cuda.deallocate_count_ = 0;
    g_mock_cuda.last_allocate_size_ = 0;
    g_mock_cuda.last_allocate_alignment_ = 0;
    g_mock_cuda.last_deallocate_ptr_ = nullptr;
  }
};

TEST_F(DeviceMemoryBufferTest, DefaultConstructedIsEmpty) {
  DeviceMemoryBuffer buf;
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.size(), 0);

  auto span = buf.as_span();
  EXPECT_EQ(span.data(), nullptr);
  EXPECT_EQ(span.size(), 0);
}

TEST_F(DeviceMemoryBufferTest, CreateAllocatesAndDestructorDeallocates) {
  {
    auto result = DeviceMemoryBuffer::create(1024, DeviceType::CUDA, 0);
    ASSERT_TRUE(result.ok());

    auto buf = std::move(result.get());
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 1024);
    EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
    EXPECT_EQ(g_mock_cuda.last_allocate_size_, 1024);
    EXPECT_EQ(g_mock_cuda.deallocate_count_, 0);
  }
  EXPECT_EQ(g_mock_cuda.deallocate_count_, 1);
  EXPECT_EQ(g_mock_cuda.last_deallocate_ptr_, g_mock_cuda.buffer_);
}

TEST_F(DeviceMemoryBufferTest, CreateFailsWithNoRegisteredAllocator) {
  auto result = DeviceMemoryBuffer::create(512, DeviceType::CPU, 0);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::NotFound);
}

TEST_F(DeviceMemoryBufferTest, MoveConstructorTransfersOwnership) {
  auto result = DeviceMemoryBuffer::create(256, DeviceType::CUDA, 0);
  ASSERT_TRUE(result.ok());
  auto original = std::move(result.get());
  void* original_ptr = original.data();

  DeviceMemoryBuffer moved(std::move(original));

  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(original.size(), 0);
  EXPECT_EQ(moved.data(), original_ptr);
  EXPECT_EQ(moved.size(), 256);
  EXPECT_EQ(g_mock_cuda.deallocate_count_, 0);
}

TEST_F(DeviceMemoryBufferTest, MoveAssignmentTransfersOwnership) {
  auto result = DeviceMemoryBuffer::create(128, DeviceType::CUDA, 0);
  ASSERT_TRUE(result.ok());
  auto original = std::move(result.get());
  void* original_ptr = original.data();

  DeviceMemoryBuffer target;
  target = std::move(original);

  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(target.data(), original_ptr);
  EXPECT_EQ(target.size(), 128);
  EXPECT_EQ(g_mock_cuda.deallocate_count_, 0);
}

TEST_F(DeviceMemoryBufferTest, DestructorNoOpForDefaultConstructed) {
  { DeviceMemoryBuffer buf; }
  EXPECT_EQ(g_mock_cuda.deallocate_count_, 0);
}

TEST_F(DeviceMemoryBufferTest, AsSpanWrapsDevicePointer) {
  auto result = DeviceMemoryBuffer::create(2048, DeviceType::CUDA, 0);
  ASSERT_TRUE(result.ok());
  auto buf = std::move(result.get());

  auto span = buf.as_span();
  EXPECT_EQ(span.data(), static_cast<uint8_t*>(buf.data()));
  EXPECT_EQ(span.size(), 2048);
}

TEST_F(DeviceMemoryBufferTest, CreateUsesDefaultAlignmentWhenUnspecified) {
  auto result = DeviceMemoryBuffer::create(1024, DeviceType::CUDA, 0);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(
      g_mock_cuda.last_allocate_alignment_, DeviceAllocator::kDefaultAlignment);
}

TEST_F(DeviceMemoryBufferTest, CreateForwardsCustomAlignmentToAllocator) {
  constexpr size_t kCustomAlignment = 512;
  auto result =
      DeviceMemoryBuffer::create(1024, DeviceType::CUDA, 0, kCustomAlignment);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(g_mock_cuda.last_allocate_alignment_, kCustomAlignment);
}
