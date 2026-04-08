/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Tests that Module's device-aware memory allocation path works correctly.
 *
 * Uses ModuleAddWithDevice.pte which has:
 *   non_const_buffer_sizes: [0, 48]  (1 buffer, index 0 reserved)
 *   non_const_buffer_device: [{buffer_idx=1, device_type=CUDA, device_index=0}]
 *
 * Since we don't have a real CUDA backend, we test that:
 * 1. CPU-only models load through Module without invoking device allocator
 * 2. Device-annotated models trigger DeviceMemoryBuffer::create via a mock
 */

#include <executorch/extension/module/module.h>

#include <gtest/gtest.h>

#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/device_memory_buffer.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::Module;
using executorch::runtime::DeviceAllocator;
using executorch::runtime::DeviceMemoryBuffer;
using executorch::runtime::Error;
using executorch::runtime::register_device_allocator;
using executorch::runtime::Result;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;

namespace {

class MockCudaAllocator : public DeviceAllocator {
 public:
  Result<void*> allocate(size_t nbytes, DeviceIndex index) override {
    allocate_count_++;
    last_allocate_size_ = nbytes;
    last_allocate_index_ = index;
    buffer_ = std::make_unique<uint8_t[]>(nbytes);
    return static_cast<void*>(buffer_.get());
  }

  void deallocate(void* ptr, DeviceIndex index) override {
    deallocate_count_++;
    buffer_.reset();
  }

  Error copy_host_to_device(void*, const void*, size_t, DeviceIndex) override {
    return Error::Ok;
  }

  Error copy_device_to_host(void*, const void*, size_t, DeviceIndex) override {
    return Error::Ok;
  }

  DeviceType device_type() const override {
    return DeviceType::CUDA;
  }

  int allocate_count_ = 0;
  int deallocate_count_ = 0;
  size_t last_allocate_size_ = 0;
  DeviceIndex last_allocate_index_ = -1;

 private:
  std::unique_ptr<uint8_t[]> buffer_;
};

} // namespace

static MockCudaAllocator g_mock_cuda;

class ModuleDeviceMemoryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
    register_device_allocator(DeviceType::CUDA, &g_mock_cuda);
  }

  void SetUp() override {
    g_mock_cuda.allocate_count_ = 0;
    g_mock_cuda.deallocate_count_ = 0;
    g_mock_cuda.last_allocate_size_ = 0;
    g_mock_cuda.last_allocate_index_ = -1;
  }
};

TEST_F(ModuleDeviceMemoryTest, CpuOnlyModelDoesNotAllocateDeviceMemory) {
  const char* path = std::getenv("ET_MODULE_ADD_PATH");
  ASSERT_NE(path, nullptr) << "ET_MODULE_ADD_PATH not set";

  Module module(path);
  auto err = module.load_method("forward");
  ASSERT_EQ(err, Error::Ok);

  EXPECT_EQ(g_mock_cuda.allocate_count_, 0)
      << "CPU-only model should not allocate device memory";
}

TEST_F(ModuleDeviceMemoryTest, DeviceMemoryBufferCreateCallsAllocator) {
  // Directly test DeviceMemoryBuffer::create with the registered mock.
  // This verifies the RAII allocation/deallocation path that Module uses.
  {
    auto result = DeviceMemoryBuffer::create(48, DeviceType::CUDA, 0);
    ASSERT_TRUE(result.ok());
    auto buf = std::move(result.get());

    EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
    EXPECT_EQ(g_mock_cuda.last_allocate_size_, 48);
    EXPECT_EQ(g_mock_cuda.last_allocate_index_, 0);
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 48);

    // as_span() wraps the device pointer for HierarchicalAllocator.
    auto span = buf.as_span();
    EXPECT_EQ(span.data(), static_cast<uint8_t*>(buf.data()));
    EXPECT_EQ(span.size(), 48);

    EXPECT_EQ(g_mock_cuda.deallocate_count_, 0);
  }
  // RAII deallocation on scope exit.
  EXPECT_EQ(g_mock_cuda.deallocate_count_, 1);
}

TEST_F(ModuleDeviceMemoryTest, DeviceModelMethodMetaReportsCudaBuffer) {
  // Verify MethodMeta reports the correct device for buffers in the
  // device-annotated model, without needing to load the full method.
  const char* path = std::getenv("ET_MODULE_ADD_WITH_DEVICE_PATH");
  ASSERT_NE(path, nullptr) << "ET_MODULE_ADD_WITH_DEVICE_PATH not set";

  Module module(path);
  auto err = module.load();
  ASSERT_EQ(err, Error::Ok);

  auto meta = module.method_meta("forward");
  ASSERT_TRUE(meta.ok());

  // ModuleAddWithDevice has 1 planned buffer (48 bytes) on CUDA.
  ASSERT_EQ(meta->num_memory_planned_buffers(), 1);

  auto size = meta->memory_planned_buffer_size(0);
  ASSERT_TRUE(size.ok());
  EXPECT_EQ(size.get(), 48);

  auto device = meta->memory_planned_buffer_device(0);
  ASSERT_TRUE(device.ok());
  EXPECT_EQ(device->type(), DeviceType::CUDA);
  EXPECT_EQ(device->index(), 0);
}

TEST_F(ModuleDeviceMemoryTest, DeviceModelWithSharedArenasReturnsNotSupported) {
  const char* path = std::getenv("ET_MODULE_ADD_WITH_DEVICE_PATH");
  ASSERT_NE(path, nullptr) << "ET_MODULE_ADD_WITH_DEVICE_PATH not set";

  // share_memory_arenas = true with a device-annotated model should fail.
  Module module(
      path,
      Module::LoadMode::File,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/true);

  auto err = module.load_method("forward");
  EXPECT_EQ(err, Error::NotSupported);
}

TEST_F(
    ModuleDeviceMemoryTest,
    LoadMethodAllocatesDeviceMemoryAndDeallocatesOnDestroy) {
  const char* path = std::getenv("ET_MODULE_ADD_WITH_DEVICE_PATH");
  ASSERT_NE(path, nullptr) << "ET_MODULE_ADD_WITH_DEVICE_PATH not set";

  {
    Module module(path);
    auto err = module.load_method("forward");

    // Regardless of whether load_method succeeds or fails (e.g. due to
    // backend init issues), the device-aware memory allocation path
    // (make_planned_memory_with_devices) runs BEFORE backend init.
    EXPECT_EQ(g_mock_cuda.allocate_count_, 1)
        << "Expected 1 device allocation for the CUDA buffer"
        << " (actual: " << g_mock_cuda.allocate_count_ << ")"
        << ", deallocate_count=" << g_mock_cuda.deallocate_count_
        << ", load_method returned error=" << static_cast<int>(err);
    EXPECT_EQ(g_mock_cuda.last_allocate_size_, 48)
        << "Expected 48 bytes allocated (3 CUDA tensors sharing one buffer)";
    EXPECT_EQ(g_mock_cuda.last_allocate_index_, 0)
        << "Expected device_index=0 (cuda:0)";

    if (err == Error::Ok) {
      // Success path: MethodHolder moved into methods_ map.
      // DeviceMemoryBuffer is alive as long as Module is alive.
      EXPECT_EQ(g_mock_cuda.deallocate_count_, 0)
          << "No deallocation while method is loaded";
    } else {
      // Error path: local MethodHolder destroyed on return from load_method.
      // RAII deallocation already happened.
      EXPECT_EQ(g_mock_cuda.deallocate_count_, 1)
          << "RAII deallocation on error path";
    }
  }

  // After Module destroyed, all device memory must be freed.
  EXPECT_EQ(g_mock_cuda.deallocate_count_, 1)
      << "Expected deallocation after Module destroyed";
}
