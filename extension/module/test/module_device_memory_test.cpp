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
 * and ModuleSharedStateWithDevice.pte, whose three methods share one state
 * planned in CUDA memory, with its initial value stored.
 *
 * Since we don't have a real CUDA backend, we test that:
 * 1. CPU-only models load through Module without invoking device allocator
 * 2. Device-annotated models trigger DeviceMemoryBuffer::create via a mock
 * 3. Methods share one device allocation of a shared state, which loading a
 *    method never resets
 */

#include <executorch/extension/module/module.h>

#include <gtest/gtest.h>

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/device_memory_buffer.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/test/mock_cuda_allocator.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::make_tensor_ptr;
using executorch::extension::Module;
using executorch::runtime::DeviceMemoryBuffer;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::get_device_allocator;
using executorch::runtime::register_device_allocator;
using executorch::runtime::etensor::DeviceType;
using executorch::runtime::testing::MockCudaAllocator;

static MockCudaAllocator g_mock_cuda;

class ModuleDeviceMemoryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
    // The registry is a process-wide static, so a second registration for the
    // same device type aborts. Repeat runs re-enter this function.
    if (get_device_allocator(DeviceType::CUDA) == nullptr) {
      register_device_allocator(&g_mock_cuda);
    }
  }

  void SetUp() override {
    g_mock_cuda.fail_allocations_ = false;
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

  ASSERT_EQ(meta->num_memory_planned_buffers(), 2);

  {
    // After turn on on-device memory planning, the output cpu tensor shares
    // the same buffer with the input cpu tensor. So the memory planned buffer
    // only needs 2 * 16 = 32 bytes.

    auto size = meta->memory_planned_buffer_size(0);
    ASSERT_TRUE(size.ok());
    EXPECT_EQ(size.get(), 32);

    auto device = meta->memory_planned_buffer_device(0);
    ASSERT_TRUE(device.ok());
    EXPECT_EQ(device->type(), DeviceType::CPU);
    EXPECT_EQ(device->index(), 0);
  }
  {
    auto size = meta->memory_planned_buffer_size(1);
    ASSERT_TRUE(size.ok());
    EXPECT_EQ(size.get(), 48);

    auto device = meta->memory_planned_buffer_device(1);
    ASSERT_TRUE(device.ok());
    EXPECT_EQ(device->type(), DeviceType::CUDA);
    EXPECT_EQ(device->index(), 0);
  }
}

TEST_F(ModuleDeviceMemoryTest, DeviceModelWithSharedArenasReturnsNotSupported) {
  const char* path = std::getenv("ET_MODULE_ADD_WITH_DEVICE_PATH");
  ASSERT_NE(path, nullptr) << "ET_MODULE_ADD_WITH_DEVICE_PATH not set";

  // The program marks no buffer shared, so share_memory_arenas cannot tell
  // which of its device buffers hold state.
  Module module(
      path,
      Module::LoadMode::File,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/true);

  EXPECT_EQ(module.load_method("forward"), Error::NotSupported);
}

// share_memory_arenas leaves the device buffers a program marks to the
// program, so such a program loads with it too.
TEST_F(ModuleDeviceMemoryTest, MethodsShareOneDeviceAllocationOfSharedState) {
#ifdef USE_ATEN_LIB
  GTEST_SKIP() << "the shared state program is built for portable kernels";
#endif
  const char* path = std::getenv("ET_MODULE_SHARED_STATE_WITH_DEVICE_PATH");
  ASSERT_NE(path, nullptr) << "ET_MODULE_SHARED_STATE_WITH_DEVICE_PATH not set";

  for (bool share_memory_arenas : {false, true}) {
    SetUp();
    {
      Module module(
          path,
          Module::LoadMode::File,
          /*event_tracer=*/nullptr,
          /*memory_allocator=*/nullptr,
          /*temp_allocator=*/nullptr,
          share_memory_arenas);
      const auto expect = [&module](
                              const char* method,
                              const std::vector<EValue>& inputs,
                              float value) {
        auto result = module.execute(method, inputs);
        ASSERT_EQ(result.error(), Error::Ok);
        EXPECT_TENSOR_EQ(
            result.get()[0].toTensor(), *make_tensor_ptr({1}, {value}));
      };
      auto x = make_tensor_ptr({1}, {2.f});
      int copies = g_mock_cuda.h2d_count_;
      // The first method to load initializes the state for all three.
      expect("get_state", {}, 1.f);
      EXPECT_GT(g_mock_cuda.h2d_count_, copies);
      expect("forward", {x}, 4.f);

      // A reader reloads, then the writer reloads, and neither copies initial
      // data. Running the methods copies state across the device boundary, so
      // only the loads are counted.
      copies = g_mock_cuda.h2d_count_;
      ASSERT_TRUE(module.unload_method("get_state"));
      ASSERT_EQ(module.load_method("get_state"), Error::Ok);
      ASSERT_TRUE(module.unload_method("forward"));
      ASSERT_EQ(module.load_method("forward"), Error::Ok);
      EXPECT_EQ(g_mock_cuda.h2d_count_, copies);
      expect("get_state", {}, 2.f);

      ASSERT_EQ(
          module.execute("set_state", make_tensor_ptr({1}, {5.f})).error(),
          Error::Ok);
      expect("forward", {x}, 8.f);
      expect("get_state", {}, 6.f);
      // One allocation of the state serves every method.
      EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
      EXPECT_EQ(g_mock_cuda.last_allocate_size_, 16);
    }
    EXPECT_EQ(g_mock_cuda.deallocate_count_, 1);
  }
}

TEST_F(ModuleDeviceMemoryTest, DeviceAllocationFailureIsReportedNotFatal) {
  const char* path = std::getenv("ET_MODULE_ADD_WITH_DEVICE_PATH");
  ASSERT_NE(path, nullptr) << "ET_MODULE_ADD_WITH_DEVICE_PATH not set";

  // Stand in for a device that is out of memory. The caller gets an error and
  // the process survives to handle it.
  g_mock_cuda.fail_allocations_ = true;

  Module module(path);
  EXPECT_EQ(module.load_method("forward"), Error::MemoryAllocationFailed);
  EXPECT_FALSE(module.is_method_loaded("forward"));
  EXPECT_EQ(g_mock_cuda.allocate_count_, 0);
  EXPECT_EQ(g_mock_cuda.deallocate_count_, 0);

  // The failure must not leave the Module half-built: with the device back,
  // the same call reaches the allocator again.
  g_mock_cuda.fail_allocations_ = false;
  (void)module.load_method("forward");
  EXPECT_EQ(g_mock_cuda.allocate_count_, 1);
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
    // (make_planned_memory) runs BEFORE backend init.
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
