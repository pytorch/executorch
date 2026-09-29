/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/memory_manager.h>

#include <executorch/runtime/core/memory_allocator.h>

#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Span;
using executorch::runtime::etensor::Device;
using executorch::runtime::etensor::DeviceType;

TEST(MemoryManagerTest, MinimalCtor) {
  MemoryAllocator method_allocator(0, nullptr);

  MemoryManager mm(&method_allocator);

  EXPECT_EQ(mm.method_allocator(), &method_allocator);
  EXPECT_EQ(mm.planned_memory(), nullptr);
  EXPECT_EQ(mm.temp_allocator(), nullptr);
}

TEST(MemoryManagerTest, CtorWithPlannedMemory) {
  MemoryAllocator method_allocator(0, nullptr);
  HierarchicalAllocator planned_memory({});

  MemoryManager mm(&method_allocator, &planned_memory);

  EXPECT_EQ(mm.method_allocator(), &method_allocator);
  EXPECT_EQ(mm.planned_memory(), &planned_memory);
  EXPECT_EQ(mm.temp_allocator(), nullptr);
}

TEST(MemoryManagerTest, CtorWithPlannedMemoryAndTemp) {
  MemoryAllocator method_allocator(0, nullptr);
  HierarchicalAllocator planned_memory({});
  MemoryAllocator temp_allocator(0, nullptr);

  MemoryManager mm(&method_allocator, &planned_memory, &temp_allocator);

  EXPECT_EQ(mm.method_allocator(), &method_allocator);
  EXPECT_EQ(mm.planned_memory(), &planned_memory);
  EXPECT_EQ(mm.temp_allocator(), &temp_allocator);
}

TEST(MemoryManagerTest, DEPRECATEDCtor) {
  MemoryAllocator method_allocator(0, nullptr);
  HierarchicalAllocator planned_memory({});
  MemoryAllocator temp_allocator(0, nullptr);
  MemoryAllocator const_allocator(0, nullptr);

  // NOLINTNEXTLINE(facebook-hte-Deprecated)
  MemoryManager mm(
      /*constant_allocator=*/&const_allocator,
      /*non_constant_allocator=*/&planned_memory,
      /*runtime_allocator=*/&method_allocator,
      /*temporary_allocator=*/&temp_allocator);

  EXPECT_EQ(mm.method_allocator(), &method_allocator);
  EXPECT_EQ(mm.planned_memory(), &planned_memory);
  EXPECT_EQ(mm.temp_allocator(), &temp_allocator);
}

TEST(MemoryManagerTest, DeprecatedCtorWithSameAllocator) {
  MemoryAllocator method_allocator(0, nullptr);
  HierarchicalAllocator planned_memory({});
  MemoryAllocator const_allocator(0, nullptr);
  ET_EXPECT_DEATH(
      MemoryManager(
          /*constant_allocator=*/&const_allocator,
          /*non_constant_allocator=*/&planned_memory,
          /*runtime_allocator=*/&method_allocator,
          /*temp_allocator=*/&method_allocator),
      "cannot be the same");
}

TEST(MemoryManagerTest, CtorWithSameAllocator) {
  MemoryAllocator method_allocator(0, nullptr);
  HierarchicalAllocator planned_memory({});
  MemoryAllocator const_allocator(0, nullptr);
  ET_EXPECT_DEATH(
      MemoryManager(
          /*runtime_allocator=*/&method_allocator,
          /*non_constant_allocator=*/&planned_memory,
          /*temp_allocator=*/&method_allocator),
      "cannot be the same");
}

TEST(MemoryManagerTest, ThreeArgCtorHasNoDeviceMemory) {
  MemoryAllocator method_allocator(0, nullptr);
  HierarchicalAllocator planned_memory({});
  MemoryAllocator temp_allocator(0, nullptr);

  MemoryManager mm(&method_allocator, &planned_memory, &temp_allocator);

  EXPECT_FALSE(mm.has_device_memory());
  EXPECT_EQ(mm.planned_buffer_devices().size(), 0);
}

TEST(MemoryManagerTest, DelegatesDeviceMetadataToHierarchicalAllocator) {
  MemoryAllocator method_allocator(0, nullptr);
  MemoryAllocator temp_allocator(0, nullptr);

  // 4 buffers: cpu:0, cpu:0, cuda:0, cuda:1. CPU buffers come first because
  // the runtime always sets up host-side planned memory before any device
  // buffers. The two CUDA entries use distinct indices to verify per-buffer
  // index tracking.
  constexpr size_t n_buffers = 4;
  uint8_t mem0[4];
  uint8_t mem1[4];
  uint8_t mem2[4];
  uint8_t mem3[4];
  Span<uint8_t> buffers[n_buffers]{
      {mem0, sizeof(mem0)},
      {mem1, sizeof(mem1)},
      {mem2, sizeof(mem2)},
      {mem3, sizeof(mem3)},
  };
  Device devices[] = {
      Device(DeviceType::CPU, 0),
      Device(DeviceType::CPU, 0),
      Device(DeviceType::CUDA, 0),
      Device(DeviceType::CUDA, 1),
  };
  Span<const Device> device_span(devices, n_buffers);

  HierarchicalAllocator planned_memory({buffers, n_buffers}, device_span);
  MemoryManager mm(&method_allocator, &planned_memory, &temp_allocator);

  EXPECT_EQ(mm.method_allocator(), &method_allocator);
  EXPECT_EQ(mm.planned_memory(), &planned_memory);
  EXPECT_EQ(mm.temp_allocator(), &temp_allocator);
  EXPECT_TRUE(mm.has_device_memory());
  EXPECT_EQ(mm.planned_buffer_devices().size(), n_buffers);
  EXPECT_EQ(mm.planned_buffer_devices()[0], Device(DeviceType::CPU, 0));
  EXPECT_EQ(mm.planned_buffer_devices()[1], Device(DeviceType::CPU, 0));
  EXPECT_EQ(mm.planned_buffer_devices()[2], Device(DeviceType::CUDA, 0));
  EXPECT_EQ(mm.planned_buffer_devices()[3], Device(DeviceType::CUDA, 1));
}

TEST(MemoryManagerTest, MinimalCtorHasNoDeviceMemory) {
  MemoryAllocator method_allocator(0, nullptr);

  MemoryManager mm(&method_allocator);

  EXPECT_FALSE(mm.has_device_memory());
  EXPECT_EQ(mm.planned_buffer_devices().size(), 0);
}
