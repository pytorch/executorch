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

TEST(MemoryManagerTest, FourArgCtorWithDeviceMetadata) {
  MemoryAllocator method_allocator(0, nullptr);
  HierarchicalAllocator planned_memory({});
  MemoryAllocator temp_allocator(0, nullptr);

  // 3 buffers: CPU, CUDA, CPU
  DeviceType devices[] = {DeviceType::CPU, DeviceType::CUDA, DeviceType::CPU};
  Span<const DeviceType> device_span(devices, 3);

  MemoryManager mm(
      &method_allocator, &planned_memory, &temp_allocator, device_span);

  EXPECT_EQ(mm.method_allocator(), &method_allocator);
  EXPECT_EQ(mm.planned_memory(), &planned_memory);
  EXPECT_EQ(mm.temp_allocator(), &temp_allocator);
  EXPECT_TRUE(mm.has_device_memory());
  EXPECT_EQ(mm.planned_buffer_devices().size(), 3);
  EXPECT_EQ(mm.planned_buffer_devices()[0], DeviceType::CPU);
  EXPECT_EQ(mm.planned_buffer_devices()[1], DeviceType::CUDA);
  EXPECT_EQ(mm.planned_buffer_devices()[2], DeviceType::CPU);
}

TEST(MemoryManagerTest, MinimalCtorHasNoDeviceMemory) {
  MemoryAllocator method_allocator(0, nullptr);

  MemoryManager mm(&method_allocator);

  EXPECT_FALSE(mm.has_device_memory());
  EXPECT_EQ(mm.planned_buffer_devices().size(), 0);
}
