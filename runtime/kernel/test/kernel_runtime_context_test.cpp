/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_runtime_context.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;

class KernelRuntimeContextTest : public ::testing::Test {
 public:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

class TestMemoryAllocator : public MemoryAllocator {
 public:
  TestMemoryAllocator(uint32_t size, uint8_t* base_address)
      : MemoryAllocator(size, base_address), last_seen_alignment(0) {}
  void* allocate(size_t size, size_t alignment) override {
    last_seen_alignment = alignment;
    return MemoryAllocator::allocate(size, alignment);
  }
  size_t last_seen_alignment;
};

TEST_F(KernelRuntimeContextTest, FailureStateDefaultsToOk) {
  KernelRuntimeContext context;

  EXPECT_EQ(context.failure_state(), Error::Ok);
}

TEST_F(KernelRuntimeContextTest, FailureStateReflectsFailure) {
  KernelRuntimeContext context;

  // Starts off Ok.
  EXPECT_EQ(context.failure_state(), Error::Ok);

  // Failing should update the failure state.
  context.fail(Error::MemoryAllocationFailed);
  EXPECT_EQ(context.failure_state(), Error::MemoryAllocationFailed);

  // State can be overwritten.
  context.fail(Error::Internal);
  EXPECT_EQ(context.failure_state(), Error::Internal);

  // And can be cleared.
  context.fail(Error::Ok);
  EXPECT_EQ(context.failure_state(), Error::Ok);
}

TEST_F(KernelRuntimeContextTest, FailureNoMemoryAllocatorProvided) {
  KernelRuntimeContext context;
  Result<void*> allocated_memory = context.allocate_temp(4);
  EXPECT_EQ(allocated_memory.error(), Error::NotFound);
}

TEST_F(KernelRuntimeContextTest, SuccessfulMemoryAllocation) {
  constexpr size_t temp_memory_allocator_pool_size = 4;
  auto temp_memory_allocator_pool =
      std::make_unique<uint8_t[]>(temp_memory_allocator_pool_size);
  MemoryAllocator temp_allocator(
      temp_memory_allocator_pool_size, temp_memory_allocator_pool.get());
  KernelRuntimeContext context(nullptr, &temp_allocator);
  Result<void*> allocated_memory = context.allocate_temp(4);
  EXPECT_EQ(allocated_memory.ok(), true);
}

TEST_F(KernelRuntimeContextTest, FailureMemoryAllocationInsufficientSpace) {
  constexpr size_t temp_memory_allocator_pool_size = 4;
  auto temp_memory_allocator_pool =
      std::make_unique<uint8_t[]>(temp_memory_allocator_pool_size);
  MemoryAllocator temp_allocator(
      temp_memory_allocator_pool_size, temp_memory_allocator_pool.get());
  KernelRuntimeContext context(nullptr, &temp_allocator);
  Result<void*> allocated_memory = context.allocate_temp(8);
  EXPECT_EQ(allocated_memory.error(), Error::MemoryAllocationFailed);
}

TEST_F(KernelRuntimeContextTest, MemoryAllocatorAlignmentPassed) {
  constexpr size_t temp_memory_allocator_pool_size = 4;
  auto temp_memory_allocator_pool =
      std::make_unique<uint8_t[]>(temp_memory_allocator_pool_size);
  TestMemoryAllocator temp_allocator(
      temp_memory_allocator_pool_size, temp_memory_allocator_pool.get());
  KernelRuntimeContext context(nullptr, &temp_allocator);
  Result<void*> allocated_memory = context.allocate_temp(4, 2);
  EXPECT_EQ(allocated_memory.ok(), true);
  EXPECT_EQ(temp_allocator.last_seen_alignment, 2);
}
