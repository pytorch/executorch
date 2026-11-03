/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <gtest/gtest.h>

namespace {
void* allocated_block;
size_t requested_bytes;
size_t requested_alignment;
void* freed_block;
void* bound_tensor;
void* bound_block;
} // namespace

void* QnnExecuTorchAllocCustomMem(size_t bytes, size_t alignment) {
  requested_bytes = bytes;
  requested_alignment = alignment;
  return allocated_block;
}

void QnnExecuTorchFreeCustomMem(void* block) {
  freed_block = block;
}

void QnnExecuTorchAddCustomMemTensorAddr(void* tensor, void* block) {
  bound_tensor = tensor;
  bound_block = block;
}

TEST(QnnIOBufferAllocatorTest, RegistersAndForwardsAllocationBindingAndFree) {
  auto& factory = executorch::extension::llm::io_buffer_allocator_factory();
  ASSERT_TRUE(factory);
  auto allocator = factory();
  ASSERT_NE(allocator, nullptr);
  alignas(64) char memory[128];
  allocated_block = memory;
  EXPECT_EQ(allocator->allocate(sizeof(memory), 64), memory);
  EXPECT_EQ(requested_bytes, sizeof(memory));
  EXPECT_EQ(requested_alignment, 64);
  allocator->bind(memory, memory + 32);
  EXPECT_EQ(bound_tensor, memory + 32);
  EXPECT_EQ(bound_block, memory);
  allocator->deallocate(memory);
  EXPECT_EQ(freed_block, memory);
}

TEST(QnnIOBufferAllocatorTest, PreservesFailedAllocationForHeapFallback) {
  auto& factory = executorch::extension::llm::io_buffer_allocator_factory();
  ASSERT_TRUE(factory);
  allocated_block = nullptr;
  auto allocator = factory();
  EXPECT_EQ(allocator->allocate(1024, 64), nullptr);
  EXPECT_EQ(requested_bytes, 1024);
  EXPECT_EQ(requested_alignment, 64);
}
