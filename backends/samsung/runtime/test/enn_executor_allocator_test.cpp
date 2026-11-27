/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#include <executorch/backends/samsung/runtime/enn_executor_allocator.h>

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

namespace torch {
namespace executor {
namespace enn {
namespace {

class FailingMemoryAllocator final : public MemoryAllocator {
 public:
  FailingMemoryAllocator() : MemoryAllocator(0, nullptr) {}

  void* allocate(size_t, size_t) override {
    return nullptr;
  }
};

TEST(EnnExecutorAllocatorTest, ConstructsExecutorInPoisonedStorage) {
  alignas(EnnExecutor) std::array<uint8_t, sizeof(EnnExecutor)> storage;
  storage.fill(0x30);
  MemoryAllocator allocator(storage.size(), storage.data());

  EnnExecutor* executor = allocateEnnExecutor(allocator);

  ASSERT_NE(executor, nullptr);
  EXPECT_EQ(executor->getInputSize(), 0);
  EXPECT_EQ(executor->getOutputSize(), 0);
}

TEST(EnnExecutorAllocatorTest, ReturnsNullWhenAllocationFails) {
  FailingMemoryAllocator allocator;

  EXPECT_EQ(allocateEnnExecutor(allocator), nullptr);
}

} // namespace
} // namespace enn
} // namespace executor
} // namespace torch
