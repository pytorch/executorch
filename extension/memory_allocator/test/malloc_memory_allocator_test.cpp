/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::MallocMemoryAllocator;

constexpr auto kDefaultAlignment = MallocMemoryAllocator::kDefaultAlignment;

class MallocMemoryAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

bool is_aligned(const void* ptr, size_t alignment) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  return addr % alignment == 0;
}

#define EXPECT_ALIGNED(ptr, alignment)        \
  EXPECT_TRUE(is_aligned((ptr), (alignment))) \
      << "Pointer " << (ptr) << " is not aligned to " << (alignment)

TEST_F(MallocMemoryAllocatorTest, IsAlignedTest) {
  struct TestCase {
    uintptr_t address;
    size_t alignment;
    bool expected;
  };
  std::vector<TestCase> tests{
      {0xffff0, 0x1, true},
      {0xffff0, 0x2, true},
      {0xffff0, 0x8, true},
      {0xffff0, 0x10, true},
      {0xffff0, 0x20, false},
      {0xffff0, 0x40, false},
      {0xffff1000, 0x1000, true},
      {0xffff1000, 0x10000, false},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(
        is_aligned(reinterpret_cast<void*>(test.address), test.alignment),
        test.expected);
  }
}

TEST_F(MallocMemoryAllocatorTest, SimpleAllocateSucceeds) {
  MallocMemoryAllocator allocator = MallocMemoryAllocator();

  auto p = allocator.allocate(16);
  EXPECT_NE(p, nullptr);
  EXPECT_ALIGNED(p, kDefaultAlignment);

  auto p2 = allocator.allocate(16);
  EXPECT_NE(p, nullptr);
  EXPECT_NE(p2, p);
  EXPECT_ALIGNED(p2, kDefaultAlignment);

  auto p3 = allocator.allocate(16);
  EXPECT_NE(p3, p2);
  EXPECT_NE(p3, p);
  EXPECT_ALIGNED(p3, kDefaultAlignment);
}

TEST_F(MallocMemoryAllocatorTest, AlignmentSmokeTest) {
  MallocMemoryAllocator allocator = MallocMemoryAllocator();

  // A set of alignments that alternate between big and small. The behavior of
  // this test will depend on the state of the heap.
  std::vector<size_t> alignments = {
      kDefaultAlignment * 64,
      kDefaultAlignment * 8,
      kDefaultAlignment * 16,
      kDefaultAlignment * 2,
      kDefaultAlignment * 32,
      kDefaultAlignment / 2,
      kDefaultAlignment * 128,
      kDefaultAlignment,
      kDefaultAlignment * 4,
  };

  static constexpr int kNumPasses = 100;
  for (int pass = 0; pass < kNumPasses; ++pass) {
    for (size_t alignment : alignments) {
      constexpr size_t kAllocationSize = 16;
      auto p = allocator.allocate(kAllocationSize, alignment);
      EXPECT_NE(p, nullptr);
      EXPECT_ALIGNED(p, alignment);
      // Write to the allocated memory. If it overruns, ASAN should catch it.
      memset(p, 0x55, kAllocationSize);
    }
  }
}

TEST_F(MallocMemoryAllocatorTest, BadAlignmentFails) {
  MallocMemoryAllocator allocator = MallocMemoryAllocator();

  // Should fail because the requested alignment is not a power of 2.
  std::vector<size_t> alignments = {0, 5, 6, 12, 34};
  for (auto alignment : alignments) {
    auto p = allocator.allocate(16, alignment);
    EXPECT_EQ(p, nullptr);
  }
}

TEST_F(MallocMemoryAllocatorTest, ResetSucceeds) {
  MallocMemoryAllocator allocator = MallocMemoryAllocator();

  auto p = allocator.allocate(16);
  EXPECT_NE(p, nullptr);
  EXPECT_ALIGNED(p, kDefaultAlignment);

  allocator.reset();

  // Continue to allocate successfully.
  p = allocator.allocate(16);
  EXPECT_NE(p, nullptr);
  EXPECT_ALIGNED(p, kDefaultAlignment);
}
