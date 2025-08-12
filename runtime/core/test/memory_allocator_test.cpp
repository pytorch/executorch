/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <vector>

#include <c10/util/irange.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/alignment.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::Error;
using executorch::runtime::MemoryAllocator;

struct TestType8 {
  char data[8];
};
static_assert(sizeof(TestType8) == 8);

struct TestType1024 {
  char data[1024];
};
static_assert(sizeof(TestType1024) == 1024);

class MemoryAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

TEST_F(MemoryAllocatorTest, MemoryAllocator) {
  constexpr size_t mem_size = 16;
  uint8_t mem_pool[mem_size];
  MemoryAllocator allocator(mem_size, mem_pool);
  ASSERT_NE(nullptr, allocator.allocate(7));
  ASSERT_NE(nullptr, allocator.allocate(6));
  ASSERT_EQ(nullptr, allocator.allocate(3));

  allocator.reset();
  ASSERT_EQ(mem_pool, allocator.allocate(0));
  ASSERT_NE(nullptr, allocator.allocate(16));
}

TEST_F(MemoryAllocatorTest, MemoryAllocatorAlignment) {
  constexpr size_t arr_size = 6;
  size_t allocation[arr_size] = {7, 6, 3, 76, 4, 1};
  size_t alignment[arr_size] = {
      MemoryAllocator::kDefaultAlignment,
      MemoryAllocator::kDefaultAlignment,
      4,
      32,
      128,
      2};

  for (const auto i : c10::irange(arr_size)) {
    auto align_size = alignment[i];
    constexpr size_t mem_size = 1000;
    uint8_t mem_pool[mem_size];
    MemoryAllocator allocator = MemoryAllocator(mem_size, mem_pool);
    for (const auto j : c10::irange(arr_size)) {
      auto size = allocation[j];
      void* start = allocator.allocate(size, align_size);
      EXPECT_ALIGNED(start, align_size);
    }
  }
}

TEST_F(MemoryAllocatorTest, MemoryAllocatorNonPowerOfTwoAlignment) {
  constexpr size_t mem_size = 128;
  uint8_t mem_pool[mem_size];
  MemoryAllocator allocator(mem_size, mem_pool);

  size_t alignment[5] = {0, 5, 6, 12, 34};
  for (const auto i : c10::irange(5)) {
    ASSERT_EQ(nullptr, allocator.allocate(8, alignment[i]));
  }
}

TEST_F(MemoryAllocatorTest, MemoryAllocatorTooLargeFailButSucceedAfterwards) {
  constexpr size_t kPoolSize = 10;
  uint8_t mem_pool[kPoolSize];
  MemoryAllocator allocator(kPoolSize, mem_pool);
  // Align to 1 byte so the entire pool is used. The default alignment could
  // skip over the first few bytes depending on the alignment of `mem_pool`.
  ASSERT_EQ(nullptr, allocator.allocate(kPoolSize + 2, /*alignment=*/1));
  ASSERT_NE(nullptr, allocator.allocate(kPoolSize - 1, /*alignment=*/1));
}

template <typename T>
static void test_allocate_instance() {
  std::array<uint8_t, 256> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Default alignment
  auto p = allocator.allocateInstance<T>();
  EXPECT_NE(p, nullptr);
  EXPECT_ALIGNED(p, alignof(T));
  memset(p, 0x55, sizeof(T));

  // Override alignment
  constexpr size_t kHigherAlignment = 64;
  EXPECT_GT(kHigherAlignment, alignof(T));
  p = allocator.allocateInstance<T>(kHigherAlignment);
  EXPECT_NE(p, nullptr);
  EXPECT_ALIGNED(p, kHigherAlignment);
  memset(p, 0x55, sizeof(T));
}

TEST_F(MemoryAllocatorTest, AllocateInstance) {
  test_allocate_instance<uint8_t>();
  test_allocate_instance<uint16_t>();
  test_allocate_instance<uint32_t>();
  test_allocate_instance<uint64_t>();
  test_allocate_instance<void*>();

  struct StructWithPointer {
    void* p;
    int i;
  };
  test_allocate_instance<StructWithPointer>();

  struct StructWithLargestType {
    std::max_align_t max;
    int i;
  };
  test_allocate_instance<StructWithLargestType>();
}

TEST_F(MemoryAllocatorTest, AllocateInstanceFailure) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate more memory than the allocator provides, which should fail.
  auto p = allocator.allocateInstance<TestType1024>();
  EXPECT_EQ(p, nullptr);
}

template <typename T>
static void test_allocate_list() {
  std::array<uint8_t, 256> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Default alignment
  constexpr size_t kNumElem = 5;
  auto p = allocator.allocateList<T>(kNumElem);
  ASSERT_NE(p, nullptr);
  EXPECT_ALIGNED(p, alignof(T));
  memset(p, 0x55, kNumElem * sizeof(T));

  // Override alignment
  constexpr size_t kHigherAlignment = 64;
  EXPECT_GT(kHigherAlignment, alignof(T));
  p = allocator.allocateList<T>(kNumElem, kHigherAlignment);
  ASSERT_NE(p, nullptr);
  EXPECT_ALIGNED(p, kHigherAlignment);
  memset(p, 0x55, kNumElem * sizeof(T));
}

TEST_F(MemoryAllocatorTest, AllocateList) {
  test_allocate_list<uint8_t>();
  test_allocate_list<uint16_t>();
  test_allocate_list<uint32_t>();
  test_allocate_list<uint64_t>();
  test_allocate_list<void*>();

  struct StructWithPointer {
    void* p;
    char c;
  };
  test_allocate_instance<StructWithPointer>();

  struct StructWithLargestType {
    std::max_align_t max;
    int i;
  };
  test_allocate_instance<StructWithLargestType>();
}

TEST_F(MemoryAllocatorTest, AllocateListFailure) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate more memory than the allocator provides, which should fail.
  auto p = allocator.allocateList<TestType8>(10);
  EXPECT_EQ(p, nullptr);
}

class HelperMacrosTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

/**
 * An Error value that doesn't map to anything defined in Error.h. Helps
 * demonstrate that the code here is being executed, and that the macro isn't
 * returning a canned Error value.
 */
static const Error kTestFailureValue = static_cast<Error>(12345);

TEST_F(HelperMacrosTest, TryAllocateSuccess) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate less memory than the allocator provides, which should succeed.
  void* p = allocator.allocate(allocator.size() / 2);
  EXPECT_NE(p, nullptr);
}

TEST_F(HelperMacrosTest, TryAllocateFailure) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate more memory than the allocator provides, which should fail.
  void* p = allocator.allocate(allocator.size() * 2);
  EXPECT_EQ(p, nullptr);
}

TEST_F(HelperMacrosTest, TryAllocateInstanceSuccess) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate less memory than the allocator provides, which should succeed.
  TestType8* p = allocator.allocateInstance<TestType8>();
  EXPECT_NE(p, nullptr);
}

TEST_F(HelperMacrosTest, TryAllocateInstanceFailure) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate more memory than the allocator provides, which should fail.
  TestType1024* p = allocator.allocateInstance<TestType1024>();
  EXPECT_EQ(p, nullptr);
}

TEST_F(HelperMacrosTest, TryAllocateListSuccess) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate less memory than the allocator provides, which should succeed.
  void* p = allocator.allocateList<uint8_t>(allocator.size() / 2);
  EXPECT_NE(p, nullptr);
}

TEST_F(HelperMacrosTest, TryAllocateListFailure) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate more memory than the allocator provides, which should fail.
  void* p = allocator.allocateList<uint8_t>(allocator.size() * 2);
  EXPECT_EQ(p, nullptr);
}
