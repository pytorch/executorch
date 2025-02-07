/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <vector>

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

  for (int i = 0; i < arr_size; i++) {
    auto align_size = alignment[i];
    constexpr size_t mem_size = 1000;
    uint8_t mem_pool[mem_size];
    MemoryAllocator allocator = MemoryAllocator(mem_size, mem_pool);
    for (int j = 0; j < arr_size; j++) {
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
  for (int i = 0; i < 5; i++) {
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

#if ET_HAVE_GNU_STATEMENT_EXPRESSIONS
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

void* try_allocate_helper(
    MemoryAllocator* allocator,
    size_t nbytes,
    Error* out_error) {
  return ET_TRY_ALLOCATE_OR(allocator, nbytes, {
    // An example that doesn't simply return.
    *out_error = kTestFailureValue;
    return nullptr;
  });
}

TEST_F(HelperMacrosTest, TryAllocateSuccess) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate less memory than the allocator provides, which should succeed.
  Error err = Error::Ok;
  void* p = try_allocate_helper(&allocator, allocator.size() / 2, &err);
  EXPECT_EQ(err, Error::Ok);
  EXPECT_NE(p, nullptr);
}

TEST_F(HelperMacrosTest, TryAllocateFailure) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate more memory than the allocator provides, which should fail.
  Error err = Error::Ok;
  void* p = try_allocate_helper(&allocator, allocator.size() * 2, &err);
  EXPECT_EQ(err, kTestFailureValue);
  EXPECT_EQ(p, nullptr);
}

Error allocate_or_return_error_helper(
    MemoryAllocator* allocator,
    size_t nbytes,
    void** out_pointer) {
  *out_pointer = ET_ALLOCATE_OR_RETURN_ERROR(allocator, nbytes);
  return Error::Ok;
}

TEST_F(HelperMacrosTest, AllocateOrReturnSuccess) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate less memory than the allocator provides, which should succeed.
  void* p;
  Error err =
      allocate_or_return_error_helper(&allocator, allocator.size() / 2, &p);
  EXPECT_EQ(err, Error::Ok);
  EXPECT_NE(p, nullptr);
}

TEST_F(HelperMacrosTest, AllocateOrReturnFailure) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate more memory than the allocator provides, which should fail.
  void* p;
  Error err =
      allocate_or_return_error_helper(&allocator, allocator.size() * 2, &p);
  EXPECT_EQ(err, Error::MemoryAllocationFailed);
}

template <typename T>
T* try_allocate_instance_helper(MemoryAllocator* allocator, Error* out_error) {
  return ET_TRY_ALLOCATE_INSTANCE_OR(allocator, T, {
    // An example that doesn't simply return.
    *out_error = kTestFailureValue;
    return nullptr;
  });
}

TEST_F(HelperMacrosTest, TryAllocateInstanceSuccess) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate less memory than the allocator provides, which should succeed.
  Error err = Error::Ok;
  TestType8* p = try_allocate_instance_helper<TestType8>(&allocator, &err);
  EXPECT_EQ(err, Error::Ok);
  EXPECT_NE(p, nullptr);
}

TEST_F(HelperMacrosTest, TryAllocateInstanceFailure) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate more memory than the allocator provides, which should fail.
  Error err = Error::Ok;
  TestType1024* p =
      try_allocate_instance_helper<TestType1024>(&allocator, &err);
  EXPECT_EQ(err, kTestFailureValue);
}

template <typename T>
Error allocate_instance_or_return_error_helper(
    MemoryAllocator* allocator,
    void** out_pointer) {
  *out_pointer = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(allocator, T);
  return Error::Ok;
}

TEST_F(HelperMacrosTest, AllocateInstanceOrReturnSuccess) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate less memory than the allocator provides, which should succeed.
  void* p;
  Error err =
      allocate_instance_or_return_error_helper<TestType8>(&allocator, &p);
  EXPECT_EQ(err, Error::Ok);
  EXPECT_NE(p, nullptr);
}

TEST_F(HelperMacrosTest, AllocateInstanceOrReturnFailure) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate more memory than the allocator provides, which should fail.
  void* p;
  Error err =
      allocate_instance_or_return_error_helper<TestType1024>(&allocator, &p);
  EXPECT_EQ(err, Error::MemoryAllocationFailed);
}

void* try_allocate_list_helper(
    MemoryAllocator* allocator,
    size_t nbytes,
    Error* out_error) {
  // Use a 1-sized type so that nbytes == nelem.
  return ET_TRY_ALLOCATE_LIST_OR(allocator, uint8_t, nbytes, {
    // An example that doesn't simply return.
    *out_error = kTestFailureValue;
    return nullptr;
  });
}

TEST_F(HelperMacrosTest, TryAllocateListSuccess) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate less memory than the allocator provides, which should succeed.
  Error err = Error::Ok;
  void* p = try_allocate_list_helper(&allocator, allocator.size() / 2, &err);
  EXPECT_EQ(err, Error::Ok);
  EXPECT_NE(p, nullptr);
}

TEST_F(HelperMacrosTest, TryAllocateListFailure) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate more memory than the allocator provides, which should fail.
  Error err = Error::Ok;
  void* p = try_allocate_list_helper(&allocator, allocator.size() * 2, &err);
  EXPECT_EQ(err, kTestFailureValue);
  EXPECT_EQ(p, nullptr);
}

Error allocate_list_or_return_error_helper(
    MemoryAllocator* allocator,
    size_t nbytes,
    void** out_pointer) {
  // Use a 1-sized type so that nbytes == nelem.
  *out_pointer = ET_ALLOCATE_LIST_OR_RETURN_ERROR(allocator, uint8_t, nbytes);
  return Error::Ok;
}

TEST_F(HelperMacrosTest, AllocateListOrReturnSuccess) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate less memory than the allocator provides, which should succeed.
  void* p;
  Error err = allocate_list_or_return_error_helper(
      &allocator, allocator.size() / 2, &p);
  EXPECT_EQ(err, Error::Ok);
  EXPECT_NE(p, nullptr);
}

TEST_F(HelperMacrosTest, AllocateListOrReturnFailure) {
  std::array<uint8_t, 16> buffer;
  MemoryAllocator allocator(buffer.size(), buffer.data());

  // Allocate more memory than the allocator provides, which should fail.
  void* p;
  Error err = allocate_list_or_return_error_helper(
      &allocator, allocator.size() * 2, &p);
  EXPECT_EQ(err, Error::MemoryAllocationFailed);
}
#endif // ET_HAVE_GNU_STATEMENT_EXPRESSIONS
