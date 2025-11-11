/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>

#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/alignment.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::Error;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::Span;

class HierarchicalAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

TEST_F(HierarchicalAllocatorTest, Smoke) {
  constexpr size_t n_buffers = 2;
  constexpr size_t size0 = 4;
  constexpr size_t size1 = 8;
  uint8_t mem0[size0];
  uint8_t mem1[size1];
  Span<uint8_t> buffers[n_buffers]{
      {mem0, size0},
      {mem1, size1},
  };

  HierarchicalAllocator allocator({buffers, n_buffers});

  // get_offset_address() success cases
  {
    // Total size is 4, so off=0 + size=2 fits.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/0, /*offset_bytes=*/0, /*size_bytes=*/2);
    ASSERT_EQ(address.error(), Error::Ok);
    ASSERT_NE(address.get(), nullptr);
    ASSERT_EQ(address.get(), mem0);
  }
  {
    // Total size is 8, so off=4 + size=4 fits exactly.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/1, /*offset_bytes=*/4, /*size_bytes=*/4);
    ASSERT_EQ(address.error(), Error::Ok);
    ASSERT_NE(address.get(), nullptr);
    ASSERT_EQ(address.get(), mem1 + 4);
  }

  // get_offset_address() failure cases
  {
    // Total size is 4, so off=0 + size=5 is too large.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/0, /*offset_bytes=*/4, /*size_bytes=*/5);
    ASSERT_FALSE(address.ok());
    ASSERT_NE(address.error(), Error::Ok);
  }
  {
    // Total size is 4, so off=8 + size=0 is off the end.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/0, /*offset_bytes=*/8, /*size_bytes=*/0);
    ASSERT_FALSE(address.ok());
    ASSERT_NE(address.error(), Error::Ok);
  }
  {
    // ID too large; only two zero-indexed entries in the allocator.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/2, /*offset_bytes=*/0, /*size_bytes=*/2);
    ASSERT_FALSE(address.ok());
    ASSERT_NE(address.error(), Error::Ok);
  }
}

// TODO(T162089316): Tests the deprecated API. Remove this when removing the
// API.
TEST_F(HierarchicalAllocatorTest, DEPRECATEDSmoke) {
  constexpr size_t n_allocators = 2;
  constexpr size_t size0 = 4;
  constexpr size_t size1 = 8;
  uint8_t mem0[size0];
  uint8_t mem1[size1];
  MemoryAllocator allocators[n_allocators]{
      MemoryAllocator(size0, mem0), MemoryAllocator(size1, mem1)};

  HierarchicalAllocator allocator(n_allocators, allocators);

  // get_offset_address() success cases
  {
    // Total size is 4, so off=0 + size=2 fits.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/0, /*offset_bytes=*/0, /*size_bytes=*/2);
    ASSERT_EQ(address.error(), Error::Ok);
    ASSERT_NE(address.get(), nullptr);
    ASSERT_EQ(address.get(), mem0);
  }
  {
    // Total size is 8, so off=4 + size=4 fits exactly.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/1, /*offset_bytes=*/4, /*size_bytes=*/4);
    ASSERT_EQ(address.error(), Error::Ok);
    ASSERT_NE(address.get(), nullptr);
    ASSERT_EQ(address.get(), mem1 + 4);
  }

  // get_offset_address() failure cases
  {
    // Total size is 4, so off=0 + size=5 is too large.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/0, /*offset_bytes=*/4, /*size_bytes=*/5);
    ASSERT_FALSE(address.ok());
    ASSERT_NE(address.error(), Error::Ok);
  }
  {
    // Total size is 4, so off=8 + size=0 is off the end.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/0, /*offset_bytes=*/8, /*size_bytes=*/0);
    ASSERT_FALSE(address.ok());
    ASSERT_NE(address.error(), Error::Ok);
  }
  {
    // ID too large; only two zero-indexed entries in the allocator.
    Result<void*> address = allocator.get_offset_address(
        /*memory_id=*/2, /*offset_bytes=*/0, /*size_bytes=*/2);
    ASSERT_FALSE(address.ok());
    ASSERT_NE(address.error(), Error::Ok);
  }
}
