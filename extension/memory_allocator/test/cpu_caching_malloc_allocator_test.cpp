#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <executorch/extension/memory_allocator/cpu_caching_malloc_allocator.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;
using executorch::extension::CPUCachingAllocator;

constexpr auto kDefaultAlignment =
    executorch::extension::kCachingAllocatorDefaultAlignment;

class CPUCachingAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

bool is_aligned(const void* ptr, size_t alignment) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  return (addr & (alignment - 1)) == 0;
}

#define EXPECT_ALIGNED(ptr, alignment)        \
  EXPECT_TRUE(is_aligned((ptr), (alignment))) \
      << "Pointer " << (ptr) << " is not aligned to " << (alignment)

TEST_F(CPUCachingAllocatorTest, SimpleAllocateSucceeds) {
  CPUCachingAllocator allocator(1024 * 1024); // 1MB max size

  auto p = allocator.allocate(16);
  EXPECT_NE(p, nullptr);
  EXPECT_ALIGNED(p, kDefaultAlignment);

  auto p2 = allocator.allocate(32);
  EXPECT_NE(p2, nullptr);
  EXPECT_ALIGNED(p2, kDefaultAlignment);

  auto p3 = allocator.allocate(64);
  EXPECT_NE(p3, nullptr);
  EXPECT_ALIGNED(p3, kDefaultAlignment);
}

TEST_F(CPUCachingAllocatorTest, CachingReusesSameSize) {
  CPUCachingAllocator allocator(1024 * 1024); // 1MB max size

  auto p1 = allocator.allocate(256);
  EXPECT_NE(p1, nullptr);
  EXPECT_ALIGNED(p1, kDefaultAlignment);

  // Reset to return the allocation to the cache
  allocator.reset();

  // Allocate the same size should reuse the cached pointer
  auto p2 = allocator.allocate(256);
  EXPECT_EQ(p1, p2);
  EXPECT_ALIGNED(p2, kDefaultAlignment);
}

TEST_F(CPUCachingAllocatorTest, DifferentSizesAllocateDifferentPtrs) {
  CPUCachingAllocator allocator(1024 * 1024); // 1MB max size

  auto p1 = allocator.allocate(128);
  auto p2 = allocator.allocate(256);
  auto p3 = allocator.allocate(512);

  EXPECT_NE(p1, nullptr);
  EXPECT_NE(p2, nullptr);
  EXPECT_NE(p3, nullptr);

  // All pointers should be different
  EXPECT_NE(p1, p2);
  EXPECT_NE(p2, p3);
  EXPECT_NE(p1, p3);

  EXPECT_ALIGNED(p1, kDefaultAlignment);
  EXPECT_ALIGNED(p2, kDefaultAlignment);
  EXPECT_ALIGNED(p3, kDefaultAlignment);
}

TEST_F(CPUCachingAllocatorTest, ResetCachesAllocations) {
  CPUCachingAllocator allocator(1024 * 1024); // 1MB max size

  auto p1 = allocator.allocate(256);
  auto p2 = allocator.allocate(256);
  EXPECT_NE(p1, p2);

  allocator.reset();

  // After reset, both cached allocations should be available
  auto p3 = allocator.allocate(256);
  auto p4 = allocator.allocate(256);

  // p3 should be one of the cached pointers (either p1 or p2)
  EXPECT_TRUE((p3 == p1) || (p3 == p2));
  EXPECT_TRUE((p4 == p1) || (p4 == p2));
  EXPECT_NE(p3, p4);
}

TEST_F(CPUCachingAllocatorTest, AlignmentParameter) {
  CPUCachingAllocator allocator(1024 * 1024); // 1MB max size

  std::vector<size_t> alignments = {
      kDefaultAlignment,
      kDefaultAlignment * 2,
      kDefaultAlignment * 4,
      kDefaultAlignment * 8,
  };

  for (size_t alignment : alignments) {
    auto p = allocator.allocate(256, alignment);
    EXPECT_NE(p, nullptr);
    EXPECT_ALIGNED(p, alignment);
  }
}

TEST_F(CPUCachingAllocatorTest, InvalidAlignmentFails) {
  CPUCachingAllocator allocator(1024 * 1024); // 1MB max size

  // Should fail because alignment is not a power of 2
  std::vector<size_t> invalid_alignments = {0, 5, 6, 12, 34};
  for (auto alignment : invalid_alignments) {
    auto p = allocator.allocate(256, alignment);
    EXPECT_EQ(p, nullptr);
  }
}

TEST_F(CPUCachingAllocatorTest, MaxSizeCanBeExceeded) {
  constexpr size_t kMaxSize = 1024; // 1KB max
  CPUCachingAllocator allocator(kMaxSize);

  // Allocate close to the max size
  auto p1 = allocator.allocate(512);
  EXPECT_NE(p1, nullptr);

  auto p2 = allocator.allocate(512);
  EXPECT_NE(p2, nullptr);

  // This should succeed even though we exceed max_size
  // The new behavior allows current_size to exceed max_size
  auto p3 = allocator.allocate(512);
  EXPECT_NE(p3, nullptr);

  // All pointers should be different
  EXPECT_NE(p1, p2);
  EXPECT_NE(p2, p3);
  EXPECT_NE(p1, p3);
}

TEST_F(CPUCachingAllocatorTest, MultipleAllocationsAndResets) {
  CPUCachingAllocator allocator(1024 * 1024); // 1MB max size

  for (int i = 0; i < 5; ++i) {
    auto p1 = allocator.allocate(256);
    auto p2 = allocator.allocate(512);
    auto p3 = allocator.allocate(1024);

    EXPECT_NE(p1, nullptr);
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p3, nullptr);

    allocator.reset();
  }
}

TEST_F(CPUCachingAllocatorTest, MemoryWriteability) {
  CPUCachingAllocator allocator(1024 * 1024); // 1MB max size

  const size_t size = 1024;
  auto p = allocator.allocate(size);
  EXPECT_NE(p, nullptr);

  // Write to allocated memory
  memset(p, 0x55, size);

  // Read back and verify
  uint8_t* bytes = reinterpret_cast<uint8_t*>(p);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(bytes[i], 0x55);
  }

  allocator.reset();
}

TEST_F(CPUCachingAllocatorTest, CachingWithMultipleSizes) {
  CPUCachingAllocator allocator(1024 * 1024); // 1MB max size

  // Allocate various sizes
  auto p1 = allocator.allocate(128);
  auto p2 = allocator.allocate(256);
  auto p3 = allocator.allocate(512);
  auto p4 = allocator.allocate(128);

  // Reset to cache them
  allocator.reset();

  // Allocate same sizes - should reuse cached pointers
  auto p5 = allocator.allocate(128);
  auto p6 = allocator.allocate(256);
  auto p7 = allocator.allocate(512);

  EXPECT_TRUE((p5 == p1) || (p5 == p4));
  EXPECT_EQ(p6, p2);
  EXPECT_EQ(p7, p3);
}

TEST_F(CPUCachingAllocatorTest, ThreadSafety) {
  CPUCachingAllocator allocator(4 * 1024 * 1024); // 4MB max size

  std::vector<std::thread> threads;
  std::vector<void*> allocated_ptrs;
  std::mutex ptrs_mutex;

  const int num_threads = 4;
  const int allocations_per_thread = 10;

  // Lambda function for thread work
  auto thread_work = [&]() {
    for (int i = 0; i < allocations_per_thread; ++i) {
      size_t size = (i + 1) * 64;
      auto p = allocator.allocate(size);
      EXPECT_NE(p, nullptr);
      EXPECT_ALIGNED(p, kDefaultAlignment);

      {
        std::lock_guard<std::mutex> guard(ptrs_mutex);
        allocated_ptrs.push_back(p);
      }
    }

    // Reset in each thread
    allocator.reset();
  };

  // Create threads
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_work);
  }

  // Wait for all threads to finish
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify all allocations were valid
  EXPECT_EQ(allocated_ptrs.size(), num_threads * allocations_per_thread);
}

TEST_F(CPUCachingAllocatorTest, LargeAllocation) {
  CPUCachingAllocator allocator(10 * 1024 * 1024); // 10MB max size

  const size_t large_size = 1024 * 1024; // 1MB allocation
  auto p = allocator.allocate(large_size);
  EXPECT_NE(p, nullptr);
  EXPECT_ALIGNED(p, kDefaultAlignment);

  // Write and verify
  memset(p, 0xAA, large_size);
  uint8_t* bytes = reinterpret_cast<uint8_t*>(p);
  for (size_t i = 0; i < 1000; ++i) { // Sample check
    EXPECT_EQ(bytes[i], 0xAA);
  }

  allocator.reset();

  // Re-allocate same size should reuse cached pointer
  auto p2 = allocator.allocate(large_size);
  EXPECT_EQ(p, p2);
}

TEST_F(CPUCachingAllocatorTest, SizeAlignmentAdjustment) {
  CPUCachingAllocator allocator(1024 * 1024); // 1MB max size

  // Test that allocation sizes get properly aligned
  auto p1 = allocator.allocate(100, 256); // Size aligned to 256
  EXPECT_NE(p1, nullptr);
  EXPECT_ALIGNED(p1, 256);

  allocator.allocate(100, 256);
  // Should not get cached pointer since size was adjusted during first
  // allocation
  allocator.reset();

  auto p3 = allocator.allocate(100, 512);
  // Should reuse p1 due to alignment adjustment
  EXPECT_NE(p1, p3);
}

TEST_F(CPUCachingAllocatorTest, ResetMultipleTimes) {
  CPUCachingAllocator allocator(1024 * 1024); // 1MB max size

  for (int i = 0; i < 3; ++i) {
    auto p = allocator.allocate(512);
    EXPECT_NE(p, nullptr);
    allocator.reset();

    auto p2 = allocator.allocate(512);
    EXPECT_EQ(p, p2);
    allocator.reset();
  }
}

TEST_F(CPUCachingAllocatorTest, ResetFreesEverythingWhenOverMaxSize) {
  constexpr size_t kMaxSize = 1024; // 1KB max
  CPUCachingAllocator allocator(kMaxSize);

  // Allocate more than max_size
  auto p1 = allocator.allocate(512);
  auto p2 = allocator.allocate(512);
  auto p3 = allocator.allocate(512);
  EXPECT_NE(p1, nullptr);
  EXPECT_NE(p2, nullptr);
  EXPECT_NE(p3, nullptr);

  // Reset should free everything since current_size (1536) > max_size (1024)
  allocator.reset();

  // Subsequent allocations should not reuse any of the old pointers
  auto p4 = allocator.allocate(512);
  auto p5 = allocator.allocate(512);
  EXPECT_NE(p4, nullptr);
  EXPECT_NE(p5, nullptr);

  // These should be new allocations, not cached ones
  // However, system allocator might cache and return the same pointesr
  // so we can't check for strict equality or inequality
}

TEST_F(CPUCachingAllocatorTest, ResetCachesWhenUnderMaxSize) {
  constexpr size_t kMaxSize = 2048; // 2KB max
  CPUCachingAllocator allocator(kMaxSize);

  // Allocate less than max_size
  auto p1 = allocator.allocate(512);
  auto p2 = allocator.allocate(512);
  EXPECT_NE(p1, nullptr);
  EXPECT_NE(p2, nullptr);

  // Reset should cache the allocations since current_size (1024) <= max_size
  // (2048)
  allocator.reset();

  // Subsequent allocations should reuse the cached pointers
  auto p3 = allocator.allocate(512);
  auto p4 = allocator.allocate(512);
  EXPECT_NE(p3, nullptr);
  EXPECT_NE(p4, nullptr);

  // Should reuse cached pointers
  EXPECT_TRUE((p3 == p1) || (p3 == p2));
  EXPECT_TRUE((p4 == p1) || (p4 == p2));
  EXPECT_NE(p3, p4);
}
