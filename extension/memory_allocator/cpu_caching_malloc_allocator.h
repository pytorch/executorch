#pragma once

#include <cstddef>
#include <mutex>

#include <executorch/runtime/core/memory_allocator.h>

#ifdef USE_C10_SMALL_VECTOR
#include <c10/util/SmallVector.h>
#else
#include <vector>
#endif

#ifdef USE_C10_FLAT_HASH_MAP
#include <c10/util/flat_hash_map.h>
#else
#include <unordered_map>
#endif

/*
 * CPUCachingAllocator:
 * This file is copied over from c10/mobile/CPUCachingAllocator.h
 * It is a thread safe caching allocator.
 */

namespace executorch::extension {

#ifdef USE_C10_SMALL_VECTOR
template <typename T, unsigned N>
using SmallVector = c10::SmallVector<T, N>;
#else
template <typename T, unsigned N>
using SmallVector = std::vector<T>;
#endif

#ifdef USE_C10_FLAT_HASH_MAP
template <typename KeyType, typename ValueType>
using FlatHashMap = ska::flat_hash_map<KeyType, ValueType>;
#else
template <typename KeyType, typename ValueType>
using FlatHashMap = std::unordered_map<KeyType, ValueType>;
#endif

constexpr size_t kCachingAllocatorDefaultAlignment = 64;
class CPUCachingAllocator : public executorch::runtime::MemoryAllocator {
  /*
   * What it does:
   * Caches all the allocations carried out by this allocator.
   * Cache key is the size of the allocation.
   * If requested size is found in the cache returns the cached pointer.
   * What it does not do:
   * No speculative allocation for any future allocations.
   */
 private:
  void free_everything();

 protected:
  // Invariants.
  // New invariants must be written.
  FlatHashMap<size_t, SmallVector<void*, 16>> available_map_;
  FlatHashMap<void*, size_t> allocation_map_;
  // Since allocation_map_ and other member variables are mutated/read via
  // all public APIs, we need a mutex to protect concurrent access to these
  // instance members.
  std::mutex mutex_;
  size_t max_size_;
  size_t current_size_;

 public:
  /*
    max_size: Maximum size of memory to cache. Never cache more than that.
  */
  explicit CPUCachingAllocator(uint32_t max_size);
  // No copies allowed
  CPUCachingAllocator(const CPUCachingAllocator&) = delete;
  CPUCachingAllocator& operator=(const CPUCachingAllocator&) = delete;
  // No moves allowed
  CPUCachingAllocator(CPUCachingAllocator&&) = delete;
  CPUCachingAllocator& operator=(CPUCachingAllocator&&) = delete;
  // Checks the cache to see if allocation of size bytes can be found.
  // If so return cached memory, else
  // allocates memory, records it for caching and returns.
  void* allocate(
      size_t size,
      size_t alignment = kCachingAllocatorDefaultAlignment) override;
  void reset() override;
  ~CPUCachingAllocator();
};

} // namespace executorch::extension
