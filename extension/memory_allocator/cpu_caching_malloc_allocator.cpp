#include <cstdlib>

#include <executorch/extension/memory_allocator/cpu_caching_malloc_allocator.h>
#include <executorch/extension/memory_allocator/memory_allocator_utils.h>

namespace executorch::extension {

CPUCachingAllocator::CPUCachingAllocator(uint32_t max_size)
    : MemoryAllocator(0, nullptr) {
  max_size_ = max_size;
  current_size_ = 0;
}

void* CPUCachingAllocator::allocate(size_t size, size_t alignment) {
  EXECUTORCH_TRACK_ALLOCATION(prof_id(), size);

  if (!isPowerOf2(alignment)) {
    ET_LOG(Error, "Alignment %zu is not a power of 2", alignment);
    return nullptr;
  }
  alignment = std::max(alignment, kCachingAllocatorDefaultAlignment);
  auto adjusted_size_value =
      executorch::extension::utils::get_aligned_size(size, alignment);
  if (!adjusted_size_value.ok()) {
    return nullptr;
  }
  size = adjusted_size_value.get();

  std::lock_guard<std::mutex> guard(mutex_);
  const auto& it = available_map_.find(size);
  // Two choices here.
  // 1. Return cached memory
  // 2. Allocate new memory
  // 2 can lead to current_size > max_size_
  if (it == available_map_.end() || it->second.empty()) {
    void* ptr = std::malloc(size);
    if (ptr == nullptr) {
      ET_LOG(Error, "Failed to allocate memory");
      return nullptr;
    }
    current_size_ += size;
    allocation_map_[ptr] = size;
    return alignPointer(ptr, alignment);
  }
  void* ptr = it->second.back();
  it->second.pop_back();
  allocation_map_[ptr] = size;
  return alignPointer(ptr, alignment);
}

void CPUCachingAllocator::free_everything() {
  // We dont lock mutex_ here because it will cause deadlock otherwise
  // we could use recursive_mutex but we just design this differently since
  // free_cache is not a public API anyways
  for (const auto& it : available_map_) {
    for (const auto ptr : it.second) {
      std::free(ptr);
    }
  }
  available_map_.clear();
  for (const auto& it : allocation_map_) {
    void* ptr = it.first;
    std::free(ptr);
  }
  allocation_map_.clear();
  // Note that purely by the design, clearing available map does not
  // mean that our current allocated size is zero.
  current_size_ = 0;
}

void CPUCachingAllocator::reset() {
  std::lock_guard<std::mutex> guard(mutex_);
  // We make the default allocations, via allcate to be either
  // a. gotten via cached memory OR
  // b. allocated via malloced and not yet cached
  // So if current_size_ (allocated) is larger than the max_size_
  // for now we simply deallocate everything.
  if (current_size_ > max_size_) {
    free_everything();
  } else {
    for (auto& it : allocation_map_) {
      void* ptr = it.first;
      size_t alloc_size = it.second;
      // Cache the memory
      available_map_[alloc_size].push_back(ptr);
    }
    allocation_map_.clear();
  }
}

CPUCachingAllocator::~CPUCachingAllocator() {
  // destructor must be called in thread safe manner
  reset();
  free_everything();
}

} // namespace executorch::extension
