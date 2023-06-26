#pragma once

#include <executorch/core/Constants.h>
#include <executorch/executor/MemoryManager.h>
#include <memory>

// mimic a typical memory pool pre-allocation in an embedded system.
// It's defined by the user and included in the application when linking to
// lib executorch

// Number of pools used. Pool zero is defaulted to constant data in flatbuffer
constexpr size_t NUM_NON_CONSTANT_POOLS = 1;

// NON_CONSTANT_POOL_SIZE is set at compilation
constexpr size_t NON_CONSTANT_POOL_SIZE = 2 * torch::executor::kMB;

// Memory used to save executor related structures
constexpr size_t EXECUTOR_POOL_SIZE = 128 * torch::executor::kKB;

namespace torch {
namespace executor {
// A simple util to create a memory manager. In this memory manager,
// there is one non constant pool, so the only variable is the size
// of this pool. It is templated.
template <size_t non_constant_pool_size, size_t runtime_pool_size>
class MemoryManagerCreator {
 public:
  MemoryManager* get_memory_manager() {
    return &memory_manager_;
  }

 private:
  MemoryAllocator const_allocator_{MemoryAllocator(0, nullptr)};

  uint8_t runtime_pool_[runtime_pool_size];
  MemoryAllocator runtime_allocator_{
      MemoryAllocator(runtime_pool_size, runtime_pool_)};

  MemoryAllocator temp_allocator_{MemoryAllocator(0, nullptr)};

  uint8_t non_constant_pool_[non_constant_pool_size];
  MemoryAllocator non_const_allocators_[NUM_NON_CONSTANT_POOLS]{
      MemoryAllocator(non_constant_pool_size, non_constant_pool_)};
  HierarchicalAllocator non_const_allocator_{
      HierarchicalAllocator(NUM_NON_CONSTANT_POOLS, non_const_allocators_)};

  MemoryManager memory_manager_{MemoryManager(
      &const_allocator_,
      &non_const_allocator_,
      &runtime_allocator_,
      &temp_allocator_)};
};

class MemoryManagerCreatorDynamic {
 public:
  MemoryManagerCreatorDynamic(
      uint32_t non_constant_pool_size,
      uint32_t runtime_pool_size)
      : non_constant_pool_size_{non_constant_pool_size},
        runtime_pool_size_{runtime_pool_size},
        runtime_pool_{new uint8_t[runtime_pool_size_]},
        runtime_allocator_{runtime_pool_size_, runtime_pool_.get()},
        non_constant_pool_{new uint8_t[non_constant_pool_size_]},
        non_const_allocators_{
            MemoryAllocator(non_constant_pool_size_, non_constant_pool_.get())},
        non_const_allocator_{HierarchicalAllocator(
            NUM_NON_CONSTANT_POOLS,
            non_const_allocators_)},
        memory_manager_{MemoryManager(
            &const_allocator_,
            &non_const_allocator_,
            &runtime_allocator_,
            &temp_allocator_)} {}

  MemoryManager* get_memory_manager() {
    return &memory_manager_;
  }

 private:
  uint32_t non_constant_pool_size_;
  uint32_t runtime_pool_size_;

  std::unique_ptr<uint8_t[]> runtime_pool_;
  MemoryAllocator runtime_allocator_;

  std::unique_ptr<uint8_t[]> non_constant_pool_;
  MemoryAllocator non_const_allocators_[NUM_NON_CONSTANT_POOLS];
  HierarchicalAllocator non_const_allocator_;

  MemoryManager memory_manager_;

  MemoryAllocator const_allocator_{MemoryAllocator(0, nullptr)};

  MemoryAllocator temp_allocator_{MemoryAllocator(0, nullptr)};
};
} // namespace executor
} // namespace torch
