/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/memory_manager.h>

namespace torch {
namespace executor {
namespace testing {

/**
 * Creates and owns a MemoryManager and the MemoryAllocators that it points to.
 * Easier to manage than creating the allocators separately.
 */
class ManagedMemoryManager {
 public:
  ManagedMemoryManager(size_t non_const_mem_bytes, size_t runtime_mem_bytes)
      : const_allocator_(0, nullptr),
        non_const_pool_(new uint8_t[non_const_mem_bytes]),
        non_const_allocators_({
            {non_const_pool_.get(), non_const_mem_bytes},
        }),
        non_const_allocator_({
            non_const_allocators_.data(),
            non_const_allocators_.size(),
        }),
        runtime_pool_(new uint8_t[runtime_mem_bytes]),
        runtime_allocator_(runtime_mem_bytes, runtime_pool_.get()),
        temp_allocator_(0, nullptr),
        memory_manager_(
            &const_allocator_,
            &non_const_allocator_,
            &runtime_allocator_,
            &temp_allocator_) {}

  MemoryManager& get() {
    return memory_manager_;
  }

 private:
  MemoryAllocator const_allocator_;

  std::unique_ptr<uint8_t[]> non_const_pool_;
  std::vector<Span<uint8_t>> non_const_allocators_;
  torch::executor::HierarchicalAllocator non_const_allocator_;

  std::unique_ptr<uint8_t[]> runtime_pool_;
  MemoryAllocator runtime_allocator_;

  MemoryAllocator temp_allocator_;

  MemoryManager memory_manager_;
};

} // namespace testing
} // namespace executor
} // namespace torch
