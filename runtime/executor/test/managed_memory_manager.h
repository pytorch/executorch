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

namespace executorch {
namespace runtime {
namespace testing {

/**
 * Creates and owns a MemoryManager and the allocators that it points to. Easier
 * to manage than creating the allocators separately.
 */
class ManagedMemoryManager {
 public:
  ManagedMemoryManager(
      size_t planned_memory_bytes,
      size_t method_allocator_bytes)
      : planned_memory_buffer_(new uint8_t[planned_memory_bytes]),
        planned_memory_span_(
            planned_memory_buffer_.get(),
            planned_memory_bytes),
        planned_memory_({&planned_memory_span_, 1}),
        method_allocator_pool_(new uint8_t[method_allocator_bytes]),
        method_allocator_(method_allocator_bytes, method_allocator_pool_.get()),
        memory_manager_(&method_allocator_, &planned_memory_) {}

  MemoryManager& get() {
    return memory_manager_;
  }

 private:
  std::unique_ptr<uint8_t[]> planned_memory_buffer_;
  Span<uint8_t> planned_memory_span_;
  HierarchicalAllocator planned_memory_;

  std::unique_ptr<uint8_t[]> method_allocator_pool_;
  MemoryAllocator method_allocator_;

  MemoryManager memory_manager_;
};

} // namespace testing
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
namespace testing {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::testing::ManagedMemoryManager;
} // namespace testing
} // namespace executor
} // namespace torch
