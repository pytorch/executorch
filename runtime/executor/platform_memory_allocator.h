/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdio.h>
#include <cinttypes>
#include <cstdint>

#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>

namespace executorch {
namespace runtime {
namespace internal {

/**
 * PlatformMemoryAllocator is a memory allocator that uses a linked list to
 * manage allocated nodes. It overrides the allocate method of MemoryAllocator
 * using the PAL fallback allocator method `et_pal_allocate`.
 */
class PlatformMemoryAllocator final : public MemoryAllocator {
 private:
  // We allocate a little more than requested and use that memory as a node in
  // a linked list, pushing the allocated buffers onto a list that's iterated
  // and freed when the KernelRuntimeContext is destroyed.
  struct AllocationNode {
    void* data;
    AllocationNode* next;
  };

  AllocationNode* head_ = nullptr;

 public:
  PlatformMemoryAllocator() : MemoryAllocator(0, nullptr) {}

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    if (!isPowerOf2(alignment)) {
      ET_LOG(Error, "Alignment %zu is not a power of 2", alignment);
      return nullptr;
    }

    // Allocate enough memory for the node, the data and the alignment bump.
    size_t alloc_size = sizeof(AllocationNode) + size + alignment;
    void* node_memory = et_pal_allocate(alloc_size);

    // If allocation failed, log message and return nullptr.
    if (node_memory == nullptr) {
      ET_LOG(Error, "Failed to allocate %zu bytes", alloc_size);
      return nullptr;
    }

    // Compute data pointer.
    uint8_t* data_ptr =
        reinterpret_cast<uint8_t*>(node_memory) + sizeof(AllocationNode);

    // Align the data pointer.
    void* aligned_data_ptr = alignPointer(data_ptr, alignment);

    // Assert that the alignment didn't overflow the allocated memory.
    ET_DCHECK_MSG(
        reinterpret_cast<uintptr_t>(aligned_data_ptr) + size <=
            reinterpret_cast<uintptr_t>(node_memory) + alloc_size,
        "aligned_data_ptr %p + size %zu > node_memory %p + alloc_size %zu",
        aligned_data_ptr,
        size,
        node_memory,
        alloc_size);

    // Construct the node.
    AllocationNode* new_node = reinterpret_cast<AllocationNode*>(node_memory);
    new_node->data = aligned_data_ptr;
    new_node->next = head_;
    head_ = new_node;

    // Return the aligned data pointer.
    return head_->data;
  }

  void reset() override {
    AllocationNode* current = head_;
    while (current != nullptr) {
      AllocationNode* next = current->next;
      et_pal_free(current);
      current = next;
    }
    head_ = nullptr;
  }

  ~PlatformMemoryAllocator() override {
    reset();
  }

 private:
  // Disable copy and move.
  PlatformMemoryAllocator(const PlatformMemoryAllocator&) = delete;
  PlatformMemoryAllocator& operator=(const PlatformMemoryAllocator&) = delete;
  PlatformMemoryAllocator(PlatformMemoryAllocator&&) noexcept = delete;
  PlatformMemoryAllocator& operator=(PlatformMemoryAllocator&&) noexcept =
      delete;
};

} // namespace internal
} // namespace runtime
} // namespace executorch
