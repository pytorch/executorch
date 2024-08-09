/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>

namespace executorch {
namespace runtime {

/**
 * A container class for allocators used during Method load and execution.
 *
 * This class consolidates all dynamic memory needs for Method load and
 * execution. This can allow for heap-based as well as heap-less execution
 * (relevant to some embedded scenarios), and overall provides more control over
 * memory use.
 *
 * This class, however, cannot ensure all allocation is accounted for since
 * kernel and backend implementations are free to use a separate way to allocate
 * memory (e.g., for things like scratch space). But we do suggest that backends
 * and kernels use these provided allocators whenever possible.
 */
class MemoryManager final {
 public:
  /**
   * Constructs a new MemoryManager.
   *
   * @param[in] method_allocator The allocator to use when loading a Method and
   *     allocating its internal structures. Must outlive the Method that uses
   *     it.
   * @param[in] planned_memory The memory-planned buffers to use for mutable
   *     tensor data when executing a Method. Must outlive the Method that uses
   *     it. May be `nullptr` if the Method does not use any memory-planned
   *     tensor data. The sizes of the buffers in this HierarchicalAllocator
   *     must agree with the corresponding
   *     `MethodMeta::num_memory_planned_buffers()` and
   *     `MethodMeta::memory_planned_buffer_size(N)` values, which are embedded
   *     in the Program.
   * @param[in] temp_allocator The allocator to use when allocating temporary
   *     data during kernel or delegate execution. Must outlive the Method that
   *     uses it. May be `nullptr` if the Method does not use kernels or
   *     delegates that allocate temporary data. This allocator will be reset
   *     after every kernel or delegate call during execution.
   */
  explicit MemoryManager(
      MemoryAllocator* method_allocator,
      HierarchicalAllocator* planned_memory = nullptr,
      MemoryAllocator* temp_allocator = nullptr)
      : method_allocator_(method_allocator),
        planned_memory_(planned_memory),
        temp_allocator_(temp_allocator) {
    ET_CHECK_MSG(
        method_allocator != temp_allocator,
        "method allocator cannot be the same as temp allocator");
  }

  /**
   * DEPRECATED: Use the constructor without `constant_allocator` instead.
   *
   * TODO(T162089316): Remove this once all users migrate to the new ctor.
   */
  __ET_DEPRECATED MemoryManager(
      // We would normally use __ET_UNUSED here, but GCC older than 9.3 has a
      // bug that triggers a syntax error when using [[maybe_unused]] on the
      // first parameter of a constructor:
      // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81429
      __attribute__((unused)) MemoryAllocator* constant_allocator,
      HierarchicalAllocator* non_constant_allocator,
      MemoryAllocator* runtime_allocator,
      MemoryAllocator* temporary_allocator)
      : MemoryManager(
            /*method_allocator=*/runtime_allocator,
            /*planned_memory=*/non_constant_allocator,
            /*temp_allocator=*/temporary_allocator) {}

  /**
   * Returns the allocator that the runtime will use to allocate internal
   * structures while loading a Method. Must not be used after its associated
   * Method has been loaded.
   */
  MemoryAllocator* method_allocator() const {
    return method_allocator_;
  }

  /**
   * Returns the memory-planned buffers to use for mutable tensor data.
   */
  HierarchicalAllocator* planned_memory() const {
    return planned_memory_;
  }

  /**
   * Returns the allocator to use for allocating temporary data during kernel or
   * delegate execution.
   *
   * This allocator will be reset after every kernel or delegate call during
   * execution.
   */
  MemoryAllocator* temp_allocator() const {
    return temp_allocator_;
  }

 private:
  MemoryAllocator* method_allocator_;
  HierarchicalAllocator* planned_memory_;
  MemoryAllocator* temp_allocator_;
};

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::MemoryManager;
} // namespace executor
} // namespace torch
