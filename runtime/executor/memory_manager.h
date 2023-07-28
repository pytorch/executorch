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

namespace torch {
namespace executor {

// The memory manager for the executor. It is responsible for keeping
// track of the memory allocation across the lifespan of the executor,
// providing allocators for the following types of objects:
//
// 1. Constants - Constant values in the program. TODO(myuan): we may not
// need it (because the constants are hardcoded in the flatbuffer) but
// we'll account for it for the time being for completeness.
//
// 2. Non-constants - Non-constant values in the program, which may or may not
// be tied to a memory plan.
//
// 3. Runtime structures - Any data needed by the executor itself.
// TODO(myuan): determine whether Delegates need to receive it in the "init"
// method for backends to use directly, or whether memory needs will be
// expressed as an argument to the delegated methods for memory planning to
// account for. Same concerns about dynamic behaviour apply.

// 4. Kernel temporary - This is to provide kernels with a way to create memory,
// without having to request it by adding an extra argument. The extra argument
// approach is fine if/when planning desires to account for such memory, but in
// certain cases a kernel may be fine just leaving this as an implementation
// detail of the kernels itself (but we still want to be able to capture such
// memory allocation).

// In general, this memory manager aims to consolidate all dynamic memory needs
// for program execution. This can allow for heap-less execution (relevant to
// some embedded scenarios), and overall have a tighter control over memory
// utilization. The manager, however, cannot ensure all allocation is accounted
// for since kernel implementations are free to use a separate way to allocate
// memory (e.g. for things like scratch space).
// TODO(myuan): analyze the stack data overhead and lifespan.

class MemoryManager {
 public:
  MemoryManager(
      MemoryAllocator* constant_allocator,
      HierarchicalAllocator* non_constant_allocator,
      MemoryAllocator* runtime_allocator,
      MemoryAllocator* kernel_temporary_allocator)
      : constant_allocator_(constant_allocator),
        non_constant_allocator_(non_constant_allocator),
        runtime_allocator_(runtime_allocator),
        kernel_temporary_allocator_(kernel_temporary_allocator) {}

  /**
   * Returns an allocator for constant values in the program.
   */
  const MemoryAllocator* get_constant_allocator() const {
    return constant_allocator_;
  }

  /**
   * Returns an hierarchical allocator for non-constant values in the program.
   */
  HierarchicalAllocator* get_non_constant_allocator() const {
    return non_constant_allocator_;
  }

  /**
   * Returns an allocator to be used for any runtime internal structures
   * (i.e. not directly program values).
   */
  MemoryAllocator* get_runtime_allocator() const {
    return runtime_allocator_;
  }

  /**
   * Returns an allocator that kernel implementations can use to
   * create temporary memory (i.e. whose lifespan is a single execution
   * of the kernel).
   */
  MemoryAllocator* get_kernel_temporary_allocator() const {
    return kernel_temporary_allocator_;
  }

  virtual ~MemoryManager() {}

 private:
  const MemoryAllocator* constant_allocator_;
  HierarchicalAllocator* non_constant_allocator_;
  MemoryAllocator* runtime_allocator_;
  MemoryAllocator* kernel_temporary_allocator_;
};

} // namespace executor
} // namespace torch
