/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/memory_allocator.h>

namespace executorch {
namespace runtime {

/**
 * BackendExecutionContext will be used to inject run time context.
 */
class BackendExecutionContext final {
 public:
  BackendExecutionContext(
      EventTracer* event_tracer = nullptr,
      MemoryAllocator* temp_allocator = nullptr)
      : event_tracer_(event_tracer), temp_allocator_(temp_allocator) {}

  /**
   * Returns a pointer to an instance of EventTracer to do profiling/debugging
   * logging inside the delegate backend. Users will need access to this pointer
   * to use any of the event tracer APIs.
   */
  EventTracer* event_tracer() {
    return event_tracer_;
  }

  /**
   * Returns a pointer to the address allocated by temp allocator. This
   * allocator will be reset after every delegate call during execution.
   */
  void* allocate(
      size_t size,
      size_t alignment = MemoryAllocator::kDefaultAlignment) {
    // TODO(chenlai): depends on the need, we may expose more functionality for
    // memory allocation.
    return temp_allocator_->allocate(size, alignment);
  }

  /**
   * Returns the temp allocator. This allocator will be reset every instruction.
   */
  MemoryAllocator* get_temp_allocator() {
    return temp_allocator_;
  }

 private:
  EventTracer* event_tracer_ = nullptr;
  MemoryAllocator* temp_allocator_ = nullptr;
};

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::BackendExecutionContext;
} // namespace executor
} // namespace torch
