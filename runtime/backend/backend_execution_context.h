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
#include <executorch/runtime/core/span.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {

/**
 * BackendExecutionContext will be used to inject run time context.
 */
class BackendExecutionContext final {
 public:
  BackendExecutionContext(
      EventTracer* event_tracer = nullptr,
      MemoryAllocator* temp_allocator = nullptr,
      const char* method_name = nullptr,
      Span<const Span<uint8_t>> scratch_buffers = {})
      : event_tracer_(event_tracer),
        temp_allocator_(temp_allocator),
        method_name_(method_name),
        scratch_buffers_(scratch_buffers) {}

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

  /**
   * Get the name of the executing method from the ExecuTorch runtime.
   */
  const char* get_method_name() const {
    return method_name_;
  }

  /**
   * The memory-planned scratch buffers this delegate declared when it was
   * lowered, in declaration order. Check size() before indexing: a program
   * that declares none, or one exported before this field existed, yields an
   * empty span and the backend must fall back to the temp allocator.
   *
   * A buffer's address is fixed for the Method's lifetime, provided the
   * planned buffers the integrator supplied outlive it. Its contents are not:
   * they are uninitialized on entry and dead on return, because the memory
   * plan may hand the same bytes to unrelated tensors between calls. Within
   * one call the planner will not overlap a buffer with this delegate's own
   * inputs and outputs, with anything live across the call, or with its
   * siblings.
   *
   * Alignment is whatever the arena provides: the planner aligns each buffer's
   * offset within its pool, to 16 bytes by default, but the pool's base
   * address comes from the integrator and the runtime does not adjust it. A
   * backend needing more must declare the slack and align the pointer itself.
   */
  Span<const Span<uint8_t>> scratch_buffers() const {
    return scratch_buffers_;
  }

 private:
  EventTracer* event_tracer_ = nullptr;
  MemoryAllocator* temp_allocator_ = nullptr;
  const char* method_name_ = nullptr;
  Span<const Span<uint8_t>> scratch_buffers_;
};

} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::ET_RUNTIME_NAMESPACE::BackendExecutionContext;
} // namespace executor
} // namespace torch
