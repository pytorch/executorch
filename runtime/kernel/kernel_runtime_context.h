/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/event_tracer_hooks.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace runtime {

/**
 * Runtime state and functionality for kernel implementations.
 *
 * NOTE: Will not be passed to operators if running in ATen mode as those
 * operators do not expect to receive a KernelRuntimeContext argument.
 */
class KernelRuntimeContext {
 public:
  /**
   * Construct a new kernel runtime context.
   *
   * KernelRuntimeContext does not take ownership
   * of these pointers, so they must outlive the context instance.
   *
   * @param[in] event_tracer The optional EventTracer to use for
   *     profiling/debugging
   * @param[in] temp_allocator The optional MemoryAllocator used to allocate
   *     temporary memory for the kernel. If not provided, an error will be
   *     returned when calling allocate_temp.
   */
  KernelRuntimeContext(
      EventTracer* event_tracer = nullptr,
      MemoryAllocator* temp_allocator = nullptr)
      : event_tracer_(event_tracer), temp_allocator_(temp_allocator) {}
  /**
   * Tells the runtime that the kernel call has failed. Prefer this over
   * ET_CHECK_*(), which fatally panics the process/system.
   *
   * If this is not called, the runtime will treat the kernel call as
   * successful.
   *
   * This unusual error-propagation path is required because kernel signatures
   * do not have a natural way to return errors directly. They are generally
   * compatible with core PyTorch ATen kernel signatures, which use exceptions
   * to report errors. But, ExecuTorch does not use exceptions.
   */
  void fail(Error error) {
    failure_state_ = error;
  }

  /// Returns the current failure state.
  ET_NODISCARD Error failure_state() const {
    return failure_state_;
  }

  /**
   * INTERNAL ONLY
   *
   * Returns a pointer to an instance of EventTracer to do profiling/debugging
   * logging inside the codegen layer. This is only for internal usage inside
   * the codegen layer and users should not be accessing this.
   */
  EventTracer* internal_event_tracer() {
    return event_tracer_;
  }

  /**
   * Allocates temporary memory that will be freed when the kernel returns. This
   * returns a pointer to the allocated memory or an error if the allocation
   * fails.
   *
   * @param[in] size Number of bytes to allocate.
   * @param[in] alignment Minimum alignment for the returned pointer. Must be a
   *     power of 2.
   *
   * @returns A result object containing either a pointer to the allocated
   *     memory or an error to indicate failure
   */
  Result<void*> allocate_temp(
      size_t size,
      size_t alignment = MemoryAllocator::kDefaultAlignment) {
    ET_CHECK_OR_RETURN_ERROR(
        temp_allocator_ != nullptr, NotFound, "No temp allocator provided");
    void* temp_memory = temp_allocator_->allocate(size, alignment);
    ET_CHECK_OR_RETURN_ERROR(
        temp_memory != nullptr,
        MemoryAllocationFailed,
        "Failed to allocate temp memory. Bytes requested: %zu",
        size);
    return temp_memory;
  }

  // TODO(T147221312): Add a way to resize a tensor.

 private:
  EventTracer* event_tracer_ = nullptr;
  MemoryAllocator* temp_allocator_ = nullptr;
  Error failure_state_ = Error::Ok;
};

} // namespace runtime
} // namespace executorch

// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
namespace torch {
namespace executor {
/// DEPRECATED: Use ::executorch::runtime::KernelRuntimeContext instead.
using ::executorch::runtime::KernelRuntimeContext;
/// DEPRECATED: Use ::executorch::runtime::KernelRuntimeContext instead.
using RuntimeContext = ::executorch::runtime::KernelRuntimeContext;
} // namespace executor
} // namespace torch
namespace executorch {
namespace aten {
/// DEPRECATED: Use ::executorch::runtime::KernelRuntimeContext instead.
using RuntimeContext = ::executorch::runtime::KernelRuntimeContext;
} // namespace aten
} // namespace executorch
// DEPRECATED: The exec_aten:: namespace is deprecated. Use executorch::aten::
// instead.
namespace exec_aten = ::executorch::aten;
