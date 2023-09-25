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
#include <executorch/runtime/platform/compiler.h>

namespace torch {
namespace executor {

/**
 * Runtime state and functionality for kernel implementations.
 *
 * NOTE: Will not be passed to operators if running in ATen mode as those
 * operators do not expect to receive a KernelRuntimeContext argument.
 */
class KernelRuntimeContext {
 public:
  /**
   * Construct a new kernel runtime context along with an optional event tracer.
   */
  KernelRuntimeContext(EventTracer* event_tracer = nullptr)
      : event_tracer_(event_tracer) {}
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
  __ET_NODISCARD Error failure_state() const {
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

  // TODO(T147221312): Add a way to allocate temporary memory.

  // TODO(T147221312): Add a way to resize a tensor.

 private:
  EventTracer* event_tracer_ = nullptr;
  Error failure_state_ = Error::Ok;
};

} // namespace executor
} // namespace torch

// TODO(T147221312): Remove these aliases once all code uses
// KernelRuntimeContext.
namespace exec_aten {
using RuntimeContext = torch::executor::KernelRuntimeContext;
} // namespace exec_aten
namespace torch {
namespace executor {
using RuntimeContext = torch::executor::KernelRuntimeContext;
} // namespace executor
} // namespace torch
