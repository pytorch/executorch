/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/event_tracer.h>

namespace torch {
namespace executor {

/**
 * BackendExecutionContext will be used to inject run time context.
 * The current plan is to add temp allocator and event tracer (for profiling) as
 * part of the runtime context.
 */
class BackendExecutionContext final {
 public:
  BackendExecutionContext(EventTracer* event_tracer = nullptr)
      : event_tracer_(event_tracer) {}

  /**
   * Returns a pointer to an instance of EventTracer to do profiling/debugging
   * logging inside the delegate backend. Users will need access to this pointer
   * to use any of the event tracer APIs.
   */
  EventTracer* event_tracer() {
    return event_tracer_;
  }

 private:
  EventTracer* event_tracer_ = nullptr;
};

} // namespace executor
} // namespace torch
