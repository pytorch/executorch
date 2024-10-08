/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/event_tracer_hooks_delegate.h>

#include <xnnpack.h>
#include <vector>

namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {
namespace profiling {

enum class XNNProfilerState { Uninitialized, Ready, Running };

class XNNProfiler {
 public:
  XNNProfiler();

  /**
   * Initialize the profiler. This must be called after model is
   * compiled and before calling begin_execution.
   */
  executorch::runtime::Error initialize(xnn_runtime_t runtime);

  /**
   * Start a new profiling session. This is typically invoked
   * immediately before invoking the XNNPACK runtime as part
   * of a forward pass.
   */
  executorch::runtime::Error start(
      executorch::runtime::EventTracer* event_tracer);

  /**
   * End a profiling session. This is typically invoked immediately
   * after the XNNPACK runtime invocation completes.
   */
  executorch::runtime::Error end();

 private:
#if defined(ET_EVENT_TRACER_ENABLED) || defined(ENABLE_XNNPACK_PROFILING)
  executorch::runtime::EventTracer* event_tracer_;
  xnn_runtime_t runtime_;
  XNNProfilerState state_;

  size_t op_count_;
  std::vector<char> op_names_;
  std::vector<uint64_t> op_timings_;
  uint64_t run_count_;
  et_timestamp_t start_time_;

#ifdef ENABLE_XNNPACK_PROFILING
  // State needed to track average timing. Track the running sum of
  // timing for each op, as well as the number of invocations. The
  // running average can be found as sum / run_count.
  std::vector<uint64_t> op_timings_sum_;
#endif

  executorch::runtime::Error get_runtime_operator_names();
  executorch::runtime::Error get_runtime_num_operators();
  executorch::runtime::Error get_runtime_operator_timings();

  void log_operator_timings();

  /**
   * Submit the trace to the ET event tracer.
   */
  void submit_trace();
#endif
};

} // namespace profiling
} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
