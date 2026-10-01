/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>

#include <executorch/extension/llm/batching/executor.h>
#include <executorch/extension/llm/batching/scheduler.h>
#include <executorch/extension/llm/serving/request_handle.h>
#include <executorch/extension/llm/serving/types.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {
namespace serving {

struct ET_EXPERIMENTAL ServingRuntimeConfig {
  std::size_t max_sessions = 1;
  // Reported metadata; 0 means unknown.
  std::size_t max_context_length = 0;
  // Queued, executing, and delivery-fenced lifecycle/generation-start
  // operations, excluding completion callbacks. Must be non-zero.
  std::size_t max_pending_operations = 64;
  // Admission is retained until terminal dispatch, then released before the
  // sink runs. The shared dispatcher may retain one additional retiring
  // request.
  std::size_t max_requests = 8;
  // Buffered nonterminal events and total buffered tokens per request. A
  // separate terminal slot is reserved, but its tokens share the token budget.
  std::size_t max_events_per_request = 16;
  std::size_t max_tokens_per_request = 256;
};

// nullopt acknowledges success. Dropping a future does not cancel its
// operation.
using LifecycleResult ET_EXPERIMENTAL = std::optional<ServingError>;
using LifecycleCallback ET_EXPERIMENTAL = std::function<void(LifecycleResult)>;

class ET_EXPERIMENTAL ServingRuntime {
 public:
  // Owns one Runner and its scheduler. The borrowed executor must outlive
  // shutdown/destruction and must not be driven by another Runner concurrently.
  // Initialization happens on the engine thread. A null scheduler or zero
  // session, operation, request, or event/token limit leaves an inert runtime:
  // no threads start, info().ready is false, and operations return
  // InvalidArgument.
  ServingRuntime(
      batching::Executor& executor,
      std::unique_ptr<batching::Scheduler> scheduler,
      ServingRuntimeConfig config);
  ~ServingRuntime();

  ServingRuntime(const ServingRuntime&) = delete;
  ServingRuntime& operator=(const ServingRuntime&) = delete;
  ServingRuntime(ServingRuntime&&) = delete;
  ServingRuntime& operator=(ServingRuntime&&) = delete;

  // Any thread. Keys must be non-empty. Accepted operations begin in per-key
  // admission order. A close/reset delivery fence defers later work for that
  // key while other keys may progress. Queue saturation reports
  // CapacityExceeded without changing state.
  // For all lifecycle operations, callbacks run once on control after
  // processing and admission release, or inline on the caller for immediate
  // rejection. They run without runtime locks and may race with submission's
  // return. Callbacks must do short, nonblocking work: no waits for runtime
  // work or runtime shutdown/destruction. Exceptions are logged, not retried.
  // An empty callback discards the result. Futures adapt this same completion
  // path. Immediate rejection is outside accepted-operation ordering.
  //
  // Accepted close/reset acknowledgements follow all earlier admitted
  // same-key generation callbacks (including terminal delivery unless the
  // sink failed) and their capture destruction, even on failure or shutdown.
  // Transports must preserve callback enqueue order for this wire guarantee.
  // A blocked sink still stalls the shared delivery thread.
  //
  // Opens may queue during initialization; success means an empty session is
  // owned, or the key was already open. An initial open failure releases its
  // slot. Opening an unavailable key fails with NotReady; use reset to retry.
  void open_session_async(std::string key, LifecycleCallback on_complete);
  std::future<LifecycleResult> open_session_async(std::string key);

  // Idempotent, including absent keys. Cancels prior same-key requests and
  // releases the logical slot before acknowledging callback quiescence.
  // Session closure is through RAII, not synchronous physical cleanup.
  void close_session_async(std::string key, LifecycleCallback on_complete);
  std::future<LifecycleResult> close_session_async(std::string key);

  // Cold replacement, retaining the key and logical slot throughout. Success
  // means reopening and old-response callback cleanup completed. Failure
  // destroys the old state and leaves the
  // key unavailable, still reserving its slot until reset succeeds or close
  // releases it. A missing key returns SessionNotFound. Cancels the old
  // request; stale completion cannot claim the replacement. No prefix reuse.
  void reset_session_async(std::string key, LifecycleCallback on_complete);
  std::future<LifecycleResult> reset_session_async(std::string key);

  // A synchronized snapshot. Ready means initialized and accepting operations,
  // not that capacity is available. Active slots include ephemeral sessions,
  // opening, reopening, and unavailable keys, but not commands still waiting
  // to reserve a slot.
  ServingInfo info() const;

  // Idempotent, including concurrent callers. Stops admission, completes all
  // accepted operations/requests, and joins control, Runner, and delivery
  // threads. Accepted callbacks have returned and their runtime-owned captures
  // are released. No new work is accepted afterward. Call only from external
  // threads, never engine/lifecycle callbacks, sinks, or commit hooks.
  // Callbacks must do bounded, nonblocking work and return for shutdown to
  // finish. Destruction calls shutdown; as usual, object lifetime must be
  // synchronized against callers still accessing the runtime.
  void shutdown();

 private:
  friend struct detail::GenerationBridge;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace serving
} // namespace llm
} // namespace extension
} // namespace executorch
