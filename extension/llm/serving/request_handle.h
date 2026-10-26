/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <executorch/extension/llm/serving/types.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {
namespace serving {

using RequestId ET_EXPERIMENTAL = std::uint64_t;

namespace detail {
struct ET_EXPERIMENTAL RequestState;
struct ET_EXPERIMENTAL GenerationBridge;
} // namespace detail

// Copyable request identity and cancellation, safe from any thread and after
// runtime destruction. Dropping a handle does not cancel its request.
class ET_EXPERIMENTAL RequestHandle {
 public:
  RequestHandle() = default;

  // Zero identifies a default or moved-from handle, never an admitted request.
  RequestId id() const noexcept;

  // Latches even before engine admission/binding. Does not wait for a callback.
  // A request whose terminal outcome was already selected keeps that outcome.
  void cancel() const;

  // Callback-lifetime barrier: true after completion processing, terminal
  // invocation, and runtime-owned callback capture destruction. Admission was
  // released before terminal invocation; this does not acknowledge transport
  // delivery. False for an invalid handle; wait() then returns at once.
  // Never wait from a sink or hook serviced by the same runtime: callbacks
  // share delivery capacity and their return is part of completion.
  bool done() const;
  void wait() const;

  // Snapshot of a selected failure, fixed before terminal invocation (and
  // always final after done()). Terminal-sink exceptions do not revise it.
  // Cancellation is not an error.
  std::optional<ServingError> error() const;

 private:
  friend class ServingRuntime;
  friend struct detail::GenerationBridge;
  explicit RequestHandle(std::shared_ptr<detail::RequestState> state);

  std::shared_ptr<detail::RequestState> state_;
};

} // namespace serving
} // namespace llm
} // namespace extension
} // namespace executorch
