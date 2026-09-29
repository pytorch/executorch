/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <executorch/extension/llm/batching/runner.h>
#include <executorch/extension/llm/serving/request_handle.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {
namespace serving {

class ET_EXPERIMENTAL ServingRuntime;

namespace detail {

struct ET_EXPERIMENTAL GenerationCompletion {
  RequestId request_id = 0;
  std::uint64_t incarnation = 0;
  // False for rejected requests or an owner retired by close/reset/shutdown.
  bool current_session = false;
  std::optional<batching::Position> position;
  batching::GenerationMetrics metrics;
  batching::GenerationUpdate terminal;
  std::optional<ServingError> error;
};

struct ET_EXPERIMENTAL GenerationRequest {
  // nullopt creates an independent ephemeral session. A named key may be opened
  // implicitly; an unavailable key must be explicitly reset before generation.
  std::optional<std::string> key;
  std::vector<batching::Token> delta;
  batching::GenConfig config;
  // Runs on the shared delivery thread, never on engine/control threads.
  // Calls are ordered, with exactly one terminal unless the sink itself throws.
  // Must do short, bounded work: no blocking I/O or waits for runtime work.
  // A throwing sink is disabled and the handle reports Internal failure.
  // Terminal token payloads are split into a nonterminal update before either
  // completion hook. The final terminal is empty. Cancellation is available
  // directly here even if callbacks precede submit returning to the caller.
  std::function<void(const batching::GenerationUpdate&, const RequestHandle&)>
      on_update;
  // Delivery-thread preparation after engine completion/metrics are stable.
  // The same nonblocking contract as on_update applies.
  // All raw tokens were delivered already. The text layer flushes text here,
  // preparing history/statistics without its public terminal event. Position
  // and current_session are filled later on control, not available here.
  // A throw becomes Internal failure; the control hook and terminal still run.
  std::function<void(const GenerationCompletion&)> on_prepare_complete;
  // Internal transaction hook, not a user sink. Runs once on the control thread
  // after prior updates were delivered and engine completion is stable, before
  // releasing the current session and before delivering the terminal update.
  // May commit history/statistics only when current_session is true. Must not
  // wait, reenter the runtime, or invoke user output. A throw invalidates the
  // current session and changes the terminal outcome to Failed.
  std::function<void(const GenerationCompletion&)> on_complete;
};

// Preadmission full/stopped/invalid rejection has no request or sink
// obligation. Every admitted request has an immediate handle and exactly one
// terminal outcome, including subsequent busy/engine rejection, cancellation,
// overflow.
using SubmissionResult ET_EXPERIMENTAL =
    std::variant<RequestHandle, ServingError>;

// Internal delta boundary for the text layer and native tests. No Session owner
// escapes. Request admission is retained through terminal delivery. Callbacks
// share delivery capacity: transports must enqueue output and handle I/O
// elsewhere. Sinks must not synchronously shut down or destroy the runtime, or
// wait for any request serviced by it.
struct ET_EXPERIMENTAL GenerationBridge {
  static SubmissionResult submit(
      ServingRuntime& runtime,
      GenerationRequest request);
};

} // namespace detail
} // namespace serving
} // namespace llm
} // namespace extension
} // namespace executorch
