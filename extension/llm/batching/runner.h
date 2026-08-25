/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Runs batches from a scheduler through an executor, and carries whole
// generations to completion on top of that.
//
// One engine thread owns the executor and all generation state. Callers reach
// it two ways, neither of which blocks it: a command inbox for session and
// generation work, which must run where the executor lives, and a cancellation
// flag that needs no round trip.
//
// Generation is completion-driven. When a batch settles, the engine thread
// emits what each output-producing task produced and submits its continuation,
// so no thread is parked per conversation and batch occupancy does not depend
// on callers happening to be blocked at the same moment.
//
// Output is pushed: an admitted generation's callbacks run on the engine
// thread as its steps settle. They must not block, because every other
// generation in the batch is waiting behind them. A callback may request
// shutdown, but must not destroy the Runner or wait for work that requires the
// engine thread.
//
// Generation starts from a caller-resolved delta. The session owns its logical
// position; the runner chunks and schedules the delta but does not reconcile
// it against session history.

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <vector>

#include <executorch/extension/llm/batching/executor.h>
#include <executorch/extension/llm/batching/scheduler.h>
#include <executorch/extension/llm/batching/types.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

// Why a generation stopped.
enum class FinishReason {
  StopToken, // a token in GenConfig::stop_tokens was produced
  NewTokenLimit, // GenConfig::max_new_tokens reached
  Cancelled, // cancelled, the handle was dropped, or the session closed
  Failed, // rejected at start, or a step failed
};

// Reports tokens as they are produced, in order, and several at once when a
// speculative executor accepts a run. The reason is set on the last call and
// only then, so the final tokens and the reason arrive together and cannot be
// reordered.
//
// Runs on the engine thread; must not block.
using GenerationCallback =
    std::function<void(const std::vector<Token>&, std::optional<FinishReason>)>;

struct GenConfig {
  std::int32_t max_new_tokens = 256;
  SamplingParams sampling;
  // Ends the generation with FinishReason::StopToken. The token is not
  // emitted.
  std::vector<Token> stop_tokens;
  // Fixed for the generation and installed before its tasks are submitted.
  // nullopt requests nondeterministic sampling.
  std::optional<std::uint64_t> seed;
};

struct RunnerConfig {
  // How finely a prompt is split. Must not exceed what the scheduler admits.
  std::int32_t max_prefill_chunk_size = 256;
};

// Shared between the runner and every handle to one generation. Defined in
// runner.cpp so its lock and flags stay out of the public header.
struct GenerationHandleState;

class RunnerImpl;

// Observes one generation. Output arrives through the callbacks, not here, so
// dropping the handle does not cancel; fire and forget is a valid use.
//
// Copyable and safe from any thread: it holds only shared state, never the
// runner, so it may outlive it.
class GenerationHandle {
 public:
  GenerationHandle() = default;

  // Requests cancellation, which lands within one step. A no-op on a
  // default-constructed handle.
  void cancel() const;

  bool done() const;

  // Blocks until the generation ends. Returns immediately if it already has.
  void wait() const;

  // Meaningful once done().
  FinishReason finish_reason() const;

 private:
  friend class RunnerImpl;
  explicit GenerationHandle(std::shared_ptr<GenerationHandleState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<GenerationHandleState> state_;
};

// A handle to one open session: the runner's internals plus an id. Copy it
// freely.
//
// Does not own the session, since closing is explicit, but does keep the
// runner's internals alive, so outliving the Runner is safe. Closing any copy
// makes all copies stale; later generation requests fail without reaching the
// executor.
class Session {
 public:
  Session() = default;

  SessionId id() const {
    return sid_;
  }
  bool valid() const {
    return impl_ != nullptr;
  }

  // Equivalent to Runner::generate_async with this session's id.
  GenerationHandle generate_async(
      std::vector<Token> delta,
      GenConfig config,
      GenerationCallback on_update) const;

  // Releases the session and anything it holds. Any live generation on it ends
  // Cancelled. Repeated closes are no-ops; this handle stays valid but stale.
  std::future<void> close() const;

 private:
  friend class RunnerImpl;
  Session(std::shared_ptr<RunnerImpl> impl, SessionId sid)
      : impl_(std::move(impl)), sid_(sid) {}

  std::shared_ptr<RunnerImpl> impl_;
  SessionId sid_ = 0;
};

class Runner {
 public:
  // Neither reference is owned; both must outlive completion of shutdown or
  // destruction on an external thread. One scheduler per runner.
  Runner(Executor& executor, Scheduler& scheduler, RunnerConfig config);
  ~Runner();

  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  // -- sessions. Any thread; queued to the engine thread and acked. ---------

  // nullopt = the executor is at capacity, or the runner is shutting down.
  std::future<std::optional<Session>> open_session();
  // Unknown, stale, and already closed ids are idempotent no-ops.
  std::future<void> close_session(SessionId session);

  // -- generations. Any thread; returns immediately. ------------------------

  // `delta` is the caller-resolved suffix to append. The session owns its
  // logical position: it opens at 0 and advances by what the executor
  // consumes, so consecutive generations continue where the previous one
  // ended. The runner does not compare the delta against resident session
  // history or recover an evicted prefix; those policies belong to the layer
  // that resolves the delta.
  //
  // The delta must be non-empty and its exclusive end must fit in Position.
  // Invalid input is reported by ending the generation with
  // FinishReason::Failed.
  //
  // Returns as soon as the start is queued; the callbacks report everything
  // after that. Every generation ends exactly once, including rejected starts.
  //
  // The session must be currently open in this Runner. Unknown, fabricated,
  // and stale ids end Failed without reaching the executor.
  //
  // A failed step poisons its session: what the executor holds may no longer
  // match what was asked for. Further generations on it end Failed until it is
  // closed and a new one opened.
  GenerationHandle generate_async(
      SessionId session,
      std::vector<Token> delta,
      GenConfig config,
      GenerationCallback on_update);

  // Idempotent. External callers block until the engine is joined, every live
  // generation has ended Cancelled, and every owned session is closed.
  // Concurrent external callers wait for the same completion.
  //
  // From a GenerationCallback, requests shutdown and returns without waiting;
  // a later external call or destructor must join. Do not destroy the Runner
  // from its callback.
  void shutdown();

 private:
  std::shared_ptr<RunnerImpl> impl_;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
