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
// emits what each output-producing task produced and submits its continuation.
// No thread is parked per conversation, so batch occupancy does not depend on
// callers being blocked at the same moment.
//
// Output is pushed: an admitted generation's callbacks run on the engine
// thread as its steps settle. They must not block, because every other
// generation in the batch is waiting behind them. A callback may request
// shutdown, but must not destroy the Runner or wait for work that requires the
// engine thread.
//
// Generation starts from a caller-resolved delta. The session owns its
// executor-committed position; the runner chunks and schedules the delta but
// does not reconcile it against session history.

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <executorch/extension/llm/batching/executor.h>
#include <executorch/extension/llm/batching/metrics.h>
#include <executorch/extension/llm/batching/scheduler.h>
#include <executorch/extension/llm/batching/types.h>
#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

// Why a generation stopped.
enum class ET_EXPERIMENTAL FinishReason {
  StopToken, // a token in GenConfig::stop_tokens was produced
  NewTokenLimit, // GenConfig::max_new_tokens reached
  Cancelled, // cancelled explicitly, by Session closure, or by shutdown
  Failed, // rejected at start, or a step failed
};

// One streaming or terminal generation event. A missing finish_reason means
// more updates may follow. A present finish_reason marks the last update;
// tokens may be empty when a generation fails or is cancelled. error_message
// is non-empty only when finish_reason is Failed.
struct ET_EXPERIMENTAL GenerationUpdate {
  std::vector<Token> tokens;
  std::optional<FinishReason> finish_reason;
  std::string error_message;
};

// Reports tokens as they are produced, in order, and several at once when a
// speculative executor accepts a run. finish_reason is set on the last call
// and only then, so the final tokens and the reason arrive together and cannot
// be reordered.
//
// Every token reported here is retained by the session. However a generation
// ends, its logical context holds what it held before, plus the delta, plus
// exactly the tokens delivered through this callback: nothing produced but
// undelivered, nothing delivered but dropped. The final delivered token may
// remain pending rather than executor-committed; the next delta carries it.
//
// A delta cancelled before it finished prefilling leaves only its consumed
// prefix in the session. Session::position() reports the executor-committed
// length, excluding any pending generated token.
//
// Admitted generations run callbacks on the engine thread. A request rejected
// before admission completes synchronously on the calling thread, potentially
// before generate_async() returns. Callbacks must not block or wait for work
// serviced by this runner. An exception from a callback is contained and ends
// that generation as Failed.
using GenerationCallback = std::function<void(const GenerationUpdate&)>;

struct ET_EXPERIMENTAL GenConfig {
  std::int32_t max_new_tokens = 256;
  SamplingParams sampling;
  // Ends the generation with FinishReason::StopToken. The matched token is
  // included in the terminal GenerationUpdate::tokens and retained by the
  // session.
  std::vector<Token> stop_tokens;
  // Fixed for the generation and installed before its tasks are submitted.
  // nullopt requests nondeterministic sampling.
  std::optional<std::uint64_t> seed;
};

// Both defined in runner.cpp: callers see neither the lock and flags the first
// holds nor the runner reference the second keeps alive.
struct GenerationHandleState;
struct SessionState;

class RunnerImpl;

// Observes one generation. Output arrives through the callbacks, not here, so
// dropping the handle does not cancel; fire and forget is a valid use.
//
// Copyable and safe from any thread: it holds only shared state, never the
// runner, so it may outlive it.
class ET_EXPERIMENTAL GenerationHandle {
 public:
  GenerationHandle() = default;
  GenerationHandle(const GenerationHandle&) = default;
  GenerationHandle& operator=(const GenerationHandle&) = default;
  GenerationHandle(GenerationHandle&&) noexcept = default;
  GenerationHandle& operator=(GenerationHandle&&) noexcept = default;

  // Whether this handle denotes a generation. False for default-constructed
  // and moved-from handles.
  bool valid() const noexcept;

  // Requests cancellation, which lands within one step. A no-op on an invalid
  // handle.
  void cancel() const;

  // Becomes true after the terminal callback returns or throws. False for an
  // invalid handle.
  bool done() const;

  // Blocks until the generation and its terminal callback have ended. Returns
  // immediately if both already have or the handle is invalid. Must not be
  // called from a callback serviced by the same runner.
  void wait() const;

  // The terminal reason once done. nullopt for an invalid or unfinished
  // handle.
  std::optional<FinishReason> finish_reason() const;

  // Diagnostic for a failed generation, when one is available. In particular,
  // preserves std::exception::what() when a callback throws. Meaningful only
  // when valid() && done(); empty when no diagnostic is available.
  std::string error_message() const;

  // This generation's timeline and counts, complete once done(). Empty on a
  // default-constructed handle.
  GenerationMetrics metrics() const;

 private:
  friend class RunnerImpl;
  friend class Session;
  explicit GenerationHandle(std::shared_ptr<GenerationHandleState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<GenerationHandleState> state_;
};

// Sole owner of one open executor session. Move it to transfer ownership.
// Destruction requests an asynchronous close and cancels any active
// generation, so retain it for as long as that generation should run.
//
// The close request never waits for the engine thread. Runner::shutdown()
// remains the deterministic boundary for executor cleanup.
class ET_EXPERIMENTAL Session {
 public:
  Session();
  ~Session();

  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // A snapshot of whether this object denotes a logically open session.
  // Returns false for default, moved-from, destructing, and shutdown-closed
  // sessions. A concurrent close may begin immediately after a true result.
  bool valid() const noexcept;

  // The absolute position of the next input to the executor. Safe from any
  // thread.
  //
  // If the last callback delivered a token that has not yet been fed back,
  // that token is retained internally as pending and will be the next input at
  // this position. A subsequent generation prepends it to the caller's delta.
  // Without a pending token, the caller's delta begins at this position.
  //
  // Before each callback, position is updated to reflect all inputs committed
  // for the updates delivered so far. It does not include in-flight or
  // discarded executor work. Once a handle is done, the value is stable until
  // the next generation. Returns 0 for a default or moved-from Session.
  Position position() const noexcept;

  // `delta` is the caller-resolved suffix to append. The session tracks its
  // committed position and any pending generated token, so consecutive
  // generations continue where prior executor work left them. Retain this
  // Session until the asynchronous generation ends;
  // destroying it requests close and completes active work as Cancelled.
  //
  // The delta must be non-empty and its exclusive end must fit in Position.
  // Invalid input and a second concurrent generation end as Failed. A default
  // or moved-from Session also completes synchronously as Failed; a retained
  // shutdown-closed Session completes synchronously as Cancelled.
  GenerationHandle generate_async(
      std::vector<Token> delta,
      GenConfig config,
      GenerationCallback on_update) const;

 private:
  friend class RunnerImpl;
  explicit Session(std::unique_ptr<SessionState> state);

  std::unique_ptr<SessionState> state_;
};

class ET_EXPERIMENTAL Runner {
 public:
  // Takes the scheduler, one per runner, which also supplies the prefill chunk
  // size prompts are split to. Owned rather than borrowed because a Session
  // keeps the runner's internals alive after ~Runner, and the close and cancel
  // paths reach the scheduler from there; a caller holding it separately would
  // have no way to see that its lifetime had been extended. Must be non-null.
  //
  // The executor is not owned and must outlive shutdown or destruction on an
  // external thread.
  Runner(Executor& executor, std::unique_ptr<Scheduler> scheduler);
  ~Runner();

  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  // Any thread; queued to the engine thread and acked.
  //
  // nullopt = the executor is at capacity, or the runner is shutting down.
  std::future<std::optional<Session>> open_session_async();

  // Idempotent. External callers block until the engine is joined, every live
  // generation has ended, and every owned session is closed. A generation that
  // was still running ends Cancelled; one that had already reached a stop
  // token, its token budget, or a failure keeps that reason, its tokens, and
  // its error message. Concurrent external callers wait for the same
  // completion.
  //
  // From a GenerationCallback, requests shutdown and returns without waiting;
  // a later external call or destructor must join. Do not destroy the Runner
  // from its callback.
  void shutdown();

  // What the engine measured. Read it after shutdown(): the counters are the
  // engine thread's, so joining it is what makes them stable and visible. A
  // call before then returns a torn snapshot.
  EngineMetrics metrics() const;

 private:
  std::shared_ptr<RunnerImpl> impl_;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
