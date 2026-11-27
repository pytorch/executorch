/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Decides which tasks run in the next batch, and accepts submissions.
//
// Bookkeeping only: never runs a callback, never calls an executor. Tasks are
// handed back and the caller decides what happens to them.
//
// Implementations must be thread safe, and cheap enough for an async context:
// bounded work, no I/O, no blocking on caller code.

#include <cstddef>
#include <vector>

#include <executorch/extension/llm/batching/types.h>
#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

class ET_EXPERIMENTAL Scheduler {
 public:
  virtual ~Scheduler() = default;

  // All or nothing: a prompt's chunks only make sense together. False means
  // rejected, and nothing was queued.
  //
  // A tid identifies a task to get_work() and cancel(), so it must be unique
  // among the tasks queued here, including the others in this vector. It is
  // free for reuse once the task has been dispatched or cancelled; ids need
  // not be unique for all time.
  virtual bool submit(std::vector<Task> tasks) = 0;

  // A hint, not a reservation. Another thread may take the work first.
  virtual bool has_work() const = 0;

  // The caller owns what it gets back and must complete each task once.
  //
  // A session may appear more than once, but only as consecutive prefill
  // chunks, which form one wider prefill. At most one of its tasks has
  // produce_output. Submitting chunks whose positions abut is the caller's
  // responsibility; the scheduler preserves their order but does not check
  // the positions.
  virtual std::vector<Task> get_work() = 0;

  // Drops the session's queued tasks and returns them, to be completed as
  // Cancelled. A task already handed out belongs to the caller, so cancelling
  // an in-flight task, or an unknown session, returns nothing.
  virtual std::vector<Task> cancel(SessionId sid) = 0;

  // Drops every queued task and returns them all, for shutdown.
  // For shutdown.
  virtual std::vector<Task> clear() = 0;

  // Largest prefill chunk this scheduler will admit. Callers split a prompt to
  // fit; a wider chunk is rejected, not split. Non-zero, so it is safe to
  // divide by.
  virtual std::size_t max_prefill_chunk_size() const = 0;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
