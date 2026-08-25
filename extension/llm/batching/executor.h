/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The seam between the batched runner and whatever actually runs a forward.
// The interface is expressed in plain tokens and positions, so a fake needs
// neither a .pte nor a GPU.
//
// Sessions live here because the cache owns their identity. Until a batched
// cache exists an implementation may number them however it likes.
//
// Called only from the runner's engine thread, so implementations need no
// locking of their own. For the same reason an implementation must not call
// back into the runner: shutdown() joins the thread it would be running on,
// and the session and generation calls are queued to it.

#include <cstdint>
#include <optional>
#include <vector>

#include <executorch/extension/llm/batching/types.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

class Executor {
 public:
  virtual ~Executor() = default;

  // A session to route tasks to. nullopt = at capacity. Every successful id
  // must be unique for the lifetime of the consuming Runner, even after close.
  virtual std::optional<SessionId> open_session() = 0;

  // Release a session and anything it holds. Unknown ids are ignored, so a
  // double close is not an error, though Runner closes each owned id once.
  virtual void close_session(SessionId session) = 0;

  // Selects the session's sampling stream, immediately before the generation's
  // tasks are submitted. A token's randomness should derive from (seed,
  // position), so results do not depend on how batches form or speculative
  // rounds roll back. nullopt requests nondeterministic seeding.
  virtual void set_sampling_seed(
      SessionId session,
      std::optional<std::uint64_t> seed) {
    (void)session;
    (void)seed;
  }

  // Run one batch. `out.outputs` is resized to batch.inputs.size() and filled
  // position-wise: outputs[i] answers inputs[i].
  //
  // The batch arrives shaped as the scheduler packed it, and every input must
  // be answered. An implementation whose model needs static shapes pads or
  // splits inside execute; it cannot constrain what the runner sends.
  //
  // An entry is nullopt exactly when its input had produce_output unset, as on
  // the leading chunks of a prompt, whose predictions are discarded. Otherwise
  // it carries every token that input produced: usually one, but a speculative
  // executor returns the tokens it accepted plus the model's own next token.
  //
  // Output::next is where the session continues, already positioned. Only the
  // executor knows what the forward actually wrote, including what a rejected
  // speculative round rolled back.
  //
  // A session may appear in more than one input of a batch when consecutive
  // prefill chunks of its prompt land together. They arrive in order, with
  // contiguous ranges, and at most one has produce_output set.
  //
  // false = the batch failed as a whole; there is no partial success. The
  // runner completes every task in it as Failed and poisons their sessions,
  // because what was written before the failure is unknown.
  virtual bool execute(const BatchInput& batch, BatchOutput& out) = 0;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
