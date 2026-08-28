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
//
// -- Session state ----------------------------------------------------------
//
// A session's state is a sequence of committed tokens, and its length is the
// session's position: the absolute position the next token will occupy. A
// freshly opened session has length 0. Nothing here says the state is
// positionally addressable -- a KV cache and a recurrent state both satisfy
// this -- only that its length is well defined.
//
// An Input's slice covers absolute positions [position + offset,
// position + offset + size). Call that range's upper bound the input's end.
//
// Committing:
//   - Every input commits its slice.
//   - An input with produce_output also commits all but the last token of its
//     Output, consecutively from the input's end, leaving the session at
//     end + tokens.size() - 1.
//   - The last token of Output::tokens is not committed. It is the model's own
//     next prediction, and it lands only when the runner feeds it back.
//   - An input without produce_output commits only its slice.
//
// Rewinding: an input whose absolute start is below the session's length
// discards everything from that position up, then commits. Equal appends.
// Above is a gap, and the batch must fail. The runner rewinds when a stop
// token, the token budget, or a cancellation ends a generation part way
// through a multi-token Output, so an implementation must retain whatever
// rewinding costs it -- a length pointer, a state snapshot -- until the
// session's next input arrives or the session closes.
//
// An implementation that cannot rewind must fail the batch. It cannot say so
// in advance, and a runner it never hands a multi-token Output will never ask
// it to.

#include <cstdint>
#include <optional>
#include <vector>

#include <executorch/extension/llm/batching/types.h>
#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

class ET_EXPERIMENTAL Executor {
 public:
  virtual ~Executor() = default;

  // A session to route tasks to. nullopt = at capacity. Every successful id
  // must be unique for the lifetime of the consuming Runner, even after close.
  virtual std::optional<SessionId> open_session() = 0;

  // Release a session and anything it holds. Unknown ids are ignored, so a
  // double close is not an error, though Runner closes each owned id once.
  virtual void close_session(SessionId session) = 0;

  // Installs the session's sampling policy, immediately before the
  // generation's tasks are submitted, and holds until the next generation
  // replaces it. Applies to every input of that session until then, so it does
  // not ride on each one.
  //
  // A token's randomness should derive from (seed, position), so results do
  // not depend on how batches form or speculative rounds roll back. A nullopt
  // seed requests nondeterministic seeding.
  virtual void set_sampling(
      SessionId session,
      const SamplingParams& params,
      std::optional<std::uint64_t> seed) = 0;

  // Run one batch. `out.outputs` is resized to batch.inputs.size() and filled
  // position-wise: outputs[i] answers inputs[i].
  //
  // The batch arrives shaped as the scheduler packed it, and every input must
  // be answered. An implementation whose model needs static shapes pads or
  // splits inside execute; it cannot constrain what the runner sends.
  //
  // An entry is nullopt exactly when its input had produce_output unset, as on
  // the leading chunks of a prompt, whose predictions are discarded. Otherwise
  // Output::tokens carries every token that input produced: usually one, but a
  // speculative executor returns the tokens it accepted plus the model's own
  // next token.
  //
  // No position is reported. The runner owns the session's length: it knows
  // what it fed and, from the rule above, which of the tokens returned here
  // were committed. A rejected speculative round is invisible to it and needs
  // to be -- those tokens never became Output::tokens, so they never entered
  // the transcript.
  //
  // Whether to continue is the runner's decision. An executor that produced
  // tokens past a stop token or the token budget will see them dropped, and
  // the session rewound below them on its next input. It has no way to end a
  // generation itself, and needs none.
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
