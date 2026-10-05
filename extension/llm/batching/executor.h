/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The seam between the batched runner and whatever actually runs a forward.
// Preparation accepts owned CPU inputs; execution carries opaque backing and
// position ranges, so a fake needs neither a .pte nor a GPU.
//
// Sessions live here because the cache owns their identity. Until a batched
// cache exists an implementation may number them however it likes.
//
// Called only from the runner's engine thread, so implementations need no
// locking of their own. Nor may an implementation call back into the runner:
// shutdown() joins that same thread, and the session and generation calls are
// queued to it.
//
// -- Session state ----------------------------------------------------------
//
// A session's state is a sequence of committed positions, and its length is the
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
// Above is a gap, and the batch must fail.
//
// The runner rewinds when a stop token, the token budget, or a cancellation
// ends a generation part way through a multi-token Output. An implementation
// must therefore retain whatever rewinding requires -- a length pointer, a
// state snapshot -- until the session's next input arrives or it closes.
//
// An implementation that cannot rewind must fail the batch; there is no way to
// declare that up front. One never handed a multi-token Output is never asked.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <executorch/extension/llm/batching/types.h>
#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

// Fixed for an executor's lifetime. Implementations must enforce workspace
// bounds before allocating/encoding, not merely inspect the final output.
struct ET_EXPERIMENTAL PreparationConfig {
  std::size_t max_positions = 1024 * 1024;
  std::size_t max_retained_bytes = 64 * 1024 * 1024;
  std::size_t max_workspace_bytes = 64 * 1024 * 1024;
  std::size_t max_total_retained_bytes = 256 * 1024 * 1024;
  std::size_t max_images = 0;

  // Capacity-based source ownership, including container storage. Returning
  // nullopt rejects unsupported vocabulary or source storage above workspace.
  std::optional<std::size_t> input_retained_bytes(
      const PreparationInput& input) const {
    if (input.segments.empty() ||
        max_workspace_bytes < sizeof(PreparationInput)) {
      return std::nullopt;
    }
    std::size_t bytes = sizeof(PreparationInput);
    if (input.segments.capacity() >
        (max_workspace_bytes - bytes) / sizeof(MultimodalInput)) {
      return std::nullopt;
    }
    bytes += input.segments.capacity() * sizeof(MultimodalInput);
    std::size_t images = 0;
    for (const auto& segment : input.segments) {
      std::size_t count = 0;
      std::size_t width = 1;
      if (segment.is_tokens()) {
        count = segment.get_tokens().capacity();
        width = sizeof(Token);
      } else if (segment.is_image()) {
        ++images;
        const auto& image = segment.get_image();
        count = image.is_uint8() ? image.get_uint8_data().capacity()
                                 : image.get_float_data().capacity();
        width = image.is_uint8() ? sizeof(uint8_t) : sizeof(float);
      } else {
        return std::nullopt;
      }
      if (count > (max_workspace_bytes - bytes) / width) {
        return std::nullopt;
      }
      bytes += count * width;
    }
    return images <= max_images ? std::optional<std::size_t>{bytes}
                                : std::nullopt;
  }
};

class ET_EXPERIMENTAL Executor {
 public:
  explicit Executor(PreparationConfig config = {})
      : preparation_config_(config) {}
  virtual ~Executor() = default;

  // Immutable, safe to inspect from any thread, including before initialize().
  const PreparationConfig& preparation_config() const {
    return preparation_config_;
  }

  // Context-free model preparation on the engine thread. Failure must not
  // mutate decoder sessions. Default text support rejects all other modalities.
  virtual bool prepare(const PreparationInput& input, PreparedInputPtr& out) {
    out.reset();
    std::size_t count = 0;
    for (const auto& segment : input.segments) {
      if (!segment.is_tokens() ||
          segment.get_tokens().size() >
              preparation_config_.max_positions - count) {
        return false;
      }
      count += segment.get_tokens().size();
    }
    const auto overhead =
        sizeof(TokenPreparedInput) + sizeof(std::vector<Token>);
    if (count == 0 || preparation_config_.max_retained_bytes < overhead ||
        count > (preparation_config_.max_retained_bytes - overhead) /
                sizeof(Token) ||
        count > preparation_config_.max_workspace_bytes / sizeof(Token)) {
      return false;
    }
    auto tokens = std::make_shared<std::vector<Token>>();
    tokens->reserve(count);
    for (const auto& segment : input.segments) {
      tokens->insert(
          tokens->end(),
          segment.get_tokens().begin(),
          segment.get_tokens().end());
    }
    return wrap_tokens(std::move(tokens), out);
  }

  // Lightweight token feedback only: no model execution or session mutation.
  // Own the supplied storage; never route decode through scheduled preparation.
  virtual bool wrap_tokens(
      std::shared_ptr<const std::vector<Token>> tokens,
      PreparedInputPtr& out) {
    out.reset();
    if (!tokens || tokens->empty() ||
        tokens->size() > preparation_config_.max_positions) {
      return false;
    }
    auto prepared =
        std::make_shared<const TokenPreparedInput>(std::move(tokens));
    if (prepared->retained_bytes() > preparation_config_.max_retained_bytes) {
      return false;
    }
    out = std::move(prepared);
    return true;
  }

  virtual bool accepts(const PreparedInput& input) const {
    return input.compatibility_tag() == TokenPreparedInput::tag();
  }

  bool valid_prepared(const PreparedInputPtr& input) const {
    return input && accepts(*input) && input->position_count() > 0 &&
        input->position_count() <= preparation_config_.max_positions &&
        input->position_count() <=
        static_cast<std::size_t>(std::numeric_limits<Position>::max()) &&
        input->retained_bytes() <= preparation_config_.max_retained_bytes;
  }

  // Validate the ENTIRE batch before any decoder state mutation, including
  // implementations called directly without Runner. Payload-specific checks
  // belong in execute(), also before its first mutation.
  bool validate_batch(const BatchInput& batch) const {
    for (const auto& input : batch.inputs) {
      if (!valid_prepared(input.prepared) || input.size == 0 ||
          input.offset > input.prepared->position_count() ||
          input.size > input.prepared->position_count() - input.offset) {
        return false;
      }
      const auto start = static_cast<std::int64_t>(input.position) +
          static_cast<std::int64_t>(input.offset);
      if (start < 0 || start > std::numeric_limits<Position>::max() ||
          input.size > static_cast<std::size_t>(
                           std::numeric_limits<Position>::max() - start)) {
        return false;
      }
    }
    return true;
  }

  // Optional one-time setup, called on the engine thread before any other
  // method.
  //
  // false or an exception = the runner admits no work, so
  // open_session_async() reports nullopt.
  virtual bool initialize() {
    return true;
  }

  // Tokens a batch should carry. At this size one execute() is a single pass
  // over the weights; a smaller batch leaves part of that pass unused, a wider
  // one is accepted but splits into slices that each re-read them. Size a
  // scheduler's batch budget to it. 0 = no preference.
  virtual std::size_t preferred_batch_tokens() const {
    return 0;
  }

  // A session to route tasks to. nullopt = at capacity. Every successful id
  // must be unique for the lifetime of the consuming Runner, even after close.
  virtual std::optional<SessionId> open_session() = 0;

  // Release a session and anything it holds. Unknown ids are ignored, so a
  // double close is not an error, though Runner closes each owned id once.
  virtual void close_session(SessionId session) = 0;

  // Independently writable state for source's committed [0, upto) prefix,
  // with a lifetime-unique id and no sampling policy. Called between forwards;
  // source may have an active generation or hold speculative state past upto.
  // The caller owns the returned session until close_session().
  //
  // nullopt = unsupported, unavailable retained history, or insufficient
  // capacity. An empty prefix may also be refused. Failure, including an
  // exception, must leave source unchanged and retain no new session.
  virtual std::optional<SessionId> clone(
      SessionId /*source*/,
      Position /*upto*/) {
    return std::nullopt;
  }

  // Installs the session's sampling policy, immediately before the generation's
  // tasks are submitted, and holds until the next generation replaces it, so it
  // does not ride on each input.
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
  // No position is reported: the runner tracks length from what it fed and, by
  // the rule above, which returned tokens committed. A rejected speculative
  // round never becomes Output::tokens, so it stays invisible.
  //
  // Whether to continue is the runner's decision. Tokens produced past a stop
  // token or the budget are dropped and the session rewound below them on its
  // next input; an executor cannot end a generation itself.
  //
  // A session may appear in more than one input of a batch when consecutive
  // prefill chunks of its prompt land together. They arrive in order, with
  // contiguous ranges, and at most one has produce_output set.
  //
  // false = the batch failed as a whole; there is no partial success. The
  // runner completes every task in it as Failed and poisons their sessions,
  // because what was written before the failure is unknown.
  virtual bool execute(const BatchInput& batch, BatchOutput& out) = 0;

 private:
  const PreparationConfig preparation_config_;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
