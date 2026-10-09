/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <executorch/extension/llm/batching/types.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/runtime/platform/compiler.h>

namespace tokenizers {
class Tokenizer;
}

namespace executorch {
namespace extension {
namespace llm {
namespace serving {

// The full prompt, not a session delta. Preparation encodes each text segment
// separately and appends ID segments verbatim, in order, without adding
// BOS/EOS. Segment boundaries therefore matter even between adjacent text
// segments. The default preparer supports text and token segments; model
// preparers may support other modalities. Invalid preparation preserves
// history.
struct ET_EXPERIMENTAL PromptInput {
  std::vector<MultimodalInput> segments;
};

struct ET_EXPERIMENTAL GenerationOptions {
  // Must be positive when set. Capped by the context remaining after the full
  // prompt; a prompt leaving no generation room is invalid. Unset uses that
  // remaining context, or the service default when the context limit is
  // unknown.
  std::optional<std::int32_t> max_new_tokens;
  // Per-request policy: finite temperature >= 0, finite top_p in (0, 1],
  // and top_k >= 0. Invalid options leave existing session history unchanged.
  batching::SamplingParams sampling;
  // Added to the service's default stop tokens. A matched token is excluded
  // from text and generated_token_ids, but retained in ordinary prompt history.
  std::vector<batching::Token> stop_tokens;
  // Non-empty strings matched across decoded pieces. The match and everything
  // after it are hidden; a match invalidates exact token replay and warm reuse.
  std::vector<std::string> stop_strings;
  // Forwarded to the executor for this generation, together with sampling.
  std::optional<std::uint64_t> seed;
};

enum class ET_EXPERIMENTAL FinishReason {
  Stop, // EOS, a stop token, or a stop string
  Length, // Generation budget exhausted
  Cancelled, // Request cancellation, session close/reset, or shutdown
  Failed, // See TerminalEvent::error
};

enum class ET_EXPERIMENTAL ErrorCode {
  InvalidArgument,
  NotReady,
  SessionNotFound,
  SessionBusy,
  CapacityExceeded,
  Internal,
};

struct ET_EXPERIMENTAL ServingError {
  ErrorCode code = ErrorCode::Internal;
  std::string message;
};

// Valid only during preparation on the runtime's control thread. The callback
// must do bounded work and may poll cancelled while preparing a large prompt.
struct ET_EXPERIMENTAL PromptPreparationContext {
  const tokenizers::Tokenizer& tokenizer;
  std::size_t max_prompt_positions;
  std::function<bool()> cancelled;
};

using PromptPreparationResult ET_EXPERIMENTAL =
    std::variant<PromptInput, ServingError>;
using PromptPreparation ET_EXPERIMENTAL =
    std::function<PromptPreparationResult(const PromptPreparationContext&)>;

using ModelPreparationResult ET_EXPERIMENTAL =
    std::variant<batching::PreparedInputPtr, ServingError>;
// Runtime-owned; consumes source storage on control after deferred source
// captures are destroyed.
using ModelPreparer ET_EXPERIMENTAL = std::function<
    ModelPreparationResult(const PromptPreparationContext&, PromptInput)>;

// Canonical token fast path for already-normalized owned tokens. Empty input
// is InvalidArgument. Tokenization/BOS policy belongs to the caller; context
// bounds remain enforced by the serving preparation pipeline.
ModelPreparationResult make_token_prepared_input(
    std::vector<batching::Token> tokens);

struct ET_EXPERIMENTAL GenerationStats {
  // Full prompt size in decoder positions, including any reused prefix.
  std::size_t prompt_tokens = 0;
  // Tokens processed by text output, excluding EOS/stop tokens but including
  // the token that completes a string stop. Later discarded tokens do not
  // count. Neither this count nor visible text describes logical session
  // history.
  std::size_t completion_tokens = 0;
  // Committed prompt prefix reused without execution. A pending prediction
  // fed by this request counts as prefilled, not reused.
  std::size_t reused_prompt_tokens = 0;
  // Consumed prompt positions; may be partial on cancellation or failure.
  std::size_t prefilled_prompt_tokens = 0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double total_ms = 0.0;
  std::string session_reset_reason;
  // Original output IDs counted by completion_tokens, excluding EOS/stop
  // tokens; not a retokenization of visible text or the full logical history
  // suffix. Present only for Stop/Length completion without a matched string
  // stop or rendering error. An empty vector is distinct from absent.
  std::optional<std::vector<batching::Token>> generated_token_ids;
};

struct ET_EXPERIMENTAL TextEvent {
  std::string text;
};

// Final serving result, fixed after engine settlement and text/history
// finalization. A terminal-sink exception or transport delivery failure does
// not revise this result or roll back resident session state.
struct ET_EXPERIMENTAL TerminalEvent {
  FinishReason finish_reason = FinishReason::Stop;
  GenerationStats stats;
  // Present exactly when finish_reason is Failed.
  std::optional<ServingError> error;
};

// Each admitted request emits zero or more ordered text events, then one
// terminal event. Synchronous rejection emits no events. A throwing sink is
// disabled and is not called again, even for the terminal.
//
// A request retains runtime admission until its terminal notification is
// selected for delivery. Before invoking the terminal sink, the runtime has
// released that admission and the request's session execution claim. Other
// limits, competing requests, or shutdown can still reject new work.
//
// Sinks must do short, nonblocking work. Terminal notification does not mean
// the callback has returned or that a transport delivered the result.
using GenerationEvent ET_EXPERIMENTAL = std::variant<TextEvent, TerminalEvent>;

struct ET_EXPERIMENTAL ServingInfo {
  bool ready = false;
  std::size_t max_context_length = 0;
  std::size_t max_sessions = 0;
  // Logical serving slots, including slots reserved during reset/reopen.
  // This is not a count of physically live executor sessions.
  std::size_t active_sessions = 0;
};

} // namespace serving
} // namespace llm
} // namespace extension
} // namespace executorch
