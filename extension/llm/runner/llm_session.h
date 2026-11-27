/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Engine/Session interfaces for model-specific LLM implementations. LLMEngine
// owns loaded model resources; LLMSession owns one logical generation state.
// Higher-level generation APIs can wrap this lower-level token-step contract.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch::extension::llm {

/// Per-decode sampling parameters. Implementations apply supported fields and
/// reject unsupported non-default values rather than silently ignoring them. -1
/// temperature means implementation default.
struct ET_EXPERIMENTAL SamplingConfig {
  float temperature = -1.0f;
  float top_p = 1.0f;
  int32_t top_k = 0; // 0 = disabled
  uint64_t seed = 0; // 0 = unset
};

/// One decoded step: the exact sampled token id and its decoded text piece
/// (raw bytes; may be a partial UTF-8 sequence the caller assembles).
///
/// `is_eos` is literal: the sampled token is an end-of-sequence token (use it
/// for the "stop" finish reason, metrics, or accounting). `is_terminal` is
/// the loop signal: generation ended at this step, either EOS or a cooperative
/// stop() took effect. A decode loop should end when is_terminal is set; every
/// EOS step is also terminal, but a stop step is terminal without being EOS.
///
/// For a cooperative stop step (requested via stop()), no token is forwarded,
/// position() must not advance, `token_id` must be 0, and `text_piece` must be
/// empty.
struct ET_EXPERIMENTAL DecodeResult {
  uint64_t token_id = 0;
  std::string text_piece;
  bool is_eos = false;
  bool is_terminal = false;
};

/// How many physical sessions an engine can host without silently multiplying
/// model memory. This is an engine-level capacity contract, distinct from how a
/// session advances a conversation.
struct ET_EXPERIMENTAL LLMServingCapacity {
  // Physical sessions creatable without duplicating model weights.
  int32_t max_physical_sessions_without_weight_duplication = 1;
  // Estimated device memory added per session, or 0 if unknown.
  int64_t estimated_bytes_per_session = 0;
};

/// One logical generation state's mutable buffers and position cursor. Created
/// by an LLMEngine.
class ET_EXPERIMENTAL LLMSession {
 public:
  virtual ~LLMSession() = default;

  /// Prefill pre-tokenized input at the current position. Must be non-empty and
  /// fit the context window.
  ///
  /// `initial_sampling` is for implementations that sample the first generated
  /// token during prefill. Implementations that sample only in decode_one() may
  /// ignore null/default configs, but should reject unsupported non-default
  /// fields.
  ///
  /// ERROR CONTRACT: an error may be returned AFTER backend state has already
  /// mutated. On any error from prefill_tokens()/decode_one(), the session is
  /// POISONED -- position() may no longer agree with resident state. The
  /// caller must call reset() (and only proceed once it returns Ok) before any
  /// further prefill/decode; it must NOT retry the failed call.
  ET_NODISCARD virtual ::executorch::runtime::Error prefill_tokens(
      const std::vector<uint64_t>& tokens,
      const SamplingConfig* initial_sampling = nullptr) = 0;

  /// Decode one token from the pending state; looping reproduces a full
  /// generation while returning exact sampled token ids. A normal decode_one()
  /// runs one forward pass and is not interruptible mid-call. If stop() is
  /// pending, decode_one() instead returns the synthetic terminal stop result
  /// documented on DecodeResult without forwarding a token.
  /// On error the session is poisoned -- see the error contract on
  /// prefill_tokens() (reset() before any further use; never retry).
  ET_NODISCARD virtual ::executorch::runtime::Result<DecodeResult> decode_one(
      const SamplingConfig& sampling) = 0;

  /// Current logical token position for this session.
  virtual int64_t position() const = 0;

  /// Clear mutable state and position for a fresh conversation.
  ET_NODISCARD virtual ::executorch::runtime::Error reset() = 0;

  /// Request that a decode_one() loop stop. This is a TOKEN-BOUNDARY,
  /// cooperative stop: it is safe to call from another thread, but it does not
  /// abort a decode_one() that is already running. It takes effect at the next
  /// decode_one(), which then returns a terminal step (is_terminal set, is_eos
  /// false) without forwarding a new token. For that synthetic step, token_id
  /// is 0, text_piece is empty, and position() does not advance. The stop is
  /// cleared by the next prefill_tokens() or reset().
  virtual void stop() = 0;
};

/// Holds immutable model resources once and creates isolated sessions. How
/// many sessions can be created without duplicating model weights is
/// backend-dependent; see serving_capacity().
class ET_EXPERIMENTAL LLMEngine {
 public:
  virtual ~LLMEngine() = default;

  /// Build a new session that reuses this engine's model resources and owns
  /// its own mutable generation state.
  ET_NODISCARD virtual ::executorch::runtime::Result<
      std::unique_ptr<LLMSession>>
  create_session() = 0;

  /// How many physical sessions this engine can host without duplicating
  /// weights, plus an optional per-session memory estimate.
  virtual LLMServingCapacity serving_capacity() const = 0;

  /// Model metadata such as context length and tokenizer-specific IDs.
  virtual const std::unordered_map<std::string, int64_t>& metadata() const = 0;
};

} // namespace executorch::extension::llm
