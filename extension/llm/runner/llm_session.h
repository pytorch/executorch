/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Model-agnostic Engine/Session interfaces. Model-specific execution lives in
// adapters that implement these (TextLLMSession over TextLLMRunner today;
// Gemma4Session etc. later); the serving code (HTTP control plane + C++ worker
// binaries) depends only on these interfaces, never on a concrete runner.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch::extension::llm {

/// Per-decode sampling parameters. An adapter applies the fields it supports
/// and rejects non-default values of the rest rather than silently ignoring
/// them (today only temperature is plumbed). -1 temperature means model
/// default.
struct SamplingConfig {
  float temperature = -1.0f;
  float top_p = 1.0f;
  int32_t top_k = 0; // 0 = disabled
  uint64_t seed = 0; // 0 = unset
};

/// One decoded step: the exact sampled token id (for prefix-cache id tracking
/// and batching) and its decoded text piece (raw bytes; may be a partial UTF-8
/// sequence the caller assembles).
///
/// `is_eos` is literal: the sampled token is an end-of-sequence token (use it
/// for the "stop" finish reason, metrics, cache/accounting). `is_terminal` is
/// the loop signal: generation ended at this step — either EOS or a cooperative
/// stop() took effect. A decode loop should end when is_terminal is set; every
/// EOS step is also terminal, but a stop step is terminal without being EOS.
struct DecodeResult {
  uint64_t token_id;
  std::string text_piece;
  bool is_eos;
  bool is_terminal;
};

/// How many physical sessions an engine can host, so the server admits logical
/// requests without silently multiplying model memory. This is a *serving
/// capacity* concern (engine-level), distinct from how a session advances a
/// conversation (LLMSession) — keep backend memory flags off LLMSession.
struct LLMServingCapacity {
  // Physical sessions creatable without duplicating packed weights.
  // Conservatively 1 (some backends repack weights per runtime, so extra
  // sessions would copy the whole model); raise only on a backend proven to
  // share packed weights.
  int32_t max_physical_sessions_without_weight_duplication = 1;
  // Planned bytes one session adds (KV + activations), for memory-budget
  // admission. 0 = unknown; the server skips the memory clamp.
  int64_t estimated_bytes_per_session = 0;
};

/// One conversation's mutable state (KV cache, position cursor). Created by an
/// LLMEngine; conversation/cache-scoped (kept warm across requests for prefix
/// reuse), not request-scoped.
class ET_EXPERIMENTAL LLMSession {
 public:
  virtual ~LLMSession() = default;

  /// Prefill pre-tokenized input at the current position (call seek() first for
  /// prefix reuse). Must be non-empty and fit the context window.
  ///
  /// `initial_sampling` (optional): the sampling config for the FIRST generated
  /// token, for backends that sample during prefill (e.g. in-graph sampling).
  /// Pass it so the first token uses the request's sampling instead of a stale
  /// default. Backends that only sample in decode_one() ignore it.
  virtual ::executorch::runtime::Error prefill_tokens(
      std::vector<uint64_t> tokens,
      const SamplingConfig* initial_sampling = nullptr) = 0;

  /// Decode one token from the pending state; looping reproduces a full
  /// generation while returning exact sampled token ids. A single decode_one()
  /// runs one forward pass and is not interruptible mid-call (see stop()).
  virtual ::executorch::runtime::Result<DecodeResult> decode_one(
      const SamplingConfig& sampling) = 0;

  /// Rewind the KV cache to `pos` (prefix reuse). Valid for full-KV models;
  /// sliding-window KV may reject a seek past its window (the caller falls back
  /// to a fresh prefill).
  virtual ::executorch::runtime::Error seek(int64_t pos) = 0;

  /// Number of tokens with resident KV (upper bound for seek()).
  virtual int64_t position() const = 0;

  /// Clear the KV cache / position for a fresh conversation.
  virtual ::executorch::runtime::Error reset() = 0;

  /// Request that a decode_one() loop stop. This is a TOKEN-BOUNDARY,
  /// cooperative stop: it is safe to call from another thread, but it does not
  /// abort a decode_one() that is already running. It takes effect at the next
  /// decode_one(), which then returns a terminal step (is_terminal set, is_eos
  /// false) without forwarding a new token. The stop is cleared by the next
  /// prefill_tokens() or reset().
  virtual void stop() = 0;
};

/// Holds the immutable model resources (program, tokenizer, metadata) once and
/// creates sessions that reuse them while isolating their own KV state. How
/// many sessions can be created without duplicating packed weights is backend-
/// dependent — see serving_capacity().
class ET_EXPERIMENTAL LLMEngine {
 public:
  virtual ~LLMEngine() = default;

  /// Build a new session that reuses this engine's program/resources when the
  /// backend supports it, with its own KV cache. serving_capacity() is the
  /// authority on how many physical sessions are safe without weight
  /// duplication.
  virtual ::executorch::runtime::Result<std::unique_ptr<LLMSession>>
  create_session() = 0;

  /// How many physical sessions this engine can host without duplicating
  /// weights (+ optional per-session memory estimate); the server clamps the
  /// number of physical sessions it creates to this.
  virtual LLMServingCapacity serving_capacity() const = 0;
  virtual const std::unordered_map<std::string, int64_t>& metadata() const = 0;
};

} // namespace executorch::extension::llm
