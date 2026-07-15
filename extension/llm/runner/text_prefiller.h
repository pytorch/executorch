/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given a text prompt, encode it using tokenizer and prefill the KV cache of a
// LLM.

#pragma once

#include <executorch/extension/llm/runner/text_decoder_runner.h>

namespace executorch {
namespace extension {
namespace llm {

class ET_EXPERIMENTAL TextPrefiller {
 public:
  TextPrefiller(
      TextDecoderRunner* text_decoder_runner,
      bool use_kv_cache,
      bool enable_parallel_prefill,
      int64_t max_seq_len = 128);

  virtual ~TextPrefiller() = default;
  /**
   * Prefill an LLM Module with the given text input.
   * @param prompt_tokens The text prompt tokens to the LLM Module. Encoded by
   * tokenizer.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * Module.
   * Equivalent to `prefill(prompt_tokens, start_pos, 0.0f)`.
   * @return The next token of the LLM Module after prefill.
   */
  virtual ::executorch::runtime::Result<uint64_t> prefill(
      std::vector<uint64_t>& prompt_tokens,
      int64_t& start_pos);

  /**
   * Like `prefill(prompt_tokens, start_pos)`, but samples the first generated
   * token with `temperature` in [0.0, 1.0].
   */
  virtual ::executorch::runtime::Result<uint64_t> prefill(
      std::vector<uint64_t>& prompt_tokens,
      int64_t& start_pos,
      float temperature);

  /**
   * Helper method to prefill a chunk of tokens.
   * @param prompt_tokens The chunk of text prompt tokens to process.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * Module.
   * Equivalent to `prefill_chunk(prompt_tokens, start_pos, 0.0f)`.
   * @return The next token of the LLM Module after prefilling this chunk.
   */
  virtual ::executorch::runtime::Result<uint64_t> prefill_chunk(
      std::vector<uint64_t>& prompt_tokens,
      int64_t& start_pos);

  /**
   * Like `prefill_chunk(prompt_tokens, start_pos)`, but samples the produced
   * token with `temperature` in [0.0, 1.0].
   */
  virtual ::executorch::runtime::Result<uint64_t> prefill_chunk(
      std::vector<uint64_t>& prompt_tokens,
      int64_t& start_pos,
      float temperature);

  /**
   * Load the necessary resources for the TextPrefiller.
   * This method should be called before using the prefill methods.
   */
  ::executorch::runtime::Error load() {
    return text_decoder_runner_->load();
  }

  /**
   * Check if the TextPrefiller has been successfully loaded.
   * @return True if the resources are loaded, false otherwise.
   */
  bool inline is_loaded() const {
    // Implementation to check if resources are loaded
    return text_decoder_runner_->is_method_loaded();
  }

 private:
  /**
   * Note: TextPrefiller does not own the TextDecoderRunner instance.
   * The responsibility of managing the lifecycle of TextDecoderRunner
   * lies with the outer class or entity (likely Runner) that creates
   * and passes the TextDecoderRunner instance to TextPrefiller.
   */
  TextDecoderRunner* text_decoder_runner_;
  bool use_kv_cache_;
  bool enable_parallel_prefill_;
  int64_t max_seq_len_;
};

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::TextPrefiller;
} // namespace executor
} // namespace torch
