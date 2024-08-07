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
#include <executorch/extension/llm/tokenizer/tokenizer.h>
#include <functional>

namespace torch::executor {

class TextPrefiller {
 public:
  TextPrefiller(
      Tokenizer* tokenizer,
      TextDecoderRunner* text_decoder_runner,
      bool use_kv_cache_,
      bool enable_parallel_prefill);
  /**
   * Prefill an LLM Module with the given text input.
   * @param prompt_tokens The text prompt tokens to the LLM Module. Encoded by
   * tokenizer.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * Module.
   * @param token_callback A callback function that will be called for each
   * token in the prompt.
   * @return The next token of the LLM Module after prefill.
   */
  Result<uint64_t> prefill(
      const std::vector<uint64_t>& prompt_tokens,
      int64_t start_pos = 0,
      std::function<void(const std::string&)> token_callback = {});

 private:
  Tokenizer* tokenizer_;
  TextDecoderRunner* text_decoder_runner_;
  bool use_kv_cache_;
  bool enable_parallel_prefill_;
};

} // namespace torch::executor
