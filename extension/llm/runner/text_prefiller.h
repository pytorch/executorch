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
// patternlint-disable-next-line executorch-cpp-nostdinc
#include <functional>

namespace executorch {
namespace extension {
namespace llm {

class ET_EXPERIMENTAL TextPrefiller {
 public:
  TextPrefiller(
      TextDecoderRunner* text_decoder_runner,
      bool use_kv_cache_,
      bool enable_parallel_prefill);
  /**
   * Prefill an LLM Module with the given text input.
   * @param prompt_tokens The text prompt tokens to the LLM Module. Encoded by
   * tokenizer.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * Module.
   * @return The next token of the LLM Module after prefill.
   */
  ::executorch::runtime::Result<uint64_t> prefill(
      std::vector<uint64_t>& prompt_tokens,
      int64_t& start_pos);

 private:
  TextDecoderRunner* text_decoder_runner_;
  bool use_kv_cache_;
  bool enable_parallel_prefill_;
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
