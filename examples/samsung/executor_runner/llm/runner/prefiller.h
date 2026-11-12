/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given a text prompt, encode it using tokenizer and prefill the KV cache of a
// LLM.

#pragma once

#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>

namespace example {

namespace llm = ::executorch::extension::llm;
namespace runtime = ::executorch::runtime;
namespace extension = ::executorch::extension;

class ET_EXPERIMENTAL Prefiller : public llm::TextPrefiller {
 public:
  // max_seq_len is the prefill method's chunk length and prefill_cache_len
  // the number of positions its KV cache holds; positions past it go through
  // the decode method.
  Prefiller(
      llm::TextDecoderRunner* text_decoder_runner,
      bool use_kv_cache,
      bool enable_parallel_prefill,
      int64_t max_seq_len,
      int64_t prefill_cache_len);

  virtual ~Prefiller() = default;

  using llm::TextPrefiller::prefill;
  using llm::TextPrefiller::prefill_chunk;

  runtime::Result<uint64_t> prefill(
      std::vector<uint64_t>& prompt_tokens,
      int64_t& start_pos,
      float temperature) override;

  runtime::Result<uint64_t> prefill_chunk(
      std::vector<uint64_t>& prompt_tokens,
      int64_t& start_pos,
      float temperature) override;

 private:
  /**
   * Note: TextPrefiller does not own the TextDecoderRunner instance.
   * The responsibility of managing the lifecycle of TextDecoderRunner
   * lies with the outer class or entity (likely Runner) that creates
   * and passes the TextDecoderRunner instance to TextPrefiller.
   */
  llm::TextDecoderRunner* text_decoder_runner_;
  bool use_kv_cache_;
  bool enable_parallel_prefill_;
  int64_t max_seq_len_;
  int64_t prefill_cache_len_;
};

} // namespace example
