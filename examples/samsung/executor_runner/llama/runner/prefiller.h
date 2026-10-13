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
  Prefiller(
      llm::TextDecoderRunner* text_decoder_runner,
      bool use_kv_cache,
      bool enable_parallel_prefill,
      int64_t max_seq_len = 128);

  virtual ~Prefiller() = default;

  virtual runtime::Result<uint64_t> prefill(
      std::vector<uint64_t>& prompt_tokens,
      int64_t& start_pos) override;

  virtual runtime::Result<uint64_t> prefill_chunk(
      std::vector<uint64_t>& prompt_tokens,
      int64_t& start_pos) override;

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
};

} // namespace example
