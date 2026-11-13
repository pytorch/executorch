/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given a text prompt, encode it using tokenizer and prefill the KV cache of a
// LLM.

#include <executorch/extension/llm/runner/text_prefiller.h>
#include <algorithm>

namespace executorch {
namespace extension {
namespace llm {

TextPrefiller::TextPrefiller(
    TextDecoderRunner* text_decoder_runner,
    bool use_kv_cache,
    bool enable_parallel_prefill,
    int64_t max_seq_len,
    TextDecoderRunner* decode_runner)
    : text_decoder_runner_(text_decoder_runner),
      decode_runner_(decode_runner),
      use_kv_cache_(use_kv_cache),
      enable_parallel_prefill_(enable_parallel_prefill),
      max_seq_len_(max_seq_len > 0 ? max_seq_len : 128) {}

::executorch::runtime::Result<uint64_t> TextPrefiller::prefill(
    std::vector<uint64_t>& prompt_tokens,
    int64_t& start_pos) {
  return prefill(prompt_tokens, start_pos, 0.0f);
}

::executorch::runtime::Result<uint64_t> TextPrefiller::prefill(
    std::vector<uint64_t>& prompt_tokens,
    int64_t& start_pos,
    float temperature) {
  ET_CHECK_MSG(!prompt_tokens.empty(), "Prompt cannot be null");
  ET_CHECK_OR_RETURN_ERROR(
      temperature >= 0.0f && temperature <= 1.0f,
      InvalidArgument,
      "Temperature must be in [0, 1], got %f",
      static_cast<double>(temperature));
  if (!text_decoder_runner_->is_method_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(text_decoder_runner_->load());
  }

  // Check if we need to chunk the prompt tokens
  int32_t num_prompt_tokens = prompt_tokens.size();

  // If prompt tokens exceed max_seq_len_, we need to chunk them
  if (num_prompt_tokens > max_seq_len_) {
    uint64_t cur_token = 0;
    int num_tokens_to_process = 0;

    while (num_tokens_to_process < num_prompt_tokens) {
      auto num_tokens_to_prefill_with = std::min<int>(
          num_prompt_tokens - num_tokens_to_process, max_seq_len_);

      std::vector<uint64_t> prompt_tokens_to_process(
          num_tokens_to_prefill_with);
      std::copy(
          prompt_tokens.begin() + num_tokens_to_process,
          prompt_tokens.begin() + num_tokens_to_process +
              num_tokens_to_prefill_with,
          prompt_tokens_to_process.begin());

      // Only the final chunk samples the first generated token.
      const bool is_last_chunk =
          num_tokens_to_process + num_tokens_to_prefill_with >=
          num_prompt_tokens;
      auto chunk_result = prefill_chunk(
          prompt_tokens_to_process,
          start_pos,
          is_last_chunk ? temperature : 0.0f);
      ET_CHECK_OK_OR_RETURN_ERROR(chunk_result.error());
      cur_token = chunk_result.get();

      num_tokens_to_process += num_tokens_to_prefill_with;
    }

    return cur_token;
  } else {
    // If prompt tokens don't exceed max_seq_len_, process them directly
    return prefill_chunk(prompt_tokens, start_pos, temperature);
  }
}

::executorch::runtime::Result<uint64_t> TextPrefiller::prefill_chunk(
    std::vector<uint64_t>& prompt_tokens,
    int64_t& start_pos) {
  return prefill_chunk(prompt_tokens, start_pos, 0.0f);
}

::executorch::runtime::Result<uint64_t> TextPrefiller::prefill_chunk(
    std::vector<uint64_t>& prompt_tokens,
    int64_t& start_pos,
    float temperature) {
  ET_CHECK_OR_RETURN_ERROR(
      temperature >= 0.0f && temperature <= 1.0f,
      InvalidArgument,
      "Temperature must be in [0, 1], got %f",
      static_cast<double>(temperature));
  // enable_parallel_prefill_ maybe set even when not using kv cache
  // When kv cache is not used, start pos is ignored
  int32_t num_prompt_tokens = prompt_tokens.size();

  if (enable_parallel_prefill_ || !use_kv_cache_) {
    const auto capacity = text_decoder_runner_->max_tokens_per_step(start_pos);
    if (capacity != 0) {
      auto tokens = from_blob(
          prompt_tokens.data(),
          {1, num_prompt_tokens},
          executorch::aten::ScalarType::Long);
      auto outputs = text_decoder_runner_->step(tokens, start_pos);
      ET_CHECK_OK_OR_RETURN_ERROR(outputs.error());
      start_pos += num_prompt_tokens;
      return text_decoder_runner_->logits_to_token(*outputs, temperature);
    }
    ET_CHECK_OR_RETURN_ERROR(
        use_kv_cache_ && decode_runner_ &&
            decode_runner_ != text_decoder_runner_,
        OutOfResources,
        "No prompt step fits the remaining context.");
    // Choose decode before execution: an execution error is not retryable.
    ET_CHECK_OK_OR_RETURN_ERROR(decode_runner_->load());
  }
  auto* runner =
      enable_parallel_prefill_ ? decode_runner_ : text_decoder_runner_;
  uint64_t cur_token = 0;
  auto tokens =
      from_blob(&cur_token, {1, 1}, executorch::aten::ScalarType::Long);
  for (size_t i = 0; i < prompt_tokens.size(); ++i) {
    cur_token = prompt_tokens[i];
    auto logits = runner->step(tokens, start_pos);
    ET_CHECK_OK_OR_RETURN_ERROR(logits.error());
    ++start_pos;
    if (i + 1 == prompt_tokens.size()) {
      cur_token = runner->logits_to_token(*logits, temperature);
    }
  }
  return cur_token;
}

} // namespace llm
} // namespace extension
} // namespace executorch
