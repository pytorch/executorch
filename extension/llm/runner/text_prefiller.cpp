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
    int64_t max_seq_len)
    : text_decoder_runner_(text_decoder_runner),
      use_kv_cache_(use_kv_cache),
      enable_parallel_prefill_(enable_parallel_prefill),
      max_seq_len_(max_seq_len > 0 ? max_seq_len : 128) {}

::executorch::runtime::Result<uint64_t> TextPrefiller::prefill(
    std::vector<uint64_t>& prompt_tokens,
    int64_t& start_pos) {
  ET_CHECK_MSG(!prompt_tokens.empty(), "Prompt cannot be null");
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

      // Process this chunk
      auto chunk_result = prefill_chunk(prompt_tokens_to_process, start_pos);
      ET_CHECK_OK_OR_RETURN_ERROR(chunk_result.error());
      cur_token = chunk_result.get();

      start_pos += num_tokens_to_prefill_with;
      num_tokens_to_process += num_tokens_to_prefill_with;
    }

    return cur_token;
  } else {
    // If prompt tokens don't exceed max_seq_len_, process them directly
    return prefill_chunk(prompt_tokens, start_pos);
  }
}

::executorch::runtime::Result<uint64_t> TextPrefiller::prefill_chunk(
    std::vector<uint64_t>& prompt_tokens,
    int64_t& start_pos) {
  // enable_parallel_prefill_ maybe set even when not using kv cache
  // When kv cache is not used, start pos is ignored
  int32_t num_prompt_tokens = prompt_tokens.size();

  // store the token
  uint64_t cur_token;
  if (enable_parallel_prefill_ || !use_kv_cache_) {
    // initialize tensor wrappers
    auto tokens = from_blob(
        prompt_tokens.data(),
        {1, num_prompt_tokens},
        executorch::aten::ScalarType::Long);

    auto outputs_res = text_decoder_runner_->step(tokens, start_pos);

    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_LOG(
        Info, "Prefill token result numel(): %zu", outputs_res.get().numel());

    start_pos += num_prompt_tokens;
    cur_token = text_decoder_runner_->logits_to_token(outputs_res.get());
  } else { // sequential prefill
    int64_t pos = 0; // position in the sequence
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    cur_token = prompt_tokens[0];

    // initialize tensor wrappers
    auto tokens =
        from_blob(&cur_token, {1, 1}, executorch::aten::ScalarType::Long);

    // run the first token and get back logits tensor. Assuming the first token
    // is bos so don't callback.
    auto logits_tensor =
        ET_UNWRAP(text_decoder_runner_->step(tokens, start_pos));

    pos += 1; // start the loop from index 1
    start_pos += 1;

    while (pos < num_prompt_tokens) {
      // Run the model
      // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
      cur_token = prompt_tokens[pos];

      logits_tensor = ET_UNWRAP(text_decoder_runner_->step(tokens, start_pos));

      pos++;
      start_pos++;
    }

    cur_token = text_decoder_runner_->logits_to_token(logits_tensor);
  }
  return cur_token;
}

} // namespace llm
} // namespace extension
} // namespace executorch
