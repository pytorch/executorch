/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given a text prompt, encode it using tokenizer and prefill the KV cache of a
// LLM.

#include <executorch/examples/samsung/executor_runner/llm/runner/prefiller.h>
#include <algorithm>
#include <cinttypes>

namespace example {

namespace llm = ::executorch::extension::llm;
namespace runtime = ::executorch::runtime;
namespace aten = ::executorch::aten;
namespace extension = ::executorch::extension;

Prefiller::Prefiller(
    llm::TextDecoderRunner* text_decoder_runner,
    bool use_kv_cache,
    bool enable_parallel_prefill,
    int64_t max_seq_len,
    int64_t prefill_cache_len)
    : TextPrefiller(
          text_decoder_runner,
          use_kv_cache,
          enable_parallel_prefill,
          max_seq_len),
      text_decoder_runner_(text_decoder_runner),
      use_kv_cache_(use_kv_cache),
      enable_parallel_prefill_(enable_parallel_prefill),
      max_seq_len_(max_seq_len > 0 ? max_seq_len : 128),
      prefill_cache_len_(prefill_cache_len) {}

runtime::Result<uint64_t> Prefiller::prefill(
    std::vector<uint64_t>& prompt_tokens,
    int64_t& start_pos,
    float temperature) {
  ET_CHECK_OR_RETURN_ERROR(
      !prompt_tokens.empty(), InvalidArgument, "Prompt cannot be empty");
  if (!text_decoder_runner_->is_method_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(text_decoder_runner_->load());
  }

  const int64_t num_prompt_tokens = prompt_tokens.size();
  ET_LOG(
      Info,
      "Micro Batch: num_prompt(%" PRId64 "), prefill::max_seq_len(%" PRId64
      "), start_pos(%" PRId64 ")",
      num_prompt_tokens,
      max_seq_len_,
      start_pos);

  uint64_t cur_token = 0;
  int64_t num_processed = 0;
  while (num_processed < num_prompt_tokens) {
    int64_t chunk_len =
        std::min(num_prompt_tokens - num_processed, max_seq_len_);
    if (enable_parallel_prefill_) {
      // Once the prefill cache is full, feed the rest one token at a time,
      // which the decoder runs through the decode method.
      chunk_len = std::max<int64_t>(
          1, std::min(chunk_len, prefill_cache_len_ - start_pos));
    }
    std::vector<uint64_t> chunk(
        prompt_tokens.begin() + num_processed,
        prompt_tokens.begin() + num_processed + chunk_len);
    num_processed += chunk_len;
    // Only the last chunk's prediction is used, so only it is sampled.
    cur_token = ET_UNWRAP(prefill_chunk(
        chunk,
        start_pos,
        num_processed == num_prompt_tokens ? temperature : 0.0f));
  }
  return cur_token;
}

runtime::Result<uint64_t> Prefiller::prefill_chunk(
    std::vector<uint64_t>& prompt_tokens,
    int64_t& start_pos,
    float temperature) {
  // enable_parallel_prefill_ maybe set even when not using kv cache
  // When kv cache is not used, start pos is ignored
  int32_t num_prompt_tokens = prompt_tokens.size();

  // store the token
  uint64_t cur_token;
  if (enable_parallel_prefill_ || !use_kv_cache_) {
    // initialize tensor wrappers
    auto tokens = extension::from_blob(
        prompt_tokens.data(),
        {1, static_cast<int>(prompt_tokens.size())},
        aten::ScalarType::Long);

    auto outputs_res = text_decoder_runner_->step(tokens, start_pos);

    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_LOG(
        Info, "Prefill token result numel(): %zu", outputs_res.get().numel());

    start_pos += num_prompt_tokens;
    cur_token =
        text_decoder_runner_->logits_to_token(outputs_res.get(), temperature);
  } else { // sequential prefill
    int64_t pos = 0; // position in the sequence
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    cur_token = prompt_tokens[0];

    ET_LOG(
        Info,
        "num_prompt_token(%d), cur_token(%ld), start_pos(%ld)",
        num_prompt_tokens,
        cur_token,
        start_pos);

    auto tokens =
        extension::from_blob(&cur_token, {1, 1}, aten::ScalarType::Long);

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

    cur_token =
        text_decoder_runner_->logits_to_token(logits_tensor, temperature);
  }
  return cur_token;
}

} // namespace example
