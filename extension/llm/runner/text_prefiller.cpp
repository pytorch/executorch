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

namespace torch::executor {

TextPrefiller::TextPrefiller(
    Tokenizer* tokenizer,
    TextDecoderRunner* text_decoder_runner,
    bool use_kv_cache,
    bool enable_parallel_prefill)
    : tokenizer_(tokenizer),
      text_decoder_runner_(text_decoder_runner),
      use_kv_cache_(use_kv_cache),
      enable_parallel_prefill_(enable_parallel_prefill) {}

Result<uint64_t> TextPrefiller::prefill(
    std::vector<uint64_t>& prompt_tokens,
    int64_t start_pos,
    std::function<void(const std::string&)> token_callback) {
  ET_CHECK_MSG(!prompt_tokens.empty(), "Prompt cannot be null");
  if (!text_decoder_runner_->is_method_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(text_decoder_runner_->load());
  }
  // enable_parallel_prefill_ maybe set even when not using kv cache
  // When kv cache is not used, start pos is ignored
  int32_t num_prompt_tokens = prompt_tokens.size();

  // store the token
  uint64_t cur_token;
  if (enable_parallel_prefill_ || !use_kv_cache_) {
    // initialize tensor wrappers
    ManagedTensor managed_tokens(
        prompt_tokens.data(), {1, num_prompt_tokens}, ScalarType::Long);

    ManagedTensor managed_start_pos(&start_pos, {1}, ScalarType::Long);

    Result<exec_aten::Tensor> outputs_res =
        text_decoder_runner_->step(managed_tokens, managed_start_pos);

    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_LOG(
        Info, "Prefill token result numel(): %zu", outputs_res.get().numel());
    ET_CHECK_MSG(
        outputs_res.get().size(1) == num_prompt_tokens,
        "Expected number of output tokens %d does not match returned value %zu.",
        num_prompt_tokens,
        outputs_res.get().size(1));
    // insert new token into prompt_tokens
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    uint64_t prev = prompt_tokens[0];
    uint64_t cur;
    for (int i = 0; i < prompt_tokens.size(); i++) {
      cur = prompt_tokens[i];
      if (cur != tokenizer_->bos_tok()) {
        token_callback(ET_UNWRAP(tokenizer_->decode(prev, cur)));
      }
      prev = cur;
    }
    cur_token = text_decoder_runner_->logits_to_token(outputs_res.get());
  } else { // sequential prefill
    int64_t pos = 0; // position in the sequence
    int64_t prev_token;
    // token & pos
    int64_t pos_data = 0;
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    cur_token = prompt_tokens[0];

    // initialize tensor wrappers
    ManagedTensor managed_tokens(&cur_token, {1, 1}, ScalarType::Long);

    ManagedTensor managed_start_pos(&pos_data, {1}, ScalarType::Long);

    // run the first token and get back logits tensor. Assuming the first token
    // is bos so don't callback.
    exec_aten::Tensor logits_tensor = ET_UNWRAP(
        text_decoder_runner_->step(managed_tokens, managed_start_pos));

    // if first token is not bos, we need to callback
    if (cur_token != tokenizer_->bos_tok()) {
      token_callback(ET_UNWRAP(tokenizer_->decode(cur_token, cur_token)));
    }
    pos = 1; // start from index 1

    while (pos < num_prompt_tokens) {
      // Run the model
      pos_data = start_pos + pos;

      prev_token = cur_token;

      // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
      cur_token = prompt_tokens[pos];

      logits_tensor = ET_UNWRAP(
          text_decoder_runner_->step(managed_tokens, managed_start_pos));

      // print the token as string, decode it with the Tokenizer object
      token_callback(ET_UNWRAP(tokenizer_->decode(prev_token, cur_token)));

      pos++;
    }

    cur_token = text_decoder_runner_->logits_to_token(logits_tensor);
  }
  return cur_token;
}

} // namespace torch::executor
