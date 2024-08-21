/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple LLaVA runner that includes preprocessing and post processing logic.
// The runner takes in a prompt string as well as a list of images as input and
// emits a string as output.

#include <executorch/examples/models/llava/runner/llava_image_prefiller.h>
#include <executorch/examples/models/llava/runner/llava_runner.h>
#include <executorch/examples/models/llava/runner/llava_text_decoder_runner.h>
#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>

#include <ctime>
#include <memory>
#include <sstream>
#include <vector>

namespace torch::executor {

bool LlavaRunner::is_loaded() {
  bool instantiated = tokenizer_ && text_decoder_runner_ && text_prefiller_ &&
      image_prefiller_ && text_token_generator_;
  if (!instantiated) {
    return false;
  }
  return text_decoder_runner_->is_method_loaded() &&
      image_prefiller_->is_method_loaded();
}

Error LlavaRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  stats_.model_load_start_ms = util::time_in_ms();

  // Load the tokenizer
  tokenizer_ = std::make_unique<BPETokenizer>();
  tokenizer_->load(tokenizer_path_);

  // Load the text decoder runner
  text_decoder_runner_ = std::make_unique<LlavaTextDecoderRunner>(
      module_.get(), tokenizer_->vocab_size(), temperature_);
  text_decoder_runner_->load();

  // Load the text prefiller
  text_prefiller_ = std::make_unique<TextPrefiller>(
      tokenizer_.get(),
      text_decoder_runner_.get(),
      /*use_kv_cache=*/true,
      /*enable_parallel_prefill=*/true);

  // Load the image prefiller
  image_prefiller_ = std::make_unique<LlavaImagePrefiller>(module_.get());
  image_prefiller_->load();

  // Load the text token generator
  text_token_generator_ = std::make_unique<TextTokenGenerator>(
      tokenizer_.get(),
      text_decoder_runner_.get(),
      /*use_kv_cache=*/true,
      std::make_unique<std::unordered_set<uint64_t>>(
          std::unordered_set<uint64_t>{tokenizer_->eos_tok()}),
      &stats_);

  stats_.model_load_end_ms = util::time_in_ms();
  return Error::Ok;
}

Error LlavaRunner::generate(
    std::vector<Image>& images,
    const std::string& prompt,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  // Wrap the token_callback with print function
  std::function<void(const std::string&)> wrapped_callback =
      [token_callback](const std::string& piece) {
        util::safe_printf(piece.c_str());
        fflush(stdout);
        if (token_callback) {
          token_callback(piece);
        }
      };

  int64_t pos = 0;

  // prefill preset prompt
  std::vector<uint64_t> preset_prompt_tokens =
      ET_UNWRAP(tokenizer_->encode(kPresetPrompt, /*bos=*/1, /*eos=*/0));
  size_t num_preset_tokens = preset_prompt_tokens.size();

  ET_UNWRAP(text_prefiller_->prefill(preset_prompt_tokens, pos));
  pos += num_preset_tokens;

  // prefill images
  for (auto& image : images) {
    auto logits = ET_UNWRAP(image_prefiller_->prefill(image, pos));
    pos += logits.size(1);
  }

  // prefill user prompt. No BOS because preset prompt already has it.
  std::vector<uint64_t> user_prompt_tokens =
      ET_UNWRAP(tokenizer_->encode(prompt, /*bos=*/0, /*eos=*/0));
  size_t num_user_tokens = user_prompt_tokens.size();

  uint64_t prefill_next_token = ET_UNWRAP(
      text_prefiller_->prefill(user_prompt_tokens, pos, wrapped_callback));
  pos += num_user_tokens;

  // Generate tokens
  int64_t num_generated_tokens = ET_UNWRAP(text_token_generator_->generate(
      {prefill_next_token}, pos, seq_len, wrapped_callback));

  // Bookkeeping
  stats_.num_prompt_tokens = num_preset_tokens + num_user_tokens;
  stats_.num_generated_tokens = num_generated_tokens;
  ::executorch::llm::print_report(stats_);
  if (stats_callback) {
    stats_callback(stats_);
  }

  return Error::Ok;
}

} // namespace torch::executor
