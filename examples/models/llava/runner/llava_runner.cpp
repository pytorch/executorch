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

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace example {

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
  stats_.model_load_start_ms = llm::time_in_ms();

  // Load the tokenizer
  tokenizer_ = std::make_unique<llm::BPETokenizer>();
  tokenizer_->load(tokenizer_path_);

  // Load the text decoder runner
  text_decoder_runner_ = std::make_unique<LlavaTextDecoderRunner>(
      module_.get(), tokenizer_->vocab_size(), temperature_);
  text_decoder_runner_->load();

  // Load the text prefiller
  text_prefiller_ = std::make_unique<llm::TextPrefiller>(
      text_decoder_runner_.get(),
      /*use_kv_cache=*/true,
      /*enable_parallel_prefill=*/true);

  // Load the image prefiller
  image_prefiller_ = std::make_unique<LlavaImagePrefiller>(module_.get());
  image_prefiller_->load();

  // Load the text token generator
  text_token_generator_ = std::make_unique<llm::TextTokenGenerator>(
      tokenizer_.get(),
      text_decoder_runner_.get(),
      /*use_kv_cache=*/true,
      std::make_unique<std::unordered_set<uint64_t>>(
          std::unordered_set<uint64_t>{tokenizer_->eos_tok()}),
      &stats_);

  stats_.model_load_end_ms = llm::time_in_ms();
  return Error::Ok;
}

Error LlavaRunner::prefill_images(
    std::vector<llm::Image>& images,
    int64_t& start_pos) {
  for (auto& image : images) {
    // pos is updated inside image prefill.
    ET_UNWRAP(image_prefiller_->prefill(image, start_pos));
  }
  return Error::Ok;
}

Result<uint64_t> LlavaRunner::prefill_prompt(
    const std::string& prompt,
    int64_t& start_pos,
    int8_t bos,
    int8_t eos) {
  std::vector<uint64_t> prompt_tokens =
      ET_UNWRAP(tokenizer_->encode(prompt, bos, eos));

  return text_prefiller_->prefill(prompt_tokens, start_pos);
}

Error LlavaRunner::generate_from_pos(
    const std::string& prompt,
    int32_t seq_len,
    int64_t start_pos,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const ::executorch::extension::llm::Stats&)>
        stats_callback,
    bool echo) {
  // prefill user prompt. No BOS because preset prompt already has it.
  if (echo) {
    token_callback(prompt);
  }

  uint64_t prefill_next_token =
      ET_UNWRAP(prefill_prompt(prompt, start_pos, /*bos=*/0, /*eos*/ 0));
  stats_.first_token_ms = llm::time_in_ms();
  stats_.prompt_eval_end_ms = llm::time_in_ms();
  stats_.num_prompt_tokens = start_pos;

  // Generate tokens
  int64_t num_generated_tokens = ET_UNWRAP(text_token_generator_->generate(
      {prefill_next_token}, start_pos, seq_len, token_callback));

  // Bookkeeping
  stats_.num_generated_tokens = num_generated_tokens;
  if (stats_callback) {
    stats_callback(stats_);
  }
  return Error::Ok;
}

Error LlavaRunner::generate(
    std::vector<llm::Image> images,
    const std::string& prompt,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const llm::Stats&)> stats_callback,
    bool echo) {
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  ET_LOG(
      Info,
      "RSS after loading model: %f MiB (0 if unsupported)",
      llm::get_rss_bytes() / 1024.0 / 1024.0);

  // Wrap the token_callback with print function
  std::function<void(const std::string&)> wrapped_callback =
      [token_callback](const std::string& piece) {
        llm::safe_printf(piece.c_str());
        fflush(stdout);
        if (token_callback) {
          token_callback(piece);
        }
      };

  int64_t pos = 0;
  stats_.inference_start_ms = llm::time_in_ms();

  // prefill preset prompt
  prefill_prompt(kPresetPrompt, pos, /*bos=*/1, /*eos*/ 0);

  // prefill images
  prefill_images(images, pos);

  ET_LOG(
      Info,
      "RSS after prompt and image prefill: %f MiB (0 if unsupported)",
      llm::get_rss_bytes() / 1024.0 / 1024.0);

  // Generate tokens
  Error err = generate_from_pos(
      prompt, seq_len, pos, wrapped_callback, stats_callback, echo);

  stats_.inference_end_ms = llm::time_in_ms();
  ::executorch::llm::print_report(stats_);

  ET_LOG(
      Info,
      "RSS after finishing text generation: %f MiB (0 if unsupported)",
      llm::get_rss_bytes() / 1024.0 / 1024.0);

  return err;
}

} // namespace example
