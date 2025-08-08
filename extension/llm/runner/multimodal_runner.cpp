/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Implementation of MultimodalRunner for multimodal input and text output LLMs

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/tiktoken.h>

namespace executorch {
namespace extension {
namespace llm {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace {
// Default preset prompt for multimodal models
const std::string kDefaultPresetPrompt =
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ";
} // namespace

MultimodalRunner::MultimodalRunner(
    std::unordered_map<std::string, int64_t> metadata,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<Module> module,
    std::unique_ptr<TextDecoderRunner> text_decoder_runner,
    std::unique_ptr<TextPrefiller> text_prefiller,
    std::unique_ptr<ImagePrefiller> image_prefiller,
    std::unique_ptr<TextTokenGenerator> text_token_generator,
    std::unique_ptr<Stats> stats)
    : metadata_(std::move(metadata)),
      tokenizer_(std::move(tokenizer)),
      module_(std::move(module)),
      text_decoder_runner_(std::move(text_decoder_runner)),
      text_prefiller_(std::move(text_prefiller)),
      image_prefiller_(std::move(image_prefiller)),
      text_token_generator_(std::move(text_token_generator)),
      stats_(std::move(stats)),
      pos_(0) {}

bool MultimodalRunner::is_loaded() {
  return text_prefiller_->is_loaded() && text_token_generator_->is_loaded();
}

Error MultimodalRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(text_prefiller_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(text_token_generator_->load());
  return Error::Ok;
}

// Don't print with the same priority during warmup
#define RUNNER_ET_LOG(warmup, format, ...) \
  if (warmup) {                            \
    ET_LOG(Debug, format, __VA_ARGS__);    \
  } else {                                 \
    ET_LOG(Info, format, __VA_ARGS__);     \
  }

Error MultimodalRunner::generate(
    const std::vector<MultimodalInput>& inputs,
    const GenerationConfig& config,
    std::function<void(const std::string&)>& token_callback,
    std::function<void(const Stats&)>& stats_callback) {
  if (inputs.empty()) {
    ET_LOG(Error, "MultimodalInput vector cannot be empty");
    return Error::InvalidArgument;
  }

  if (!is_loaded()) {
    stats_->model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_->model_load_end_ms = time_in_ms();
  }

  if (config.warming) {
    ET_LOG(Info, "Doing a warmup run...");
  }

  RUNNER_ET_LOG(
      config.warming,
      "RSS after loading model: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  // Wrap the token_callback with print function
  std::function<void(const std::string&)> wrapped_callback =
      [token_callback, config](const std::string& piece) {
        if (!config.warming) {
          safe_printf(piece.c_str());
          fflush(stdout);
        }
        if (token_callback) {
          token_callback(piece);
        }
      };

  // Reset internal state and start inference
  pos_ = 0;
  stats_->inference_start_ms = time_in_ms();

  uint64_t prefill_next_token = 0;
  // Process multimodal inputs in order
  for (const auto& input : inputs) {
    prefill_next_token = ET_UNWRAP(prefill_input(input, 0, 0));
  }

  stats_->first_token_ms = time_in_ms();
  stats_->prompt_eval_end_ms = time_in_ms();
  stats_->num_prompt_tokens = pos_;

  wrapped_callback(ET_UNWRAP_TOKENIZER(
      tokenizer_->decode(prefill_next_token, prefill_next_token)));

  RUNNER_ET_LOG(
      config.warming,
      "RSS after multimodal input processing: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  // Resolve max_new_tokens based on config
  int64_t max_context_len =
      metadata_.at(kMaxContextLen) - 0; // No start_pos offset
  int32_t max_new_tokens = config.resolve_max_new_tokens(max_context_len, pos_);

  ET_LOG(
      Info,
      "Max new tokens resolved: %d, pos_ %" PRId64 ", max_context_len %" PRId64,
      max_new_tokens,
      pos_,
      max_context_len);

  ET_CHECK_OR_RETURN_ERROR(
      max_new_tokens > 0,
      InvalidArgument,
      "Max new tokens %d is less than or equal to 0",
      max_new_tokens);

  // Generate tokens using the text token generator
  std::vector<uint64_t> prompt_tokens = {prefill_next_token};
  int64_t num_generated_tokens = ET_UNWRAP(text_token_generator_->generate(
      /*tokens=*/prompt_tokens,
      /*start_pos=*/pos_,
      /*max_new_tokens=*/max_new_tokens -
          1, // Subtract 1 because prefill already generated 1 token
      /*temperature=*/config.temperature,
      /*token_callback=*/wrapped_callback));

  pos_ += num_generated_tokens;
  // Update stats
  stats_->num_generated_tokens = num_generated_tokens;
  // Finalize stats and call callback
  stats_->inference_end_ms = time_in_ms();
  if (!config.warming) {
    printf("\n");
  }

  if (config.warming) {
    ET_LOG(Info, "Warmup run finished!");
  } else {
    // Do not print report during warmup
    print_report(*stats_);
  }

  if (stats_callback) {
    stats_callback(*stats_);
  }

  return Error::Ok;
}

Result<uint64_t> MultimodalRunner::prefill_input(
    const MultimodalInput& input,
    int8_t bos,
    int8_t eos) {
  if (input.is_text()) {
    const std::string& text = input.get_text();
    ::tokenizers::Result<std::vector<uint64_t>> encode_res =
        tokenizer_->encode(text, bos, eos);

    ET_CHECK_TK_OK_OR_RETURN_ERROR(
        encode_res.error(), "Failed to encode text %s", text.c_str());

    std::vector<uint64_t> tokens = encode_res.get();
    return text_prefiller_->prefill(tokens, pos_);
  } else if (input.is_image()) {
    // Create a mutable copy for the prefiller
    Image mutable_image = input.get_image();
    return image_prefiller_->prefill(mutable_image, pos_);
  } else {
    ET_LOG(Error, "Unsupported input type");
    return Error::InvalidArgument;
  }
}

// Helper functions moved to llm_runner_helper.cpp
// Include the helper header for the implementations
// #include <executorch/extension/llm/runner/llm_runner_helper.h>

} // namespace llm
} // namespace extension
} // namespace executorch
