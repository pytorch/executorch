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
#include <pytorch/tokenizers/sentencepiece.h>

#ifdef CUDA_AVAILABLE
#include <executorch/backends/cuda/runtime/memory_tracker.h>
#endif

namespace executorch::extension::llm {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

MultimodalRunner::MultimodalRunner(
    std::unordered_map<std::string, int64_t> metadata,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<Module> module,
    std::unique_ptr<MultimodalDecoderRunner> text_decoder_runner,
    std::unique_ptr<MultimodalPrefiller> multimodal_prefiller,
    std::unique_ptr<IOManager> io_manager,
    std::unique_ptr<TextTokenGenerator> text_token_generator,
    std::unique_ptr<Stats> stats)
    : metadata_(std::move(metadata)),
      tokenizer_(std::move(tokenizer)),
      module_(std::move(module)),
      text_decoder_runner_(std::move(text_decoder_runner)),
      multimodal_prefiller_(std::move(multimodal_prefiller)),
      io_manager_(std::move(io_manager)),
      text_token_generator_(std::move(text_token_generator)),
      stats_(std::move(stats)),
      pos_(0) {
#ifdef CUDA_AVAILABLE
  cuda_memory_tracker_ =
      std::make_unique<::executorch::backends::cuda::CudaMemoryTracker>();
  stats_->gpu_total_bytes = cuda_memory_tracker_->total_bytes();
  stats_->gpu_free_before_load_bytes = cuda_memory_tracker_->last_free_bytes();
#endif
}

bool MultimodalRunner::is_loaded() const {
  return multimodal_prefiller_->is_method_loaded() &&
      text_token_generator_->is_loaded();
}

Error MultimodalRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  stats_->model_load_start_ms = time_in_ms();
  ET_CHECK_OK_OR_RETURN_ERROR(multimodal_prefiller_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(text_token_generator_->load());
  stats_->model_load_end_ms = time_in_ms();

#ifdef CUDA_AVAILABLE
  cuda_memory_tracker_->log_sample("after_load");
  stats_->gpu_total_bytes = cuda_memory_tracker_->total_bytes();
  stats_->gpu_free_after_load_bytes = cuda_memory_tracker_->last_free_bytes();
  stats_->gpu_peak_usage_mb = cuda_memory_tracker_->peak_usage_mb();
#endif

  return Error::Ok;
}

// Don't print with the same priority during warmup
#define RUNNER_ET_LOG(warmup, format, ...) \
  if (warmup) {                            \
    ET_LOG(Debug, format, __VA_ARGS__);    \
  } else {                                 \
    ET_LOG(Info, format, __VA_ARGS__);     \
  }

Error MultimodalRunner::prefill(
    const std::vector<MultimodalInput>& inputs,
    const GenerationConfig& config) {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    const MultimodalInput& input = inputs[i];
    ET_LOG(
        Info,
        "Prefilling input %zu/%zu, type: %s",
        i,
        inputs.size(),
        input.type_name());

    // BOS/EOS handling: add to first text input, or prepend BOS token if
    // first input is non-text and pos_ == 0
    int32_t bos = 0;
    int32_t eos = 0;
    if (i == 0 && pos_ == 0) {
      if (input.is_text() || input.is_tokens()) {
        bos = config.num_bos;
        eos = config.num_eos;
      } else if (config.num_bos > 0) {
        // Non-text first input: prepend BOS token(s) via token embedding
        auto bos_it = metadata_.find(kBosId);
        if (bos_it != metadata_.end()) {
          std::vector<uint64_t> bos_tokens(config.num_bos, bos_it->second);
          MultimodalInput bos_input(std::move(bos_tokens));
          auto bos_result = multimodal_prefiller_->prefill(bos_input, pos_);
          if (!bos_result.ok()) {
            return bos_result.error();
          }
          prefill_next_token_ = bos_result.get();
        }
      }
    }

    auto prefill_result =
        multimodal_prefiller_->prefill(input, pos_, bos, eos);
    if (!prefill_result.ok()) {
      return prefill_result.error();
    }
    prefill_next_token_ = prefill_result.get();
  }
  return Error::Ok;
}

Error MultimodalRunner::generate(
    const std::string& prompt,
    const GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  if (!prompt.empty()) {
    // Wrap text prompt as multimodal input and delegate
    std::vector<MultimodalInput> inputs;
    inputs.emplace_back(MultimodalInput(prompt));
    return generate(inputs, config, token_callback, stats_callback);
  }

  // Empty prompt: run decode loop from current KV cache state
  ET_CHECK_OR_RETURN_ERROR(
      prefill_next_token_.has_value(),
      InvalidArgument,
      "Empty prompt requires a prior prefill() call");

  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

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

  stats_->inference_start_ms = time_in_ms();

  uint64_t cur_token = prefill_next_token_.value();
  prefill_next_token_.reset();

  stats_->first_token_ms = time_in_ms();
  stats_->prompt_eval_end_ms = time_in_ms();
  stats_->num_prompt_tokens = pos_;

  // Decode the first token
  auto decode_result = tokenizer_->decode(cur_token, cur_token);
  if (!decode_result.ok()) {
    ET_LOG(
        Error,
        "Tokenizers error code %d",
        static_cast<uint32_t>(decode_result.error()));
    return Error::InvalidArgument;
  }
  wrapped_callback(std::move(*decode_result));

  // Resolve max_new_tokens
  int64_t max_context_len = metadata_.at(kMaxContextLen);
  int32_t max_new_tokens = config.resolve_max_new_tokens(max_context_len, pos_);

  ET_CHECK_OR_RETURN_ERROR(
      max_new_tokens > 0,
      InvalidArgument,
      "Max new tokens %d is less than or equal to 0",
      max_new_tokens);

  text_token_generator_->set_ignore_eos(config.ignore_eos);

  std::vector<uint64_t> tokens = {cur_token};
  auto generate_result = text_token_generator_->generate(
      tokens, pos_, max_new_tokens - 1, config.temperature, wrapped_callback);
  if (!generate_result.ok()) {
    return generate_result.error();
  }
  int64_t num_generated_tokens = generate_result.get();

  pos_ += num_generated_tokens;
  stats_->num_generated_tokens = num_generated_tokens;
  stats_->inference_end_ms = time_in_ms();

#ifdef CUDA_AVAILABLE
  cuda_memory_tracker_->log_sample("after_generate");
  stats_->gpu_free_after_generate_bytes =
      cuda_memory_tracker_->last_free_bytes();
  stats_->gpu_peak_usage_mb = cuda_memory_tracker_->peak_usage_mb();
#endif

  if (!config.warming) {
    printf("\n");
    print_report(*stats_);
  } else {
    ET_LOG(Info, "Warmup run finished!");
  }

  if (stats_callback) {
    stats_callback(*stats_);
  }

  return Error::Ok;
}

Error MultimodalRunner::generate(
    const std::vector<MultimodalInput>& inputs,
    const GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  if (inputs.empty()) {
    ET_LOG(Error, "MultimodalInput vector cannot be empty");
    return Error::InvalidArgument;
  }

  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
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

  stats_->inference_start_ms = time_in_ms();

  // Echo last text input if requested
  if (config.echo) {
    for (auto it = inputs.rbegin(); it != inputs.rend(); ++it) {
      if (it->is_text()) {
        wrapped_callback(it->get_text());
        break;
      }
    }
  }

  // Prefill all inputs
  auto prefill_err = prefill(inputs, config);
  if (prefill_err != Error::Ok) {
    return prefill_err;
  }

  ET_CHECK_OR_RETURN_ERROR(
      prefill_next_token_.has_value(),
      Internal,
      "Prefill did not produce a next token");

  uint64_t cur_token = prefill_next_token_.value();
  prefill_next_token_.reset();

  stats_->first_token_ms = time_in_ms();
  stats_->prompt_eval_end_ms = time_in_ms();
  stats_->num_prompt_tokens = pos_;

  auto decode_result = tokenizer_->decode(cur_token, cur_token);
  if (!decode_result.ok()) {
    ET_LOG(
        Error,
        "Tokenizers error code %d",
        static_cast<uint32_t>(decode_result.error()));
    return Error::InvalidArgument;
  }
  wrapped_callback(std::move(*decode_result));

  RUNNER_ET_LOG(
      config.warming,
      "RSS after multimodal input processing: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  // Resolve max_new_tokens
  int64_t max_context_len = metadata_.at(kMaxContextLen);
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

  text_token_generator_->set_ignore_eos(config.ignore_eos);

  std::vector<uint64_t> tokens = {cur_token};
  auto generate_result = text_token_generator_->generate(
      tokens,
      pos_,
      max_new_tokens - 1,
      config.temperature,
      wrapped_callback);
  if (!generate_result.ok()) {
    return generate_result.error();
  }
  int64_t num_generated_tokens = generate_result.get();

  pos_ += num_generated_tokens;
  stats_->num_generated_tokens = num_generated_tokens;
  stats_->inference_end_ms = time_in_ms();

#ifdef CUDA_AVAILABLE
  cuda_memory_tracker_->log_sample("after_generate");
  stats_->gpu_free_after_generate_bytes =
      cuda_memory_tracker_->last_free_bytes();
  stats_->gpu_peak_usage_mb = cuda_memory_tracker_->peak_usage_mb();
#endif

  if (!config.warming) {
    printf("\n");
  }

  if (config.warming) {
    ET_LOG(Info, "Warmup run finished!");
  } else {
    print_report(*stats_);
  }

  if (stats_callback) {
    stats_callback(*stats_);
  }

  return Error::Ok;
}

} // namespace executorch::extension::llm
