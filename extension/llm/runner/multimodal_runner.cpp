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
  // Probe immediately after creating the tracker to capture GPU state before
  // any model loading happens.
  stats_->gpu_total_bytes = cuda_memory_tracker_->total_bytes();
  stats_->gpu_free_before_load_bytes = cuda_memory_tracker_->last_free_bytes();
#endif

  // Initialize ring buffer configuration from metadata
  if (metadata_.count(kIsRingBuffer) && metadata_.at(kIsRingBuffer) > 0) {
    is_ring_buffer_ = true;
    // Sliding window size defaults to max_context_len if not specified
    if (metadata_.count(kSlidingWindowSize)) {
      sliding_window_size_ = metadata_.at(kSlidingWindowSize);
    } else if (metadata_.count(kMaxContextLen)) {
      sliding_window_size_ = metadata_.at(kMaxContextLen);
    }

    // Validate sliding_window_size
    if (sliding_window_size_ <= 0) {
      ET_LOG(
          Error,
          "Ring buffer enabled but sliding_window_size is %" PRId64
          ". Disabling ring buffer mode.",
          sliding_window_size_);
      is_ring_buffer_ = false;
    } else {
      ET_LOG(
          Info,
          "Ring buffer KV cache enabled with sliding window size: %" PRId64,
          sliding_window_size_);
    }
  }
}

bool MultimodalRunner::is_loaded() {
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

Error MultimodalRunner::prefill(const std::vector<MultimodalInput>& inputs) {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }
  for (auto& input : inputs) {
    auto prefill_result = multimodal_prefiller_->prefill(input, pos_);
    if (!prefill_result.ok()) {
      return prefill_result.error();
    }
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

  // Reset internal state and start inference
  stats_->inference_start_ms = time_in_ms();

  uint64_t prefill_next_token = 0;
  // Process multimodal inputs in order
  for (size_t i = 0; i < inputs.size(); ++i) {
    const MultimodalInput& input = inputs[i];
    ET_LOG(
        Info,
        "Prefilling input %zu/%zu, type: %s",
        i,
        inputs.size(),
        input.type_name());
    if (config.echo && i == inputs.size() - 1 && input.is_text()) {
      wrapped_callback(input.get_text());
    }
    auto prefill_result = multimodal_prefiller_->prefill(input, pos_);
    if (!prefill_result.ok()) {
      return prefill_result.error();
    }
    prefill_next_token = prefill_result.get();
  }

  stats_->first_token_ms = time_in_ms();
  stats_->prompt_eval_end_ms = time_in_ms();
  stats_->num_prompt_tokens = pos_;

  auto decode_result =
      tokenizer_->decode(prefill_next_token, prefill_next_token);
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

  // Resolve max_new_tokens based on config and ring buffer mode
  int64_t max_context_len = metadata_.at(kMaxContextLen);
  int64_t effective_context_len;
  int32_t max_new_tokens;

  if (is_ring_buffer_) {
    // Ring buffer mode: positions wrap around, allowing continuous generation.
    // The model's KV cache uses a ring buffer that overwrites old entries,
    // so we can generate beyond the initial context length.
    effective_context_len = sliding_window_size_;

    // In ring buffer mode, we're not limited by pos_ since positions wrap
    // around. Use the effective position within the sliding window.
    int64_t effective_pos = pos_ % sliding_window_size_;
    max_new_tokens =
        config.resolve_max_new_tokens(effective_context_len, effective_pos);

    // Log ring buffer status
    if (pos_ >= sliding_window_size_) {
      ET_LOG(
          Info,
          "Ring buffer active: logical pos %" PRId64
          " >= window size %" PRId64 ", positions will wrap",
          pos_,
          sliding_window_size_);
    }
  } else {
    // Non-ring buffer mode: original behavior with hard context limit.
    effective_context_len = max_context_len - pos_;

    ET_CHECK_OR_RETURN_ERROR(
        pos_ < max_context_len,
        InvalidArgument,
        "pos_ %" PRId64 " >= max_context_len %" PRId64
        ", context exhausted - please increase max context len or enable ring "
        "buffer KV cache",
        pos_,
        max_context_len);

    max_new_tokens =
        config.resolve_max_new_tokens(effective_context_len, 0);
  }

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

  // Set ignore_eos based on config
  text_token_generator_->set_ignore_eos(config.ignore_eos);

  // Generate tokens using the text token generator
  std::vector<uint64_t> prompt_tokens = {prefill_next_token};
  auto generate_result = text_token_generator_->generate(
      /*tokens=*/prompt_tokens,
      /*start_pos=*/pos_,
      /*max_new_tokens=*/max_new_tokens -
          1, // Subtract 1 because prefill already generated 1 token
      /*temperature=*/config.temperature,
      /*token_callback=*/wrapped_callback);
  if (!generate_result.ok()) {
    return generate_result.error();
  }
  int64_t num_generated_tokens = generate_result.get();

  pos_ += num_generated_tokens;
  // Update stats
  stats_->num_generated_tokens = num_generated_tokens;
  // Finalize stats and call callback
  stats_->inference_end_ms = time_in_ms();

#ifdef CUDA_AVAILABLE
  cuda_memory_tracker_->log_sample("after_generate");
  stats_->gpu_free_after_generate_bytes =
      cuda_memory_tracker_->last_free_bytes();
  // update peak in case it changed after generation
  stats_->gpu_peak_usage_mb = cuda_memory_tracker_->peak_usage_mb();
#endif

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

} // namespace executorch::extension::llm
