/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Runner stats for LLM
#pragma once
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>
#include <cinttypes>
#include <sstream>
#include <string>

namespace executorch {
namespace extension {
namespace llm {

struct ET_EXPERIMENTAL Stats {
  // Scaling factor for timestamps - in this case, we use ms.
  const long SCALING_FACTOR_UNITS_PER_SECOND = 1000;
  // Time stamps for the different stages of the execution
  // model_load_start_ms: Start of model loading.
  long model_load_start_ms;
  // model_load_end_ms: End of model loading.
  long model_load_end_ms;
  // inference_start_ms: Immediately after the model is loaded (or we check
  // for model load), measure the inference time.
  // NOTE: It's actually the tokenizer encode + model execution time.
  long inference_start_ms;
  // End of the tokenizer encode time.
  long token_encode_end_ms;
  // Start of the model execution (forward function) time.
  long model_execution_start_ms;
  // End of the model execution (forward function) time.
  long model_execution_end_ms;
  // prompt_eval_end_ms: Prompt array allocation and tokenization. Ends right
  // before the inference loop starts
  long prompt_eval_end_ms;
  // first_token: Timestamp when the first generated token is emitted
  long first_token_ms;
  // inference_end_ms: End of inference/generation.
  long inference_end_ms;
  // Keep a running total of the time spent in sampling.
  long aggregate_sampling_time_ms;
  // Token count from prompt
  int64_t num_prompt_tokens;
  // Token count from generated (total - prompt)
  int64_t num_generated_tokens;
  // GPU memory stats (optional; may be zero if not available)
  // GPU memory stats (optional). Use sentinel UINT64_MAX / -1.0 to indicate
  // "not available".
  uint64_t gpu_total_bytes = static_cast<uint64_t>(-1);
  uint64_t gpu_free_before_load_bytes = static_cast<uint64_t>(-1);
  uint64_t gpu_free_after_load_bytes = static_cast<uint64_t>(-1);
  uint64_t gpu_free_after_generate_bytes = static_cast<uint64_t>(-1);
  double gpu_peak_usage_mb = -1.0;
  inline void on_sampling_begin() {
    aggregate_sampling_timer_start_timestamp = time_in_ms();
  }
  inline void on_sampling_end() {
    aggregate_sampling_time_ms +=
        time_in_ms() - aggregate_sampling_timer_start_timestamp;
    aggregate_sampling_timer_start_timestamp = 0;
  }

  void reset(bool all_stats = false) {
    // Not resetting model_load_start_ms and model_load_end_ms because reset is
    // typically called after warmup and before running the actual run.
    // However, we don't load the model again during the actual run after
    // warmup. So, we don't want to reset these timestamps unless we are
    // resetting everything.
    if (all_stats) {
      model_load_start_ms = 0;
      model_load_end_ms = 0;
    }
    inference_start_ms = 0;
    prompt_eval_end_ms = 0;
    first_token_ms = 0;
    inference_end_ms = 0;
    aggregate_sampling_time_ms = 0;
    num_prompt_tokens = 0;
    num_generated_tokens = 0;
    gpu_total_bytes = static_cast<uint64_t>(-1);
    gpu_free_before_load_bytes = static_cast<uint64_t>(-1);
    gpu_free_after_load_bytes = static_cast<uint64_t>(-1);
    gpu_free_after_generate_bytes = static_cast<uint64_t>(-1);
    gpu_peak_usage_mb = -1.0;
    aggregate_sampling_timer_start_timestamp = 0;
  }

 private:
  long aggregate_sampling_timer_start_timestamp = 0;
};

inline std::string stats_to_json_string(const Stats& stats) {
  std::stringstream ss;
  ss << "{\"prompt_tokens\":" << stats.num_prompt_tokens << ","
     << "\"generated_tokens\":" << stats.num_generated_tokens << ","
     << "\"model_load_start_ms\":" << stats.model_load_start_ms << ","
     << "\"model_load_end_ms\":" << stats.model_load_end_ms << ","
     << "\"inference_start_ms\":" << stats.inference_start_ms << ","
     << "\"inference_end_ms\":" << stats.inference_end_ms << ","
     << "\"prompt_eval_end_ms\":" << stats.prompt_eval_end_ms << ","
     << "\"first_token_ms\":" << stats.first_token_ms << ","
     << "\"aggregate_sampling_time_ms\":" << stats.aggregate_sampling_time_ms
     << ",";
  // Only include GPU fields in the JSON if gpu_total_bytes is valid (not
  // equal to sentinel -1)
  if (stats.gpu_total_bytes != static_cast<uint64_t>(-1)) {
    ss << "\"gpu_total_bytes\":" << stats.gpu_total_bytes;
    if (stats.gpu_free_before_load_bytes != static_cast<uint64_t>(-1)) {
      ss << ",\"gpu_free_before_load_bytes\":"
         << stats.gpu_free_before_load_bytes;
    }
    if (stats.gpu_free_after_load_bytes != static_cast<uint64_t>(-1)) {
      ss << ",\"gpu_free_after_load_bytes\":"
         << stats.gpu_free_after_load_bytes;
    }
    if (stats.gpu_free_after_generate_bytes != static_cast<uint64_t>(-1)) {
      ss << ",\"gpu_free_after_generate_bytes\":"
         << stats.gpu_free_after_generate_bytes;
    }
    if (stats.gpu_peak_usage_mb >= 0.0) {
      ss << ",\"gpu_peak_usage_mb\":" << stats.gpu_peak_usage_mb;
    }
    ss << ",";
  }
  ss << "\"SCALING_FACTOR_UNITS_PER_SECOND\":"
     << stats.SCALING_FACTOR_UNITS_PER_SECOND << "}";
  return ss.str();
}

inline void print_report(const Stats& stats) {
  printf("PyTorchObserver %s\n", stats_to_json_string(stats).c_str());

  ET_LOG(
      Info,
      "\tPrompt Tokens: %" PRIu64 "    Generated Tokens: %" PRIu64,
      stats.num_prompt_tokens,
      stats.num_generated_tokens);

  ET_LOG(
      Info,
      "\tModel Load Time:\t\t%f (seconds)",
      ((double)(stats.model_load_end_ms - stats.model_load_start_ms) /
       stats.SCALING_FACTOR_UNITS_PER_SECOND));
  double inference_time_ms =
      (double)(stats.inference_end_ms - stats.inference_start_ms);
  ET_LOG(
      Info,
      "\tTotal inference time:\t\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      inference_time_ms / stats.SCALING_FACTOR_UNITS_PER_SECOND,

      (stats.num_generated_tokens) /
          (double)(stats.inference_end_ms - stats.inference_start_ms) *
          stats.SCALING_FACTOR_UNITS_PER_SECOND);
  double prompt_eval_time =
      (double)(stats.prompt_eval_end_ms - stats.inference_start_ms);
  ET_LOG(
      Info,
      "\t\tPrompt evaluation:\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      prompt_eval_time / stats.SCALING_FACTOR_UNITS_PER_SECOND,
      (stats.num_prompt_tokens) / prompt_eval_time *
          stats.SCALING_FACTOR_UNITS_PER_SECOND);

  double eval_time =
      (double)(stats.inference_end_ms - stats.prompt_eval_end_ms);
  ET_LOG(
      Info,
      "\t\tGenerated %" PRIu64
      " tokens:\t%f (seconds)\t\t Rate: \t%f (tokens/second)",
      stats.num_generated_tokens,
      eval_time / stats.SCALING_FACTOR_UNITS_PER_SECOND,
      stats.num_generated_tokens / eval_time *
          stats.SCALING_FACTOR_UNITS_PER_SECOND);

  // Time to first token is measured from the start of inference, excluding
  // model load time.
  ET_LOG(
      Info,
      "\tTime to first generated token:\t%f (seconds)",
      ((double)(stats.first_token_ms - stats.inference_start_ms) /
       stats.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tSampling time over %" PRIu64 " tokens:\t%f (seconds)",
      stats.num_prompt_tokens + stats.num_generated_tokens,
      (double)stats.aggregate_sampling_time_ms /
          stats.SCALING_FACTOR_UNITS_PER_SECOND);

  // GPU memory reporting (only meaningful if GPU fields were populated)
  if (stats.gpu_total_bytes != static_cast<uint64_t>(-1)) {
    ET_LOG(
        Info,
        "\tGPU total memory: %.2f MB",
        stats.gpu_total_bytes / 1024.0 / 1024.0);
    if (stats.gpu_free_before_load_bytes != static_cast<uint64_t>(-1)) {
      ET_LOG(
          Info,
          "\tGPU free before load: %.2f MB",
          stats.gpu_free_before_load_bytes / 1024.0 / 1024.0);
    }
    if (stats.gpu_free_after_load_bytes != static_cast<uint64_t>(-1)) {
      ET_LOG(
          Info,
          "\tGPU free after load: %.2f MB",
          stats.gpu_free_after_load_bytes / 1024.0 / 1024.0);
    }
    if (stats.gpu_free_after_generate_bytes != static_cast<uint64_t>(-1)) {
      ET_LOG(
          Info,
          "\tGPU free after generate: %.2f MB",
          stats.gpu_free_after_generate_bytes / 1024.0 / 1024.0);
    }
    if (stats.gpu_peak_usage_mb >= 0.0) {
      ET_LOG(Info, "\tGPU peak usage: %.2f MB", stats.gpu_peak_usage_mb);
    }
  }
}

} // namespace llm
} // namespace extension
} // namespace executorch

namespace executorch {
namespace llm {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::print_report;
using ::executorch::extension::llm::Stats;
} // namespace llm
} // namespace executorch
