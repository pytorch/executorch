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
// patternlint-disable-next-line executorch-cpp-nostdinc
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
  long inference_start_ms;
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
  inline void on_sampling_begin() {
    aggregate_sampling_timer_start_timestamp = time_in_ms();
  }
  inline void on_sampling_end() {
    aggregate_sampling_time_ms +=
        time_in_ms() - aggregate_sampling_timer_start_timestamp;
    aggregate_sampling_timer_start_timestamp = 0;
  }

 private:
  long aggregate_sampling_timer_start_timestamp = 0;
};

static constexpr auto kTopp = 0.9f;

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
     << "," << "\"SCALING_FACTOR_UNITS_PER_SECOND\":"
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
}

} // namespace llm
} // namespace extension
} // namespace executorch

namespace executorch {
namespace llm {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::kTopp;
using ::executorch::extension::llm::print_report;
using ::executorch::extension::llm::Stats;
} // namespace llm
} // namespace executorch
