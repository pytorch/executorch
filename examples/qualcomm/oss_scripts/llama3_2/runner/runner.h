/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama3.2 runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <executorch/examples/qualcomm/oss_scripts/llama3_2/runner/io_memory.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/llm/tokenizer/tokenizer.h>
#include <executorch/extension/module/module.h>

namespace example {

class Runner {
 public:
  explicit Runner(
      const std::vector<std::string>& models_path,
      const std::string& tokenizer_path,
      const float temperature,
      const int eval_mode);

  struct Stats {
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
  };

  bool is_loaded() const;
  executorch::runtime::Error load();
  executorch::runtime::Error generate(
      const std::string& prompt,
      const std::string& system_prompt,
      int32_t seq_len,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {});
  void stop();
  std::vector<executorch::runtime::Result<executorch::runtime::MethodMeta>>
  get_methods_meta();

 private:
  enum EvalMode {
    kBatchPrefill = 0,
    kKVCached,
    kUnsupported,
  };
  template <typename T>
  T getMetadataHelper(std::string method_name, T default_val);
  template <typename T>
  int32_t logitsToToken(const executorch::aten::Tensor& logits_tensor);
  void run_model_step(
      std::vector<std::vector<executorch::runtime::EValue>>& inputs);
  // metadata
  int32_t max_seq_len_;
  int32_t vocab_size_;
  int32_t bos_id_;
  std::unordered_set<uint64_t> eos_id_;
  const int32_t n_bos_;
  const int32_t n_eos_;
  std::vector<std::shared_ptr<executorch::extension::Module>> modules_;
  std::string tokenizer_path_;
  float temperature_;
  std::unique_ptr<executorch::extension::llm::Tokenizer> tokenizer_;
  std::unique_ptr<executorch::extension::llm::Sampler> sampler_;
  Stats stats_;
  std::unique_ptr<Memory> io_mem_;
  int32_t eval_mode_;
};

} // namespace example
