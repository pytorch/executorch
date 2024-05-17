/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <executorch/examples/models/llama2/sampler/sampler.h>
#include <executorch/examples/models/llama2/tokenizer/tokenizer.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/managed_tensor.h>

namespace torch {
namespace executor {

class Runner {
 public:
  explicit Runner(
      const std::vector<std::string>& model_path_list,
      const std::string& tokenizer_path,
      const float temperature = 0.8f);

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
  Error load();
  Error generate(
      const std::string& prompt,
      int32_t seq_len,
      std::vector<std::vector<ManagedTensor>>& managed_kv_inputs,
      std::vector<std::vector<float>>& freqs_inputs,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {});
  void stop();
  std::vector<Result<MethodMeta>> get_methods_meta();

 private:
  // metadata
  template <typename T>
  T getMetadataHelper(Module*, std::string method_name, T default_val);
  template <typename T>
  int32_t logitsToToken(const exec_aten::Tensor& logits_tensor);
  Result<torch::executor::Tensor> run_model_step(
      int64_t input_token,
    Tensor& token,
    Tensor& start_pos,
    Tensor& atten_mask,
    Tensor& freqs_cos,
    Tensor& freqs_sin,
    std::vector<std::vector<Tensor>>& kv_tensors,
    std::vector<std::vector<Tensor>>& kv_outputs);
  // metadata
  int32_t vocab_size_;
  int64_t bos_id_;
  int64_t eos_id_;
  int32_t n_bos_;
  int32_t n_eos_;
  int32_t max_seq_len_;
  int32_t head_dim_;
  std::unordered_set<std::string> model_methods_;
  std::vector<std::unique_ptr<Module>> modules_;
  std::string tokenizer_path_;
  float temperature_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Sampler> sampler_;
  bool shouldStop_{false};
  Stats stats_;
};

} // namespace executor
} // namespace torch
