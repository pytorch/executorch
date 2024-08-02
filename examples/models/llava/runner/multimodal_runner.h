/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple multimodal LLM runner that includes preprocessing and post
// processing logic. The module takes in an image as well a string as input and
// emits a string as output.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/llm/tokenizer/tokenizer.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/managed_tensor.h>

namespace torch::executor {

struct Image {
  // Assuming NCHW format
  std::vector<uint8_t> data;
  int32_t width;
  int32_t height;
  int32_t channels;
};

class MultiModalRunner {
 public:
  explicit MultiModalRunner(
      const std::string& model_path,
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

  Result<torch::executor::Tensor> prefill_image(
      Image& image,
      int64_t start_pos);

  Result<torch::executor::Tensor> prefill_prompt(
      const std::string& prompt,
      int64_t start_pos,
      bool add_bos = false,
      std::function<void(const std::string&)> token_callback = {});

  Error generate(
      std::vector<Image> images,
      const std::string& prompt,
      int32_t seq_len = 1024,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {});
  void stop();

 private:
  inline static const std::string kPresetPrompt =
      "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ";
  // metadata
  int32_t logits_to_token(const torch::executor::Tensor& logits_tensor);

  Result<torch::executor::Tensor> step(
      ManagedTensor& tokens,
      ManagedTensor& start_pos);

  // metadata
  int32_t vocab_size_;
  int32_t bos_id_;
  int32_t eos_id_;
  int32_t n_bos_;
  int32_t n_eos_;
  int32_t max_seq_len_;
  bool append_eos_;
  std::unordered_set<std::string> model_methods_;
  std::string model_path_;
  std::unique_ptr<Module> module_;
  std::string tokenizer_path_;
  float temperature_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Sampler> sampler_;
  bool shouldStop_{false};
  MultiModalRunner::Stats stats_;
};

} // namespace torch::executor
