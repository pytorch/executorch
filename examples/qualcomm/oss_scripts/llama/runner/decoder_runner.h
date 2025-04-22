/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

namespace example {
class DecoderRunner {
 public:
  DecoderRunner(
      executorch::extension::Module* module,
      int32_t vocab_size,
      float temperature);
  /**
   * Run LLM text decoder with inputs to generate next token.
   * @param inputs The inputs to the LLM Module.
   * @return The output of the LLM Module. This will be a tensor of logits.
   */
  executorch::runtime::Result<executorch::aten::Tensor> step(
      const std::string& method_name,
      std::vector<executorch::runtime::EValue>& inputs);

  /**
   * Once KV Cache output data pointer change, need to set
   * the output for specify method name in the module.
   * @return The error code.
   */
  executorch::runtime::Error set_outputs(
      const std::string& method_name,
      std::vector<executorch::aten::Tensor> output_values);

  /**
   * Load the Module for text decode purpose.
   * @return The error code.
   */
  executorch::runtime::Error load(const std::vector<std::string>& method_names);
  /**
   * Check if the required methods in the Module is loaded.
   * @return True if the Module is loaded, false otherwise.
   */
  bool is_method_loaded(const std::vector<std::string>& method_names);

  /**
   * Sample the next token from the logits tensor.
   * @param logits_tensor The logits tensor.
   * @return The next token.
   */
  inline int32_t logits_to_token(
      const executorch::aten::Tensor& logits_tensor,
      int64_t pos) {
    auto* logits = logits_tensor.mutable_data_ptr<uint16_t>();
    auto num_tokens = logits_tensor.size(1);
    auto vocab_size = logits_tensor.size(2);
    static std::vector<float> logits_f(vocab_size);
    auto* logits_last = logits;
    // offset to the meaningful logit we want for prefill model.
    if (num_tokens > 1) {
      logits_last += pos * vocab_size;
    }
    // Discard dequantization (converting uint16_t to float) because the
    // relative order of elements remains the same without conversion
    for (int i = 0; i < vocab_size; i++) {
      logits_f[i] = logits_last[i];
    }
    return sampler_->sample(logits_f.data());
  }

 protected:
  executorch::extension::Module* module_;
  std::unique_ptr<executorch::extension::llm::Sampler> sampler_;
};
} // namespace example
