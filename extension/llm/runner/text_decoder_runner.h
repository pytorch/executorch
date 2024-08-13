/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder in LLM and return the output.

#pragma once

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/managed_tensor.h>
// patternlint-disable-next-line executorch-cpp-nostdinc
#include <functional>

namespace torch::executor {

class TextDecoderRunner {
 public:
  TextDecoderRunner(
      Module* module,
      bool use_kv_cache,
      int32_t vocab_size,
      float temperature);
  /**
   * Run LLM text decoder with inputs to generate next token.
   * @param input The input to the LLM Module.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * Module.
   * @return The output of the LLM Module. This will be a tensor of logits.
   */
  Result<exec_aten::Tensor> step(
      ManagedTensor& input,
      ManagedTensor& start_pos);

  /**
   * Load the Module for a given method name.
   * @param method_name The name of the method to load.
   * @return The error code.
   */
  inline Error load(const std::string& method_name = "forward") {
    return module_->load_method(method_name);
  }

  /**
   * Check if the Module is loaded.
   * @return True if the Module is loaded, false otherwise.
   */
  inline bool is_method_loaded(const std::string& method_name = "forward") {
    return module_->is_method_loaded(method_name);
  }

  inline void stop() {
    should_stop_ = true;
  }

  /**
   * Sample the next token from the logits tensor.
   * @param logits_tensor The logits tensor.
   * @return The next token.
   */
  inline int32_t logits_to_token(const exec_aten::Tensor& logits_tensor) {
    ET_CHECK_MSG(logits_tensor.dim() == 3, "Logits tensor must be 3D");
    auto num_tokens = logits_tensor.size(1);
    auto vocab_size = logits_tensor.size(2);

    switch (logits_tensor.scalar_type()) {
      case ScalarType::Float: {
        float* logits = logits_tensor.mutable_data_ptr<float>();
        float* logits_last = logits;
        logits_last += (num_tokens - 1) * vocab_size;
        return sampler_->sample(logits_last);
      }
      case ScalarType::Half: {
        exec_aten::Half* logits =
            logits_tensor.mutable_data_ptr<exec_aten::Half>();
        exec_aten::Half* logits_last = logits;
        logits_last += (num_tokens - 1) * vocab_size;
        return sampler_->sample(logits_last);
      }
      default:
        ET_CHECK_MSG(
            false,
            "Unsupported dtype output %hhd",
            static_cast<int8_t>(logits_tensor.scalar_type()));
    }
  }

 protected:
  // TODO: use shared_ptr for module
  Module* module_;
  std::unique_ptr<Sampler> sampler_;
  bool use_kv_cache_;
  bool should_stop_{false};
};

} // namespace torch::executor
