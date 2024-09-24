/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple phi-3-mini runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#pragma once

#include <memory>
#include <string>

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/llm/tokenizer/tokenizer.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace example {

class Runner {
 public:
  explicit Runner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const float temperature = 0.8f);

  /**
   * Generates response for a given prompt.
   *
   * @param[in] prompt The prompt to generate a response for.
   * @param[in] max_seq_len The maximum length of the sequence to generate,
   * including prompt.
   */
  void generate(const std::string& prompt, std::size_t max_seq_len);

 private:
  uint64_t logits_to_token(const exec_aten::Tensor& logits_tensor);
  uint64_t prefill(std::vector<uint64_t>& tokens);
  uint64_t run_model_step(uint64_t token);

  std::unique_ptr<executorch::extension::Module> module_;
  std::unique_ptr<executorch::extension::llm::Tokenizer> tokenizer_;
  std::unique_ptr<executorch::extension::llm::Sampler> sampler_;
};

} // namespace example
