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
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include "sentence_piece_tokenizer.h"

namespace torch::executor {

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
   * @param[in] seq_len The length of the sequence to generate, including
   * prompt.
   */
  void generate(const std::string& prompt, std::size_t seq_len);

 private:
  int64_t logits_to_token(const exec_aten::Tensor& logits_tensor);
  int64_t prefill(const std::vector<int64_t>& tokens);
  int64_t run_model_step(int64_t token);

  std::unique_ptr<Module> module_;
  std::unique_ptr<SentencePieceTokenizer> tokenizer_;
  std::unique_ptr<Sampler> sampler_;
};

} // namespace torch::executor
