/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#pragma once

#include <memory>
#include <unordered_map>

#include <executorch/examples/models/llama2/sampler/sampler.h>
#include <executorch/examples/models/llama2/tokenizer/tokenizer.h>
#include <executorch/extension/module/module.h>

namespace torch {
namespace executor {

class Runner {
 public:
  explicit Runner(const char* model_path, const char* tokenizer_path);

  Error generate(const char* prompt, bool eos = false);

 private:
  std::vector<int32_t> readMetadata(
      std::unordered_set<std::string> method_names);
  // metadata
  int32_t vocab_size_;
  int32_t bos_id_;
  int32_t eos_id_;
  int32_t n_bos_;
  int32_t n_eos_;
  int32_t max_seq_len_;
  // module
  std::unique_ptr<Module> module_;
  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;
  // sampler
  std::unique_ptr<Sampler> sampler_;
};

} // namespace executor
} // namespace torch
