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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <executorch/examples/models/llama2/sampler/sampler.h>
#include <executorch/examples/models/llama2/tokenizer/tokenizer.h>
#include <executorch/extension/module/module.h>

namespace torch {
namespace executor {

class Runner {
 public:
  explicit Runner(const char* model_path, const char* tokenizer_path);

  Error generate(
      const std::string& prompt,
      std::function<void(const std::string&)> callback = {});

 private:
  // metadata
  template <typename T>
  T getMetadataHelper(std::string method_name, T default_val);
  template <typename T>
  int32_t
  logitsToToken(const exec_aten::Tensor& logits_tensor, int64_t pos, T _);
  std::vector<exec_aten::SizesType> getKVCacheShape();
  // metadata
  int32_t vocab_size_;
  int32_t bos_id_;
  int32_t eos_id_;
  int32_t n_bos_;
  int32_t n_eos_;
  int32_t max_seq_len_;
  bool use_kv_cache_;
  bool append_eos_;
  std::unordered_set<std::string> model_methods_;
  // module
  std::unique_ptr<Module> module_;
  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;
  // sampler
  std::unique_ptr<Sampler> sampler_;
};

} // namespace executor
} // namespace torch
