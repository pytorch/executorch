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

namespace torch::executor {

class Runner {
 public:
  explicit Runner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const float temperature = 0.8f);

  bool is_loaded() const;
  Error load();
  Error generate(
      const std::string& prompt,
      int32_t seq_len = 128,
      std::function<void(const std::string&)> callback = {});
  void stop();

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
  bool use_sdpa_with_kv_cache_;
  bool append_eos_;
  std::unordered_set<std::string> model_methods_;
  std::unique_ptr<Module> module_;
  std::string tokenizer_path_;
  float temperature_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Sampler> sampler_;
  bool shouldStop_{false};
};

} // namespace torch::executor
