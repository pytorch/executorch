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

#include <executorch/examples/models/llama2/tokenizer/llama_tiktoken.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>
#include <executorch/extension/llm/tokenizer/tiktoken.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "llama_runner/LlamaConfig.h"
#include "llama_runner/LlamaRuntime.h"
using Stats = ::executorch::llm::Stats;

using example::LlamaModelOptions;
using example::LlamaModelPaths;
using example::LlamaRuntime;
using executorch::extension::llm::Tokenizer;
using executorch::runtime::Error;
using executorch::runtime::Result;

class MTKLlamaRunner {
 public:
  explicit MTKLlamaRunner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const float temperature = 0.8f);

  bool is_loaded() const;
  Error load();
  Error generate(
      const std::string& prompt,
      int32_t seq_len = 128,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {});
  void stop();

  LlamaModelOptions get_model_options();
  LlamaModelPaths get_model_paths();
  Result<uint64_t> digest_prompt(
      LlamaRuntime& llama_runtime,
      const std::unique_ptr<Tokenizer>& tokenizer,
      const std::vector<uint64_t> input_tokens);
  Error gen_response(
      LlamaRuntime& llama_runtime,
      const std::unique_ptr<Tokenizer>& tokenizer,
      const uint64_t input_token,
      std::function<void(const std::string&)> token_callback);
  Error inference(
      LlamaRuntime& llama_runtime,
      const std::unique_ptr<Tokenizer>& tokenizer,
      const std::string& prompt,
      std::function<void(const std::string&)> token_callback);
  std::unique_ptr<Tokenizer> load_tokenizer();

 private:
  // model
  const LlamaModelOptions modeloptions_;
  const LlamaModelPaths modelpaths_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<LlamaRuntime> runtime_;
};
