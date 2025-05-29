/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama3.2 runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/token_generator.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>
namespace example {

enum LlamaVersion {
  kLlama2 = 0,
  kLlama3,
};
class Runner {
 public:
  explicit Runner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& performance_output_path,
      const float temperature = 0.8f,
      const int eval_mode = EvalMode::kKVCached,
      const std::string& kv_updater = "SmartMask");

  bool is_loaded() const;
  executorch::runtime::Error load();
  // TODO: Support echo and warming
  executorch::runtime::Error generate(
      const std::string& prompt,
      int32_t seq_len,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {},
      bool echo = true,
      bool warming = false);
  void stop() {};
  executorch::runtime::Result<LlamaVersion> get_llama_version();

 private:
  enum EvalMode {
    kKVCached = 0,
    kHybrid,
    kUnsupported,
  };

  std::unique_ptr<executorch::extension::Module> module_;
  int32_t context_len_{0};

  int64_t cur_pos_{0};

  std::string tokenizer_path_;
  std::string performance_output_path_;
  float temperature_;
  EvalMode eval_mode_;
  LlamaVersion llama_version_;
  KVManagerMode kv_updater_;
  std::unique_ptr<IMemAlloc> buffer_manager_;
  std::unique_ptr<KVManager> kv_manager_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<DecoderRunner> decoder_runner_;
  std::unique_ptr<PromptProcessor> prompt_processor_;
  std::unique_ptr<TokenGenerator> token_generator_;

  // stats
  executorch::llm::Stats stats_;
};
} // namespace example
