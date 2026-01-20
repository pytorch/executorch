/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// C++ runner for static attention LLM models exported with
// export_static_llm_coreml.py. Subclasses TextDecoderRunner to maintain
// the standard LLM runner interface.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <executorch/examples/models/llama/runner/static_attention_io_manager.h>
#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/method.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace example {

/**
 * Configuration for the static LLM model.
 */
struct StaticLLMConfig {
  size_t n_layers = 16;
  size_t n_kv_heads = 8;
  size_t head_dim = 64;
  size_t input_len = 32;
  size_t cache_len = 992;
  size_t vocab_size = 128256;
  bool generate_full_logits = true;
  float rope_base = 500000.0f;
};

/**
 * Configuration for lookahead (speculative) decoding.
 */
struct LookaheadConfig {
  bool enabled = false;
  size_t ngram_size = 4;
  size_t window_size = 5;
  size_t n_verifications = 3;
};

/**
 * IOManager adapter that wraps StaticAttentionIOManager to implement
 * the executorch::extension::llm::IOManager interface.
 *
 * This bridges the gap between TextDecoderRunner's expected interface
 * and the static attention model's I/O requirements.
 */
class StaticLLMIOManager : public ::executorch::extension::llm::IOManager {
 public:
  using CacheT = __fp16;
  using MaskT = __fp16;
  using RopeT = __fp16;
  using LogitT = __fp16;
  using TokenT = int32_t;

  StaticLLMIOManager(
      ::executorch::extension::Module& module,
      const StaticLLMConfig& config);

  ~StaticLLMIOManager() override = default;

  ::executorch::runtime::Error load(
      const std::string& prefill_method,
      const std::string& decode_method) override;

  ::executorch::runtime::Error reset(
      const std::string& prefill_method,
      const std::string& decode_method) override;

  ::executorch::runtime::Result<std::vector<::executorch::runtime::EValue>>
  prepare_prefill(
      const ::executorch::extension::TensorPtr& input,
      const ::executorch::extension::TensorPtr& start_pos,
      const std::string& prefill_method) override;

  ::executorch::runtime::Result<std::vector<::executorch::runtime::EValue>>
  prepare_decode(
      const ::executorch::extension::TensorPtr& input,
      const ::executorch::extension::TensorPtr& start_pos,
      const std::string& decode_method) override;

  ::executorch::runtime::Error update_prefill(
      const std::vector<::executorch::runtime::EValue>& model_outputs,
      const std::string& prefill_method) override;

  ::executorch::runtime::Error update_decode(
      const std::vector<::executorch::runtime::EValue>& model_outputs,
      const std::string& decode_method) override;

  /**
   * Get the underlying StaticAttentionIOManager for advanced operations.
   */
  StaticAttentionIOManager<CacheT, MaskT, RopeT>* get_static_io_manager() {
    return static_io_manager_.get();
  }

  const StaticLLMConfig& config() const {
    return config_;
  }

  /**
   * Set up CoreML output buffers on the Method.
   * Must be called before first inference.
   */
  void setup_output_buffers(::executorch::runtime::Method& method);

 private:
  void compute_rope_frequencies();

  ::executorch::extension::Module& module_;
  StaticLLMConfig config_;

  std::vector<RopeT> freqs_cos_;
  std::vector<RopeT> freqs_sin_;
  std::vector<TokenT> input_buffer_;
  std::vector<LogitT> logits_output_;
  std::vector<std::vector<CacheT>> k_update_buffers_;
  std::vector<std::vector<CacheT>> v_update_buffers_;

  std::unique_ptr<StaticAttentionIOManager<CacheT, MaskT, RopeT>>
      static_io_manager_;

  size_t actual_input_len_ = 0;
  bool output_buffers_set_ = false;
};

/**
 * TextDecoderRunner subclass for static attention LLM models.
 *
 * Overrides step() to use StaticAttentionIOManager's prepare/execute flow.
 */
class StaticLLMTextDecoderRunner
    : public ::executorch::extension::llm::TextDecoderRunner {
 public:
  StaticLLMTextDecoderRunner(
      ::executorch::extension::Module* module,
      StaticLLMIOManager* io_manager);

  ~StaticLLMTextDecoderRunner() override = default;

  ::executorch::runtime::Result<::executorch::aten::Tensor> step(
      ::executorch::extension::TensorPtr& input,
      int64_t start_pos) override;

  ::executorch::runtime::Error load() override;

 private:
  StaticLLMIOManager* static_io_manager_;
};

/**
 * Main runner class that orchestrates text generation.
 */
class StaticLLMRunner {
 public:
  StaticLLMRunner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const StaticLLMConfig& config);

  ~StaticLLMRunner() = default;

  ::executorch::runtime::Error load();
  bool is_loaded() const;
  void reset();

  ::executorch::runtime::Error generate(
      const std::string& prompt,
      int32_t max_new_tokens,
      float temperature = 0.0f,
      std::function<void(const std::string&)> token_callback = nullptr);

  ::executorch::runtime::Error generate_with_lookahead(
      const std::string& prompt,
      int32_t max_new_tokens,
      const LookaheadConfig& lookahead_config,
      std::function<void(const std::string&)> token_callback = nullptr);

  const ::executorch::extension::llm::Stats& stats() const {
    return stats_;
  }

  /**
   * Get the TextDecoderRunner for external use.
   */
  StaticLLMTextDecoderRunner* decoder_runner() {
    return decoder_runner_.get();
  }

 private:
  using TokenT = int32_t;
  using LogitT = __fp16;

  TokenT sample_token(::executorch::runtime::Method& method, size_t pos = 0);
  std::vector<TokenT> sample_all_tokens(::executorch::runtime::Method& method);

  std::string model_path_;
  std::string tokenizer_path_;
  StaticLLMConfig config_;

  std::unique_ptr<::executorch::extension::Module> module_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<StaticLLMIOManager> io_manager_;
  std::unique_ptr<StaticLLMTextDecoderRunner> decoder_runner_;

  std::unordered_set<uint64_t> eos_ids_;
  ::executorch::extension::llm::Stats stats_;
};

/**
 * Create a StaticLLMRunner with configuration auto-detected from model metadata.
 * Reads input_len, cache_len, n_layers, n_kv_heads, head_dim, vocab_size from
 * the model's method metadata. Only rope_base is read from params.json.
 */
std::unique_ptr<StaticLLMRunner> create_static_llm_runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& params_path);

} // namespace example
