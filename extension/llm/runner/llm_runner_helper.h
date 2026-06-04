/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Helper utilities for creating and configuring LLM runners

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace executorch::extension::llm {

// Forward declarations
class TextLLMRunner;
class MultimodalRunner;

/**
 * @brief Loads a tokenizer from the specified path
 *
 * This function creates and initializes a tokenizer from a file, with options
 * to customize special tokens and regex patterns. It tries different tokenizer
 * types in order: HF JSON, TikToken, SentencePiece, and BPE.
 *
 * @param tokenizer_path Path to the tokenizer file
 * @param special_tokens Optional list of special tokens to add to the tokenizer
 * @param pattern Optional regex pattern for tokenization
 * @param bos_token_index Index of the beginning-of-sequence token
 * @param eos_token_index Index of the end-of-sequence token
 * @return std::unique_ptr<tokenizers::Tokenizer> Initialized tokenizer
 * instance, or nullptr on failure
 */
ET_EXPERIMENTAL std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(
    const std::string& tokenizer_path,
    std::unique_ptr<std::vector<std::string>> special_tokens = nullptr,
    std::optional<std::string> pattern = std::nullopt,
    size_t bos_token_index = 0,
    size_t eos_token_index = 1);

/**
 * @brief Gets LLM metadata from the model and tokenizer
 *
 * This function extracts metadata from the model such as vocabulary size,
 * context length, and other configuration parameters. It reads metadata
 * methods from the model and combines them with tokenizer information.
 *
 * @param tokenizer Initialized tokenizer instance
 * @param module The model module
 * @return Result<std::unordered_map<std::string, int64_t>> Metadata key-value
 * pairs on success, or Error::InvalidArgument if required metadata (e.g.,
 * kMaxSeqLen) is missing from the model
 */
ET_EXPERIMENTAL ::executorch::runtime::Result<
    std::unordered_map<std::string, int64_t>>
get_llm_metadata(tokenizers::Tokenizer* tokenizer, Module* module);

/**
 * @brief Gets EOS token IDs from the model and tokenizer
 *
 * This function extracts the end-of-sequence token IDs from the model.
 * It first tries to get EOS IDs from the model's metadata, falling back
 * to the tokenizer's default EOS token.
 *
 * @param tokenizer Initialized tokenizer instance
 * @param module The model module
 * @return std::unordered_set<uint64_t> Set of EOS token IDs
 */
ET_EXPERIMENTAL std::unordered_set<uint64_t> get_eos_ids(
    tokenizers::Tokenizer* tokenizer,
    Module* module);

/**
 * @brief Creates a TextLLMRunner instance with dependency injection
 *
 * This factory function creates and initializes a TextLLMRunner with all
 * necessary components for text generation using the specified model and
 * tokenizer.
 *
 * @param model_path Path to the model file
 * @param tokenizer Initialized tokenizer instance
 * @param data_path Optional path to additional data required by the model
 * @param temperature Optional temperature parameter for controlling randomness
 * (deprecated)
 * @param method_name Name of the method to execute in the model
 * @param load_mode Loading strategy for the model file. Defaults to
 * MmapUseMlockIgnoreErrors which uses mmap to avoid loading the entire
 * model into RAM and attempts to pin pages with mlock for lower inference
 * latency, gracefully falling back to standard mmap if mlock is unavailable.
 * @return std::unique_ptr<TextLLMRunner> Initialized TextLLMRunner instance, or
 * nullptr on failure
 */
ET_EXPERIMENTAL std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path,
    float temperature = -1.0f,
    const std::string& method_name = "forward",
    Module::LoadMode load_mode = Module::LoadMode::MmapUseMlockIgnoreErrors);

/**
 * @brief Creates a TextLLMRunner instance with dependency injection
 *
 * This factory function creates and initializes a TextLLMRunner with all
 * necessary components for text generation using the specified model and
 * tokenizer.
 *
 * @param model_path Path to the model file
 * @param tokenizer Initialized tokenizer instance
 * @param data_files Vector of paths to additional data required by the model
 * @param temperature Optional temperature parameter for controlling randomness
 * (deprecated)
 * @param event_tracer Optional event tracer for profiling
 * @param method_name Name of the method to execute in the model
 * @param load_mode Loading strategy for the model file. Defaults to
 * MmapUseMlockIgnoreErrors which uses mmap to avoid loading the entire
 * model into RAM and attempts to pin pages with mlock for lower inference
 * latency, gracefully falling back to standard mmap if mlock is unavailable.
 * @return std::unique_ptr<TextLLMRunner> Initialized TextLLMRunner instance, or
 * nullptr on failure
 */
ET_EXPERIMENTAL std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::vector<std::string> data_files = {},
    float temperature = -1.0f,
    std::unique_ptr<::executorch::runtime::EventTracer> event_tracer = nullptr,
    const std::string& method_name = "forward",
    Module::LoadMode load_mode = Module::LoadMode::MmapUseMlockIgnoreErrors);

/**
 * @brief Creates a TextLLMRunner over an already-loaded Program.
 *
 * Unlike create_text_llm_runner(model_path, ...), this does not load the model
 * file again: the resulting runner's Module reuses `program` while owning its
 * own method state and KV cache. This is the per-session construction path for
 * TextLLMEngine — N sessions reuse one loaded Program but isolate their mutable
 * KV state. Whether they also avoid re-materializing packed weights per session
 * is backend-dependent (serving_capacity() is authoritative).
 *
 * The caller must keep the DataLoader backing `program` alive for the lifetime
 * of every runner created from it (TextLLMEngine holds the loader Module).
 *
 * @param program Shared, already-loaded program.
 * @param tokenizer Initialized tokenizer instance (owned by the new runner).
 * @param temperature Optional temperature (deprecated; prefer
 * GenerationConfig).
 * @param method_name Name of the method to execute in the model.
 * @return std::unique_ptr<TextLLMRunner> on success, or nullptr on failure.
 */
ET_EXPERIMENTAL std::unique_ptr<TextLLMRunner>
create_text_llm_runner_from_program(
    std::shared_ptr<Program> program,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    float temperature = -1.0f,
    const std::string& method_name = "forward");

/**
 * @brief Engine for multi-session text generation over one loaded Program.
 *
 * Loads the model's Program (weights/constants) once; create_session() builds a
 * TextLLMRunner that reuses that Program but owns its own method/KV state. This
 * is the correctness-first foundation for serving multiple conversations.
 * Backend execution should be serialized by the caller until per-backend thread
 * safety is proven (Module::execute is not assumed thread-safe). Whether extra
 * sessions avoid duplicating packed weights is backend-dependent and reported
 * by serving_capacity() (conservatively one).
 */
class ET_EXPERIMENTAL TextLLMEngine : public LLMEngine {
 public:
  static std::unique_ptr<TextLLMEngine> create(
      const std::string& model_path,
      const std::string& tokenizer_path,
      std::optional<const std::string> data_path = std::nullopt,
      float temperature = -1.0f,
      const std::string& method_name = "forward",
      Module::LoadMode load_mode = Module::LoadMode::MmapUseMlockIgnoreErrors);

  // Returns a TextLLMSession (LLMSession) that reuses this engine's loaded
  // Program (physical weight sharing is backend-dependent; see
  // serving_capacity).
  ::executorch::runtime::Result<std::unique_ptr<LLMSession>> create_session()
      override;
  // Conservative: a single physical session (no proven cross-session weight
  // sharing). Raise on a backend proven to share packed weights.
  LLMServingCapacity serving_capacity() const override {
    return LLMServingCapacity{};
  }
  const std::unordered_map<std::string, int64_t>& metadata() const override {
    return metadata_;
  }

  TextLLMEngine(const TextLLMEngine&) = delete;
  TextLLMEngine& operator=(const TextLLMEngine&) = delete;

 private:
  TextLLMEngine(
      std::unique_ptr<Module> loader_module,
      std::shared_ptr<Program> program,
      std::string tokenizer_path,
      float temperature,
      std::string method_name,
      std::unordered_map<std::string, int64_t> metadata);

  // Keeps the shared Program's DataLoader alive for the lifetime of sessions.
  std::unique_ptr<Module> loader_module_;
  std::shared_ptr<Program> program_;
  std::string tokenizer_path_;
  float temperature_;
  std::string method_name_;
  std::unordered_map<std::string, int64_t> metadata_;
};

/**
 * @brief Creates a MultimodalRunner instance with dependency injection
 *
 * This factory function creates and initializes a MultimodalRunner with all
 * necessary components for multimodal text generation.
 *
 * @param model_path Path to the model file
 * @param tokenizer Initialized tokenizer instance
 * @param data_path Optional path to additional .ptd required by the model
 * @return std::unique_ptr<MultimodalRunner> Initialized MultimodalRunner
 * instance, or nullptr on failure
 */
ET_EXPERIMENTAL std::unique_ptr<MultimodalRunner> create_multimodal_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path = std::nullopt,
    Module::LoadMode load_mode = Module::LoadMode::File);

} // namespace executorch::extension::llm
