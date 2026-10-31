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
 * @brief Gets the largest number of prompt tokens one prefill call can take
 *
 * For the text runner's [1, N] token input: returns `max_seq_len` (the model's
 * get_max_seq_len), lowered to the upper bound of the method's input 0,
 * dimension 1, when that bound is smaller. The bound is the size serialized for
 * a bounded dynamic dimension; if it cannot be read, `max_seq_len` is returned.
 * export_llm's default KV-cache dynamic shapes in ExecuTorch 1.1 through 1.5
 * bound that dimension at max_seq_len - 1 while the program publishes
 * get_max_seq_len = max_seq_len, so prefill chunks sized from the metadata
 * alone are one token larger than the program accepts.
 *
 * @param module The model module
 * @param method_name The method that runs prefill
 * @param max_seq_len The model's get_max_seq_len
 * @return int64_t The prefill chunk size to use
 */
ET_EXPERIMENTAL int64_t get_max_prefill_chunk_size(
    Module* module,
    const std::string& method_name,
    int64_t max_seq_len);

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
    const std::string& method_name = kForwardMethod,
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
 * @param method_name Name of the method to execute in the model. Falls back to
 * kPrefillMethod when absent from the model. When the model also exports
 * kDecodeMethod, the resolved method drives prefill only and kDecodeMethod
 * drives decode.
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
    const std::string& method_name = kForwardMethod,
    Module::LoadMode load_mode = Module::LoadMode::MmapUseMlockIgnoreErrors);

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
