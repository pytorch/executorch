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
 * @return std::unordered_map<std::string, int64_t> Metadata key-value pairs
 */
ET_EXPERIMENTAL std::unordered_map<std::string, int64_t> get_llm_metadata(
    tokenizers::Tokenizer* tokenizer,
    Module* module);

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
 * @return std::unique_ptr<TextLLMRunner> Initialized TextLLMRunner instance, or
 * nullptr on failure
 */
ET_EXPERIMENTAL std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path = std::nullopt,
    float temperature = -1.0f);

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
    std::optional<const std::string> data_path = std::nullopt);

} // namespace executorch::extension::llm
