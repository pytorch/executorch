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
#include <optional>
#include <string>
#include <unordered_map>

#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace example {

namespace llm = ::executorch::extension::llm;

std::unique_ptr<llm::TextLLMRunner> create_llama_runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    std::optional<const std::string> data_path = std::nullopt,
    float temperature = -1.0f);

std::unique_ptr<tokenizers::Tokenizer> load_llama_tokenizer(
    const std::string& tokenizer_path);

} // namespace example
