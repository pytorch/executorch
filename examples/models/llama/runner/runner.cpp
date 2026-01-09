/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llama/runner/runner.h>
#include <executorch/extension/module/module.h>

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>

namespace example {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace llm = ::executorch::extension::llm;

std::unique_ptr<::tokenizers::Tokenizer> load_llama_tokenizer(
    const std::string& tokenizer_path,
    Version version) {
  auto special_tokens = get_special_tokens(version);
  return llm::load_tokenizer(tokenizer_path, std::move(special_tokens));
}

std::unique_ptr<llm::TextLLMRunner> create_llama_runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    std::optional<const std::string> data_path,
    float temperature,
    std::unique_ptr<::executorch::runtime::EventTracer> event_tracer) {
  if (data_path.has_value()) {
    std::vector<std::string> data_files;
    data_files.push_back(data_path.value());
    return create_llama_runner(
        model_path,
        tokenizer_path,
        std::move(data_files),
        temperature,
        std::move(event_tracer));
  }
  return create_llama_runner(
      model_path,
      tokenizer_path,
      std::vector<std::string>(),
      temperature,
      std::move(event_tracer));
}

std::unique_ptr<llm::TextLLMRunner> create_llama_runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    std::vector<std::string> data_files,
    float temperature,
    std::unique_ptr<::executorch::runtime::EventTracer> event_tracer) {
  ET_LOG(
      Info,
      "Creating LLaMa runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());

  // Create and load tokenizer
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer =
      load_llama_tokenizer(tokenizer_path, Version::Default);

  if (tokenizer == nullptr) {
    ET_LOG(
        Info,
        "Failed to load %s as a Tiktoken, Sentencepiece or Llama2.c tokenizer, make sure the artifact is one of these types",
        tokenizer_path.c_str());
    return nullptr;
  }
  return llm::create_text_llm_runner(
      model_path,
      std::move(tokenizer),
      data_files,
      temperature,
      std::move(event_tracer));
}

} // namespace example
