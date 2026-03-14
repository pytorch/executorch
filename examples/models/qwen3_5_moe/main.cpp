/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <fstream>
#include <sstream>
#include <vector>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(
    data_path,
    "",
    "Comma-separated data files (.ptd) for CUDA backend.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(prompt, "Hello", "Prompt text.");
DEFINE_double(temperature, 0.8, "Sampling temperature (0 = greedy).");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");

namespace llm = ::executorch::extension::llm;

static std::vector<std::string> split_comma(const std::string& input) {
  std::vector<std::string> result;
  if (input.empty())
    return result;
  std::stringstream ss(input);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(0, item.find_first_not_of(" \t"));
    item.erase(item.find_last_not_of(" \t") + 1);
    if (!item.empty())
      result.push_back(item);
  }
  return result;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_path.empty()) {
    ET_LOG(Error, "Must specify --model_path");
    return 1;
  }
  if (FLAGS_tokenizer_path.empty()) {
    ET_LOG(Error, "Must specify --tokenizer_path");
    return 1;
  }

  std::vector<std::string> data_files = split_comma(FLAGS_data_path);

  // Load tokenizer
  auto tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  auto tok_status = tokenizer->load(FLAGS_tokenizer_path);
  if (tok_status != tokenizers::Error::Ok) {
    ET_LOG(Error, "Failed to load tokenizer from %s", FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // Create LLM runner
  auto runner = llm::create_text_llm_runner(
      FLAGS_model_path,
      std::move(tokenizer),
      data_files,
      FLAGS_temperature);

  if (runner == nullptr) {
    ET_LOG(Error, "Failed to create runner");
    return 1;
  }

  // Generate
  llm::GenerationConfig config;
  config.temperature = FLAGS_temperature;
  config.max_new_tokens = FLAGS_max_new_tokens;

  auto error = runner->generate(FLAGS_prompt.c_str(), config);
  if (error != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Generation failed");
    return 1;
  }

  return 0;
}
