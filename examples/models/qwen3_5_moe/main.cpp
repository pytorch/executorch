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
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <string>
#include <vector>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(data_path, "", "Data file (.ptd) for CUDA backend.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(prompt, "Hello", "Prompt text.");
DEFINE_double(temperature, 0.8, "Sampling temperature (0 = greedy).");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");

namespace llm = ::executorch::extension::llm;

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

  std::vector<std::string> data_files;
  if (!FLAGS_data_path.empty()) {
    data_files.push_back(FLAGS_data_path);
  }

  // Load tokenizer
  auto tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  auto tok_status = tokenizer->load(FLAGS_tokenizer_path);
  if (tok_status != tokenizers::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to load tokenizer from %s",
        FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // Create LLM runner
  fprintf(stderr, "Creating runner from %s...\n", FLAGS_model_path.c_str());
  auto runner = llm::create_text_llm_runner(
      FLAGS_model_path, std::move(tokenizer), data_files, FLAGS_temperature);

  if (runner == nullptr) {
    fprintf(stderr, "FATAL: Failed to create runner\n");
    return 1;
  }
  fprintf(stderr, "Runner created successfully\n");

  // Generate
  llm::GenerationConfig config;
  config.temperature = FLAGS_temperature;
  config.max_new_tokens = FLAGS_max_new_tokens;

  fprintf(stderr, "Starting generation with prompt: %s\n", FLAGS_prompt.c_str());
  try {
    auto error = runner->generate(FLAGS_prompt.c_str(), config);
    if (error != executorch::runtime::Error::Ok) {
      fprintf(stderr, "Generation failed with error code: %d\n", static_cast<int>(error));
      return 1;
    }
    fprintf(stderr, "Generation completed successfully\n");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception during generation: %s\n", e.what());
    return 1;
  } catch (...) {
    fprintf(stderr, "Unknown exception during generation\n");
    return 1;
  }

  return 0;
}
