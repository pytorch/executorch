/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <optional>
#include <string>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(data_path, "", "Data file (.ptd) for CUDA backend.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(prompt, "Hello", "Prompt text.");
DEFINE_double(temperature, 0.8, "Sampling temperature (0 = greedy).");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;

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

  // Single-method runner: "forward" handles both prefill (T>1) and decode (T=1)
  // via torch.cond dispatch inside the model.
  fprintf(stderr, "Loading model from %s...\n", FLAGS_model_path.c_str());
  std::optional<const std::string> data_path =
      FLAGS_data_path.empty() ? std::nullopt
                              : std::optional<const std::string>(FLAGS_data_path);
  auto runner = llm::create_text_llm_runner(
      FLAGS_model_path,
      std::move(tokenizer),
      data_path,
      FLAGS_temperature);
  fprintf(stderr, "Runner created successfully\n");

  // Generate
  llm::GenerationConfig config;
  config.temperature = FLAGS_temperature;
  config.max_new_tokens = FLAGS_max_new_tokens;

  fprintf(stderr, "Starting generation with prompt: %s\n", FLAGS_prompt.c_str());
  try {
    auto error = runner->generate(FLAGS_prompt.c_str(), config);
    if (error != Error::Ok) {
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
