/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>
#include <fstream>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(tokenizer_path, "tokenizer.json", "Tokenizer path.");

DEFINE_string(prompt, "Hello, world!", "Text prompt.");

DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    max_new_tokens,
    100,
    "Maximum number of tokens to generate.");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

DEFINE_bool(warmup, false, "Whether to run a warmup run.");

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char* model_path = FLAGS_model_path.c_str();
  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();
  const char* prompt = FLAGS_prompt.c_str();
  float temperature = FLAGS_temperature;
  int32_t max_new_tokens = FLAGS_max_new_tokens;
  int32_t cpu_threads = FLAGS_cpu_threads;
  bool warmup = FLAGS_warmup;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif

  std::unique_ptr<::tokenizers::Tokenizer> tokenizer =
      ::executorch::extension::llm::load_tokenizer(tokenizer_path);
  if (tokenizer == nullptr) {
    ET_LOG(Error, "Failed to load tokenizer from: %s", tokenizer_path);
    return 1;
  }

  // Create text LLM runner
  std::unique_ptr<::executorch::extension::llm::TextLLMRunner> runner =
      ::executorch::extension::llm::create_text_llm_runner(
          model_path, std::move(tokenizer));

  if (runner == nullptr) {
    ET_LOG(Error, "Failed to create text LLM runner");
    return 1;
  }

  // Load runner
  auto load_error = runner->load();
  if (load_error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load text LLM runner");
    return 1;
  }

  // Format prompt with Gemma3 chat template
  std::string formatted_prompt = std::string("<start_of_turn>user\n") +
      std::string(prompt) + std::string("<end_of_turn>\n<start_of_turn>model\n");

  ::executorch::extension::llm::GenerationConfig config;
  config.max_new_tokens = max_new_tokens;
  config.temperature = temperature;

  // Run warmup if requested
  if (warmup) {
    ET_LOG(Info, "Running warmup...");
    auto warmup_error = runner->warmup(formatted_prompt, max_new_tokens);
    if (warmup_error != ::executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to run warmup");
      return 1;
    }
    runner->reset();
  }

  ET_LOG(Info, "Generating response...");

  // Note: TextLLMRunner::generate() already handles printing tokens and stats
  // internally, so we don't need to pass callbacks for printing
  auto error = runner->generate(formatted_prompt, config);

  if (error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to generate with text LLM runner\n");
    return 1;
  }

  return 0;
}
