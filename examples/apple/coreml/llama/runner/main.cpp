/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Main executable for running static attention LLM models.
//
// Usage:
//   ./run_static_llm \
//       --model /path/to/model.pte \
//       --params /path/to/params.json \
//       --tokenizer /path/to/tokenizer.model \
//       --prompt "Once upon a time," \
//       --max_new_tokens 100
//
// With lookahead decoding:
//   ./run_static_llm \
//       --model /path/to/model.pte \
//       --params /path/to/params.json \
//       --tokenizer /path/to/tokenizer.model \
//       --prompt "Once upon a time," \
//       --max_new_tokens 100 \
//       --lookahead \
//       --ngram_size 4 \
//       --window_size 5 \
//       --n_verifications 3

#include <iostream>
#include <string>

#include <executorch/examples/apple/coreml/llama/runner/static_llm_runner.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <gflags/gflags.h>

DEFINE_string(model, "", "Path to the .pte model file (required)");
DEFINE_string(params, "", "Path to params.json file (optional, for rope_theta)");
DEFINE_string(tokenizer, "", "Path to tokenizer model file (required)");
DEFINE_string(prompt, "Once upon a time,", "Input prompt");
DEFINE_int32(max_new_tokens, 100, "Maximum number of tokens to generate");
DEFINE_double(temperature, 0.0, "Sampling temperature (0 = greedy)");

// Lookahead decoding options
DEFINE_bool(lookahead, false, "Enable lookahead (speculative) decoding");
DEFINE_int32(ngram_size, 4, "N-gram size for lookahead decoding");
DEFINE_int32(window_size, 5, "Window size for lookahead decoding");
DEFINE_int32(n_verifications, 3, "Number of verification branches for lookahead decoding");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Validate required arguments
  if (FLAGS_model.empty()) {
    std::cerr << "Error: --model is required" << std::endl;
    return 1;
  }
  if (FLAGS_params.empty()) {
    std::cerr << "Error: --params is required" << std::endl;
    return 1;
  }
  if (FLAGS_tokenizer.empty()) {
    std::cerr << "Error: --tokenizer is required" << std::endl;
    return 1;
  }

  // Initialize runtime
  executorch::runtime::runtime_init();

  // Create runner (config is auto-detected from model metadata)
  auto runner = example::create_static_llm_runner(
      FLAGS_model,
      FLAGS_tokenizer,
      FLAGS_params);

  if (!runner) {
    std::cerr << "Error: Failed to create runner" << std::endl;
    return 1;
  }

  // Load model
  auto load_err = runner->load();
  if (load_err != executorch::runtime::Error::Ok) {
    std::cerr << "Error: Failed to load model" << std::endl;
    return 1;
  }

  // Print prompt
  std::cout << "\n" << FLAGS_prompt << std::flush;

  // Generate
  executorch::runtime::Error gen_err;

  if (FLAGS_lookahead) {
    // Use lookahead decoding
    example::LookaheadConfig lookahead_config;
    lookahead_config.enabled = true;
    lookahead_config.ngram_size = static_cast<size_t>(FLAGS_ngram_size);
    lookahead_config.window_size = static_cast<size_t>(FLAGS_window_size);
    lookahead_config.n_verifications = static_cast<size_t>(FLAGS_n_verifications);

    gen_err = runner->generate_with_lookahead(
        FLAGS_prompt,
        FLAGS_max_new_tokens,
        lookahead_config,
        [](const std::string& token) { std::cout << token << std::flush; });
  } else {
    // Use standard decoding
    gen_err = runner->generate(
        FLAGS_prompt,
        FLAGS_max_new_tokens,
        static_cast<float>(FLAGS_temperature),
        [](const std::string& token) { std::cout << token << std::flush; });
  }

  if (gen_err != executorch::runtime::Error::Ok) {
    std::cerr << "\nError: Generation failed" << std::endl;
    return 1;
  }

  std::cout << "\n" << std::endl;

  return 0;
}
