/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <gflags/gflags.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <iostream>

using executorch::extension::llm::TextLLMRunner;

DEFINE_string(
    model_path,
    "phi-3-mini.pte",
    "File path for model serialized in flatbuffer format.");

DEFINE_string(tokenizer_path, "tokenizer.bin", "File path for tokenizer.");

DEFINE_string(prompt, "Tell me a story", "Prompt.");

DEFINE_double(
    temperature,
    0.8f,
    "Temperature; Default is 0.8f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output).");

int main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char* model_path = FLAGS_model_path.c_str();

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  const char* prompt = FLAGS_prompt.c_str();

  double temperature = FLAGS_temperature;

  int32_t seq_len = FLAGS_seq_len;

  std::unique_ptr<tokenizers::Tokenizer> tokenizer =
      std::make_unique<tokenizers::Llama2cTokenizer>();
  tokenizer->load(tokenizer_path);

  auto runner = executorch::extension::llm::create_text_llm_runner(
      model_path, std::move(tokenizer));

  runner->generate(
      prompt,
      {.seq_len = seq_len, .temperature = static_cast<float>(temperature)});

  return 0;
}
