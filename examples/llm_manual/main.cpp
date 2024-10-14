/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>

#include "basic_sampler.h"
#include "basic_tokenizer.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::EValue;
using executorch::runtime::Result;

// The value of the gpt2 `<|endoftext|>` token.
#define ENDOFTEXT_TOKEN 50256

std::string generate(
    Module& llm_model,
    std::string& prompt,
    BasicTokenizer& tokenizer,
    BasicSampler& sampler,
    size_t max_input_length,
    size_t max_output_length) {
  // Convert the input text into a list of integers (tokens) that represents it,
  // using the string-to-token mapping that the model was trained on. Each token
  // is an integer that represents a word or part of a word.
  std::vector<int64_t> input_tokens = tokenizer.encode(prompt);
  std::vector<int64_t> output_tokens;

  for (auto i = 0u; i < max_output_length; i++) {
    // Convert the input_tokens from a vector of int64_t to EValue. EValue is a
    // unified data type in the ExecuTorch runtime.
    auto inputs = from_blob(
        input_tokens.data(),
        {1, static_cast<int>(input_tokens.size())},
        ScalarType::Long);

    // Run the model. It will return a tensor of logits (log-probabilities).
    auto logits_evalue = llm_model.forward(inputs);

    // Convert the output logits from EValue to std::vector, which is what the
    // sampler expects.
    Tensor logits_tensor = logits_evalue.get()[0].toTensor();
    std::vector<float> logits(
        logits_tensor.data_ptr<float>(),
        logits_tensor.data_ptr<float>() + logits_tensor.numel());

    // Sample the next token from the logits.
    int64_t next_token = sampler.sample(logits);

    // Break if we reached the end of the text.
    if (next_token == ENDOFTEXT_TOKEN) {
      break;
    }

    // Add the next token to the output.
    output_tokens.push_back(next_token);

    std::cout << tokenizer.decode({next_token});
    std::cout.flush();

    // Update next input.
    input_tokens.push_back(next_token);
    if (input_tokens.size() > max_input_length) {
      input_tokens.erase(input_tokens.begin());
    }
  }

  std::cout << std::endl;

  // Convert the output tokens into a human-readable string.
  std::string output_string = tokenizer.decode(output_tokens);
  return output_string;
}

int main() {
  // Set up the prompt. This provides the seed text for the model to elaborate.
  std::cout << "Enter model prompt: ";
  std::string prompt;
  std::getline(std::cin, prompt);

  // The tokenizer is used to convert between tokens (used by the model) and
  // human-readable strings.
  BasicTokenizer tokenizer("vocab.json");

  // The sampler is used to sample the next token from the logits.
  BasicSampler sampler = BasicSampler();

  // Load the exported nanoGPT program, which was generated via the previous
  // steps.
  Module model("nanogpt.pte", Module::LoadMode::MmapUseMlockIgnoreErrors);

  const auto max_input_tokens = 1024;
  const auto max_output_tokens = 30;
  std::cout << prompt;
  generate(
      model, prompt, tokenizer, sampler, max_input_tokens, max_output_tokens);
}
