/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// main.cpp

#include <iostream>

#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/managed_tensor.h>

#include "sentence_piece_tokenizer.h"

using namespace torch::executor;

// The value of the phi-3-mini `<|endoftext|>` token.
#define ENDOFTEXT_TOKEN 32000
#define VOCABULARY_SIZE 32064

// TODO(lunwenh): refactor and share with llama
void generate(
    Module& llm_model,
    std::string& prompt,
    SentencePieceTokenizer& tokenizer,
    size_t max_output_length) {
  // Convert the input text into a list of integers (tokens) that represents
  // it, using the string-to-token mapping that the model was trained on.
  // Each token is an integer that represents a word or part of a word.
  std::vector<int64_t> input_tokens = tokenizer.encode(prompt);

  std::cout << "Generating tokens ..." << std::endl;

  std::vector<int64_t> output_tokens;

  for (size_t i = 0; i < max_output_length; i++) {
    ManagedTensor tensor_tokens(
        input_tokens.data(),
        {1, static_cast<int>(input_tokens.size())},
        ScalarType::Long);
    std::vector<EValue> inputs = {tensor_tokens.get_aliasing_tensor()};

    Result<std::vector<EValue>> result_evalue = llm_model.forward(inputs);

    const auto error = result_evalue.error();
    Tensor logits_tensor = result_evalue.get()[0].toTensor();
    const auto sentence_length = logits_tensor.size(1);
    std::vector<float> logits(
        logits_tensor.data_ptr<float>() +
            (sentence_length - 1) * VOCABULARY_SIZE,
        logits_tensor.data_ptr<float>() + sentence_length * VOCABULARY_SIZE);

    // Sample the next token from the logits.
    int64_t next_token =
        std::max_element(logits.begin(), logits.end()) - logits.begin();

    std::cout << next_token << "\t";
    std::cout.flush();

    // Break if we reached the end of the text.
    if (next_token == ENDOFTEXT_TOKEN) {
      break;
    }

    output_tokens.push_back(next_token);

    // Update next input.
    input_tokens.push_back(next_token);
  }

  std::cout << std::endl;
  std::cout << tokenizer.decode(output_tokens) << std::endl;
}

int main() {
  // Set up the prompt. This provides the seed text for the model to elaborate.
  std::cout << "Enter model prompt: ";
  std::string prompt;
  std::getline(std::cin, prompt);

  SentencePieceTokenizer tokenizer("tokenizer.model");

  Module model("phi-3-mini.pte", Module::LoadMode::MmapUseMlockIgnoreErrors);

  const auto max_output_tokens = 128;
  generate(model, prompt, tokenizer, max_output_tokens);
}
