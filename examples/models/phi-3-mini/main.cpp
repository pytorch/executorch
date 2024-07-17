/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// main.cpp

#include <iostream>

#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/managed_tensor.h>
#include <executorch/runtime/platform/assert.h>

using namespace torch::executor;

// The value of the phi-3-mini `<|endoftext|>` token.
#define ENDOFTEXT_TOKEN 32000
#define VOCABULARY_SIZE 32064

// TODO(lunwenh): refactor and share with llama
void generate(
    Module& llm_model,
    std::string& prompt,
    std::unique_ptr<Tokenizer> tokenizer,
    size_t max_output_length) {
  // Convert the input text into a list of integers (tokens) that represents
  // it, using the string-to-token mapping that the model was trained on.
  // Each token is an integer that represents a word or part of a word.
  auto encode_res = tokenizer->encode(prompt, 0, 0);
  ET_CHECK_MSG(encode_res.ok(), "Failed to encode prompt %s", prompt.c_str());
  std::vector<uint64_t> input_tokens = encode_res.get();

  std::cout << "Generating tokens ..." << std::endl;

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
    uint64_t next_token =
        std::max_element(logits.begin(), logits.end()) - logits.begin();

    auto decode_res = tokenizer->decode(input_tokens.back(), next_token);
    ET_CHECK_MSG(decode_res.ok(), "Failed to decode token %" PRIu64, next_token);

    std::cout << decode_res.get();
    std::cout.flush();

    // Break if we reached the end of the text.
    if (next_token == ENDOFTEXT_TOKEN) {
      break;
    }

    // Update next input.
    input_tokens.push_back(next_token);
  }

  std::cout << std::endl;
}

int main() {
  // Set up the prompt. This provides the seed text for the model to elaborate.
  std::cout << "Enter model prompt: ";
  std::string prompt;
  std::getline(std::cin, prompt);

  std::unique_ptr<Tokenizer> tokenizer = std::make_unique<BPETokenizer>();
  ET_CHECK_MSG(tokenizer->load("tokenizer.bin") == Error::Ok, "Failed to load tokenizer");

  Module model("phi-3-mini.pte", Module::LoadMode::MmapUseMlockIgnoreErrors);

  const auto max_output_tokens = 128;
  generate(model, prompt, std::move(tokenizer), max_output_tokens);
}
