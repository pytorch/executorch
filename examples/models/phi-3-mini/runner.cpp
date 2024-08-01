/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/phi-3-mini/runner.h>

#include <ctime>
#include <iostream>

#include <executorch/extension/runner_util/managed_tensor.h>
#include <executorch/runtime/platform/log.h>

namespace torch::executor {

#define SAMPLER_TOP 0.9f
#define ENDOFTEXT_TOKEN 32000
#define VOCABULARY_SIZE 32064

Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const float temperature)
    : module_(std::make_unique<Module>(model_path, Module::LoadMode::File)),
      tokenizer_(std::make_unique<SentencePieceTokenizer>(tokenizer_path)),
      sampler_(std::make_unique<Sampler>(
          VOCABULARY_SIZE,
          temperature,
          SAMPLER_TOP,
          static_cast<unsigned long long>(std::time(nullptr)))) {
  ET_LOG(
      Info,
      "Creating Phi-3-mini runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());
}

void Runner::generate(const std::string& prompt, std::size_t seq_len) {
  std::vector<int64_t> input_tokens = tokenizer_->encode(prompt);
  std::vector<int64_t> output_tokens;

  std::cout << "Prefilling tokens ..." << std::endl;
  for (auto token : input_tokens) {
    std::cout << token << " ";
  }
  std::cout << std::endl;
  std::cout.flush();
  auto current_token = prefill(input_tokens);
  output_tokens.push_back(current_token);

  std::cout << "Generating tokens ..." << std::endl;
  std::cout << current_token << " ";
  std::cout.flush();

  while (current_token != ENDOFTEXT_TOKEN &&
         output_tokens.size() < seq_len - input_tokens.size()) {
    current_token = run_model_step(current_token);
    output_tokens.push_back(current_token);
    std::cout << current_token << " ";
    std::cout.flush();
  }

  std::cout << std::endl;

  std::cout << "Decoding tokens ..." << std::endl;
  std::cout << tokenizer_->decode(output_tokens) << std::endl;
}

int64_t Runner::logits_to_token(const exec_aten::Tensor& logits_tensor) {
  return sampler_->sample(logits_tensor.data_ptr<float>());
}

int64_t Runner::prefill(const std::vector<int64_t>& tokens) {
  int64_t current_token = 0;

  for (auto token : tokens) {
    current_token = run_model_step(token);
  }

  return current_token;
}

int64_t Runner::run_model_step(int64_t token) {
  ManagedTensor input_token(&token, {1, 1}, ScalarType::Long);
  std::vector<EValue> inputs = {input_token.get_aliasing_tensor()};

  auto result = module_->forward(inputs);
  ET_CHECK_MSG(
      result.error() == Error::Ok,
      "Failed to run forward() for token %" PRIu64,
      token);

  return logits_to_token(result.get()[0].toTensor());
}

} // namespace torch::executor
