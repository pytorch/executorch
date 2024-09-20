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

#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>

using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::extension::llm::BPETokenizer;
using executorch::extension::llm::Sampler;
using executorch::runtime::Error;

namespace example {

#define SAMPLER_TOP 0.9f
#define ENDOFTEXT_TOKEN 32000
#define VOCABULARY_SIZE 32064

Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const float temperature)
    : module_(std::make_unique<Module>(model_path, Module::LoadMode::File)),
      tokenizer_(std::make_unique<BPETokenizer>()),
      sampler_(std::make_unique<Sampler>(
          VOCABULARY_SIZE,
          temperature,
          SAMPLER_TOP,
          static_cast<unsigned long long>(std::time(nullptr)))) {
  ET_CHECK_MSG(
      tokenizer_->load(tokenizer_path) == Error::Ok,
      "Failed to load tokenizer at %s",
      tokenizer_path.c_str());
  ET_LOG(
      Info,
      "Created Phi-3-mini runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());
}

void Runner::generate(const std::string& prompt, std::size_t max_seq_len) {
  auto encode_res = tokenizer_->encode(prompt, 0, 0);
  ET_CHECK_MSG(
      encode_res.error() == Error::Ok, "Failed to encode %s", prompt.c_str());
  auto input_tokens = encode_res.get();

  std::cout << "Prefilling tokens ..." << std::endl;
  for (auto token : input_tokens) {
    std::cout << token << " ";
  }
  std::cout << std::endl;
  std::cout.flush();
  auto prev_token = input_tokens.back();
  auto current_token = prefill(input_tokens);

  std::cout << "Generating tokens ..." << std::endl;
  std::cout << tokenizer_->decode(prev_token, current_token).get();
  std::cout.flush();

  std::size_t seq_len = input_tokens.size() + 1;

  while (current_token != ENDOFTEXT_TOKEN && seq_len < max_seq_len) {
    prev_token = current_token;
    current_token = run_model_step(current_token);
    std::cout << tokenizer_->decode(prev_token, current_token).get();
    std::cout.flush();

    ++seq_len;
  }

  std::cout << std::endl;
}

uint64_t Runner::logits_to_token(const exec_aten::Tensor& logits_tensor) {
  return sampler_->sample(logits_tensor.data_ptr<float>());
}

uint64_t Runner::prefill(std::vector<uint64_t>& tokens) {
  auto result = module_->forward(executorch::extension::from_blob(
      tokens.data(),
      {1, static_cast<exec_aten::SizesType>(tokens.size())},
      ScalarType::Long));
  ET_CHECK_MSG(result.error() == Error::Ok, "Failed to prefill tokens");

  return logits_to_token(result.get()[0].toTensor());
}

uint64_t Runner::run_model_step(uint64_t token) {
  auto result = module_->forward(
      executorch::extension::from_blob(&token, {1, 1}, ScalarType::Long));
  ET_CHECK_MSG(
      result.error() == Error::Ok,
      "Failed to run forward() for token %" PRIu64,
      token);

  return logits_to_token(result.get()[0].toTensor());
}

} // namespace example
