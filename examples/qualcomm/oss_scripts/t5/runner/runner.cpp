/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/t5/runner/runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <ctime>
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::from_blob;
using executorch::extension::make_tensor_ptr;
using executorch::extension::llm::Sampler;
using executorch::extension::llm::time_in_ms;
using executorch::llm::kTopp;
using executorch::runtime::Error;
using executorch::runtime::Result;

namespace example {
namespace {
static constexpr auto kEosId = "get_eos_id";
static constexpr auto kMaxContextLen = "get_max_context_len";
static constexpr auto kMaxHiddenSeqLen = "max_hidden_seq_length";
} // namespace
Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_model_path)
    : tokenizer_model_path_(tokenizer_model_path) {
  encoder_ = std::make_unique<T5Encoder>(model_path);
  decoder_ = std::make_unique<T5Decoder>(model_path);
  tokenizer_ = std::make_unique<tokenizers::SPTokenizer>();
}

bool Runner::is_loaded() const {
  return encoder_->is_method_loaded() && decoder_->is_method_loaded() &&
      tokenizer_->is_loaded() && sampler_;
}

Error Runner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(encoder_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(decoder_->load());
  if (tokenizer_->load(tokenizer_model_path_) != tokenizers::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to load tokenizer with %s",
        tokenizer_model_path_.c_str());
    return Error::Internal;
  }
  eos_ids_ = std::make_unique<std::unordered_set<uint64_t>>(
      std::unordered_set<uint64_t>{tokenizer_->eos_tok()});

  // create sampler
  sampler_ = std::make_unique<Sampler>(
      tokenizer_->vocab_size(),
      0,
      kTopp,
      static_cast<unsigned long long>(std::time(nullptr)));

  // Initialize metadata with default values
  metadata_ = {
      {kMaxContextLen, 128},
      {kMaxHiddenSeqLen, 384},
  };

  // Read metadata from the model
  auto method_names_result = decoder_->method_names();
  if (method_names_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed reading method names");
    return Error::Internal;
  }
  const auto method_names = method_names_result.get();

  for (auto& [method_name, value] : metadata_) {
    if (method_names.count(method_name)) {
      auto get_result = decoder_->get(method_name);

      auto result = get_result.get();
      value =
          get_result.get().toScalar().to<decltype(metadata_)::mapped_type>();
    } else {
      ET_LOG(
          Info,
          "Method %s not found, using the default value %" PRId64,
          method_name.c_str(),
          value);
    }
    ET_LOG(Info, "Metadata: %s = %" PRId64, method_name.c_str(), value);
  }

  // Get EOS IDs if available
  if (method_names.count(kEosId)) {
    eos_ids_->clear();
    auto execute_result = decoder_->execute(kEosId);
    if (execute_result.error() != Error::Ok) {
      ET_LOG(Error, "Failed to execute %s", kEosId);
      return Error::Internal;
    }
    for (const auto& eos_id : execute_result.get()) {
      auto value = eos_id.toScalar().to<int64_t>();
      eos_ids_->emplace(value);
      ET_LOG(Info, "eos_id = %" PRId64, value);
    }
  }

  return Error::Ok;
}

uint64_t Runner::logits_to_token(
    const executorch::aten::Tensor& logits_tensor) {
  return sampler_->sample(logits_tensor.data_ptr<float>());
}

Error Runner::generate(
    int32_t seq_len,
    std::vector<std::vector<uint8_t>>& inputs,
    std::function<void(const std::string&)> token_callback) {
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  ET_CHECK_MSG(inputs.size() == 3, "The input size of t5 should be three.");

  ET_LOG(Info, "Start Encoding");
  stats_.encoder_inference_start_ms = time_in_ms();
  auto hidden_seq_len = static_cast<int>(metadata_.at(kMaxHiddenSeqLen));
  executorch::extension::TensorPtr prompt_tokens =
      from_blob(inputs[0].data(), {1, hidden_seq_len}, ScalarType::Long);
  executorch::extension::TensorPtr prompt_attn_mask =
      from_blob(inputs[1].data(), {1, 1, 1, hidden_seq_len}, ScalarType::Float);

  auto encoder_output = encoder_->encode(prompt_tokens, prompt_attn_mask);

  ET_CHECK_OK_OR_RETURN_ERROR(encoder_output.error());
  auto encoder_hidden_states_tensor_ptr = make_tensor_ptr(encoder_output.get());
  stats_.encoder_inference_end_ms = time_in_ms();
  auto max_seq_len = metadata_.at(kMaxContextLen);

  seq_len = (seq_len > 0 && seq_len <= max_seq_len) ? seq_len : max_seq_len;

  int64_t pos = 0;
  num_generated_token_ = 0;

  // use decoder_input_id as first token
  ET_CHECK_MSG(!inputs[2].empty(), "decoder_input_ids is empty.");
  uint64_t prev_token = inputs[2][0], cur_token = prev_token;

  ET_LOG(Info, "Start Decoding");
  std::vector<int64_t> output_token_ids;
  std::vector<float> attention_mask_data(max_seq_len, -255.0);
  stats_.decoder_inference_start_ms = time_in_ms();
  while (pos < seq_len) {
    auto decoder_input_ids_tensor_ptr =
        from_blob(&cur_token, {1, 1}, ScalarType::Long);
    attention_mask_data[pos] = 0;
    auto attention_mask_tensor_ptr = from_blob(
        attention_mask_data.data(),
        {1, 1, 1, static_cast<int>(max_seq_len)},
        ScalarType::Float);
    auto pos_tensor_ptr = from_blob(&pos, {1}, ScalarType::Long);
    Result<Tensor> logits = decoder_->step(
        decoder_input_ids_tensor_ptr,
        attention_mask_tensor_ptr,
        encoder_hidden_states_tensor_ptr,
        prompt_attn_mask,
        pos_tensor_ptr);

    prev_token = cur_token;
    cur_token = logits_to_token(logits.get());
    ++pos;
    output_token_ids.push_back(cur_token);

    if (token_callback) {
      token_callback(
          ET_UNWRAP_TOKENIZER(tokenizer_->decode(prev_token, cur_token)));
    }
    if (eos_ids_->count(cur_token) > 0) {
      ET_LOG(Info, "\nReached to the end of generation");
      break;
    }
  }
  stats_.decoder_inference_end_ms = time_in_ms();
  if (pos == seq_len) {
    ET_LOG(Info, "\nSequence length (%i tokens) reached!", seq_len);
  }
  num_generated_token_ = pos;
  print_performance();
  return Error::Ok;
}

Error Runner::print_performance() {
  ET_LOG(Info, "\tTotal Generated token:\t\t\t\t%ld", num_generated_token_);

  ET_LOG(
      Info,
      "\tModel Load Time:\t\t\t\t%f (seconds)",
      ((double)(stats_.model_load_end_ms - stats_.model_load_start_ms) /
       stats_.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tEncoding Time:\t\t\t\t\t%f (seconds)",
      ((double)(stats_.encoder_inference_end_ms -
                stats_.encoder_inference_start_ms) /
       stats_.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tDecoding Time:\t\t\t%f (seconds)",
      ((double)(stats_.decoder_inference_end_ms -
                stats_.decoder_inference_start_ms) /
       stats_.SCALING_FACTOR_UNITS_PER_SECOND));

  ET_LOG(
      Info,
      "\tAverage Decoding Time:\t\t\t%f (seconds)",
      ((double)((stats_.decoder_inference_end_ms -
                 stats_.decoder_inference_start_ms) /
                num_generated_token_) /
       (stats_.SCALING_FACTOR_UNITS_PER_SECOND)));

  return Error::Ok;
}

} // namespace example
