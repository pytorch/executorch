/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/whisper/runner/runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
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
static constexpr auto kDecoderStartTokenId = "decoder_start_token_id";
static constexpr auto kEosId = "get_eos_id";
static constexpr auto kMaxContextLen = "get_max_context_len";
} // namespace
WhisperRunner::WhisperRunner(
    const std::string& model_path,
    const std::string& tokenizer_json_path)
    : tokenizer_json_path_(tokenizer_json_path) {
  encoder_ = std::make_unique<WhisperEncoder>(model_path);
  decoder_ = std::make_unique<WhisperDecoder>(model_path);
  tokenizer_ = std::make_unique<tokenizers::HFTokenizer>();
}
bool WhisperRunner::is_loaded() const {
  return encoder_->is_method_loaded() && decoder_->is_method_loaded() &&
      tokenizer_->is_loaded() && sampler_;
}

Error WhisperRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(encoder_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(decoder_->load());
  if (tokenizer_->load(tokenizer_json_path_) != tokenizers::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to load tokenizer with %s",
        tokenizer_json_path_.c_str());
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
      {kDecoderStartTokenId, 50258},
      {kMaxContextLen, 128},
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
uint64_t WhisperRunner::logits_to_token(
    const executorch::aten::Tensor& logits_tensor) {
  return sampler_->sample(logits_tensor.data_ptr<float>());
}
/**
 * @param inputs: A vector containing one element: a vector of bytes that
 * encodes a float tensor in little-endian byte order.
 *
 */
Error WhisperRunner::transcribe(
    int32_t seq_len,
    std::vector<std::vector<char>>& inputs,
    std::function<void(const std::string&)> token_callback) {
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  ET_CHECK_MSG(inputs.size() == 1, "The input size of whisper should be one.");

  ET_LOG(Info, "Start Encoding");
  stats_.encoder_inference_start_ms = time_in_ms();
  auto input_features_tensor_ptr = from_blob(
      inputs[0].data(),
      // (1, processor.feature_extractor.feature_size,
      // processor.feature_extractor.nb_max_frames)
      {1, 80, 3000},
      ScalarType::Float);
  Result<Tensor> encoder_out = encoder_->encode(input_features_tensor_ptr);
  auto encoder_out_tensor_ptr = make_tensor_ptr(encoder_out.get());
  stats_.encoder_inference_end_ms = time_in_ms();
  auto max_seq_len = metadata_.at(kMaxContextLen);

  seq_len = (seq_len > 0 && seq_len <= max_seq_len) ? seq_len : max_seq_len;

  int64_t pos = 0;
  num_generated_token_ = 0;
  uint64_t prev_token = metadata_.at(kDecoderStartTokenId),
           cur_token = prev_token;
  ET_LOG(Info, "Start Decoding");
  std::vector<float> attention_mask_data(max_seq_len, -255.0);
  stats_.decoder_inference_start_ms = time_in_ms();
  while (pos < seq_len) {
    attention_mask_data[pos] = 0;
    auto decoder_input_ids_tensor_ptr =
        from_blob(&cur_token, {1, 1}, ScalarType::Long);
    auto pos_tensor_ptr = from_blob(&pos, {1}, ScalarType::Long);

    auto attention_mask_tensor_ptr = from_blob(
        attention_mask_data.data(),
        {1, 1, 1, static_cast<int>(max_seq_len)},
        ScalarType::Float);
    Result<Tensor> logits = decoder_->step(
        decoder_input_ids_tensor_ptr,
        attention_mask_tensor_ptr,
        encoder_out_tensor_ptr,
        pos_tensor_ptr);

    prev_token = cur_token;
    cur_token = logits_to_token(logits.get());
    ++pos;

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

Error WhisperRunner::print_performance() {
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
