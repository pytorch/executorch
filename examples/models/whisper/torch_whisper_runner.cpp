/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torch_whisper_runner.h"

#include <inttypes.h>

#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::examples::whisper {
namespace {

using ::executorch::extension::asr::AsrTranscribeConfig;
using ::executorch::extension::llm::Sampler;
using ::executorch::extension::llm::time_in_ms;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

torch::Tensor ensure_contiguous_cpu(torch::Tensor tensor) {
  auto target = tensor;
  if (!target.is_cpu()) {
    target = target.to(torch::kCPU);
  }
  if (!target.is_contiguous()) {
    target = target.contiguous();
  }
  return target;
}

torch::Tensor maybe_to_float(torch::Tensor tensor) {
  if (tensor.scalar_type() == torch::kFloat32) {
    return tensor;
  }
  return tensor.to(torch::kFloat32);
}

} // namespace

TorchWhisperRunner::TorchWhisperRunner(
    std::string encoder_library,
    std::string decoder_library,
    std::string tokenizer_path,
    std::optional<std::string> encoder_weights,
    std::optional<std::string> decoder_weights,
    std::string device)
    : encoder_library_(std::move(encoder_library)),
      decoder_library_(std::move(decoder_library)),
      tokenizer_path_(std::move(tokenizer_path)),
      encoder_weights_(std::move(encoder_weights)),
      decoder_weights_(std::move(decoder_weights)),
      device_(std::move(device)) {}

Error TorchWhisperRunner::load_tokenizer() {
  if (tokenizer_ && tokenizer_->is_loaded()) {
    return Error::Ok;
  }
  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(tokenizer_path_);
  ET_CHECK_OR_RETURN_ERROR(
      tokenizer,
      Error::Internal,
      "Failed to load tokenizer from %s",
      tokenizer_path_.c_str());
  tokenizer_ = std::move(tokenizer);
  eos_token_ids_.clear();
  eos_token_ids_.insert(static_cast<int64_t>(tokenizer_->eos_tok()));
  return Error::Ok;
}

bool TorchWhisperRunner::is_loaded() const {
  return encoder_model_ && decoder_model_ && tokenizer_ &&
      tokenizer_->is_loaded();
}

Error TorchWhisperRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  stats_.model_load_start_ms = time_in_ms();
  encoder_model_ = std::make_unique<TorchAOTIModel>(
      encoder_library_, encoder_weights_, device_);
  decoder_model_ = std::make_unique<TorchAOTIModel>(
      decoder_library_, decoder_weights_, device_);
  ET_CHECK_OK_OR_RETURN_ERROR(encoder_model_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(decoder_model_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(load_tokenizer());
  stats_.model_load_end_ms = time_in_ms();
  return Error::Ok;
}

torch::Tensor TorchWhisperRunner::maybe_cast_encoder_input(
    const torch::Tensor& tensor) {
  auto prepared = ensure_contiguous_cpu(tensor);
  // Whisper encoder AOTI expects float/bfloat16. Default to float.
  if (prepared.scalar_type() != torch::kFloat32 &&
      prepared.scalar_type() != torch::kBFloat16) {
    prepared = prepared.to(torch::kFloat32);
  }
  return prepared;
}

torch::Tensor TorchWhisperRunner::build_decoder_input(int64_t token_id) {
  auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  return torch::full({1, 1}, token_id, options);
}

torch::Tensor TorchWhisperRunner::build_cache_position_tensor(
    int64_t cache_position) {
  auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  return torch::full({1}, cache_position, options);
}

Result<std::vector<int64_t>> TorchWhisperRunner::transcribe(
    const torch::Tensor& preprocessed_features,
    AsrTranscribeConfig config,
    std::function<void(const std::string&)> token_callback) {
  ET_CHECK_OR_RETURN_ERROR(
      preprocessed_features.defined(), Error::InvalidArgument, "Input is empty");
  ET_CHECK_OR_RETURN_ERROR(
      config.max_new_tokens > 0,
      Error::InvalidArgument,
      "max_new_tokens must be > 0");

  ET_CHECK_OK_OR_RETURN_ERROR(load());

  stats_.inference_start_ms = time_in_ms();

  torch::Tensor encoder_input = maybe_cast_encoder_input(preprocessed_features);
  auto encoder_outputs = encoder_model_->run({encoder_input});
  ET_CHECK_OK_OR_RETURN_ERROR(
      encoder_outputs.error(),
      "Encoder execution failed with error %d",
      static_cast<int>(encoder_outputs.error()));

  auto outputs = std::move(encoder_outputs).get();
  ET_CHECK_OR_RETURN_ERROR(
      outputs.size() == 1,
      Error::Internal,
      "Encoder returned %zu outputs",
      outputs.size());
  torch::Tensor encoder_output = ensure_contiguous_cpu(outputs[0]);

  stats_.prompt_eval_end_ms = time_in_ms();
  stats_.num_prompt_tokens = 0;

  std::vector<int64_t> tokens = {config.decoder_start_token_id};
  int64_t cache_position = 0;
  int64_t generated_tokens = 0;
  bool first_token_emitted = false;
  torch::Tensor decoder_input = build_decoder_input(tokens.back());
  torch::Tensor cache_position_tensor =
      build_cache_position_tensor(cache_position);

  Sampler sampler(tokenizer_->vocab_size(), config.temperature);

  const std::unordered_set<int64_t>* eos_tokens = &eos_token_ids_;
  if (!config.eos_token_ids.empty()) {
    eos_tokens = &config.eos_token_ids;
  }
  ET_CHECK_OR_RETURN_ERROR(
      !eos_tokens->empty(),
      Error::InvalidArgument,
      "EOS token list must not be empty");

  while (generated_tokens < config.max_new_tokens) {
    decoder_input.fill_(tokens.back());
    cache_position_tensor.fill_(cache_position);

    auto decoder_result = decoder_model_->run(
        {decoder_input, encoder_output, cache_position_tensor});
    ET_CHECK_OK_OR_RETURN_ERROR(decoder_result.error());
    auto decoder_outputs = std::move(decoder_result).get();
    ET_CHECK_OR_RETURN_ERROR(
        decoder_outputs.size() == 1,
        Error::Internal,
        "Decoder returned %zu outputs",
        decoder_outputs.size());
    auto logits_tensor = ensure_contiguous_cpu(decoder_outputs[0]);
    logits_tensor = maybe_to_float(logits_tensor);
    const int64_t vocab_size = logits_tensor.numel();
    ET_CHECK_OR_RETURN_ERROR(
        vocab_size > 0, Error::Internal, "Decoder logits tensor empty");

    int64_t next_token =
        sampler.sample(logits_tensor.data_ptr<float>());

    if (!first_token_emitted) {
      stats_.first_token_ms = time_in_ms();
      first_token_emitted = true;
    }

    int64_t prev_token = tokens.back();
    tokens.push_back(next_token);
    ++generated_tokens;
    ++cache_position;

    if (token_callback) {
      auto piece = tokenizer_->decode(
          static_cast<uint64_t>(prev_token),
          static_cast<uint64_t>(next_token));
      if (piece.ok()) {
        token_callback(piece.get());
      } else {
        ET_LOG(
            Error,
            "Tokenizer failed to decode pair (%" PRId64 ", %" PRId64 ")",
            prev_token,
            next_token);
      }
    }

    if (eos_tokens->count(next_token) > 0) {
      break;
    }
  }

  stats_.num_generated_tokens = generated_tokens;
  stats_.inference_end_ms = time_in_ms();
  ::executorch::extension::llm::print_report(stats_);

  return tokens;
}

} // namespace executorch::examples::whisper
