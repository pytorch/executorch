/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/asr/runner/runner.h>

#include <inttypes.h>
#include <algorithm>
#include <optional>

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::extension::asr {
namespace {

constexpr const char* kEncoderMethodName = "encoder";
constexpr const char* kDecoderMethodName = "text_decoder";
constexpr const char* kSamplerMethodName = "sampler";

} // namespace

AsrRunner::AsrRunner(
    const std::string& module_path,
    std::optional<std::string> data_path,
    const std::string& tokenizer_path)
    : module_path_(module_path),
      data_path_(data_path.value_or("")),
      tokenizer_path_(tokenizer_path) {
  if (data_path_.empty()) {
    module_ = std::make_unique<Module>(module_path_, Module::LoadMode::Mmap);
  } else {
    module_ = std::make_unique<Module>(
        module_path_, data_path_, Module::LoadMode::Mmap);
  }
}

bool AsrRunner::is_loaded() const {
  return module_ && encoder_method_loaded_ && decoder_method_loaded_ &&
      (!sampler_method_present_ || sampler_method_loaded_) && tokenizer_ &&
      tokenizer_->is_loaded() && !eos_token_ids_.empty();
}

Error AsrRunner::load_tokenizer() {
  if (tokenizer_ && tokenizer_->is_loaded()) {
    return Error::Ok;
  }

  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(tokenizer_path_);
  ET_CHECK_OR_RETURN_ERROR(
      tokenizer,
      Internal,
      "Failed to create tokenizer from %s",
      tokenizer_path_.c_str());

  tokenizer_ = std::move(tokenizer);
  if (!tokenizer_->is_loaded()) {
    ET_LOG(
        Error,
        "Tokenizer reported unloaded state after load: %s",
        tokenizer_path_.c_str());
    return Error::Internal;
  }

  eos_token_ids_.clear();
  eos_token_ids_.insert(static_cast<int64_t>(tokenizer_->eos_tok()));
  return Error::Ok;
}

Error AsrRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }

  stats_.model_load_start_ms = ::executorch::extension::llm::time_in_ms();

  ET_CHECK_OR_RETURN_ERROR(
      module_ != nullptr,
      InvalidArgument,
      "Module handle is null for path %s",
      module_path_.c_str());

  ET_CHECK_OK_OR_RETURN_ERROR(module_->load());

  auto method_names_result = module_->method_names();
  ET_CHECK_OK_OR_RETURN_ERROR(method_names_result.error());
  const auto& method_names = method_names_result.get();

  sampler_method_present_ = method_names.count(kSamplerMethodName);

  ET_CHECK_OR_RETURN_ERROR(
      method_names.count(kEncoderMethodName) &&
          method_names.count(kDecoderMethodName),
      InvalidArgument,
      "Required methods not found. encoder=%d, text_decoder=%d",
      static_cast<int>(method_names.count(kEncoderMethodName)),
      static_cast<int>(method_names.count(kDecoderMethodName)));

  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kEncoderMethodName));
  encoder_method_loaded_ = true;

  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kDecoderMethodName));
  decoder_method_loaded_ = true;

  if (sampler_method_present_) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kSamplerMethodName));
    sampler_method_loaded_ = true;
  }
#ifdef CUDA_AVAILABLE
  // Skip copying outputs to CPU. When a sampler exists, keep both encoder and
  // decoder outputs on device and pass decoder logits directly into sampler.
  executorch::runtime::BackendOptions<1> backend_options;
  std::string skip_methods = kEncoderMethodName;
  if (sampler_method_present_) {
    skip_methods.append(",").append(kDecoderMethodName);
  }
  ET_CHECK_OK_OR_RETURN_ERROR(backend_options.set_option(
      "skip_copy_output_to_cpu_for_method", skip_methods.c_str()));
  const auto opt_err =
      executorch::runtime::set_option("CudaBackend", backend_options.view());
  if (opt_err != ::executorch::runtime::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to set CUDA backend options: %d",
        static_cast<int>(opt_err));
  }
#endif
  ET_CHECK_OK_OR_RETURN_ERROR(load_tokenizer());
  auto eos_ids = get_eos_ids(tokenizer_.get(), module_.get());
  if (!eos_ids.empty()) {
    eos_token_ids_.clear();
    for (uint64_t eos_id : eos_ids) {
      eos_token_ids_.insert(static_cast<int64_t>(eos_id));
    }
  }

  stats_.model_load_end_ms = ::executorch::extension::llm::time_in_ms();

  return Error::Ok;
}

Result<std::string> AsrRunner::transcribe(
    ::executorch::extension::TensorPtr preprocessed_features,
    AsrTranscribeConfig config,
    std::optional<TokenCallback> token_callback) {
  ET_CHECK_OR_RETURN_ERROR(
      config.max_new_tokens > 0,
      InvalidArgument,
      "max_new_tokens must be positive, got %" PRId64,
      config.max_new_tokens);

  ET_LOG(
      Info,
      "Preprocessed features shape: [%zu, %zu, %zu]",
      static_cast<size_t>(preprocessed_features->size(0)),
      static_cast<size_t>(preprocessed_features->size(1)),
      static_cast<size_t>(preprocessed_features->size(2)));

  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  ET_LOG(
      Info,
      "RSS after loading model: %f MiB (0 if unsupported)",
      ::executorch::extension::llm::get_rss_bytes() / 1024.0 / 1024.0);

  // Reset internal state and start inference
  stats_.inference_start_ms = ::executorch::extension::llm::time_in_ms();

  const std::unordered_set<int64_t>* eos_tokens = &eos_token_ids();
  if (!config.eos_token_ids.empty()) {
    eos_tokens = &config.eos_token_ids;
  }
  ET_CHECK_OR_RETURN_ERROR(
      !eos_tokens->empty(),
      InvalidArgument,
      "EOS token set must not be empty.");
  ::executorch::extension::llm::Sampler sampler(
      tokenizer_->vocab_size(), config.temperature);

  // Check expected dtype for encoder input
  auto encoder_method_meta_result = module_->method_meta(kEncoderMethodName);
  ET_CHECK_OK_OR_RETURN_ERROR(encoder_method_meta_result.error());
  auto encoder_method_meta = encoder_method_meta_result.get();

  ::executorch::aten::ScalarType expected_dtype =
      ::executorch::aten::ScalarType::Float;
  if (encoder_method_meta.num_inputs() > 0) {
    auto input_meta_result = encoder_method_meta.input_tensor_meta(0);
    if (input_meta_result.error() == ::executorch::runtime::Error::Ok) {
      expected_dtype = input_meta_result.get().scalar_type();
    }
  }

  // Convert preprocessed_features to expected dtype if needed
  if (preprocessed_features->scalar_type() != expected_dtype) {
    if (expected_dtype == ::executorch::aten::ScalarType::BFloat16) {
      ET_LOG(
          Info,
          "Converting audio features from %s to BFloat16. Before converting, first value = %f",
          ::executorch::runtime::toString(preprocessed_features->scalar_type()),
          preprocessed_features->mutable_data_ptr<float>()[0]);
      auto convert_result = ::executorch::extension::llm::convert_to_bfloat16(
          preprocessed_features);
      ET_CHECK_OK_OR_RETURN_ERROR(convert_result.error());
      preprocessed_features = convert_result.get();
      ET_LOG(
          Info,
          "Conversion complete, first value = %f",
          static_cast<float>(
              preprocessed_features->mutable_data_ptr<float>()[0]));
    }
  }

  auto encoder_result =
      module_->execute(kEncoderMethodName, preprocessed_features);
  ET_CHECK_OK_OR_RETURN_ERROR(encoder_result.error());

  stats_.prompt_eval_end_ms = ::executorch::extension::llm::time_in_ms();
  stats_.num_prompt_tokens = 0;

  auto encoder_outputs = std::move(*encoder_result);
  ET_CHECK_OR_RETURN_ERROR(
      encoder_outputs.size() == 1 && encoder_outputs[0].isTensor(),
      Internal,
      "Encoder returned %zu outputs; expected a single tensor.",
      encoder_outputs.size());

  ::executorch::aten::Tensor encoder_output_tensor =
      std::move(encoder_outputs[0]).toTensor();

  ET_LOG(
      Info,
      "Encoder output shape: [%zu, %zu, %zu]",
      static_cast<size_t>(encoder_output_tensor.size(0)),
      static_cast<size_t>(encoder_output_tensor.size(1)),
      static_cast<size_t>(encoder_output_tensor.size(2)));

  auto encoder_output_ptr = std::make_shared<::executorch::aten::Tensor>(
      std::move(encoder_output_tensor));

  std::vector<int64_t> tokens = {config.decoder_start_token_id};
  std::string transcription;

  int64_t input_id = config.decoder_start_token_id;
  int64_t cache_position = 0;
  int64_t generated_tokens = 0;
  bool first_token_generated = false;
  auto decoder_input_ptr = ::executorch::extension::from_blob(
      &input_id,
      {static_cast<::executorch::aten::SizesType>(1),
       static_cast<::executorch::aten::SizesType>(1)},
      ::executorch::aten::ScalarType::Long);

  auto cache_position_ptr = ::executorch::extension::from_blob(
      &cache_position,
      {static_cast<::executorch::aten::SizesType>(1)},
      ::executorch::aten::ScalarType::Long);

  std::vector<::executorch::runtime::EValue> decoder_inputs;
  decoder_inputs.reserve(3);
  decoder_inputs.emplace_back(decoder_input_ptr);
  decoder_inputs.emplace_back(encoder_output_ptr);
  decoder_inputs.emplace_back(cache_position_ptr);
  // Add some green coloring for the first generated token
  // token_callback("\033[1;32m");
  const bool use_sampler_method = sampler_method_loaded_;
  while (generated_tokens < config.max_new_tokens) {
    input_id = tokens.back();
    auto decoder_result = module_->execute(kDecoderMethodName, decoder_inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(decoder_result.error());

    auto decoder_outputs = std::move(*decoder_result);
    ET_CHECK_OR_RETURN_ERROR(
        decoder_outputs.size() == 1 && decoder_outputs[0].isTensor(),
        Internal,
        "Decoder returned %zu outputs; expected a single tensor.",
        decoder_outputs.size());

    int64_t next_token = 0;
    if (!use_sampler_method || config.temperature != 0.0f) {
      ::executorch::aten::Tensor logits_tensor =
          std::move(decoder_outputs[0]).toTensor();
      const int64_t vocab_size = logits_tensor.numel();
      ET_CHECK_OR_RETURN_ERROR(
          vocab_size > 0, Internal, "Decoder logits tensor is empty.");
      next_token =
          static_cast<int64_t>(::executorch::extension::llm::logits_to_token(
              logits_tensor, config.temperature));
    } else {
      auto sampler_result =
          module_->execute(kSamplerMethodName, decoder_outputs);
      ET_CHECK_OK_OR_RETURN_ERROR(sampler_result.error());

      auto sampler_outputs = std::move(*sampler_result);
      ET_CHECK_OR_RETURN_ERROR(
          sampler_outputs.size() == 1 && sampler_outputs[0].isTensor(),
          Internal,
          "Sampler returned %zu outputs; expected a single tensor.",
          sampler_outputs.size());

      ::executorch::aten::Tensor token_tensor =
          std::move(sampler_outputs[0]).toTensor();
      ET_CHECK_OR_RETURN_ERROR(
          token_tensor.numel() > 0,
          Internal,
          "Sampler logits tensor is empty.");
      next_token = token_tensor.mutable_data_ptr<int64_t>()[0];
    }

    if (!first_token_generated) {
      stats_.first_token_ms = ::executorch::extension::llm::time_in_ms();
      first_token_generated = true;
    }

    const int64_t prev_token = input_id;
    tokens.push_back(next_token);
    ++generated_tokens;
    ++cache_position;
    input_id = next_token;

    auto piece_result = tokenizer_->decode(
        static_cast<uint64_t>(prev_token), static_cast<uint64_t>(next_token));
    if (piece_result.ok()) {
      transcription += piece_result.get();
      if (token_callback.has_value()) {
        (*token_callback)(piece_result.get());
      }
    } else {
      ET_LOG(
          Error,
          "Tokenizer failed to decode token pair (%" PRId64 ", %" PRId64
          ") with error %d",
          prev_token,
          next_token,
          static_cast<int>(piece_result.error()));
    }

    if (eos_tokens->count(next_token) > 0) {
      break;
    }
  }
  // Reset coloring
  // token_callback("\033[0m");
  // Update stats and print report
  stats_.num_generated_tokens = generated_tokens;
  stats_.inference_end_ms = ::executorch::extension::llm::time_in_ms();
  printf("\n");
  print_report(stats_);

  return transcription;
}

} // namespace executorch::extension::asr
