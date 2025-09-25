/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generic encoder prefiller that handles multimodal inputs (text, image and
// audio (to be implemented)) to prefill the KV cache of a multimodal LLM.
// @lint-ignore-every CLANGTIDY facebook-hte-Deprecated

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor.h>

namespace executorch::extension::llm {

MultimodalPrefiller::MultimodalPrefiller(
    Module* module,
    MultimodalDecoderRunner* decoder_runner,
    Tokenizer* tokenizer,
    IOManager* io_manager)
    : module_(module),
      text_decoder_runner_(decoder_runner),
      tokenizer_(tokenizer),
      io_manager_(io_manager) {}

/**
 * Prefill an LLM Module with the given multimodal input.
 * @param input The multimodal input (text, image or audio) to the multimodal
 * LLM.
 * @param start_pos The starting position in KV cache of the input in the LLM
 * @return logits of the prefill.
 */
Result<uint64_t> MultimodalPrefiller::prefill(
    const MultimodalInput& input,
    int64_t& start_pos) {
  // 1. Run encoder model.
  ::executorch::runtime::EValue encoder_output;
  if (input.is_image()) {
    Image image = input.get_image();

    auto method_meta = ET_UNWRAP(
        module_->method_meta(kVisionEncoderMethod),
        "Failed to get method_meta for %s",
        kVisionEncoderMethod);

    ET_CHECK_OR_RETURN_ERROR(
        method_meta.num_inputs() > 0,
        InvalidArgument,
        "Image encoder should have at least 1 input");
    auto input_meta = ET_UNWRAP(
        method_meta.input_tensor_meta(0),
        "Cannot get input tensor meta at index 0");
    auto expected_dtype = input_meta.scalar_type();

    if (expected_dtype == ::executorch::aten::ScalarType::Float) {
      ET_CHECK_OR_RETURN_ERROR(
          image.is_float(),
          InvalidArgument,
          "Model expects float image data, but image has uint8_t data.");
    } else if (expected_dtype == ::executorch::aten::ScalarType::Byte) {
      ET_CHECK_OR_RETURN_ERROR(
          image.is_uint8(),
          InvalidArgument,
          "Model expects uint8_t image data, but image has float data.");
    } else {
      ET_LOG(
          Error,
          "Unsupported image encoder input dtype: %s",
          ::executorch::runtime::toString(expected_dtype));
      return ::executorch::runtime::Error::NotSupported;
    }

    // The model might expect a 4D tensor (NCHW), but toTensor() returns a 3D
    // tensor (CHW). Add a batch dimension of 1 if needed.
    auto expected_dims = input_meta.sizes();
    auto image_tensor = ET_UNWRAP(
        image.toTensor(/*with_batch*/ expected_dims.size() == 4),
        "Failed to convert image to tensor");
    ET_LOG(
        Info,
        "Image tensor dim: %zu, dtype: %s",
        image_tensor->dim(),
        ::executorch::runtime::toString(image_tensor->scalar_type()));
    // Run image encoder
    auto image_encoder_outputs =
        ET_UNWRAP(module_->execute(kVisionEncoderMethod, image_tensor));

    encoder_output = image_encoder_outputs[0];
  } else if (input.is_audio()) {
    Audio audio = input.get_audio();

    // Use Audio::toTensor() for tensor creation
    auto audio_tensor =
        ET_UNWRAP(audio.toTensor(), "Failed to convert audio to tensor");
    ET_LOG(
        Info,
        "Audio tensor dim: %zu, dtype: %s",
        audio_tensor->dim(),
        ::executorch::runtime::toString(audio_tensor->scalar_type()));
    // Run audio encoder
    auto audio_encoder_result =
        module_->execute(kAudioEncoderMethod, audio_tensor);
    if (audio_encoder_result.error() != ::executorch::runtime::Error::Ok) {
      return ::executorch::runtime::Error::Internal;
    }
    auto audio_encoder_outputs = audio_encoder_result.get();

    encoder_output = audio_encoder_outputs[0];
  } else if (input.is_text() || input.is_tokens()) {
    std::vector<uint64_t> tokens;
    if (input.is_text()) {
      auto& text = input.get_text();
      tokens = ET_UNWRAP_TOKENIZER(tokenizer_->encode(text));
    } else {
      tokens = input.get_tokens();
    }

    auto text_tensor = executorch::extension::from_blob(
        tokens.data(),
        {1, static_cast<aten::SizesType>(tokens.size())},
        ::executorch::aten::ScalarType::Long);

    // Run text encoder (token embeddings)
    auto token_embedding_outputs =
        ET_UNWRAP(module_->execute(kTokenEmbeddingMethod, text_tensor));

    encoder_output = token_embedding_outputs[0];
  } else {
    ET_LOG(Error, "Unsupported input type");
    // For any other input types, return error
    return ::executorch::runtime::Error::NotSupported;
  }

  // 2. Run decoder model for prefill.

  // Get expected shape of cache position tensor, which should be the second
  // argument

  int64_t seq_len = encoder_output.toTensor().size(1);
  if (seq_len == 0) {
    ET_LOG(Error, "The encoder returned an empty output.");
    return ::executorch::runtime::Error::InvalidState;
  }
  std::vector<int64_t> cache_positions;

  auto cache_position_tensor = ET_UNWRAP(populate_start_pos_or_cache_position(
      module_, start_pos, cache_positions, seq_len, kTextModelMethod));

  auto prefill_result = module_->execute(
      kTextModelMethod, {encoder_output, cache_position_tensor});
  if (prefill_result.error() != ::executorch::runtime::Error::Ok) {
    return prefill_result.error();
  }
  // Check if prefill_outputs is empty, if it is return error and log that the
  // specified encoder returned empty results when used to prefill decoder.
  auto prefill_outputs = prefill_result.get();
  if (prefill_outputs.empty()) {
    ET_LOG(
        Error, "Encoder returned empty results when used to prefill decoder");
    return ::executorch::runtime::Error::InvalidState;
  }
  auto outputs_res = prefill_outputs[0].toTensor();

  // Update start_pos, tracking the current cache position.
  start_pos += seq_len;

  return static_cast<uint64_t>(
      text_decoder_runner_->logits_to_token(outputs_res));
}

/**
 * Load the Module for encoder prefill purpose.
 * @return The error code.
 */
::executorch::runtime::Error MultimodalPrefiller::load() {
  if (is_method_loaded()) {
    return ::executorch::runtime::Error::Ok;
  }
  // token_embeddings and text_model have to show up in method names.
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kTokenEmbeddingMethod));
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kTextModelMethod));

  std::unordered_set<std::string> methods =
      ET_UNWRAP(module_->method_names(), "Failed to get method names");

  // Load image_encoder method if exists.
  if (methods.find(kVisionEncoderMethod) != methods.end()) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kVisionEncoderMethod));
  }

  if (methods.find(kAudioEncoderMethod) != methods.end()) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kAudioEncoderMethod));
  }

  return ::executorch::runtime::Error::Ok;
}

/**
 * Check if the required methods in the Module is loaded.
 * @return True if the Module is loaded, false otherwise.
 */
bool MultimodalPrefiller::is_method_loaded() {
  ::executorch::runtime::Result<std::unordered_set<std::string>> methods_res =
      module_->method_names();
  if (!module_->is_method_loaded(kTokenEmbeddingMethod)) {
    return false;
  }
  if (!module_->is_method_loaded(kTextModelMethod)) {
    return false;
  }
  if (methods_res.error() != ::executorch::runtime::Error::Ok) {
    ET_CHECK_MSG(false, "Failed to get method names");
  }
  std::unordered_set<std::string> methods = methods_res.get();
  if (methods.find(kVisionEncoderMethod) != methods.end()) {
    return module_->is_method_loaded(kVisionEncoderMethod);
  }
  return true;
}

} // namespace executorch::extension::llm
