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
    auto image_tensor = executorch::extension::from_blob(
        image.data.data(),
        {3, image.height, image.width},
        ::executorch::aten::ScalarType::Byte);

    // Run image encoder
    auto image_encoder_outputs =
        ET_UNWRAP(module_->execute(kImageEncoderMethod, image_tensor));

    encoder_output = image_encoder_outputs[0];
  } else if (input.is_audio()) {
    Audio audio = input.get_audio();

    // Use the original tensor shape as intended
    auto audio_tensor = executorch::extension::from_blob(
        audio.data.data(),
        {audio.batch_size, audio.n_bins, audio.n_frames},
        ::executorch::aten::ScalarType::Float);

    // Run audio encoder
    auto audio_encoder_result =
        module_->execute(kAudioEncoderMethod, audio_tensor);
    if (audio_encoder_result.error() != ::executorch::runtime::Error::Ok) {
      return ::executorch::runtime::Error::Internal;
    }
    auto audio_encoder_outputs = audio_encoder_result.get();

    encoder_output = audio_encoder_outputs[0];
  } else if (input.is_text()) {
    auto& text = input.get_text();
    std::vector<uint64_t> tokens =
        ET_UNWRAP_TOKENIZER(tokenizer_->encode(text));

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

  // Get expected shape of cache position tensor, which should be the first (0th
  // index) argument
  auto method_meta = ET_UNWRAP(module_->method_meta(kTextModelMethod));
  auto first_input_info = ET_UNWRAP(method_meta.input_tensor_meta(0));
  auto first_input_sizes = first_input_info.sizes();
  auto numel = first_input_sizes[0];

  int64_t seq_len = encoder_output.toTensor().size(1);
  if (seq_len == 0) {
    ET_LOG(Error, "The encoder returned an empty output.");
    return ::executorch::runtime::Error::InvalidState;
  }

  executorch::extension::TensorPtr cache_position_tensor;
  if (numel > 1) {
    // `cache_position` goes from start_pos to start_pos +
    // encoder_output.size(1). e.g. if start_pos = 2 and encoder_output.size(1)
    // = 5, cache_position_tensor should be [2, 3, 4, 5, 6].
    std::vector<int64_t> cache_positions(seq_len);
    for (int64_t i = 0; i < seq_len; ++i) {
      cache_positions[i] = start_pos + i;
    }
    cache_position_tensor = ::executorch::extension::from_blob(
        cache_positions.data(),
        {static_cast<int>(seq_len)},
        executorch::aten::ScalarType::Long);
  } else {
    // Cache position is size 1.
    cache_position_tensor = ::executorch::extension::from_blob(
        &start_pos, {1}, executorch::aten::ScalarType::Long);
  }
  auto prefill_result = module_->execute(
      kTextModelMethod, {cache_position_tensor, encoder_output});
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
  if (methods.find(kImageEncoderMethod) != methods.end()) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kImageEncoderMethod));
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
  if (methods.find(kImageEncoderMethod) != methods.end()) {
    return module_->is_method_loaded(kImageEncoderMethod);
  }
  return true;
}

} // namespace executorch::extension::llm
