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
  // Check if input is image
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
  } else if (input.is_text()) {
    // For text input, we don't need to run the image encoder.
    // Instead, we run the text encoder to get the encoder output.
    auto& text = input.get_text();
    std::vector<uint64_t> tokens =
        ET_UNWRAP_TOKENIZER(tokenizer_->encode(text));
    auto text_tensor = executorch::extension::from_blob(
        tokens.data(),
        {1, static_cast<aten::SizesType>(tokens.size())},
        ::executorch::aten::ScalarType::Long);

    // Run token embedding
    auto token_embedding_outputs =
        ET_UNWRAP(module_->execute(kTokenEmbeddingMethod, text_tensor));

    encoder_output = token_embedding_outputs[0];
  } else {
    ET_LOG(Error, "Unsupported input type");
    // For all other input types (e.g., audio), return error
    return ::executorch::runtime::Error::NotSupported;
  }

  auto outputs_res =
      ET_UNWRAP(text_decoder_runner_->decode(encoder_output, start_pos));

  // Update the start_pos, which is only available inside this function.
  // outputs_res can have only one logits.
  start_pos += encoder_output.toTensor().size(1);

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
