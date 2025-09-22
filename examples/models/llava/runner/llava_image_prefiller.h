/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given a image tensor, prefill the KV cache of LLaVA.

#pragma once

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/image_prefiller.h>
#include <executorch/extension/tensor/tensor.h>

namespace example {

using executorch::extension::llm::kTextModelMethod;
using executorch::extension::llm::kVisionEncoderMethod;

class ET_EXPERIMENTAL LlavaImagePrefiller {
 public:
  explicit LlavaImagePrefiller(::executorch::extension::Module* module)
      : module_(module) {}

  /**
   * Prefill an LLM Module with the given image input.
   * @param image The image input to LLaVa.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * @return logits of the image prefill.
   */
  inline ::executorch::runtime::Result<executorch::aten::Tensor> prefill(
      ::executorch::extension::llm::Image& image,
      int64_t& start_pos) {
    auto image_tensor = executorch::extension::from_blob(
        image.data.data(),
        {3, image.height, image.width},
        ::executorch::aten::ScalarType::Byte);
    // Run image encoder
    auto image_encoder_outputs =
        ET_UNWRAP(module_->execute(kVisionEncoderMethod, image_tensor));

    // inputs:[start_pos, embeds]
    auto start_pos_tensor = executorch::extension::from_blob(
        &start_pos, {1}, ::executorch::aten::ScalarType::Long);

    // Run text model
    auto outputs_res = ET_UNWRAP(module_->execute(
        kTextModelMethod, {image_encoder_outputs[0], start_pos_tensor}));
    ET_CHECK_MSG(
        outputs_res[0].isTensor(),
        "Non Tensor Output returned from executing image prefill");

    // Update the start_pos, which is only available inside this function.
    // outputs_res can have only one logits.
    start_pos += image_encoder_outputs[0].toTensor().size(1);

    return outputs_res[0].toTensor();
  }

  /**
   * Load the Module for image prefill purpose.
   * @return The error code.
   */
  inline ::executorch::runtime::Error load() {
    if (is_method_loaded()) {
      return ::executorch::runtime::Error::Ok;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kVisionEncoderMethod));
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kTextModelMethod));
    return ::executorch::runtime::Error::Ok;
  }

  /**
   * Check if the required methods in the Module is loaded.
   * @return True if the Module is loaded, false otherwise.
   */
  inline bool is_method_loaded() {
    ::executorch::runtime::Result<std::unordered_set<std::string>> methods_res =
        module_->method_names();
    if (methods_res.error() != ::executorch::runtime::Error::Ok) {
      ET_CHECK_MSG(false, "Failed to get method names");
    }
    std::unordered_set<std::string> methods = methods_res.get();
    bool methods_exist = methods.find(kVisionEncoderMethod) != methods.end() &&
        methods.find(kTextModelMethod) != methods.end();
    if (!methods_exist) {
      for (const auto& method : methods) {
        ET_LOG(Error, "Method: %s", method.c_str());
      }
      ET_CHECK_MSG(
          methods_exist,
          "Missing required methods (%s, %s) in the model",
          kVisionEncoderMethod,
          kTextModelMethod);
    }
    bool methods_loaded = module_->is_method_loaded(kVisionEncoderMethod) &&
        module_->is_method_loaded(kTextModelMethod);
    return methods_loaded;
  }

 private:
  ::executorch::extension::Module* module_;
};

} // namespace example
