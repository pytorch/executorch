/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given a image tensor, prefill the KV cache of LLaVA.

#pragma once

#include <executorch/extension/llm/runner/image_prefiller.h>
#include <executorch/extension/runner_util/managed_tensor.h>

namespace torch::executor {

class LlavaImagePrefiller : public ImagePrefiller {
 public:
  LlavaImagePrefiller(Module* module) : ImagePrefiller(module){};
  /**
   * Prefill an LLM Module with the given image input.
   * @param image The image input to LLaVa.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * @return logits of the image prefill.
   */
  inline Result<exec_aten::Tensor> prefill(
      Image& image,
      int64_t start_pos = 0) {
    ManagedTensor managed_images(
        image.data.data(), {3, image.height, image.width}, ScalarType::Byte);
    // Run image encoder
    std::vector<EValue> image_encoder_outputs = ET_UNWRAP(module_->execute(
        kImageEncoderMethod, {managed_images.get_aliasing_tensor()}));

    // inputs:[start_pos, embeds]
    ManagedTensor managed_start_pos(&start_pos, {1}, ScalarType::Long);
    auto start_pos_tensor = managed_start_pos.get_aliasing_tensor();

    // Run text model
    std::vector<EValue> outputs_res = ET_UNWRAP(module_->execute(
        kTextModelMethod, {start_pos_tensor, image_encoder_outputs[0]}));
    ET_CHECK_MSG(
        outputs_res[0].isTensor(),
        "Non Tensor Output returned from executing image prefill");

    return outputs_res[0].toTensor();
  }

  /**
   * Load the Module for image prefill purpose.
   * @return The error code.
   */
  inline Error load() override {
    if (is_method_loaded()) {
      return Error::Ok;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kImageEncoderMethod));
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kTextModelMethod));
    return Error::Ok;
  }

  /**
   * Check if the required methods in the Module is loaded.
   * @return True if the Module is loaded, false otherwise.
   */
  inline bool is_method_loaded() override {
    Result<std::unordered_set<std::string>> methods_res =
        module_->method_names();
    if (methods_res.error() != Error::Ok) {
      ET_CHECK_MSG(false, "Failed to get method names");
    }
    std::unordered_set<std::string> methods = methods_res.get();
    bool methods_exist = methods.find(kImageEncoderMethod) != methods.end() &&
        methods.find(kTextModelMethod) != methods.end();
    if (!methods_exist) {
      for (const auto& method : methods) {
        ET_LOG(Error, "Method: %s", method.c_str());
      }
      ET_CHECK_MSG(
          methods_exist,
          "Missing required methods (%s, %s) in the model",
          kImageEncoderMethod.c_str(),
          kTextModelMethod.c_str());
    }
    bool methods_loaded = module_->is_method_loaded(kImageEncoderMethod) &&
        module_->is_method_loaded(kTextModelMethod);
    return methods_loaded;
  }

  inline static const std::string kImageEncoderMethod = "image_encoder";
  inline static const std::string kTextModelMethod = "text_model";
};

} // namespace torch::executor
