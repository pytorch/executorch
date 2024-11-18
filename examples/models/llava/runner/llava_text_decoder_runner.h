/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder in Llava and return the output.

#pragma once

#include <executorch/extension/llm/runner/text_decoder_runner.h>

namespace example {

class ET_EXPERIMENTAL LlavaTextDecoderRunner
    : public executorch::extension::llm::TextDecoderRunner {
 public:
  LlavaTextDecoderRunner(
      executorch::extension::Module* module,
      int32_t vocab_size,
      float temperature)
      : TextDecoderRunner(module, true, vocab_size, temperature){};

  inline executorch::runtime::Result<exec_aten::Tensor> step(
      executorch::extension::TensorPtr& tokens,
      executorch::extension::TensorPtr& start_pos) override {
    // run token embedding
    auto token_embedding_outputs =
        ET_UNWRAP(module_->execute(kTokenEmbeddingMethod, tokens));

    // run text model
    auto outputs_res = ET_UNWRAP(module_->execute(
        kTextModelMethod, {start_pos, token_embedding_outputs[0]}));

    ET_CHECK_MSG(
        outputs_res.size() == 1,
        "More then one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    // Return the logits tensor
    return outputs_res[0].toTensor();
  }

  /**
   * Load the Module for text decode purpose.
   * @return The error code.
   */
  inline executorch::runtime::Error load() override {
    if (is_method_loaded()) {
      return executorch::runtime::Error::Ok;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kTokenEmbeddingMethod));
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kTextModelMethod));
    return executorch::runtime::Error::Ok;
  }

  /**
   * Check if the required methods in the Module is loaded.
   * @return True if the Module is loaded, false otherwise.
   */
  inline bool is_method_loaded() override {
    executorch::runtime::Result<std::unordered_set<std::string>> methods_res =
        module_->method_names();
    if (methods_res.error() != executorch::runtime::Error::Ok) {
      ET_CHECK_MSG(false, "Failed to get method names");
    }
    std::unordered_set<std::string> methods = methods_res.get();
    bool methods_exist = methods.find(kTokenEmbeddingMethod) != methods.end() &&
        methods.find(kTextModelMethod) != methods.end();
    if (!methods_exist) {
      for (const auto& method : methods) {
        ET_LOG(Error, "Method: %s", method.c_str());
      }
      ET_CHECK_MSG(
          methods_exist,
          "Missing required methods (%s, %s) in the model",
          kTokenEmbeddingMethod.c_str(),
          kTextModelMethod.c_str());
    }
    bool methods_loaded = module_->is_method_loaded(kTokenEmbeddingMethod) &&
        module_->is_method_loaded(kTextModelMethod);
    return methods_loaded;
  }

  inline static const std::string kTokenEmbeddingMethod = "token_embedding";
  inline static const std::string kTextModelMethod = "text_model";
};

} // namespace example
