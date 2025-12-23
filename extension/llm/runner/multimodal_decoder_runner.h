/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>

namespace executorch::extension::llm {

class ET_EXPERIMENTAL MultimodalDecoderRunner
    : public executorch::extension::llm::TextDecoderRunner {
 public:
  explicit MultimodalDecoderRunner(Module* module, IOManager* io_manager)
      : TextDecoderRunner(module, io_manager) {}

  /**
   * Step the LLM Decoder with the given tokens and start position.
   * @param tokens The tokens to the LLM.
   * @param start_pos The start position of the tokens.
   * @return The logits tensor.
   */
  inline executorch::runtime::Result<executorch::aten::Tensor> step(
      executorch::extension::TensorPtr& tokens,
      int64_t start_pos) override {
    // run token embedding
    auto token_embedding_result =
        module_->execute(kTokenEmbeddingMethod, tokens);
    if (!token_embedding_result.ok()) {
      return token_embedding_result.error();
    }
    auto token_embedding_outputs = std::move(*token_embedding_result);

    // Return the logits tensor
    return decode(token_embedding_outputs[0], start_pos);
  }

  /**
   * Decode the embeddings to logits.
   * @param embeddings The embeddings tensor.
   * @param start_pos The start position of the embeddings.
   * @return The logits tensor.
   */
  inline executorch::runtime::Result<executorch::aten::Tensor> decode(
      const runtime::EValue& embeddings,
      int64_t start_pos) {
    auto start_pos_tensor = ::executorch::extension::from_blob(
        &start_pos, {1}, executorch::aten::ScalarType::Long);
    // run text model
    auto outputs_result =
        module_->execute(kTextModelMethod, {embeddings, start_pos_tensor});
    if (!outputs_result.ok()) {
      return outputs_result.error();
    }
    auto outputs_res = std::move(*outputs_result);

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
          kTokenEmbeddingMethod,
          kTextModelMethod);
    }
    bool methods_loaded = module_->is_method_loaded(kTokenEmbeddingMethod) &&
        module_->is_method_loaded(kTextModelMethod);
    return methods_loaded;
  }
};

} // namespace executorch::extension::llm
