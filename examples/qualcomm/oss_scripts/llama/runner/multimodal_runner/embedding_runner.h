/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <string>
#include <vector>

namespace example {

/**
 * @class EmbeddingRunner
 * @brief Class for running embedding module, similar to DecoderRunner
 */
class EmbeddingRunner {
 public:
  EmbeddingRunner(executorch::extension::Module* module);

  /**
   * Run embedding module with inputs to generate embeddings.
   * @param method_name The method name to execute.
   * @param inputs The inputs to the embedding module.
   * @return The output embeddings tensor.
   */
  executorch::runtime::Result<executorch::aten::Tensor> step(
      const std::string& method_name,
      std::vector<executorch::runtime::EValue>& inputs);

  executorch::runtime::Error set_outputs(
      const std::string& method_name,
      std::vector<executorch::aten::Tensor> output_values);

  /**
   * Load the Module for token embedding.
   * @return The error code.
   */
  executorch::runtime::Error load(const std::vector<std::string>& method_names);

  /**
   * Check if the required methods in the Module are loaded.
   * @return True if the Module is loaded, false otherwise.
   */
  bool is_method_loaded(const std::vector<std::string>& method_names);

  /**
   * @brief Check if embedding module is loaded
   * @return true if module is loaded, false otherwise
   */
  bool is_loaded() const;
  executorch::extension::Module* module_;
};

} // namespace example
