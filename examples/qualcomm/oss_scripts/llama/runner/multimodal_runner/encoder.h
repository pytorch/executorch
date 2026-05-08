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
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>
#include <list>
#include <memory>
#include <string>
#include <vector>

namespace example {

/**
 * @class EncoderRunner
 * @brief Class for running vision encoder to generate image hidden states.
 */
class EncoderRunner {
 public:
  /**
   * @brief Constructor for EncoderRunner
   * @param model_path Path to the encoder model PTE file
   */
  explicit EncoderRunner(executorch::extension::Module* module);

  /**
   * @brief Check if the encoder method is loaded
   * @return true if method is loaded, false otherwise
   */
  bool is_method_loaded() const;

  /**
   * @brief Load the encoder method
   * @return Error status
   */
  executorch::runtime::Error load();

  /**
   * @brief Encode input tensor to hidden states
   * @param input_tensor Input tensor
   * @return Result containing the hidden states tensor
   */
  executorch::runtime::Result<executorch::aten::Tensor> encode(
      executorch::extension::TensorPtr& input_tensor);

 private:
  executorch::extension::Module* module_;
  inline static const std::string kEncoderForwardName = "forward";
  std::vector<executorch::runtime::EValue> encoder_output_;
};

} // namespace example
