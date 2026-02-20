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
  explicit EncoderRunner(const std::string& model_path);

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
   * @brief Get the image sequence length from encoder output metadata
   * @return Image sequence length
   */
  int32_t get_image_seq_len() const;

  /**
   * @brief Encode image tensor to hidden states
   * @param image_tensor Input image tensor (B, C, H, W)
   * @return Result containing the image hidden states tensor
   */
  executorch::runtime::Result<executorch::aten::Tensor> encode(
      executorch::extension::TensorPtr& image_tensor);

  /**
   * @brief Encode image from raw file
   * @param image_file_path Path to raw image file
   * @return Result containing the image hidden states tensor
   */
  executorch::runtime::Result<executorch::aten::Tensor> encode_from_file(
      const std::string& image_file_path);

 private:
  std::unique_ptr<executorch::extension::Module> module_;
  inline static const std::string kEncoderForwardName = "forward";
  int32_t image_seq_len_;
};

} // namespace example
