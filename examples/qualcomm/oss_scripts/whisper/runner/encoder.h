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
#include <memory>
#include <string>
#include <vector>

namespace example {

class WhisperEncoder {
 public:
  explicit WhisperEncoder(const std::string& model_path);

  bool is_method_loaded() const;
  executorch::runtime::Error load();
  executorch::runtime::Result<executorch::aten::Tensor> encode(
      executorch::extension::TensorPtr& input_feature);

 private:
  std::unique_ptr<executorch::extension::Module> module_;
  static constexpr const char* kEncoderForwardName = "encoder";
};

} // namespace example
