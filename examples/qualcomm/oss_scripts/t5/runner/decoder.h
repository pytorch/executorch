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
#include <executorch/runtime/core/evalue.h>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace example {

class T5Decoder {
 public:
  explicit T5Decoder(const std::string& model_path);

  bool is_method_loaded() const;
  executorch::runtime::Error load();
  executorch::runtime::Result<executorch::aten::Tensor> step(
      executorch::extension::TensorPtr& input_ids,
      executorch::extension::TensorPtr& attention_mask,
      executorch::extension::TensorPtr& encoder_hidden_states,
      executorch::extension::TensorPtr& encoder_attention_mask,
      executorch::extension::TensorPtr& cache_position);
  executorch::runtime::Result<std::unordered_set<std::string>> method_names() {
    return module_->method_names();
  }
  executorch::runtime::Result<executorch::runtime::EValue> get(
      const std::string& method_name) {
    return module_->get(method_name);
  }

  executorch::runtime::Result<std::vector<executorch::runtime::EValue>> execute(
      const std::string& method_name) {
    return module_->execute(method_name);
  }

 private:
  std::unique_ptr<executorch::extension::Module> module_;
  static constexpr const char* kDecoderForwardName = "decoder";
};

} // namespace example
