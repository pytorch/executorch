/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <cstring>
#include <fstream>

using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

namespace example {

EncoderRunner::EncoderRunner(executorch::extension::Module* module)
    : module_(module) {}

bool EncoderRunner::is_method_loaded() const {
  return module_->is_method_loaded(kEncoderForwardName);
}

Error EncoderRunner::load() {
  if (is_method_loaded()) {
    return Error::Ok;
  }

  auto load_result = module_->load_method(kEncoderForwardName);
  if (load_result != Error::Ok) {
    ET_LOG(Error, "Failed to load encoder method");
    return load_result;
  }

  // Get image sequence length from output metadata
  Result<MethodMeta> method_meta = module_->method_meta(kEncoderForwardName);
  if (!method_meta.ok()) {
    ET_LOG(Error, "Failed to get encoder method metadata");
    return method_meta.error();
  }

  return Error::Ok;
}

Result<Tensor> EncoderRunner::encode(TensorPtr& image_tensor) {
  ET_CHECK_MSG(is_method_loaded(), "Encoder method not loaded");

  auto tensor_ptr = image_tensor.get();
  ET_LOG(Info, "Encoding image tensor with numel: %zu", tensor_ptr->numel());

  std::vector<executorch::runtime::EValue> encoder_inputs;
  encoder_inputs.emplace_back(*tensor_ptr);

  auto encoder_result = module_->forward(encoder_inputs);
  ET_CHECK_MSG(encoder_result.ok(), "Encoder execution failed");

  auto encoder_output = encoder_result.get();
  auto image_hidden_states = encoder_output[0].toTensor();
  ET_LOG(Info, "Encoder execution completed, got image hidden states");

  return image_hidden_states;
}

} // namespace example
