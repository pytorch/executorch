/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/whisper/runner/encoder.h>

using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::Result;
namespace example {
WhisperEncoder::WhisperEncoder(const std::string& model_path) {
  module_ = std::make_unique<Module>(
      model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  ET_LOG(Info, "creating encoder module: model_path=%s", model_path.c_str());
}

bool WhisperEncoder::is_method_loaded() const {
  return module_->is_method_loaded(kEncoderForwardName);
}

Error WhisperEncoder::load() {
  if (is_method_loaded()) {
    return Error::Ok;
  }
  return module_->load_method(kEncoderForwardName);
}
Result<Tensor> WhisperEncoder::encode(TensorPtr& input_feature) {
  auto outputs_res = module_->execute(kEncoderForwardName, input_feature);
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  ET_CHECK_MSG(
      outputs_res.get().size() == 1,
      "More then one output returned from executing encoder.");
  ET_CHECK_MSG(
      outputs_res.get()[0].isTensor(),
      "Non Tensor Output returned from executing encoder");

  // Return the hidden state tensor
  return outputs_res.get()[0].toTensor();
}
} // namespace example
