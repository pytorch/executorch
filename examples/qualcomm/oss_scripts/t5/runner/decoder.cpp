/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/t5/runner/decoder.h>

using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::Result;

namespace example {
T5Decoder::T5Decoder(const std::string& model_path) {
  module_ = std::make_unique<Module>(
      model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  ET_LOG(Info, "creating decoder module: model_path=%s", model_path.c_str());
}

bool T5Decoder::is_method_loaded() const {
  return module_->is_method_loaded(kDecoderForwardName);
}

Error T5Decoder::load() {
  if (is_method_loaded()) {
    return Error::Ok;
  }
  return module_->load_method(kDecoderForwardName);
}
Result<Tensor> T5Decoder::step(
    TensorPtr& input_ids,
    TensorPtr& attention_mask,
    TensorPtr& encoder_hidden_states,
    TensorPtr& encoder_attention_mask,
    TensorPtr& cache_position) {
  auto outputs_res = module_->execute(
      kDecoderForwardName,
      {input_ids,
       attention_mask,
       encoder_hidden_states,
       encoder_attention_mask,
       cache_position});
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  ET_CHECK_MSG(
      outputs_res.get().size() == 1,
      "More then one output returned from executing decoder.");
  ET_CHECK_MSG(
      outputs_res.get()[0].isTensor(),
      "Non Tensor Output returned from executing decoder");

  // Return the logits tensor
  return outputs_res.get()[0].toTensor();
}
} // namespace example
