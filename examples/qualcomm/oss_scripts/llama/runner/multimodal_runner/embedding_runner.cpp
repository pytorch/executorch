/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/embedding_runner.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;

namespace example {

EmbeddingRunner::EmbeddingRunner(Module* module) : module_(module) {}

Result<Tensor> EmbeddingRunner::step(
    const std::string& method_name,
    std::vector<EValue>& inputs) {
  // Execute embedding module
  Result<std::vector<EValue>> outputs_res =
      module_->execute(method_name, inputs);

  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  ET_CHECK_MSG(
      outputs_res.get()[0].isTensor(),
      "Non Tensor Output returned from executing Token Embedding");

  // Get the embedding tensor from result
  return outputs_res.get()[0].toTensor();
}

Error EmbeddingRunner::set_outputs(
    const std::string& method_name,
    std::vector<executorch::aten::Tensor> output_values) {
  for (size_t i = 0; i < output_values.size(); ++i) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        module_->set_output(method_name, output_values[i], i));
  }
  return Error::Ok;
}

Error EmbeddingRunner::load(const std::vector<std::string>& method_names) {
  if (is_method_loaded(method_names)) {
    return Error::Ok;
  }
  for (const std::string& method_name : method_names) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(method_name));
  }
  return Error::Ok;
}

bool EmbeddingRunner::is_method_loaded(
    const std::vector<std::string>& method_names) {
  bool method_loaded = true;
  for (const std::string& method_name : method_names) {
    method_loaded &= module_->is_method_loaded(method_name);
  }
  return method_loaded;
}

bool EmbeddingRunner::is_loaded() const {
  return module_ != nullptr && module_->is_loaded();
}

} // namespace example
