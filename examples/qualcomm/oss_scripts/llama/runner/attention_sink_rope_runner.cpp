/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// With the provided kv cache inputs, apply RoPE to update the key cache
// rotation, evict the token, and then return the kv cache.

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/attention_sink_rope_runner.h>

using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using executorch::runtime::Span;
namespace example {

AttentionSinkRopeRunner::AttentionSinkRopeRunner(Module* module)
    : module_(module) {}

Error AttentionSinkRopeRunner::set_outputs(
    const std::string& method_name,
    std::vector<executorch::runtime::EValue> output_values) {
  for (size_t i = 0; i < output_values.size(); ++i) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        module_->set_output(method_name, output_values[i], i));
  }
  return Error::Ok;
}

Error AttentionSinkRopeRunner::load(
    const std::vector<std::string>& method_names) {
  if (is_method_loaded(method_names)) {
    return Error::Ok;
  }
  for (const std::string& method_name : method_names) {
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(method_name));
  }
  eviction_batch_size_ = ET_UNWRAP(module_->get("get_eviction_batch_size"))
                             .toScalar()
                             .to<int64_t>();
  return Error::Ok;
}

bool AttentionSinkRopeRunner::is_method_loaded(
    const std::vector<std::string>& method_names) {
  bool method_loaded = true;
  for (const std::string& method_name : method_names) {
    method_loaded &= module_->is_method_loaded(method_name);
  }
  return method_loaded;
}

Error AttentionSinkRopeRunner::evict_token(
    const std::string& method_name,
    std::vector<EValue>& inputs) {
  Result<std::vector<EValue>> outputs_res =
      module_->execute(method_name, inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  position_shift_ += eviction_batch_size_;

  return Error::Ok;
}

} // namespace example
