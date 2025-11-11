/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/training/module/training_module.h>

namespace executorch {
namespace extension {
namespace training {

namespace {

std::string make_parameters_method_name(const std::string& method_name) {
  return "__et_training_parameters_index_" + method_name;
}

std::string make_gradients_method_name(const std::string& method_name) {
  return "__et_training_gradients_index_" + method_name;
}

std::string make_fqn_method_name(const std::string& method_name) {
  return "__et_training_fqn_" + method_name;
}

} // namespace

runtime::Result<std::vector<runtime::EValue>>
TrainingModule::execute_forward_backward(
    const std::string& method_name,
    const std::vector<runtime::EValue>& input) {
  // Find where the user outputs end.
  const std::string gradients_method_name =
      make_gradients_method_name(method_name);
  auto res = executorch::extension::Module::execute(gradients_method_name);
  if (!res.ok()) {
    return res.error();
  }
  uint64_t grad_start = res.get()[0].toInt();

  const std::string parameters_method_name =
      make_parameters_method_name(method_name);
  // get params start.
  auto param_res =
      executorch::extension::Module::execute(parameters_method_name);
  if (!param_res.ok()) {
    return param_res.error();
  }

  uint64_t param_start = param_res.get()[0].toInt();

  // Execute the forward and backward pass.
  auto outputs = torch::executor::Module::execute(method_name, input);
  if (!outputs.ok()) {
    return outputs.error();
  }

  // Extract the user outputs.
  std::vector<runtime::EValue> user_outputs;
  user_outputs.reserve(grad_start);
  for (size_t i = 0; i < grad_start; ++i) {
    user_outputs.push_back(outputs.get().at(i));
  }

  // Extract and store the gradients and params if this is the first time seeing
  // this method.
  if (method_named_gradients_.find(method_name) ==
      method_named_gradients_.end()) {
    // Fully qualified names
    std::vector<runtime::EValue> fqn_list;
    method_named_gradients_.insert({method_name, {}});

    auto& gradients_map = method_named_gradients_.at(method_name);

    // Get names if we havent seen this method before.
    const std::string fqn_method_name = make_fqn_method_name(method_name);
    auto fqn_res = executorch::extension::Module::execute(fqn_method_name);
    if (!fqn_res.ok()) {
      return fqn_res.error();
    }
    fqn_list = fqn_res.get();

    // Only have to initialize the dict once because the tensors in the dict and
    // the tensors in the method alias the same TensorImpl, so updating one will
    // update the other.
    size_t name_index = 0;
    for (size_t grad_index = grad_start; grad_index < param_start;
         ++grad_index, ++name_index) {
      std::string_view fqn = fqn_list.at(name_index).toString();
      gradients_map.insert({fqn, outputs.get().at(grad_index).toTensor()});
    }
  }

  return user_outputs;
}

runtime::Result<const std::map<std::string_view, executorch::aten::Tensor>>
TrainingModule::named_parameters(const std::string& method_name) {
  // If we haven't seen this method before, populate the dict.
  if (method_named_parameters_.find(method_name) ==
      method_named_parameters_.end()) {
    const std::string fqn_method_name = make_fqn_method_name(method_name);
    const std::string parameters_method_name =
        make_parameters_method_name(method_name);

    method_named_parameters_.insert({method_name, {}});

    // get names.
    auto fqn_res = executorch::extension::Module::execute(fqn_method_name);
    if (!fqn_res.ok()) {
      return fqn_res.error();
    }
    const auto& fqn_list = fqn_res.get();

    // get params start.
    auto param_res =
        executorch::extension::Module::execute(parameters_method_name);
    if (!param_res.ok()) {
      return param_res.error();
    }

    uint64_t param_start = param_res.get()[0].toInt();

    // Load the method if it is not already loaded.
    auto e = executorch::extension::Module::load_method(method_name);
    if (e != runtime::Error::Ok) {
      return e;
    }
    auto& method = methods_.at(method_name).method;

    // populate dict
    size_t name_index = 0;
    for (size_t param_index = param_start; param_index < method->outputs_size();
         ++param_index, ++name_index) {
      std::string_view fqn = fqn_list.at(name_index).toString();
      executorch::aten::Tensor param =
          method->get_output(param_index).toTensor();
      method_named_parameters_.at(method_name).insert({fqn, param});
    }
  }
  return method_named_parameters_.at(method_name);
}

runtime::Result<const std::map<std::string_view, executorch::aten::Tensor>>
TrainingModule::named_gradients(const std::string& method_name) {
  if (method_named_gradients_.find(method_name) ==
      method_named_gradients_.end()) {
    ET_LOG(Error, "No gradients found for method %s", method_name.c_str());
    return executorch::runtime::Error::InvalidArgument;
  }
  return method_named_gradients_.at(method_name);
}

runtime::Result<const std::map<std::string_view, executorch::aten::Tensor>>
TrainingModule::named_attributes(const std::string& method_name) {
  // If we haven't seen this method before, populate the dict.
  if (method_named_attributes_.find(method_name) ==
      method_named_attributes_.end()) {
    method_named_attributes_.insert({method_name, {}});

    // get method metadata
    auto meta_res = method_meta(method_name);
    if (!meta_res.ok()) {
      return meta_res.error();
    }
    // get method
    auto e = load_method(method_name);
    if (e != runtime::Error::Ok) {
      return e;
    }
    auto& method = methods_.at(method_name).method;
    // get tensor by name
    for (int idx = 0; idx < meta_res->num_attributes(); idx++) {
      const auto tensor_res = meta_res->attribute_tensor_meta(idx);
      if (!tensor_res.ok()) {
        return tensor_res.error();
      }
      const auto tensorName = tensor_res.get().name();
      const auto attribute_res = method->get_attribute(tensorName);
      if (!attribute_res.ok()) {
        return attribute_res.error();
      }
      method_named_attributes_.at(method_name)
          .insert({tensorName, attribute_res.get()});
    }
  }
  return method_named_attributes_.at(method_name);
}

} // namespace training
} // namespace extension
} // namespace executorch
