/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Constant metadata can be serialized in .pte files, this helper enables
 * easy access to the metadata.
 */
#pragma once

#include <executorch/extension/module/module.h>

namespace torch::executor {
template <typename T>
T get_module_metadata(const Module* module, const std::string& method_name, T default_val) {
  const auto method_names = module->method_names();
  ET_CHECK_MSG(method_names.ok(), "Failed to read method names from model");
  model_methods = method_names.get();

  T res = default_val;
  if (model_methods.count(method_name)) {
    Result<std::vector<EValue>> outputs = module->execute(method_name);
    if (outputs.ok()) {
      std::vector<EValue> outs = outputs.get();
      if (outs.size() > 0) {
        res = outs[0].to<T>();
      }
    }
  } else {
    ET_LOG(
        Info,
        "The model does not contain %s method, using default value %lld",
        method_name.c_str(),
        (long long)default_val);
  }
  ET_LOG(Info, "%s: %lld", method_name.c_str(), (long long)res);
  return res;
}
} // namespace torch::executor
