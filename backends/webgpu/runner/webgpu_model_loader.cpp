/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runner/webgpu_model_loader.h>

#include <unordered_set>
#include <utility>

namespace executorch::backends::webgpu {

runtime::Result<std::unique_ptr<extension::Module>> load_webgpu_model(
    WebGPUModelLoadSpec spec) {
  if (spec.pte_path.empty() || spec.required_methods.empty()) {
    return runtime::Error::InvalidArgument;
  }
  std::unordered_set<std::string> methods;
  for (const auto& method : spec.required_methods) {
    if (method.empty() || !methods.insert(method).second) {
      return runtime::Error::InvalidArgument;
    }
  }
  std::unordered_set<std::string> data_files;
  for (const auto& path : spec.ptd_paths) {
    if (path.empty() || path == spec.pte_path ||
        !data_files.insert(path).second) {
      return runtime::Error::InvalidArgument;
    }
  }

  auto module = std::make_unique<extension::Module>(
      spec.pte_path, std::move(spec.ptd_paths), spec.load_mode);
  for (const auto& method : spec.required_methods) {
    const runtime::Error error = module->load_method(method);
    if (error != runtime::Error::Ok) {
      return error;
    }
  }
  return module;
}

} // namespace executorch::backends::webgpu
