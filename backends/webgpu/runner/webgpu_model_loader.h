/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/result.h>

#include <memory>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

struct WebGPUModelLoadSpec {
  std::string pte_path;
  std::vector<std::string> ptd_paths;
  std::vector<std::string> required_methods;
  extension::Module::LoadMode load_mode = extension::Module::LoadMode::File;
};

runtime::Result<std::unique_ptr<extension::Module>> load_webgpu_model(
    WebGPUModelLoadSpec spec);

} // namespace executorch::backends::webgpu
