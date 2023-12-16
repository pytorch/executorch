/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/runner/runner.h>

namespace torch::executor {

class Module : public Runner {
 public:
  explicit Module(const std::string& filePath);

  Error forward(
      const std::vector<EValue>& inputs,
      std::vector<EValue>& outputs) {
    return run("forward", inputs, outputs);
  }
};

} // namespace torch::executor
