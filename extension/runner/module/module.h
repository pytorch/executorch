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

/**
 * A specialized subclass of Runner designed to load the ExecuTorch program from
 * a file using MmapDataLoader and MallocMemoryAllocator, and running the
 * 'forward' method commonly found in Torch modules.
 */
class Module : public Runner {
 public:
  /**
   * Constructs an instance by loading a program from a file.
   *
   * @param[in] filePath The path to the ExecuTorch program file to load.
   *
   * @throws std::runtime_error if the file fails to load.
   */
  explicit Module(const std::string& filePath);

  /**
   * Run the 'forward' method with the given inputs and retrieve outputs.
   * Loads the method before running if needed.
   *
   * @param[in] inputs A vector of input values for the 'forward' method.
   * @param[out] outputs A vector of output values from the 'forward' method.
   *
   * @returns An Error to indicate success or failure.
   */
  Error forward(
      const std::vector<EValue>& inputs,
      std::vector<EValue>& outputs) {
    return run("forward", inputs, outputs);
  }
};

} // namespace torch::executor
