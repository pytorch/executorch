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
 * A specialized class designed to load the ExecuTorch program from
 * a file using MmapDataLoader and MallocMemoryAllocator, and running the
 * 'forward' method commonly found in Torch modules.
 */
class Module : public Runner {
 public:
  /**
   * Enum to define memory locking behavior.
   */
  enum class MlockConfig {
    /// Do not use memory locking.
    NoMlock,
    /// Use memory locking and handle errors.
    UseMlock,
    /// Use memory locking and ignore errors.
    UseMlockIgnoreErrors,
  };

  /**
   * Constructs an instance by loading a program from a file with specified
   * memory locking behavior.
   *
   * @param[in] filePath The path to the ExecuTorch program file to load.
   * @param[in] mlockConfig The memory locking configuration to use.
   */
  explicit Module(
      const std::string& filePath,
      const MlockConfig mlockConfig = MlockConfig::UseMlock);

  /**
   * Loads the ExecuTorch program from the specified file path and memory
   * locking options.
   *
   * @returns An Error to indicate success or failure of the loading process.
   */
  Error load() override;

  /**
   * Run the 'forward' method with the given inputs and retrieve outputs.
   * Loads the program and method before running if needed.
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

 private:
  const std::string filePath_;
  const MlockConfig mlockConfig_;
  std::unique_ptr<Runner> runner_;
};

} // namespace torch::executor
