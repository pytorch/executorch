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
class Module {
 public:
  /**
   * Constructs an instance by loading a program from a file with additional
   * memory locking options.
   *
   * @param[in] filePath The path to the ExecuTorch program file to load.
   * @param[in] useMlock Determines whether to use memory locking (mlock).
   * @param[in] ignoreMlockErrors If true, ignores errors during memory locking;
   * effective only if useMlock is true.
   *
   * @throws std::runtime_error if the file fails to load.
   */
  explicit Module(
      const std::string& filePath,
      const bool useMlock = true,
      const bool ignoreMlockErrors = true);

  /**
   * Loads the ExecuTorch program from the specified file path and memory
   * locking options.
   *
   * @returns An Error to indicate success or failure of the loading process.
   */
  Error load();

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
      std::vector<EValue>& outputs);

  /**
   * Get a list of method names available in the loaded program.
   *
   * @returns A vector of strings containing the names of the methods, or an
   * empty vector if the program failed to load.
   */
  std::vector<std::string> methodNames() const;

 private:
  const std::string filePath_;
  const bool useMlock_;
  const bool ignoreMlockErrors_;
  std::unique_ptr<Runner> runner_;
};

} // namespace torch::executor
