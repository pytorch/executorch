/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/runtime/executor/program.h>

namespace torch::executor {

/**
 * A facade class for loading programs and executing methods within them.
 */
class Runner {
 public:
  /**
   * Constructs an instance with the provided data loader and memory allocator.
   *
   * @param[in] dataLoader A DataLoader used for loading program data.
   * @param[in] memoryAllocator A MemoryAllocator used for memory management.
   *
   * @throws std::runtime_error if the program fails to load.
   */
  explicit Runner(
      std::unique_ptr<DataLoader> dataLoader,
      std::unique_ptr<MemoryAllocator> memoryAllocator);
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  /**
   * Loads the program using the data loader.
   *
   * @returns An Error to indicate success or failure of the loading process.
   */
  Error load();

  /**
   * Run a specific method with the given inputs and retrieve outputs.
   * Loads the program and method before running if needed.
   *
   * @param[in] methodName The name of the method to execute.
   * @param[in] inputs A vector of input values to be passed to the method.
   * @param[out] outputs A vector to store the output values from the method.
   *
   * @returns An Error to indicate success or failure.
   */
  Error run(
      const std::string& methodName,
      const std::vector<EValue>& inputs,
      std::vector<EValue>& outputs);

  /**
   * Get a list of method names available in the loaded program.
   *
   * @returns A vector of strings containing the names of the methods, or an
   * empty vector if the program failed to load.
   */
  std::vector<std::string> methodNames() const;

  /**
   * Load a specific method from the program and set up memory management if
   * needed. The loaded method is cached to reuse the next time it's run.
   *
   * @param[in] methodName The name of the method to load.
   *
   * @returns An Error to indicate success or failure.
   */
  Error loadMethod(const std::string& methodName);

 private:
  struct MethodHolder {
    std::vector<std::vector<uint8_t>> plannedBuffers;
    std::vector<Span<uint8_t>> plannedSpans;
    std::unique_ptr<HierarchicalAllocator> plannedMemory;
    std::unique_ptr<MemoryManager> memoryManager;
    std::unique_ptr<Method> method;
  };

 private:
  std::unique_ptr<DataLoader> dataLoader_;
  std::unique_ptr<MemoryAllocator> memoryAllocator_;
  std::unique_ptr<Program> program_;
  std::unordered_map<std::string, MethodHolder> methods_;
};

} // namespace torch::executor
