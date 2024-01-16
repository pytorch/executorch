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
  virtual ~Runner() = default;

  /**
   * Loads the program using the specified data loader and memory allocator.
   *
   * @returns An Error to indicate success or failure of the loading process.
   */
  virtual Error load();

  /**
   * Checks if the program is loaded.
   *
   * @returns true if the program is loaded, false otherwise.
   */
  bool isLoaded() const;

  /**
   * Get a list of method names available in the loaded program.
   * Loads the program and method if needed.
   *
   * @returns A vector of strings containing the names of the methods, or an
   * error if the program or method failed to load.
   */
  Result<std::vector<std::string>> methodNames();

  /**
   * Load a specific method from the program and set up memory management if
   * needed. The loaded method is cached to reuse the next time it's run.
   *
   * @param[in] methodName The name of the method to load.
   *
   * @returns An Error to indicate success or failure.
   */
  Error loadMethod(const std::string& methodName);

  /**
   * Checks if a specific method is loaded.
   *
   * @param[in] methodName The name of the method to check.
   * @returns true if the method specified by methodName is loaded, false
   * otherwise.
   */
  bool isMethodLoaded(const std::string& methodName) const;

  /**
   * Get a method metadata struct by method name.
   * Loads the program and method if needed.
   *
   * @param[in] methodName The name of the method to get the metadata for.
   *
   * @returns A method metadata, or an error if the program or method failed to
   * load.
   */
  Result<MethodMeta> methodMeta(const std::string& methodName);

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

 private:
  struct MethodHolder {
    std::vector<std::vector<uint8_t>> plannedBuffers;
    std::vector<Span<uint8_t>> plannedSpans;
    std::unique_ptr<HierarchicalAllocator> plannedMemory;
    std::unique_ptr<MemoryManager> memoryManager;
    std::unique_ptr<Method> method;
  };

 protected:
  std::unique_ptr<DataLoader> dataLoader_;

 private:
  std::unique_ptr<MemoryAllocator> memoryAllocator_;
  std::unique_ptr<Program> program_;
  std::unordered_map<std::string, MethodHolder> methods_;
};

} // namespace torch::executor
