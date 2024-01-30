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
#include <unordered_set>
#include <vector>

#include <executorch/runtime/executor/program.h>

namespace torch::executor {

/**
 * A facade class for loading programs and executing methods within them.
 */
class Module final {
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
   * Constructs an instance with the provided data loader and memory allocator.
   *
   * @param[in] dataLoader A DataLoader used for loading program data.
   * @param[in] memoryAllocator A MemoryAllocator used for memory management.
   */
  explicit Module(
      std::unique_ptr<DataLoader> dataLoader,
      std::unique_ptr<MemoryAllocator> memoryAllocator = nullptr,
      std::unique_ptr<EventTracer> eventTracer = nullptr);
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&&) = default;
  Module& operator=(Module&&) = default;

  /**
   * Loads the program using the specified data loader and memory allocator.
   *
   * @param[in] verification The type of verification to do before returning
   * success.
   * @returns An Error to indicate success or failure of the loading process.
   */
  __ET_NODISCARD
  Error load(
      const Program::Verification verification =
          Program::Verification::Minimal);

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
   * @returns A set of strings containing the names of the methods, or an error
   * if the program or method failed to load.
   */
  Result<std::unordered_set<std::string>> methodNames();

  /**
   * Load a specific method from the program and set up memory management if
   * needed. The loaded method is cached to reuse the next time it's executed.
   *
   * @param[in] methodName The name of the method to load.
   *
   * @returns An Error to indicate success or failure.
   */
  __ET_NODISCARD
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
   * Execute a specific method with the given input and retrieve output.
   * Loads the program and method before executing if needed.
   *
   * @param[in] methodName The name of the method to execute.
   * @param[in] input A vector of input values to be passed to the method.
   *
   * @returns A Result object containing either a vector of output values
   *          from the method or an error to indicate failure.
   */
  __ET_NODISCARD
  Result<std::vector<EValue>> execute(
      const std::string& methodName,
      const std::vector<EValue>& input);

  /**
   * Execute a specific method without any input values.
   * Loads the program and method before executing if needed.
   *
   * @param[in] methodName The name of the method to execute.
   *
   * @returns A Result object containing either a vector of output values
   *          from the method or an error to indicate failure.
   */
  __ET_NODISCARD
  Result<std::vector<EValue>> execute(const std::string& methodName) {
    return execute(methodName, {});
  }

  /**
   * Execute the 'forward' method with the given input and retrieve output.
   * Loads the program and method before executing if needed.
   *
   * @param[in] input A vector of input values for the 'forward' method.
   *
   * @returns A Result object containing either a vector of output values
   *          from the 'forward' method or an error to indicate failure.
   */
  __ET_NODISCARD
  Result<std::vector<EValue>> forward(const std::vector<EValue>& input) {
    return execute("forward", input);
  }

  /**
   * Execute the 'forward' method without any input values.
   * Loads the program and method before executing if needed.
   *
   * @returns A Result object containing either a vector of output values
   *          from the 'forward' method or an error to indicate failure.
   */
  __ET_NODISCARD
  Result<std::vector<EValue>> forward() {
    return forward({});
  }

  /**
   * Retrieves the EventTracer instance being used by the Module.
   * EventTracer is used for tracking and logging events during the execution
   * of methods.
   *
   * @returns A pointer to the EventTracer instance. Returns nullptr if no
   * EventTracer is set.
   */
  EventTracer* eventTracer() const {
    return eventTracer_.get();
  }

 private:
  struct MethodHolder {
    std::vector<std::vector<uint8_t>> plannedBuffers;
    std::vector<Span<uint8_t>> plannedSpans;
    std::unique_ptr<HierarchicalAllocator> plannedMemory;
    std::unique_ptr<MemoryManager> memoryManager;
    std::unique_ptr<Method> method;
  };

 private:
  std::string filePath_;
  MlockConfig mlockConfig_{MlockConfig::NoMlock};
  std::unique_ptr<DataLoader> dataLoader_;
  std::unique_ptr<MemoryAllocator> memoryAllocator_;
  std::unique_ptr<EventTracer> eventTracer_;
  std::unique_ptr<Program> program_;
  std::unordered_map<std::string, MethodHolder> methods_;
};

} // namespace torch::executor
