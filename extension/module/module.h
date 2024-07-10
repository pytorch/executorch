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
   * Enum to define loading behavior.
   */
  enum class LoadMode {
    /// Load the whole file as a buffer.
    File,
    /// Use mmap to load pages into memory.
    Mmap,
    /// Use memory locking and handle errors.
    MmapUseMlock,
    /// Use memory locking and ignore errors.
    MmapUseMlockIgnoreErrors,
  };

  /**
   * Constructs an instance by loading a program from a file with specified
   * memory locking behavior.
   *
   * @param[in] file_path The path to the ExecuTorch program file to load.
   * @param[in] load_mode The loading mode to use.
   */
  explicit Module(
      const std::string& file_path,
      const LoadMode load_mode = LoadMode::MmapUseMlock,
      std::unique_ptr<EventTracer> event_tracer = nullptr);

  /**
   * Constructs an instance with the provided data loader and memory allocator.
   *
   * @param[in] data_loader A DataLoader used for loading program data.
   * @param[in] memory_allocator A MemoryAllocator used for memory management.
   * @param[in] temp_allocator A MemoryAllocator to use when allocating
   * temporary data during kernel or delegate execution.
   * @param[in] event_tracer A EventTracer used for tracking and logging events.
   */
  explicit Module(
      std::unique_ptr<DataLoader> data_loader,
      std::unique_ptr<MemoryAllocator> memory_allocator = nullptr,
      std::unique_ptr<MemoryAllocator> temp_allocator = nullptr,
      std::unique_ptr<EventTracer> event_tracer = nullptr);
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&&) = default;
  Module& operator=(Module&&) = default;

  /**
   * Loads the program using the specified data loader and memory allocator.
   *
   * @param[in] verification The type of verification to do before returning
   * success.
   *
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
  bool is_loaded() const;

  /**
   * Get a list of method names available in the loaded program.
   * Loads the program and method if needed.
   *
   * @returns A set of strings containing the names of the methods, or an error
   * if the program or method failed to load.
   */
  Result<std::unordered_set<std::string>> method_names();

  /**
   * Load a specific method from the program and set up memory management if
   * needed. The loaded method is cached to reuse the next time it's executed.
   *
   * @param[in] method_name The name of the method to load.
   *
   * @returns An Error to indicate success or failure.
   */
  __ET_NODISCARD
  Error load_method(const std::string& method_name);

  /**
   * Checks if a specific method is loaded.
   *
   * @param[in] method_name The name of the method to check.
   *
   * @returns true if the method specified by method_name is loaded, false
   * otherwise.
   */
  bool is_method_loaded(const std::string& method_name) const;

  /**
   * Get a method metadata struct by method name.
   * Loads the program and method if needed.
   *
   * @param[in] method_name The name of the method to get the metadata for.
   *
   * @returns A method metadata, or an error if the program or method failed to
   * load.
   */
  Result<MethodMeta> method_meta(const std::string& method_name);

  /**
   * Execute a specific method with the given input and retrieve output.
   * Loads the program and method before executing if needed.
   *
   * @param[in] method_name The name of the method to execute.
   * @param[in] input A vector of input values to be passed to the method.
   *
   * @returns A Result object containing either a vector of output values
   *          from the method or an error to indicate failure.
   */
  __ET_NODISCARD
  Result<std::vector<EValue>> execute(
      const std::string& method_name,
      const std::vector<EValue>& input);

  /**
   * Execute a specific method without any input values.
   * Loads the program and method before executing if needed.
   *
   * @param[in] method_name The name of the method to execute.
   *
   * @returns A Result object containing either a vector of output values
   *          from the method or an error to indicate failure.
   */
  __ET_NODISCARD
  Result<std::vector<EValue>> execute(const std::string& method_name) {
    return execute(method_name, {});
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
  EventTracer* event_tracer() const {
    return event_tracer_.get();
  }

  /**
   * Set output data pointer for forward method.
   *
   * @param[in] output_tensor A Tensor for the output of 'forward' method.
   * @param[in] output_index Index of the output in 'forward' method.
   *
   * @returns An Error to indicate success or failure of the loading process.
   */
  Error set_output_data_ptr(Tensor& output_tensor, size_t output_index);

 private:
  struct MethodHolder {
    std::vector<std::vector<uint8_t>> planned_buffers;
    std::vector<Span<uint8_t>> planned_spans;
    std::unique_ptr<HierarchicalAllocator> planned_memory;
    std::unique_ptr<MemoryManager> memory_manager;
    std::unique_ptr<Method> method;
  };

 private:
  std::string file_path_;
  LoadMode load_mode_{LoadMode::MmapUseMlock};
  std::unique_ptr<DataLoader> data_loader_;
  std::unique_ptr<MemoryAllocator> memory_allocator_;
  std::unique_ptr<MemoryAllocator> temp_allocator_;
  std::unique_ptr<EventTracer> event_tracer_;
  std::unique_ptr<Program> program_;
  std::unordered_map<std::string, MethodHolder> methods_;
};

} // namespace torch::executor
