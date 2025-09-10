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

#ifdef USE_ATEN_LIB
#define ET_MODULE_NAMESPACE module::aten
#else // !USE_ATEN_LIB
#define ET_MODULE_NAMESPACE module
#endif // USE_ATEN_LIB

namespace executorch {
namespace extension {

using ET_RUNTIME_NAMESPACE::Method;
using ET_RUNTIME_NAMESPACE::MethodMeta;
using ET_RUNTIME_NAMESPACE::NamedDataMap;
using ET_RUNTIME_NAMESPACE::Program;

class ExecuTorchJni;

namespace ET_MODULE_NAMESPACE {
/**
 * A facade class for loading programs and executing methods within them.
 */
class Module {
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
   * @param[in] event_tracer A EventTracer used for tracking and logging events.
   */
  explicit Module(
      const std::string& file_path,
      const LoadMode load_mode = LoadMode::File,
      std::unique_ptr<runtime::EventTracer> event_tracer = nullptr);

  /**
   * Constructs an instance by loading a program from a file with specified
   * memory locking behavior.
   *
   * @param[in] file_path The path to the ExecuTorch program file to load.
   * @param[in] data_map_path The path to a .ptd file
   * @param[in] load_mode The loading mode to use.
   * @param[in] event_tracer A EventTracer used for tracking and logging events.
   */
  explicit Module(
      const std::string& file_path,
      const std::string& data_map_path,
      const LoadMode load_mode = LoadMode::File,
      std::unique_ptr<runtime::EventTracer> event_tracer = nullptr);

  /**
   * Constructs an instance with the provided data loader and memory allocator.
   *
   * @param[in] data_loader A DataLoader used for loading program data.
   * @param[in] memory_allocator A MemoryAllocator used for memory management.
   * @param[in] temp_allocator A MemoryAllocator to use when allocating
   * temporary data during kernel or delegate execution.
   * @param[in] event_tracer A EventTracer used for tracking and logging events.
   * @param[in] data_map_loader A DataLoader used for loading external weights.
   */
  explicit Module(
      std::unique_ptr<runtime::DataLoader> data_loader,
      std::unique_ptr<runtime::MemoryAllocator> memory_allocator = nullptr,
      std::unique_ptr<runtime::MemoryAllocator> temp_allocator = nullptr,
      std::unique_ptr<runtime::EventTracer> event_tracer = nullptr,
      std::unique_ptr<runtime::DataLoader> data_map_loader = nullptr);

  /**
   * Constructs an instance using an existing shared program.
   *
   * @param[in] program The shared program to use. It's required the data loader
   * the program uses is valid for the lifetime of the program.
   * @param[in] memory_allocator A MemoryAllocator used for memory management.
   * @param[in] temp_allocator A MemoryAllocator to use when allocating
   * temporary data.
   * @param[in] event_tracer A EventTracer used for tracking and logging events.
   * @param[in] data_map_loader A DataLoader used for loading external weights.
   */
  explicit Module(
      std::shared_ptr<Program> program,
      std::unique_ptr<runtime::MemoryAllocator> memory_allocator = nullptr,
      std::unique_ptr<runtime::MemoryAllocator> temp_allocator = nullptr,
      std::unique_ptr<runtime::EventTracer> event_tracer = nullptr,
      std::unique_ptr<runtime::DataLoader> data_map_loader = nullptr);

  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&&) = delete;
  Module& operator=(Module&&) = delete;
  virtual ~Module() = default;
  /**
   * Loads the program if needed.
   *
   * @param[in] verification The type of verification to do before returning
   * success.
   *
   * @returns An Error to indicate success or failure of the loading process.
   */
  ET_NODISCARD virtual runtime::Error load(
      const Program::Verification verification =
          Program::Verification::Minimal);

  /**
   * Checks if the program is loaded.
   *
   * @returns true if the program is loaded, false otherwise.
   */
  virtual inline bool is_loaded() const {
    return program_ != nullptr;
  }

  /**
   * Get the program. The data loader used by the program is guaranteed to be
   * valid for the lifetime of the program.
   *
   * @returns Shared pointer to the program or nullptr if it's not yet loaded.
   */
  inline std::shared_ptr<Program> program() const {
    return program_;
  }

  /**
   * Get the number of methods available in the loaded program.
   *
   * @returns A Result object containing either the number of methods available
   *          or an error to indicate failure.
   */
  runtime::Result<size_t> num_methods();

  /**
   * Get a list of method names available in the loaded program.
   * Loads the program and method if needed.
   *
   * @returns A set of strings containing the names of the methods, or an error
   * if the program or method failed to load.
   */
  runtime::Result<std::unordered_set<std::string>> method_names();

  /**
   * Load a specific method from the program and set up memory management if
   * needed. The loaded method is cached to reuse the next time it's executed.
   *
   * @param[in] method_name The name of the method to load.
   * @param[in] planned_memory The memory-planned buffers to use for mutable
   * tensor data when executing a method.
   * @param[in] event_tracer Per-method event tracer to profile/trace methods
   * individually. When not given, the event tracer passed to the Module
   * constructor is used. Otherwise, this per-method event tracer takes
   * precedence.
   *
   * @returns An Error to indicate success or failure.
   */
  ET_NODISCARD
  runtime::Error load_method(
      const std::string& method_name,
      runtime::HierarchicalAllocator* planned_memory = nullptr,
      torch::executor::EventTracer* event_tracer = nullptr);

  ET_DEPRECATED ET_NODISCARD runtime::Error inline load_method(
      const std::string& method_name,
      torch::executor::EventTracer* event_tracer) {
    return load_method(method_name, nullptr, event_tracer);
  }

  /**
   * Unload a specific method from the program.
   *
   * @param[in] method_name The name of the method to unload.
   *
   * @returns True if the method is unloaded, false if no-op.
   */
  inline bool unload_method(const std::string& method_name) {
    return methods_.erase(method_name);
  }

  /**
   * DEPRECATED: Module manages each Method exclusively.
   *
   * Get a method by it's name. Not recommended to use this method directly as
   * an end user. It's exposed to allow for composability of module in apis that
   * operate on method.
   *
   * @param[in] method_name The name of the method to get.
   *
   * @returns A Result object containing either a pointer to the requested
   *          method or an error to indicate failure.
   */
  ET_DEPRECATED ET_NODISCARD runtime::Result<Method*> method(
      const std::string& method_name);

  /**
   * Load the 'forward' method from the program and set up memory management if
   * needed. The loaded method is cached to reuse the next time it's executed.
   *
   * @param[in] planned_memory The memory-planned buffers to use for mutable
   * tensor data when executing the 'forward' method.
   * @param[in] event_tracer An event tracer used for tracking and logging
   * events.
   *
   * @returns An Error to indicate success or failure.
   */
  ET_NODISCARD inline runtime::Error load_forward(
      runtime::HierarchicalAllocator* planned_memory = nullptr,
      torch::executor::EventTracer* event_tracer = nullptr) {
    return load_method("forward", planned_memory, event_tracer);
  }

  ET_DEPRECATED ET_NODISCARD inline runtime::Error load_forward(
      torch::executor::EventTracer* event_tracer) {
    return load_forward(nullptr, event_tracer);
  }

  /**
   * Unload the 'forward' method from the program.
   *
   * @returns True if the 'forward' method is unloaded, false if no-op.
   */
  inline bool unload_forward() {
    return unload_method("forward");
  }

  /**
   * Checks if a specific method is loaded.
   *
   * @param[in] method_name The name of the method to check.
   *
   * @returns true if the method specified by method_name is loaded, false
   * otherwise.
   */
  inline bool is_method_loaded(const std::string& method_name) const {
    return methods_.count(method_name);
  }

  /**
   * Get a method metadata struct by method name.
   * Loads the program if needed.
   *
   * @param[in] method_name The name of the method to get the metadata for.
   *
   * @returns A method metadata, or an error if the program or method failed to
   * load.
   */
  runtime::Result<MethodMeta> method_meta(const std::string& method_name);

  /**
   * Execute a specific method with the given input values and retrieve the
   * output values. Loads the program and method before executing if needed.
   *
   * @param[in] method_name The name of the method to execute.
   * @param[in] input_values A vector of input values to be passed to the
   * method.
   *
   * @returns A Result object containing either a vector of output values
   *          from the method or an error to indicate failure.
   */
  ET_NODISCARD virtual runtime::Result<std::vector<runtime::EValue>> execute(
      const std::string& method_name,
      const std::vector<runtime::EValue>& input_values);

  /**
   * Execute a specific method with a single input value.
   * Loads the program and method before executing if needed.
   *
   * @param[in] method_name The name of the method to execute.
   * @param[in] input_value A value to be passed to the method.
   *
   * @returns A Result object containing either a vector of output values
   *          from the method or an error to indicate failure.
   */
  ET_NODISCARD inline runtime::Result<std::vector<runtime::EValue>> execute(
      const std::string& method_name,
      const runtime::EValue& input_value) {
    return execute(method_name, std::vector<runtime::EValue>{input_value});
  }

  /**
   * Execute a specific method without any input values.
   * Loads the program and method before executing if needed.
   *
   * @param[in] method_name The name of the method to execute.
   *
   * @returns A Result object containing either a vector of output values
   *          from the method or an error to indicate failure.
   */
  ET_NODISCARD inline runtime::Result<std::vector<runtime::EValue>> execute(
      const std::string& method_name) {
    return execute(method_name, std::vector<runtime::EValue>{});
  }

  /**
   * Retrieve the output value of a specific method with the given input values.
   * Loads the program and method before execution if needed.
   *
   * @param[in] method_name The name of the method to execute.
   * @param[in] input_values A vector of input values to be passed to the
   * method.
   *
   * @returns A Result object containing either the first output value from the
   * method or an error to indicate failure.
   */
  ET_NODISCARD inline runtime::Result<runtime::EValue> get(
      const std::string& method_name,
      const std::vector<runtime::EValue>& input_values) {
    auto result = ET_UNWRAP(execute(method_name, input_values));
    if (result.empty()) {
      return runtime::Error::InvalidArgument;
    }
    return result[0];
  }

  /**
   * Retrieve the output value of a specific method with a single input value.
   * Loads the program and method before execution if needed.
   *
   * @param[in] method_name The name of the method to execute.
   * @param[in] input_value A value to be passed to the method.
   *
   * @returns A Result object containing either the first output value from the
   * method or an error to indicate failure.
   */
  ET_NODISCARD inline runtime::Result<runtime::EValue> get(
      const std::string& method_name,
      const runtime::EValue& input_value) {
    return get(method_name, std::vector<runtime::EValue>{input_value});
  }

  /**
   * Retrieve the output value of a specific method without any input values.
   * Loads the program and method before execution if needed.
   *
   * @param[in] method_name The name of the method to execute.
   *
   * @returns A Result object containing either the first output value from the
   * method or an error to indicate failure.
   */
  ET_NODISCARD inline runtime::Result<runtime::EValue> get(
      const std::string& method_name) {
    return get(method_name, std::vector<runtime::EValue>{});
  }

  /**
   * Execute the 'forward' method with the given input values and retrieve the
   * output values. Loads the program and method before executing if needed.
   *
   * @param[in] input_values A vector of input values for the 'forward' method.
   *
   * @returns A Result object containing either a vector of output values
   *          from the 'forward' method or an error to indicate failure.
   */
  ET_NODISCARD inline runtime::Result<std::vector<runtime::EValue>> forward(
      const std::vector<runtime::EValue>& input_values) {
    return execute("forward", input_values);
  }

  /**
   * Execute the 'forward' method with a single value.
   * Loads the program and method before executing if needed.
   *
   * @param[in] input_value A value for the 'forward' method.
   *
   * @returns A Result object containing either a vector of output values
   *          from the 'forward' method or an error to indicate failure.
   */
  ET_NODISCARD inline runtime::Result<std::vector<runtime::EValue>> forward(
      const runtime::EValue& input_value) {
    return forward(std::vector<runtime::EValue>{input_value});
  }

  /**
   * Execute the 'forward' method without any input values.
   * Loads the program and method before executing if needed.
   *
   * @returns A Result object containing either a vector of output values
   *          from the 'forward' method or an error to indicate failure.
   */
  ET_NODISCARD inline runtime::Result<std::vector<runtime::EValue>> forward() {
    return forward(std::vector<runtime::EValue>{});
  }

  /**
   * Sets a single input value for a specific method.
   *
   * @param[in] method_name The name of the method.
   * @param[in] input_value The EValue to set as the method input.
   * @param[in] input_index Zero-based index of the input to set.
   *
   * @returns An Error to indicate success or failure.
   */
  ET_NODISCARD
  runtime::Error set_input(
      const std::string& method_name,
      const runtime::EValue& input_value,
      size_t input_index);

  /**
   * Sets a single input value for the "forward" method.
   *
   * @param[in] input_value The EValue to set as the method input.
   * @param[in] input_index Zero-based index of the input to set.
   *
   * @returns An Error to indicate success or failure.
   */
  ET_NODISCARD
  inline runtime::Error set_input(
      const runtime::EValue& input_value,
      size_t input_index) {
    return set_input("forward", input_value, input_index);
  }

  /**
   * Sets all input values for a specific method.
   *
   * @param[in] method_name The name of the method.
   * @param[in] input_values A vector of EValues to set as the method inputs.
   *
   * @returns An Error to indicate success or failure.
   */
  ET_NODISCARD
  runtime::Error set_inputs(
      const std::string& method_name,
      const std::vector<runtime::EValue>& input_values);

  /**
   * Sets all input values for the "forward" method.
   *
   * @param[in] input_values A vector of EValues to set as the method inputs.
   *
   * @returns An Error to indicate success or failure.
   */
  ET_NODISCARD
  inline runtime::Error set_inputs(
      const std::vector<runtime::EValue>& input_values) {
    return set_inputs("forward", input_values);
  }

  /**
   * Sets the output tensor for a specific method.
   *
   * @param[in] method_name The name of the method.
   * @param[in] output_value The EValue containing the Tensor to set as the
   * method output.
   * @param[in] output_index Zero-based index of the output to set.
   *
   * @returns An Error to indicate success or failure.
   *
   * @note Only Tensor outputs are currently supported for setting.
   */
  ET_NODISCARD
  runtime::Error set_output(
      const std::string& method_name,
      runtime::EValue output_value,
      size_t output_index = 0);

  /**
   * Sets the output tensor for the "forward" method.
   *
   * @param[in] output_value The EValue containing the Tensor to set as the
   * method output.
   * @param[in] output_index Zero-based index of the output to set.
   *
   * @returns An Error to indicate success or failure.
   *
   * @note Only Tensor outputs are currently supported for setting.
   */
  ET_NODISCARD
  inline runtime::Error set_output(
      runtime::EValue output_value,
      size_t output_index = 0) {
    return set_output("forward", std::move(output_value), output_index);
  }

  /**
   * Sets all output tensors for a specific method.
   *
   * Loads the program and method if needed, and for each output uses
   * the provided tensor's data buffer as the method's output buffer.
   *
   * @param[in] method_name The name of the method.
   * @param[in] output_values A vector of EValues to set as the method outputs.
   *
   * @returns An Error to indicate success or failure.
   *
   * @note Only Tensor outputs are currently supported for setting.
   * @note Will fail for outputs that are memory-planned or constants.
   */
  ET_NODISCARD
  runtime::Error set_outputs(
      const std::string& method_name,
      const std::vector<runtime::EValue>& output_values);

  /**
   * Sets all output tensors for the "forward" method.
   *
   * @param[in] output_values A vector of EValues to set as the method outputs.
   *
   * @returns An Error to indicate success or failure.
   *
   * @note Only Tensor outputs are currently supported for setting.
   * @note Will fail for outputs that are memory-planned or constants.
   */
  ET_NODISCARD
  inline runtime::Error set_outputs(
      const std::vector<runtime::EValue>& output_values) {
    return set_outputs("forward", output_values);
  }

  /**
   * Retrieve all current output values of a specific method without executing
   * it. Loads the program and method before retrieval if needed.
   *
   * @param[in] method_name The name of the method.
   *
   * @returns A Result containing the vector of output values, or an error.
   */
  ET_NODISCARD
  runtime::Result<std::vector<runtime::EValue>> get_outputs(
      const std::string& method_name);

  /**
   * Retrieve all current output values of the "forward" method without
   * executing it. Loads the program and method before retrieval if needed.
   *
   * @returns A Result containing the vector of output values, or an error.
   */
  ET_NODISCARD
  inline runtime::Result<std::vector<runtime::EValue>> get_outputs() {
    return get_outputs("forward");
  }

  /**
   * Retrieve a single current output value of a specific method without
   * executing it. Loads the program and method before retrieval if needed.
   *
   * @param[in] method_name The name of the method.
   * @param[in] output_index Zero-based index of the output to retrieve.
   *
   * @returns A Result containing the requested output value, or an error.
   */
  ET_NODISCARD
  runtime::Result<runtime::EValue> get_output(
      const std::string& method_name,
      size_t output_index = 0);

  /**
   * Retrieve a single current output value of the "forward" method without
   * executing it. Loads the program and method before retrieval if needed.
   *
   * @param[in] output_index Zero-based index of the output to retrieve.
   *
   * @returns A Result containing the requested output value, or an error.
   */
  ET_NODISCARD
  inline runtime::Result<runtime::EValue> get_output(size_t output_index = 0) {
    return get_output("forward", output_index);
  }

  /**
   * Retrieves the EventTracer instance being used by the Module.
   * EventTracer is used for tracking and logging events during the execution
   * of methods.
   *
   * @returns A pointer to the EventTracer instance. Returns nullptr if no
   * EventTracer is set.
   */
  inline runtime::EventTracer* event_tracer() const {
    return event_tracer_.get();
  }

  // Note: this debug_buffer will always be empty. The one being used is in
  // the event_tracer attached to module. Please use that one.
  ET_DEPRECATED ET_NODISCARD runtime::Span<uint8_t> debug_buffer() {
    return runtime::Span<uint8_t>(debug_buffer_.data(), debug_buffer_.size());
  }

 private:
  struct MethodHolder {
    std::vector<std::vector<uint8_t>> planned_buffers;
    std::vector<runtime::Span<uint8_t>> planned_spans;
    std::unique_ptr<runtime::HierarchicalAllocator> planned_memory;
    std::unique_ptr<runtime::MemoryManager> memory_manager;
    std::unique_ptr<Method> method;
  };

  std::string file_path_;
  std::string data_map_path_;
  LoadMode load_mode_{LoadMode::File};
  std::shared_ptr<Program> program_;
  std::unique_ptr<runtime::DataLoader> data_loader_;
  std::unique_ptr<runtime::MemoryAllocator> memory_allocator_;
  std::unique_ptr<runtime::MemoryAllocator> temp_allocator_;
  std::unique_ptr<runtime::EventTracer> event_tracer_;
  std::unique_ptr<runtime::DataLoader> data_map_loader_;
  std::unique_ptr<NamedDataMap> data_map_;
  ET_DEPRECATED std::vector<uint8_t> debug_buffer_;

 protected:
  std::unordered_map<std::string, MethodHolder> methods_;

  friend class executorch::extension::ExecuTorchJni;
};

} // namespace ET_MODULE_NAMESPACE
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::ET_MODULE_NAMESPACE::Module;
} // namespace executor
} // namespace torch

namespace executorch {
namespace extension {
// backward compatible namespace alias
using ::executorch::extension::ET_MODULE_NAMESPACE::Module;
} // namespace extension
} // namespace executorch
