/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/executor/program.h>

namespace executorch {
namespace extension {
namespace training {

/**
 * A facade class for loading programs for on-device training and executing
 * methods within them.
 */
class ET_EXPERIMENTAL TrainingModule final : executorch::extension::Module {
 public:
  explicit TrainingModule(
      std::unique_ptr<runtime::DataLoader> data_loader,
      std::unique_ptr<runtime::MemoryAllocator> memory_allocator = nullptr,
      std::unique_ptr<runtime::MemoryAllocator> temp_allocator = nullptr,
      std::unique_ptr<runtime::EventTracer> event_tracer = nullptr)
      : executorch::extension::Module(
            std::move(data_loader),
            std::move(memory_allocator),
            std::move(temp_allocator),
            std::move(event_tracer)),
        method_named_gradients_({}) {}

  explicit TrainingModule(const Module&) = delete;
  TrainingModule& operator=(const Module&) = delete;
  explicit TrainingModule(Module&&) = delete;
  TrainingModule& operator=(Module&&) = delete;

  /**
   * Execute a specific method with the given input and retrieve output. Only
   * valid if the specified method is a joint graph. Loads the program and
   * method before executing if needed.
   *
   * @param[in] method_name The name of the joint graph method to execute.
   * @param[in] input A vector of input values to be passed to the method.
   *
   * @returns A Result object containing the output values from the method or an
   * error to indicate failure.
   */
  ET_EXPERIMENTAL runtime::Result<std::vector<runtime::EValue>>
  execute_forward_backward(
      const std::string& method_name,
      const std::vector<runtime::EValue>& input);

  /**
   * Retrieve the trainable parameters for a joint graph method.
   *
   * @param[in] method_name The name of the joint graph method to get the
   * parameters for.
   *
   * @returns A Result object containing a map of the fully qualified name to
   * parameter tensor, or an error if the method is not a joint graph.
   */
  ET_EXPERIMENTAL
  runtime::Result<
      const std::map<executorch::aten::string_view, executorch::aten::Tensor>>
  named_parameters(const std::string& method_name);

  /**
   * Retrieve the latest gradients for a joint graph method.
   *
   * @param[in] method_name The name of the joint graph method to get the
   * gradients for.
   *
   * @returns A Result object containing a map of the fully qualified name to
   * gradient tensor associated with that parameter from the latest
   * forward_backward execution, or an error if the method is not a joint graph
   * or has not been executed yet.
   */
  ET_EXPERIMENTAL
  runtime::Result<
      const std::map<executorch::aten::string_view, executorch::aten::Tensor>>
  named_gradients(const std::string& method_name);

 private:
  std::unordered_map<
      std::string,
      std::map<executorch::aten::string_view, executorch::aten::Tensor>>
      method_named_gradients_;
};

} // namespace training
} // namespace extension
} // namespace executorch
