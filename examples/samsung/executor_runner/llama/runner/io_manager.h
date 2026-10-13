/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <executorch/extension/llm/runner/io_manager/io_manager.h>

namespace example {

/**
 * @brief Base class for managing input/output operations for LLM inference.
 *
 * IOManager provides an interface for handling the input preparation and
 * output processing for both prefill and decode phases of LLM inference.
 * Derived classes must implement the virtual methods to provide specific IO
 * management functionality.
 */
class ET_EXPERIMENTAL LlmIOManager
    : public executorch::extension::llm::IOManager {
 public:
  /**
   * @brief Construct an IOManager bound to a Module.
   *
   * @param module The Module used for querying method metadata and execution.
   */
  explicit LlmIOManager(executorch::extension::Module& module)
      : IOManager(module), module_(module) {}

  /**
   * @brief Virtual destructor to allow proper cleanup in derived classes.
   */
  virtual ~LlmIOManager() = default;

  /**
   * @brief Load the IO manager with method metadata for prefill and
   * decode operations.
   *
   * @param prefill_method The prefill method to initialize with.
   * @param decode_method The decode method to initialize with.
   */
  ET_NODISCARD virtual executorch::runtime::Error load(
      const std::string& prefill_method,
      const std::string& decode_method) override {
    return executorch::runtime::Error::Ok;
  }

  /**
   * @brief Reset the IO manager state.
   *
   * @param prefill_method The prefill method to reset with.
   * @param decode_method The decode method to reset with.
   */
  ET_NODISCARD virtual executorch::runtime::Error reset(
      const std::string& prefill_method,
      const std::string& decode_method) override {
    (void)prefill_method;
    (void)decode_method;
    return executorch::runtime::Error::Ok;
  }

  /**
   * @brief Prepare inputs for the prefill phase of LLM inference.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position of the current
   * input within the context.
   * @param prefill_method The prefill method to prepare inputs for.
   * @return std::vector<executorch::runtime::EValue> Vector of prepared inputs
   * for the prefill method.
   */
  virtual executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
  prepare_prefill(
      const executorch::extension::TensorPtr& input,
      const executorch::extension::TensorPtr& start_pos,
      const std::string& prefill_method) override {
    auto method_meta = module_.method_meta(prefill_method);
    if (!method_meta.ok()) {
      return method_meta.error();
    }
    if (method_meta->num_inputs() != 2) {
      ET_LOG(
          Error,
          "Expected 2 inputs for prefill method, got %zu. Likely the model takes the caches or mask as an argument which this IOManager does not support.",
          method_meta->num_inputs());
      return executorch::runtime::Error::InvalidState;
    }
    // Cpu IO Manager supports dynamic shapes for prefill, so no work to be done
    // here.
    return std::vector<executorch::runtime::EValue>{input, start_pos};
  }

  /**
   * @brief Prepare inputs for the prefill phase using the default method name.
   *
   * Uses "forward" as the prefill method.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position.
   * @return Vector of prepared inputs for the prefill method.
   */
  executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
  prepare_prefill(
      const executorch::extension::TensorPtr& input,
      const executorch::extension::TensorPtr& start_pos) {
    return prepare_prefill(input, start_pos, "forward");
  }

  /**
   * @brief Prepare inputs for the decode phase of LLM inference.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position of the current
   * input within the context.
   * @param decode_method The decode method to prepare inputs for.
   * @return std::vector<executorch::runtime::EValue> Vector of prepared inputs
   * for the decode method.
   */
  virtual executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
  prepare_decode(
      const executorch::extension::TensorPtr& input,
      const executorch::extension::TensorPtr& start_pos,
      const std::string& decode_method) override {
    auto method_meta = module_.method_meta(decode_method);
    if (!method_meta.ok()) {
      return method_meta.error();
    }
    if (method_meta->num_inputs() != 2) {
      ET_LOG(
          Error,
          "Expected 2 inputs for decode method, got %zu. Likely the model takes the caches or mask as an argument which this IOManager does not support.",
          method_meta->num_inputs());
      return executorch::runtime::Error::InvalidState;
    }
    // Cpu IO Manager supports dynamic shapes for prefill, so no work to be done
    // here.
    return std::vector<executorch::runtime::EValue>{input, start_pos};
  }

  /**
   * @brief Prepare inputs for the decode phase using the default method name.
   *
   * Uses "forward" as the decode method.
   *
   * @param input The input tensor containing token IDs.
   * @param start_pos The tensor containing the starting position.
   * @return Vector of prepared inputs for the decode method.
   */
  executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
  prepare_decode(
      const executorch::extension::TensorPtr& input,
      const executorch::extension::TensorPtr& start_pos) {
    return prepare_decode(input, start_pos, "forward");
  }

  /**
   * @brief Process and update internal state with outputs from the prefill
   * phase.
   *
   * @param prefill_method The prefill method to update with outputs.
   * @param model_outputs Vector of outputs from the prefill method execution.
   */
  ET_NODISCARD virtual executorch::runtime::Error update_prefill(
      const std::vector<executorch::runtime::EValue>& model_outputs,
      const std::string& prefill_method) override {
    (void)model_outputs;
    (void)prefill_method;
    // No post inference work to do.
    return executorch::runtime::Error::Ok;
  }

  /**
   * @brief Process outputs from the prefill phase using the default method.
   *
   * Uses "forward" as the prefill method.
   *
   * @param model_outputs Vector of outputs from the prefill execution.
   * @return Error code.
   */
  ET_NODISCARD executorch::runtime::Error update_prefill(
      const std::vector<executorch::runtime::EValue>& model_outputs) {
    return update_prefill(model_outputs, "forward");
  }

  /**
   * @brief Process and update internal state with outputs from the decode
   * phase.
   *
   * @param decode_method The decode method to update with outputs.
   * @param model_outputs Vector of outputs from the decode method execution.
   */
  ET_NODISCARD virtual executorch::runtime::Error update_decode(
      const std::vector<executorch::runtime::EValue>& model_outputs,
      const std::string& decode_method) override {
    (void)model_outputs;
    (void)decode_method;
    // No post inference work to do.
    return executorch::runtime::Error::Ok;
  }

  /**
   * @brief Process outputs from the decode phase using the default method.
   *
   * Uses "forward" as the decode method.
   *
   * @param model_outputs Vector of outputs from the decode execution.
   * @return Error code.
   */
  ET_NODISCARD executorch::runtime::Error update_decode(
      const std::vector<executorch::runtime::EValue>& model_outputs) {
    return update_decode(model_outputs, "forward");
  }

 private:
  /**
   * @brief Reference to the Module used for method metadata and execution.
   */
  executorch::extension::module::Module& module_;
};

} // namespace example
