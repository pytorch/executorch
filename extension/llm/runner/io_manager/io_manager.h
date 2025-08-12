/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>

namespace executorch {
namespace extension {
namespace llm {

/**
 * @brief Base class for managing input/output operations for LLM inference.
 *
 * IOManager provides an interface for handling the input preparation and
 * output processing for both prefill and decode phases of LLM inference.
 * Derived classes must implement the virtual methods to provide specific IO
 * management functionality.
 */
class ET_EXPERIMENTAL IOManager {
 public:
  /**
   * @brief Virtual destructor to allow proper cleanup in derived classes.
   */
  virtual ~IOManager() = default;

  /**
   * @brief Load the IO manager with method metadata for prefill and
   * decode operations.
   *
   * @param program The program prefill and decode methods are loaded from.
   * @param prefill_method The prefill method to initialize with.
   * @param decode_method The decode method to initialize with.
   */
  ET_NODISCARD virtual runtime::Error load(
      const executorch::ET_RUNTIME_NAMESPACE::Program& program,
      executorch::ET_RUNTIME_NAMESPACE::Method& prefill_method,
      executorch::ET_RUNTIME_NAMESPACE::Method& decode_method) {
    (void)program;
    (void)prefill_method;
    (void)decode_method;
    return runtime::Error::Ok;
  }

  /**
   * @brief Reset the IO manager state.
   *
   * @param prefill_method The prefill method to reset with.
   * @param decode_method The decode method to reset with.
   */
  ET_NODISCARD virtual runtime::Error reset(
      executorch::ET_RUNTIME_NAMESPACE::Method& prefill_method,
      executorch::ET_RUNTIME_NAMESPACE::Method& decode_method) {
    (void)prefill_method;
    (void)decode_method;
    return runtime::Error::Ok;
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
  virtual runtime::Result<std::vector<executorch::runtime::EValue>>
  prepare_prefill(
      const executorch::extension::TensorPtr& input,
      const executorch::extension::TensorPtr& start_pos,
      executorch::ET_RUNTIME_NAMESPACE::Method& prefill_method) {
    if (prefill_method.inputs_size() != 2) {
      ET_LOG(
          Error,
          "Expected 2 inputs for prefill method, got %zu. Likely the model takes the caches or mask as an argument which this IOManager does not support.",
          prefill_method.inputs_size());
      return runtime::Error::InvalidState;
    }
    // Cpu IO Manager supports dynamic shapes for prefill, so no work to be done
    // here.
    return std::vector<runtime::EValue>{input, start_pos};
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
  virtual runtime::Result<std::vector<executorch::runtime::EValue>>
  prepare_decode(
      const executorch::extension::TensorPtr& input,
      const executorch::extension::TensorPtr& start_pos,
      executorch::ET_RUNTIME_NAMESPACE::Method& decode_method) {
    if (decode_method.inputs_size() != 2) {
      ET_LOG(
          Error,
          "Expected 2 inputs for decode method, got %zu. Likely the model takes the caches or mask as an argument which this IOManager does not support.",
          decode_method.inputs_size());
      return runtime::Error::InvalidState;
    }
    // Cpu IO Manager supports dynamic shapes for prefill, so no work to be done
    // here.
    return std::vector<runtime::EValue>{input, start_pos};
  }

  /**
   * @brief Process and update internal state with outputs from the prefill
   * phase.
   *
   * @param prefill_method The prefill method to update with outputs.
   * @param model_outputs Vector of outputs from the prefill method execution.
   */
  ET_NODISCARD virtual runtime::Error update_prefill(
      executorch::ET_RUNTIME_NAMESPACE::Method& prefill_method,
      const std::vector<executorch::runtime::EValue>& model_outputs) {
    (void)prefill_method;
    (void)model_outputs;
    // No post inference work to do.
    return runtime::Error::Ok;
  }

  /**
   * @brief Process and update internal state with outputs from the decode
   * phase.
   *
   * @param decode_method The decode method to update with outputs.
   * @param model_outputs Vector of outputs from the decode method execution.
   */
  ET_NODISCARD virtual runtime::Error update_decode(
      const executorch::ET_RUNTIME_NAMESPACE::Method& decode_method,
      const std::vector<executorch::runtime::EValue>& model_outputs) {
    (void)decode_method;
    (void)model_outputs;
    // No post inference work to do.
    return runtime::Error::Ok;
  }
};

} // namespace llm
} // namespace extension
} // namespace executorch
