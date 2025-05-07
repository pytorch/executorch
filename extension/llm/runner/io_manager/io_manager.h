/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

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
 * IOManagerBase provides an interface for handling the input preparation and
 * output processing for both prefill and decode phases of LLM inference.
 * Derived classes must implement the virtual methods to provide specific IO
 * management functionality.
 */
class ET_EXPERIMENTAL IOManagerBase {
 public:
  /**
   * @brief Virtual destructor to allow proper cleanup in derived classes.
   */
  virtual ~IOManagerBase() = default;

  /**
   * @brief Initialize the IO manager with method metadata for prefill and
   * decode operations.
   *
   * @param prefill_method The prefill method to initialize with.
   * @param decode_method The decode method to initialize with.
   */
  ET_NODISCARD virtual runtime::Error init(
      executorch::ET_RUNTIME_NAMESPACE::Method& prefill_method,
      executorch::ET_RUNTIME_NAMESPACE::Method& decode_method) = 0;

  /**
   * @brief Reset the IO manager state.
   *
   * @param prefill_method The prefill method to reset with.
   * @param decode_method The decode method to reset with.
   */
  ET_NODISCARD virtual runtime::Error reset(
      executorch::ET_RUNTIME_NAMESPACE::Method& prefill_method,
      executorch::ET_RUNTIME_NAMESPACE::Method& decode_method) = 0;

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
      executorch::ET_RUNTIME_NAMESPACE::Method& prefill_method) = 0;

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
      executorch::ET_RUNTIME_NAMESPACE::Method& decode_method) = 0;

  /**
   * @brief Process and update internal state with outputs from the prefill
   * phase.
   *
   * @param prefill_method The prefill method to update with outputs.
   * @param model_outputs Vector of outputs from the prefill method execution.
   */
  ET_NODISCARD virtual runtime::Error update_prefill(
      executorch::ET_RUNTIME_NAMESPACE::Method& prefill_method,
      const std::vector<executorch::runtime::EValue>& model_outputs) = 0;

  /**
   * @brief Process and update internal state with outputs from the decode
   * phase.
   *
   * @param decode_method The decode method to update with outputs.
   * @param model_outputs Vector of outputs from the decode method execution.
   */
  ET_NODISCARD virtual runtime::Error update_decode(
      const executorch::ET_RUNTIME_NAMESPACE::Method& decode_method,
      const std::vector<executorch::runtime::EValue>& model_outputs) = 0;
};

} // namespace llm
} // namespace extension
} // namespace executorch
