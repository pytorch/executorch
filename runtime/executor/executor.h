/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/compiler.h>

namespace torch {
namespace executor {

// TODO(T158932073): Remove this once all clients use the new name/APIs.
/// DEPRECATED: Use `Method` instead.
using ExecutionPlan __ET_DEPRECATED = Method;

// TODO(T158932073): Remove this once Program::load_method is available and all
// clients use it.
/// DEPRECATED: Use `Program::load_method()` instead.
class __ET_DEPRECATED Executor {
 public:
  // Executes a PyTorch executor program.
  Executor(const Program* program, MemoryManager* memory_manager);

  /**
   * DEPRECATED: Use `Program::load_method()` instead.
   *
   * Initializes the execution plan to use the specified entry point
   * of the model. `execution_plan()` returns this plan.
   *
   * May only be called once for the lifetime of the Executor.
   *
   * @param[in] index The index of the entry point to use for this plan.
   *     Defaults to using the `forward()` method.
   * @retval Error::Ok on successful initialization.
   */
  __ET_DEPRECATED __ET_NODISCARD Error
  init_execution_plan(size_t index = Program::kForwardMethodIndex);

  /**
   * DEPRECATED: Use `Program::load_method()` instead.
   *
   * Initializes the execution plan to use the specified entry point of the
   * model. `execution_plan()` returns this plan.
   *
   * May only be called once for the lifetime of the Executor.
   *
   * @param[in] name The name of the entry point to use for this plan.
   * @retval Error::Ok on successful initialization.
   */
  __ET_DEPRECATED __ET_NODISCARD Error
  init_execution_plan(const char* method_name);

  /**
   * DEPRECATED: Use `Program::load_method()` instead.
   *
   * Returns the plan that was initialized by `init_execution_plan()`.
   */
  __ET_DEPRECATED ExecutionPlan& execution_plan() {
    return plan_;
  }

  ~Executor() = default;

 private:
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;
  Executor(Executor&&) = delete;
  Executor& operator=(Executor&&) = delete;

  const Program* program_;
  ExecutionPlan plan_;
};

} // namespace executor
} // namespace torch
