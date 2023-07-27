#pragma once

#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/compiler.h>

namespace torch {
namespace executor {

// TODO(T158932073): Remove this once all clients use the new name/APIs.
using ExecutionPlan = Method;

// TODO(T158932073): Remove this once Program::load_method is available and all
// clients use it.
class Executor {
 public:
  // Executes a PyTorch executor program.
  Executor(const Program* program, MemoryManager* memory_manager);

  /**
   * DEPRECATED: Use init_execution_plan(const char*)
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
   * Initializes the execution plan to use the specified entry point of the
   * model. `execution_plan()` returns this plan.
   *
   * May only be called once for the lifetime of the Executor.
   *
   * @param[in] name The name of the entry point to use for this plan.
   * @retval Error::Ok on successful initialization.
   */
  __ET_NODISCARD Error init_execution_plan(const char* method_name);

  /**
   * Returns the plan that was initialized by `init_execution_plan()`.
   */
  ExecutionPlan& execution_plan() {
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
