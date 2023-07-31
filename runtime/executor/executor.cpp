/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/executor.h>

#include <cinttypes>
#include <cstdint>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/schema/schema_generated.h>

namespace torch {
namespace executor {

Executor::Executor(const Program* program, MemoryManager* memory_manager)
    : program_(program), plan_(program, memory_manager) {}

Error Executor::init_execution_plan(size_t index) {
  EXECUTORCH_SCOPE_PROF("ExecPlan::init_execution_plan");
  auto serialization_plan =
      program_->get_internal_program()->execution_plan()->GetMutableObject(
          index);
  return plan_.init(serialization_plan);
}

Error Executor::init_execution_plan(const char* method_name) {
  EXECUTORCH_SCOPE_PROF("ExecPlan::init_execution_plan");
  auto internal_program = static_cast<const executorch_flatbuffer::Program*>(
      program_->get_internal_program());
  auto execution_plans = internal_program->execution_plan();
  for (size_t i = 0; i < execution_plans->size(); i++) {
    auto serialization_plan = execution_plans->GetMutableObject(i);
    if (std::strcmp(serialization_plan->name()->c_str(), method_name) == 0) {
      return plan_.init(serialization_plan);
    }
  }
  return Error::InvalidArgument;
}

} // namespace executor
} // namespace torch
