/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <filesystem>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/executor.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/util.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using torch::executor::Error;
using torch::executor::EValue;
using torch::executor::Executor;
using torch::executor::Program;
using torch::executor::Result;
using torch::executor::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

// TODO(T158932073): Tests the deprecated Executor APIs. Remove this file when
// Executor is deleted.
class ExecutionPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a loader for the serialized ModuleAdd program.
    const char* path = std::getenv("ET_MODULE_ADD_PATH");
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));

    // Use it to load the program.
    Result<Program> program = Program::Load(
        loader_.get(), Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);
    program_ = std::make_unique<Program>(std::move(program.get()));
  }

 private:
  // Must outlive program_, but tests shouldn't need to touch it.
  std::unique_ptr<FileDataLoader> loader_;

 protected:
  std::unique_ptr<Program> program_;
};

TEST_F(ExecutionPlanTest, SuccessfulInitSmoke) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Executor executor(program_.get(), &mmm.get());
  auto& plan = executor.execution_plan();

  // Cannot execute an uninitialized plan.
  Error err = plan.execute();
  ASSERT_NE(err, Error::Ok);

  // First successful initialization.
  err = executor.init_execution_plan();
  ASSERT_EQ(err, Error::Ok);

  // Can execute now that the plan has been initialized. Note that it's not safe
  // to prepare input tensors before the plan has been initialized.
  exec_aten::ArrayRef<void*> inputs =
      torch::executor::util::PrepareInputTensors(plan);
  err = plan.execute();
  ASSERT_EQ(err, Error::Ok);

  // Cannot initialize again.
  err = executor.init_execution_plan();
  ASSERT_NE(err, Error::Ok);

  // But can still execute it.
  err = plan.execute();
  ASSERT_EQ(err, Error::Ok);

  torch::executor::util::FreeInputs(inputs);
}

TEST_F(ExecutionPlanTest, FailedInitSmoke) {
  // A memory manager that provides no memory, which will cause the plan init to
  // fail.
  ManagedMemoryManager mmm(/*non_const_mem_bytes=*/0, /*runtime_mem_bytes=*/0);

  Executor executor(program_.get(), &mmm.get());
  auto& plan = executor.execution_plan();

  // Cannot execute an uninitialized plan.
  Error err = plan.execute();
  ASSERT_NE(err, Error::Ok);

  // Init fails by running out of memory.
  err = executor.init_execution_plan();
  ASSERT_EQ(err, Error::MemoryAllocationFailed);

  // Cannot execute a plan that failed to initialize.
  err = plan.execute();
  ASSERT_NE(err, Error::Ok);

  // Other operations that fail on an uninitialized plan.
  EValue val;
  err = plan.set_input(val, /*input_idx=*/0);
  ASSERT_EQ(err, Error::InvalidState);

  ArrayRef<EValue> vals(
      /*data=*/nullptr,
      // '+' stops promotion of 0 to nullptr, which would be ambiguous here.
      /*length=*/+0);
  err = plan.set_inputs(vals);
  ASSERT_EQ(err, Error::InvalidState);

  err = plan.get_outputs(/*output_evalues=*/nullptr, /*length=*/0);
  ASSERT_EQ(err, Error::InvalidState);

  // Cannot re-initialize the failed plan.
  err = executor.init_execution_plan();
  ASSERT_EQ(err, Error::InvalidState);
}
