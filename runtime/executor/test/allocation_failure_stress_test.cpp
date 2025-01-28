/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <filesystem>
#include <memory>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::Scalar;
using exec_aten::Tensor;
using executorch::extension::FileDataLoader;
using executorch::extension::prepare_input_tensors;
using executorch::runtime::Error;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::testing::ManagedMemoryManager;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

class AllocationFailureStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();

    // Create a loader for the serialized ModuleAdd program.
    const char* path = std::getenv("ET_MODULE_ADD_PATH");
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));

    // Use it to load the program.
    Result<Program> program = Program::load(
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

/**
 * Slowly increases the amount of available runtime memory until load_method()
 * and execute() succeed. This should cause every runtime allocation to fail at
 * some point, exercising every allocation failure path reachable by the test
 * model.
 */
TEST_F(AllocationFailureStressTest, End2EndIncreaseRuntimeMemUntilSuccess) {
  size_t runtime_mem_bytes = 0;
  Error err = Error::Internal;
  size_t num_load_failures = 0;
  while (runtime_mem_bytes < kDefaultRuntimeMemBytes && err != Error::Ok) {
    ManagedMemoryManager mmm(kDefaultNonConstMemBytes, runtime_mem_bytes);

    // Loading should fail several times from allocation failures.
    Result<Method> method = program_->load_method("forward", &mmm.get());
    if (method.error() != Error::Ok) {
      runtime_mem_bytes += sizeof(size_t);
      num_load_failures++;
      continue;
    }

    // Execution does not use the runtime allocator, so it should always succeed
    // once load was successful.
    auto input_cleanup = prepare_input_tensors(*method);
    ASSERT_EQ(input_cleanup.error(), Error::Ok);
    err = method->execute();
    ASSERT_EQ(err, Error::Ok);
  }
  EXPECT_GT(num_load_failures, 0) << "Expected at least some failures";
  EXPECT_EQ(err, Error::Ok)
      << "Did not succeed after increasing runtime_mem_bytes to "
      << runtime_mem_bytes;
}

/**
 * Slowly increases the amount of available non-constant memory until
 * load_method() and execute() succeed. This should cause every non-const
 * allocation to fail at some point, exercising every allocation failure path
 * reachable by the test model.
 */
TEST_F(AllocationFailureStressTest, End2EndNonConstantMemUntilSuccess) {
  size_t non_constant_mem_bytes = 0;
  Error err = Error::Internal;
  size_t num_load_failures = 0;
  while (non_constant_mem_bytes < kDefaultNonConstMemBytes &&
         err != Error::Ok) {
    ManagedMemoryManager mmm(non_constant_mem_bytes, kDefaultRuntimeMemBytes);

    // Loading should fail several times from allocation failures.
    Result<Method> method = program_->load_method("forward", &mmm.get());
    if (method.error() != Error::Ok) {
      non_constant_mem_bytes += sizeof(size_t);
      num_load_failures++;
      continue;
    }

    // Execution does not use the runtime allocator, so it should always succeed
    // once load was successful.
    auto input_cleanup = prepare_input_tensors(*method);
    ASSERT_EQ(input_cleanup.error(), Error::Ok);
    err = method->execute();
    ASSERT_EQ(err, Error::Ok);
  }
  EXPECT_GT(num_load_failures, 0) << "Expected at least some failures";
  EXPECT_EQ(err, Error::Ok)
      << "Did not succeed after increasing non_constant_mem_bytes to "
      << non_constant_mem_bytes;
}
