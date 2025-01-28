/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cctype>
#include <filesystem>

#include <cstring>
#include <memory>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Kernel;
using executorch::runtime::KernelKey;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::register_kernel;
using executorch::runtime::Result;
using executorch::runtime::TensorMeta;
using executorch::runtime::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

class KernelResolutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
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

  std::unique_ptr<FileDataLoader> loader_;
  std::unique_ptr<Program> program_;
};

/**
 * Test if the program can initialize properly.
 */
TEST_F(KernelResolutionTest, InitExecutionPlanSuccess) {
  // register kernel with fallback kernel key
  Kernel kernel_1 = Kernel(
      "aten::add.out", {}, [](KernelRuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  auto s1 = register_kernel(kernel_1);
  EXPECT_EQ(s1, executorch::runtime::Error::Ok);

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  auto method = program_->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);
}

/**
 * Test if we can resolve the kernel key correctly.
 */
TEST_F(KernelResolutionTest, ResolveKernelKeySuccess) {
  // getting all these TensorMeta from args to this kernel_call in the program.
  // particularly for aten::add.out:
  // add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) ->
  // Tensor(a!) The arguments are: `self, other, out, out` (we repeat out
  // argument in the program) Also since we traced using randn(2, 2), all the
  // args are Float with dim order (0, 1)

  // Construct a kernel key with the following meta:
  // exec_aten::DimOrderType contiguous[] = {0, 1};
  // TensorMeta float_contiguous[] = {
  //     TensorMeta(ScalarType::Float, contiguous),
  //     TensorMeta(ScalarType::Float, contiguous),
  //     TensorMeta(ScalarType::Float, contiguous),
  //     TensorMeta(ScalarType::Float, contiguous)};
  KernelKey key = KernelKey("v1/6;0,1|6;0,1|6;0,1|6;0,1");
  Kernel kernel_1 = Kernel(
      "aten::add.out", key, [](KernelRuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  auto s1 = register_kernel(kernel_1);
  EXPECT_EQ(s1, executorch::runtime::Error::Ok);

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  auto method = program_->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);
}
