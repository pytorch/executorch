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
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::ArrayRef;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Kernel;
using executorch::runtime::KernelKey;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

/**
 * Used to control and observe the behavior of a kernel.
 */
struct KernelControl {
 public:
  // The number of times the kernel has been called.
  int call_count = 0;

  // If true, the kernel should call `context.fail(error_to_set)`. If false,
  // the kernel should not call `context.fail()`.
  bool call_context_fail = true;

  // The error value that the kernel should pass to `context.fail()` before
  // returning.
  Error fail_value = Error::Ok;

  void reset() {
    call_count = 0;
    call_context_fail = false;
    fail_value = Error::Ok;
  }

  /**
   * Registers a kernel that uses the singleton instance to record and control
   * its behavior.
   */
  static void register_singleton() {
    if (registered_) {
      return;
    }

    // This test helper installs itself as aten::add.out:
    //
    // add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) ->
    //     Tensor(a!)
    //
    // The arguments are: `self, other, out, out` (we repeat the out argument in
    // the program). And since we traced using randn(2, 2), all the args are
    // Float with dim order (0, 1)

    // Construct a kernel key with the following meta:
    // exec_aten::DimOrderType contiguous[] = {0, 1};
    // TensorMeta float_contiguous[] = {
    //     TensorMeta(ScalarType::Float, contiguous), // self
    //     TensorMeta(ScalarType::Float, contiguous), // other
    //     TensorMeta(ScalarType::Float, contiguous), // out
    //     TensorMeta(ScalarType::Float, contiguous)}; // out (repeated)
    KernelKey key =
        executorch::runtime::KernelKey("v1/6;0,1|6;0,1|6;0,1|6;0,1");
    Kernel kernel = executorch::runtime::Kernel(
        "aten::add.out", key, KernelControl::kernel_hook);
    Error err = executorch::runtime::register_kernels({kernel});
    EXPECT_EQ(err, Error::Ok);

    registered_ = true;
  }

  static KernelControl* singleton() {
    return &singleton_;
  }

 private:
  /**
   * An OpFunction-compatible function that uses the singleton KernelControl
   * to record and determine its behavior.
   */
  static void kernel_hook(
      KernelRuntimeContext& context,
      __ET_UNUSED EValue** args) {
    auto* control = KernelControl::singleton();
    control->call_count++;
    if (control->call_context_fail) {
      context.fail(control->fail_value);
    }
  }

  static bool registered_;
  static KernelControl singleton_;
};

bool KernelControl::registered_ = false;
KernelControl KernelControl::singleton_;

class KernelIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();

    // Register the controllable kernel hook.
    KernelControl::register_singleton();
    // Ensure that its state is clear.
    KernelControl::singleton()->reset();
    // Provide the singleton to the tests.
    control_ = KernelControl::singleton();

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

    // Load the forward method.
    mmm_ = std::make_unique<ManagedMemoryManager>(
        kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
    Result<Method> method = program_->load_method("forward", &mmm_->get());
    ASSERT_EQ(method.error(), Error::Ok);
    method_ = std::make_unique<Method>(std::move(method.get()));

    // Set up its inputs.
    inputs_ = torch::executor::util::PrepareInputTensors(*method_);
  }

  void TearDown() override {
    torch::executor::util::FreeInputs(inputs_);
    inputs_ = {};
  }

 private:
  // Must outlive program_
  std::unique_ptr<FileDataLoader> loader_;

  // Must outlive method_
  std::unique_ptr<Program> program_;
  std::unique_ptr<ManagedMemoryManager> mmm_;
  ArrayRef<void*> inputs_;

 protected:
  // An executable method that will call the kernel associated with control_.
  // Its inputs will have been allocated and initialized.
  std::unique_ptr<Method> method_;

  // The KernelControl associated with method_.
  KernelControl* control_;
};

TEST_F(KernelIntegrationTest, KernelHookIsCalled) {
  // Demonstrate that the kernel hook is called in the default state.
  EXPECT_EQ(control_->call_count, 0);
  Error err = method_->execute();
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(control_->call_count, 1);

  // Calling it again bumps the count.
  err = method_->execute();
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(control_->call_count, 2);
}

TEST_F(KernelIntegrationTest, FailurePropagates) {
  // Tell the kernel to fail.
  control_->call_context_fail = true;

  // We should see the error from the kernel.
  control_->fail_value = Error::InvalidArgument;
  Error err = method_->execute();
  EXPECT_EQ(err, Error::InvalidArgument);
  EXPECT_EQ(control_->call_count, 1);

  // Have it fail with a different error to show that it's not a coincidence.
  control_->fail_value = Error::MemoryAllocationFailed;
  err = method_->execute();
  EXPECT_EQ(err, Error::MemoryAllocationFailed);
  EXPECT_EQ(control_->call_count, 2);

  // Returning an Ok does not cause the execution to fail.
  control_->fail_value = Error::Ok;
  err = method_->execute();
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(control_->call_count, 3);
}
