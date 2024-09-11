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
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::ArrayRef;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Kernel;
using executorch::runtime::KernelKey;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::MemoryAllocator;
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

  // If true, the kernel should allocate temporary memory.
  bool allocate_temp_memory = false;

  // If true, the kernel should simulate allocating temporary memory.
  bool simulate_temp_memory_allocation = false;

  // The size of the temporary memory to allocate.
  int temp_memory_size = 0;

  // The total size of all allocations.
  int total_allocated_size = 0;

  void reset() {
    call_count = 0;
    call_context_fail = false;
    fail_value = Error::Ok;
    allocate_temp_memory = false;
    simulate_temp_memory_allocation = false;
    temp_memory_size = 0;
    total_allocated_size = 0;
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
    Error err = executorch::runtime::register_kernel(kernel);
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
      ET_UNUSED EValue** args) {
    auto* control = KernelControl::singleton();
    control->call_count++;
    if (control->call_context_fail) {
      context.fail(control->fail_value);
    }

    // Allocate temporary memory.
    if (control->allocate_temp_memory) {
      Result<void*> temp_mem_res =
          context.allocate_temp(control->temp_memory_size);
      if (temp_mem_res.ok()) {
        control->total_allocated_size += control->temp_memory_size;
        // We actually use the memory, to test default memory allocation was
        // successful.
        uint8_t* array = (uint8_t*)(temp_mem_res.get());
        for (int i = 0; i < control->temp_memory_size; i++) {
          array[i] = i % 256;
        }
      }
    }

    // Simulate allocating temporary memory. We use this, for testing that when
    // a temp allocator is provided, the kernel will use it, instead of
    // allocating memory with the default platform memory allocator.
    // The provided TempMemoryAllocator class in this file, simulates allocating
    // memory instead of actually allocating anything.
    if (control->simulate_temp_memory_allocation) {
      Result<void*> temp_mem_res =
          context.allocate_temp(control->temp_memory_size);
      control->total_allocated_size += control->temp_memory_size;
      EXPECT_EQ(temp_mem_res.error(), Error::Ok);
    }
  }

  static bool registered_;
  static KernelControl singleton_;
};

bool KernelControl::registered_ = false;
KernelControl KernelControl::singleton_;

/**
 * MemoryAllocator that keeps track of the number/sizes of its allocations,
 * to test the case where the user provides a temp allocator.
 */
class TempMemoryAllocator final : public MemoryAllocator {
 public:
  TempMemoryAllocator() : MemoryAllocator(0, nullptr) {}

  // The number of times allocate() has been called.
  int number_of_allocations = 0;

  // The number of times reset() has been called.
  int number_of_resets = 0;

  // The amount of memory currently allocated (should go to 0 when reset is
  // called).
  int currently_allocated_size = 0;

  // The total size of all allocations.
  int total_allocated_size = 0;

  void* allocate(size_t size, ET_UNUSED size_t alignment = kDefaultAlignment)
      override {
    number_of_allocations += 1;
    currently_allocated_size += size;
    total_allocated_size += size;
    // This is a simulation, we don't actually allocate memory. But we need to
    // return a non-null pointer, so we return a bad, non-zero address that will
    // crash if anyone tries to dereference it.
    return (void*)1;
  }

  void reset() override {
    number_of_resets += 1;
    currently_allocated_size = 0;
  }
};

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
        kDefaultNonConstMemBytes,
        kDefaultRuntimeMemBytes,
        temp_allocator_.get());
    Result<Method> method = program_->load_method("forward", &mmm_->get());
    ASSERT_EQ(method.error(), Error::Ok);
    method_ = std::make_unique<Method>(std::move(method.get()));

    // Set up its inputs.
    auto inputs_cleanup =
        executorch::extension::prepare_input_tensors(*method_);
    ASSERT_EQ(inputs_cleanup.error(), Error::Ok);
    inputs_cleanup_ = std::make_unique<executorch::extension::BufferCleanup>(
        std::move(*inputs_cleanup));
  }

  void TearDown() override {
    inputs_cleanup_.reset();
  }

 private:
  // Must outlive program_
  std::unique_ptr<FileDataLoader> loader_;

  // Must outlive method_
  std::unique_ptr<Program> program_;
  std::unique_ptr<ManagedMemoryManager> mmm_;
  std::unique_ptr<executorch::extension::BufferCleanup> inputs_cleanup_;

 protected:
  // An executable method that will call the kernel associated with control_.
  // Its inputs will have been allocated and initialized.
  std::unique_ptr<Method> method_;

  // The KernelControl associated with method_.
  KernelControl* control_;

  // The temp memory allocator provided by the user. By default, none is
  // provided.
  std::unique_ptr<TempMemoryAllocator> temp_allocator_ = nullptr;
};

class KernelTempMemoryAllocatorIntegrationTest : public KernelIntegrationTest {
 protected:
  void SetUp() override {
    // Create a temp allocator for the test before calling the parent SetUp.
    temp_allocator_ = std::make_unique<TempMemoryAllocator>();
    KernelIntegrationTest::SetUp();
  }
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

TEST_F(KernelIntegrationTest, DefaultPlatformMemoryAllocator) {
  // Tell the kernel to allocate memory. Since no temp allocator is provided,
  // this will allocate memory using the default platform memory allocator.
  control_->allocate_temp_memory = true;

  control_->temp_memory_size = 4;
  // This is not a simulation. This actually allocates memory, using the
  // default platform memory allocator.
  Error err = method_->execute();
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(control_->call_count, 1);
  EXPECT_EQ(control_->total_allocated_size, 4);

  control_->temp_memory_size = 8;
  // This is not a simulation. This actually allocates memory, using the
  // default platform memory allocator.
  err = method_->execute();
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(control_->call_count, 2);
  EXPECT_EQ(control_->total_allocated_size, 12);
}

TEST_F(KernelTempMemoryAllocatorIntegrationTest, UsingTempMemoryAllocator) {
  // In this test we provide a temp allocator to the method, and tell the kernel
  // to allocate memory using it. We want to make sure that the kernel uses the
  // temp allocator, and that the temp allocator is reset after the execution.
  // Since we are testing that the kernel uses the temp allocator, and not the
  // temp allocator itself, we don't need to test the actual allocation of
  // memory. Therefore, we set simulate_temp_memory_allocation to true, so that
  // the kernel will not actually allocate memory, but will instead simulate
  // allocating memory.
  // The provided TempMemoryAllocator, simulates allocating memory by increasing
  // total_allocated_size and currently_allocated_size by the requested size.
  // We simulate resetting the allocator by setting currently_allocated_size
  // back to 0.
  control_->simulate_temp_memory_allocation = true;

  control_->temp_memory_size = 4;
  Error err = method_->execute();
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(control_->call_count, 1);
  EXPECT_EQ(control_->total_allocated_size, 4);
  EXPECT_EQ(temp_allocator_->number_of_allocations, 1);
  EXPECT_EQ(temp_allocator_->total_allocated_size, 4);
  // The temp allocator should have been reset after the execution.
  EXPECT_EQ(temp_allocator_->number_of_resets, 1);
  EXPECT_EQ(temp_allocator_->currently_allocated_size, 0);

  control_->temp_memory_size = 8;
  err = method_->execute();
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(control_->call_count, 2);
  EXPECT_EQ(control_->total_allocated_size, 12);
  EXPECT_EQ(temp_allocator_->number_of_allocations, 2);
  EXPECT_EQ(temp_allocator_->total_allocated_size, 12);
  // The temp allocator should have been reset after the execution.
  EXPECT_EQ(temp_allocator_->number_of_resets, 2);
  EXPECT_EQ(temp_allocator_->currently_allocated_size, 0);
}
