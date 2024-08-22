/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/runner_util/inputs.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::extension::BufferCleanup;
using executorch::extension::FileDataLoader;
using executorch::extension::prepare_input_tensors;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::Tag;
using executorch::runtime::testing::ManagedMemoryManager;

class InputsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::executor::runtime_init();

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

    mmm_ = std::make_unique<ManagedMemoryManager>(
        /*planned_memory_bytes=*/32 * 1024U,
        /*method_allocator_bytes=*/32 * 1024U);

    // Load the forward method.
    Result<Method> method = program_->load_method("forward", &mmm_->get());
    ASSERT_EQ(method.error(), Error::Ok);
    method_ = std::make_unique<Method>(std::move(method.get()));
  }

 private:
  // Must outlive method_, but tests shouldn't need to touch them.
  std::unique_ptr<FileDataLoader> loader_;
  std::unique_ptr<ManagedMemoryManager> mmm_;
  std::unique_ptr<Program> program_;

 protected:
  std::unique_ptr<Method> method_;
};

TEST_F(InputsTest, Smoke) {
  Result<BufferCleanup> input_buffers = prepare_input_tensors(*method_);
  ASSERT_EQ(input_buffers.error(), Error::Ok);

  // We can't look at the input tensors, but we can check that the outputs make
  // sense after executing the method.
  Error status = method_->execute();
  ASSERT_EQ(status, Error::Ok);

  // Get the single output, which should be a floating-point Tensor.
  ASSERT_EQ(method_->outputs_size(), 1);
  const EValue& output_value = method_->get_output(0);
  ASSERT_EQ(output_value.tag, Tag::Tensor);
  Tensor output = output_value.toTensor();
  ASSERT_EQ(output.scalar_type(), ScalarType::Float);

  // ModuleAdd adds its two inputs together, so if the input elements were set
  // to 1, the output elemements should all be 2.
  Span<float> elements(output.mutable_data_ptr<float>(), output.numel());
  EXPECT_GT(elements.size(), 0); // Make sure we're actually testing something.
  for (float e : elements) {
    EXPECT_EQ(e, 2.0);
  }

  // Although it's tough to test directly, ASAN should let us know if
  // BufferCleanup doesn't behave properly: either freeing too soon or leaking
  // the pointers.
}

TEST(BufferCleanupTest, Smoke) {
  // Returns the size of the buffer at index `i`.
  auto test_buffer_size = [](size_t i) {
    // Use multiples of OS page sizes. As this gets bigger, we're more
    // likely to allocate outside the main heap in a separate page, making
    // it easier to catch uses-after-free.
    return 4096 << i;
  };

  // Create some buffers.
  constexpr size_t kNumBuffers = 8;
  void** buffers = (void**)malloc(kNumBuffers * sizeof(void*));
  for (int i = 0; i < kNumBuffers; i++) {
    size_t nbytes = test_buffer_size(i);
    buffers[i] = malloc(nbytes);
    memset(reinterpret_cast<char*>(buffers[i]), 0x00, nbytes);
  }

  std::unique_ptr<BufferCleanup> bc2;
  {
    // bc1 should own `buffers` and the buffers that its entries point to.
    BufferCleanup bc1({buffers, kNumBuffers});

    // They're still alive; no segfaults or ASAN complaints if we write to them.
    for (int i = 0; i < kNumBuffers; i++) {
      size_t nbytes = test_buffer_size(i);
      memset(reinterpret_cast<char*>(buffers[i]), 0xff, nbytes);
    }

    // Move ownership to a new object.
    bc2 = std::make_unique<BufferCleanup>(std::move(bc1));

    // Still alive.
    for (int i = 0; i < kNumBuffers; i++) {
      size_t nbytes = test_buffer_size(i);
      memset(reinterpret_cast<char*>(buffers[i]), 0x00, nbytes);
    }

    // bc1 goes out of scope here. If it thinks it owns the buffers, it will
    // try to free them.
  }

  // bc2 should own the buffers now, and they should still be alive.
  for (int i = 0; i < kNumBuffers; i++) {
    size_t nbytes = test_buffer_size(i);
    memset(reinterpret_cast<char*>(buffers[i]), 0xff, nbytes);
  }

  // Destroy bc2, which should destroy the buffers. There's no way for us to
  // check that it happened, but the sanitizer should complain if there's a
  // memory leak. And if bc1 freed them before, we should get a double-free
  // complaint.
  bc2.reset();
}
