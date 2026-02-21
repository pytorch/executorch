/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/runner_util/inputs.h>

#include <cstdlib>
#include <cstring>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::BufferCleanup;
using executorch::extension::FileDataLoader;
using executorch::extension::prepare_input_tensors;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::Tag;
using executorch::runtime::testing::ManagedMemoryManager;

class InputsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::executor::runtime_init();

    // Load ModuleAdd
    const char* add_path = std::getenv("ET_MODULE_ADD_PATH");
    ASSERT_NE(add_path, nullptr)
        << "ET_MODULE_ADD_PATH environment variable must be set";
    Result<FileDataLoader> add_loader = FileDataLoader::from(add_path);
    ASSERT_EQ(add_loader.error(), Error::Ok);
    add_loader_ = std::make_unique<FileDataLoader>(std::move(add_loader.get()));

    Result<Program> add_program = Program::load(
        add_loader_.get(), Program::Verification::InternalConsistency);
    ASSERT_EQ(add_program.error(), Error::Ok);
    add_program_ = std::make_unique<Program>(std::move(add_program.get()));

    add_mmm_ = std::make_unique<ManagedMemoryManager>(
        /*planned_memory_bytes=*/32 * 1024U,
        /*method_allocator_bytes=*/32 * 1024U);

    Result<Method> add_method =
        add_program_->load_method("forward", &add_mmm_->get());
    ASSERT_EQ(add_method.error(), Error::Ok);
    add_method_ = std::make_unique<Method>(std::move(add_method.get()));

    // Load ModuleIntBool
    const char* intbool_path = std::getenv("ET_MODULE_INTBOOL_PATH");
    ASSERT_NE(intbool_path, nullptr)
        << "ET_MODULE_INTBOOL_PATH environment variable must be set";
    Result<FileDataLoader> intbool_loader = FileDataLoader::from(intbool_path);
    ASSERT_EQ(intbool_loader.error(), Error::Ok);
    intbool_loader_ =
        std::make_unique<FileDataLoader>(std::move(intbool_loader.get()));

    Result<Program> intbool_program = Program::load(
        intbool_loader_.get(), Program::Verification::InternalConsistency);
    ASSERT_EQ(intbool_program.error(), Error::Ok);
    intbool_program_ =
        std::make_unique<Program>(std::move(intbool_program.get()));

    intbool_mmm_ = std::make_unique<ManagedMemoryManager>(
        /*planned_memory_bytes=*/32 * 1024U,
        /*method_allocator_bytes=*/32 * 1024U);

    Result<Method> intbool_method =
        intbool_program_->load_method("forward", &intbool_mmm_->get());
    ASSERT_EQ(intbool_method.error(), Error::Ok);
    intbool_method_ = std::make_unique<Method>(std::move(intbool_method.get()));
  }

 private:
  std::unique_ptr<FileDataLoader> add_loader_;
  std::unique_ptr<Program> add_program_;
  std::unique_ptr<ManagedMemoryManager> add_mmm_;

  std::unique_ptr<FileDataLoader> intbool_loader_;
  std::unique_ptr<Program> intbool_program_;
  std::unique_ptr<ManagedMemoryManager> intbool_mmm_;

 protected:
  std::unique_ptr<Method> add_method_;
  std::unique_ptr<Method> intbool_method_;
};

TEST_F(InputsTest, Smoke) {
  Result<BufferCleanup> input_buffers = prepare_input_tensors(*add_method_);
  ASSERT_EQ(input_buffers.error(), Error::Ok);
  auto input_err = add_method_->set_input(executorch::runtime::EValue(1.0), 2);
  ASSERT_EQ(input_err, Error::Ok);

  // We can't look at the input tensors, but we can check that the outputs make
  // sense after executing the method.
  Error status = add_method_->execute();
  ASSERT_EQ(status, Error::Ok);

  // Get the single output, which should be a floating-point Tensor.
  ASSERT_EQ(add_method_->outputs_size(), 1);
  const EValue& output_value = add_method_->get_output(0);
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

TEST_F(InputsTest, ExceedingInputCountLimitFails) {
  // The smoke test above demonstrated that we can prepare inputs with the
  // default limits. It should fail if we lower the max below the number of
  // actual inputs.
  MethodMeta method_meta = add_method_->method_meta();
  size_t num_inputs = method_meta.num_inputs();
  ASSERT_GE(num_inputs, 1);
  executorch::extension::PrepareInputTensorsOptions options;
  options.max_inputs = num_inputs - 1;

  Result<BufferCleanup> input_buffers =
      prepare_input_tensors(*add_method_, options);
  ASSERT_NE(input_buffers.error(), Error::Ok);
}

TEST_F(InputsTest, ExceedingInputAllocationLimitFails) {
  // The smoke test above demonstrated that we can prepare inputs with the
  // default limits. It should fail if we lower the max below the actual
  // allocation size.
  executorch::extension::PrepareInputTensorsOptions options;
  // The input tensors are float32, so 1 byte will always be smaller than any
  // non-empty input tensor.
  options.max_total_allocation_size = 1;

  Result<BufferCleanup> input_buffers =
      prepare_input_tensors(*add_method_, options);
  ASSERT_NE(input_buffers.error(), Error::Ok);
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

TEST_F(InputsTest, DoubleInputWrongSizeFails) {
  MethodMeta method_meta = add_method_->method_meta();

  // ModuleAdd has 3 inputs: tensor, tensor, double (alpha)
  ASSERT_EQ(method_meta.num_inputs(), 3);

  // Verify input 2 is a Double
  auto tag = method_meta.input_tag(2);
  ASSERT_TRUE(tag.ok());
  ASSERT_EQ(tag.get(), Tag::Double);

  // Create input_buffers with wrong size for the Double input
  std::vector<std::pair<char*, size_t>> input_buffers;

  // Allocate correct buffers for tensors (inputs 0 and 1)
  auto tensor0_meta = method_meta.input_tensor_meta(0);
  auto tensor1_meta = method_meta.input_tensor_meta(1);
  ASSERT_TRUE(tensor0_meta.ok());
  ASSERT_TRUE(tensor1_meta.ok());

  std::vector<char> buf0(tensor0_meta->nbytes(), 0);
  std::vector<char> buf1(tensor1_meta->nbytes(), 0);

  // ModuleAdd expects alpha=1.0. Need to set this correctly, otherwise
  // set_input fails validation before the buffer overflow happens.
  double alpha = 1.0;
  // Double is size 8; use a larger buffer to invoke overflow.
  char large_buffer[16];
  memcpy(large_buffer, &alpha, sizeof(double));

  input_buffers.push_back({buf0.data(), buf0.size()});
  input_buffers.push_back({buf1.data(), buf1.size()});
  input_buffers.push_back({large_buffer, sizeof(large_buffer)});

  Result<BufferCleanup> result =
      prepare_input_tensors(*add_method_, {}, input_buffers);
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

TEST_F(InputsTest, IntBoolInputWrongSizeFails) {
  MethodMeta method_meta = intbool_method_->method_meta();

  // ModuleIntBool has 3 inputs: tensor, int, bool
  ASSERT_EQ(method_meta.num_inputs(), 3);

  // Verify input types
  auto int_tag = method_meta.input_tag(1);
  ASSERT_TRUE(int_tag.ok());
  ASSERT_EQ(int_tag.get(), Tag::Int);

  auto bool_tag = method_meta.input_tag(2);
  ASSERT_TRUE(bool_tag.ok());
  ASSERT_EQ(bool_tag.get(), Tag::Bool);

  // Allocate correct buffer for tensor (input 0)
  auto tensor0_meta = method_meta.input_tensor_meta(0);
  ASSERT_TRUE(tensor0_meta.ok());
  std::vector<char> buf0(tensor0_meta->nbytes(), 0);

  // Prepare scalar values
  int64_t y = 1;
  bool z = true;

  // Test 1: Int input with wrong size
  {
    std::vector<std::pair<char*, size_t>> input_buffers;

    // Int is size 8; use a larger buffer to invoke overflow.
    char large_int_buffer[16];
    memcpy(large_int_buffer, &y, sizeof(int64_t));

    char bool_buffer[sizeof(bool)];
    memcpy(bool_buffer, &z, sizeof(bool));

    input_buffers.push_back({buf0.data(), buf0.size()});
    input_buffers.push_back({large_int_buffer, sizeof(large_int_buffer)});
    input_buffers.push_back({bool_buffer, sizeof(bool_buffer)});

    Result<BufferCleanup> result =
        prepare_input_tensors(*intbool_method_, {}, input_buffers);
    EXPECT_EQ(result.error(), Error::InvalidArgument);
  }

  // Test 2: Bool input with wrong size
  {
    std::vector<std::pair<char*, size_t>> input_buffers;

    char int_buffer[sizeof(int64_t)];
    memcpy(int_buffer, &y, sizeof(int64_t));

    // Bool is size 1; use a larger buffer to invoke overflow.
    char large_bool_buffer[8];
    memcpy(large_bool_buffer, &z, sizeof(bool));

    input_buffers.push_back({buf0.data(), buf0.size()});
    input_buffers.push_back({int_buffer, sizeof(int_buffer)});
    input_buffers.push_back({large_bool_buffer, sizeof(large_bool_buffer)});

    Result<BufferCleanup> result =
        prepare_input_tensors(*intbool_method_, {}, input_buffers);
    EXPECT_EQ(result.error(), Error::InvalidArgument);
  }
}
