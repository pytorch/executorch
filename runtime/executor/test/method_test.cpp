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
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using executorch::extension::prepare_input_tensors;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

class MethodTest : public ::testing::Test {
 protected:
  void load_program(const char* path, const char* module_name) {
    // Create a loader for the serialized program.
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    loaders_.insert(
        {module_name,
         std::make_unique<FileDataLoader>(std::move(loader.get()))});

    // Use it to load the program.
    Result<Program> program = Program::load(
        loaders_[module_name].get(),
        Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);
    programs_.insert(
        {module_name, std::make_unique<Program>(std::move(program.get()))});
  }

  void SetUp() override {
    executorch::runtime::runtime_init();

    load_program(std::getenv("ET_MODULE_ADD_PATH"), "add");
    load_program(std::getenv("ET_MODULE_INDEX_PATH"), "index");
    load_program(
        std::getenv("ET_MODULE_DYNAMIC_CAT_UNALLOCATED_IO_PATH"), "cat");
    load_program(std::getenv("ET_MODULE_LINEAR_PATH"), "linear");
    load_program(
        std::getenv("DEPRECATED_ET_MODULE_LINEAR_CONSTANT_BUFFER_PATH"),
        "linear_constant_buffer");
  }

 private:
  // Must outlive program_, but tests shouldn't need to touch it.
  std::unordered_map<std::string, std::unique_ptr<FileDataLoader>> loaders_;

 protected:
  std::unordered_map<std::string, std::unique_ptr<Program>> programs_;
};

TEST_F(MethodTest, MoveTest) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = programs_["add"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  // Can execute the method.
  auto input_cleanup = prepare_input_tensors(*method);
  ASSERT_EQ(input_cleanup.error(), Error::Ok);
  Error err = method->execute();
  ASSERT_EQ(err, Error::Ok);

  // Move into a new Method.
  Method new_method(std::move(method.get()));

  // Can't execute the old method.
  err = method->execute();
  ASSERT_NE(err, Error::Ok);

  // Can execute the new method.
  err = new_method.execute();
  ASSERT_EQ(err, Error::Ok);
}

TEST_F(MethodTest, GetInputTests) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = programs_["add"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  size_t num_inputs = method->inputs_size();
  ASSERT_GT(num_inputs, 0);

  // In-range inputs should succeed without aborting.
  method->get_input(0);
  method->get_input(num_inputs - 1);

  // Out-of-range inputs should abort.
  ET_EXPECT_DEATH(method->get_input(num_inputs), "");
  ET_EXPECT_DEATH(method->get_input(num_inputs + 1), "");
}

TEST_F(MethodTest, MutableInputTests) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = programs_["add"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  size_t num_inputs = method->inputs_size();
  ASSERT_GT(num_inputs, 0);

  // In-range inputs should succeed without aborting.
  method->mutable_input(0);
  method->mutable_input(num_inputs - 1);

  // Out-of-range inputs should abort.
  ET_EXPECT_DEATH(method->mutable_input(num_inputs), "");
  ET_EXPECT_DEATH(method->mutable_input(num_inputs + 1), "");
}

TEST_F(MethodTest, GetOutputTests) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = programs_["add"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  size_t num_outputs = method->outputs_size();
  ASSERT_GT(num_outputs, 0);

  // In-range outputs should succeed without aborting.
  method->get_output(0);
  method->get_output(num_outputs - 1);

  // Out-of-range outputs should abort.
  ET_EXPECT_DEATH(method->get_output(num_outputs), "");
  ET_EXPECT_DEATH(method->get_output(num_outputs + 1), "");
}

TEST_F(MethodTest, MutableOutputTests) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = programs_["add"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  size_t num_outputs = method->outputs_size();
  ASSERT_GT(num_outputs, 0);

  // In-range outputs should succeed without aborting.
  method->mutable_output(0);
  method->mutable_output(num_outputs - 1);

  // Out-of-range outputs should abort.
  ET_EXPECT_DEATH(method->mutable_output(num_outputs), "");
  ET_EXPECT_DEATH(method->mutable_output(num_outputs + 1), "");
}

TEST_F(MethodTest, SetPrimInputTest) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = programs_["add"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  // Can execute the method.
  auto input_cleanup = prepare_input_tensors(*method);
  ASSERT_EQ(input_cleanup.error(), Error::Ok);

  // The args to the method are x, y, alpha. x and y are tensors handled above
  // alpha is a prim.

  // Traced prim input was '1.0' so 3.0 should error.
  auto input_err = method->set_input(EValue(3.0), 2);
  EXPECT_EQ(input_err, Error::InvalidArgument);

  // Traced prim input was '1.0' so '1.0' should be ok.
  input_err = method->set_input(EValue(1.0), 2);
  ASSERT_EQ(input_err, Error::Ok);

  Error err = method->execute();
  EXPECT_EQ(err, Error::Ok);
}

TEST_F(MethodTest, MethodMetaTest) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = programs_["add"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  auto method_meta = method->method_meta();

  EXPECT_EQ(method_meta.num_inputs(), method->inputs_size());
  EXPECT_EQ(method_meta.num_outputs(), method->outputs_size());
}

TEST_F(MethodTest, AliasedIOTest) {
  // TODO(T163238401)
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = programs_["cat"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  // Set up io. Input and Output should share the same memory.
  constexpr int buffer_size = 16;
  float buffer[buffer_size]; // Initial input is (2,4) we then cat a (1,4) to it
                             // twice for a final shape of (4,4)
  for (int i = 0; i < buffer_size; ++i) {
    buffer[i] = 0.f;
  }
  int32_t sizes[2] = {2, 4};
  uint8_t dim_order[2] = {0, 1};
  int32_t strides[2] = {4, 1};
  exec_aten::TensorImpl impl(
      exec_aten::ScalarType::Float, 2, sizes, buffer, dim_order, strides);

  auto input_err = method->set_input(EValue(exec_aten::Tensor(&impl)), 0);
  ASSERT_EQ(input_err, Error::Ok);

  auto output_err = method->set_output_data_ptr(buffer, sizeof(buffer), 0);
  ASSERT_EQ(output_err, Error::Ok);
  ASSERT_EQ(method->get_output(0).toTensor().const_data_ptr(), buffer);

  // Execute the method once. Cat a 1x4 to a 2x4.
  auto execute_error = method->execute();
  ASSERT_EQ(execute_error, Error::Ok);

  auto output = method->get_output(0);
  ASSERT_TRUE(output.isTensor());
  EXPECT_EQ(output.toTensor().sizes()[0], 3);
  EXPECT_EQ(output.toTensor().sizes()[1], 4);
  // Original input should be 0.
  for (size_t i = 0; i < 2 * 4; i++) {
    EXPECT_FLOAT_EQ(output.toTensor().const_data_ptr<float>()[i], 0.f);
  }
  // Section that was cat on should be 1.
  for (size_t i = 0; i < 1 * 4; i++) {
    EXPECT_FLOAT_EQ(
        output.toTensor().const_data_ptr<float>()[(2 * 4) + i], 1.f);
  }

  // Set the input again to update the size.
  sizes[0] = output.toTensor().sizes()[0];
  exec_aten::TensorImpl impl_2(
      exec_aten::ScalarType::Float, 2, sizes, buffer, dim_order, strides);
  input_err = method->set_input(EValue(exec_aten::Tensor(&impl_2)), 0);
  ASSERT_EQ(input_err, Error::Ok);

  // Execute the method again. Cat a 1x4 to a 3x4.
  execute_error = method->execute();
  ASSERT_EQ(execute_error, Error::Ok);

  output = method->get_output(0);
  EXPECT_EQ(output.toTensor().sizes()[0], 4);
  EXPECT_EQ(output.toTensor().sizes()[1], 4);
  // Original input should be 0.
  for (size_t i = 0; i < 2 * 4; i++) {
    EXPECT_FLOAT_EQ(output.toTensor().const_data_ptr<float>()[i], 0.f);
  }
  // Previous section and the new one that were cat on should be 1.
  for (size_t i = 0; i < 2 * 4; i++) {
    EXPECT_FLOAT_EQ(
        output.toTensor().const_data_ptr<float>()[(2 * 4) + i], 1.f);
  }
}

TEST_F(MethodTest, ConstantSegmentTest) {
  // Execute model with constants stored in segment.
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method =
      programs_["linear"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  // Can execute the method.
  Error err = method->execute();
  ASSERT_EQ(err, Error::Ok);
}

TEST_F(MethodTest, ConstantBufferTest) {
  // Execute model with constants stored in the program flatbuffer.
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method =
      programs_["linear_constant_buffer"]->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  // Can execute the method.
  Error err = method->execute();
  ASSERT_EQ(err, Error::Ok);
}

/*
 * TODO(T161163608): Test is disabled due to a resize bug in tensor_index_out of
 * the portable op lib

TEST_F(MethodTest, OptionalTensorListDeserialization) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes,
  kDefaultRuntimeMemBytes); Result<Method> method =
  index_program_->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  // Can execute the method.
  auto input_cleanup = prepare_input_tensors(*method);
  ASSERT_EQ(input_cleanup.error(), Error::Ok);
  Error err = method->execute();
  ASSERT_EQ(err, Error::Ok);

  EXPECT_EQ(method->inputs_size(), 1);

  auto outputs = method->get_output(0);
  EXPECT_EQ(outputs.toTensor().dim(), 3);
  EXPECT_EQ(outputs.toTensor().size(0), 5);
  EXPECT_EQ(outputs.toTensor().size(1), 2);
  EXPECT_EQ(outputs.toTensor().size(2), 10);
}
*/
