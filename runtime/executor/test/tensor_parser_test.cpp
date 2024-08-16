/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/tensor_parser.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/schema/program_generated.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::deserialization::parseTensor;
using executorch::runtime::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

class TensorParserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Load the serialized ModuleAdd data.
    const char* path = std::getenv("ET_MODULE_ADD_PATH");
    Result<FileDataLoader> float_loader = FileDataLoader::from(path);
    ASSERT_EQ(float_loader.error(), Error::Ok);
    float_loader_ =
        std::make_unique<FileDataLoader>(std::move(float_loader.get()));

    // Load the serialized ModuleAddHalf data.
    const char* half_path = std::getenv("ET_MODULE_ADD_HALF_PATH");
    Result<FileDataLoader> half_loader = FileDataLoader::from(half_path);
    ASSERT_EQ(half_loader.error(), Error::Ok);
    half_loader_ =
        std::make_unique<FileDataLoader>(std::move(half_loader.get()));
  }

  std::unique_ptr<FileDataLoader> float_loader_;
  std::unique_ptr<FileDataLoader> half_loader_;
};

namespace executorch {
namespace runtime {
namespace testing {
// Provides access to private Program methods.
class ProgramTestFriend final {
 public:
  const static executorch_flatbuffer::Program* GetInternalProgram(
      const Program* program) {
    return program->internal_program_;
  }
};
} // namespace testing
} // namespace runtime
} // namespace executorch

using executorch::runtime::testing::ProgramTestFriend;

void test_module_add(
    std::unique_ptr<FileDataLoader>& loader,
    ScalarType scalar_type,
    int type_size) {
  Result<Program> program =
      Program::load(loader.get(), Program::Verification::Minimal);
  EXPECT_EQ(program.error(), Error::Ok);

  const Program* program_ = &program.get();
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);

  const executorch_flatbuffer::Program* internal_program =
      ProgramTestFriend::GetInternalProgram(program_);
  executorch_flatbuffer::ExecutionPlan* execution_plan =
      internal_program->execution_plan()->GetMutableObject(0);
  auto flatbuffer_values = execution_plan->values();

  int tensor_count = 0;
  int double_count = 0;
  for (size_t i = 0; i < flatbuffer_values->size(); ++i) {
    auto serialization_value = flatbuffer_values->Get(i);
    if (serialization_value->val_type() ==
        executorch_flatbuffer::KernelTypes::Tensor) {
      tensor_count++;
      Result<Tensor> tensor = parseTensor(
          program_, &mmm.get(), serialization_value->val_as_Tensor());
      Tensor t = tensor.get();
      ASSERT_EQ(scalar_type, t.scalar_type());
      ASSERT_EQ(2, t.dim()); // [2, 2]
      ASSERT_EQ(4, t.numel());
      ASSERT_EQ(type_size * t.numel(), t.nbytes());
    } else if (
        serialization_value->val_type() ==
        executorch_flatbuffer::KernelTypes::Double) {
      double_count++;
      ASSERT_EQ(1.0, serialization_value->val_as_Double()->double_val());
    }
  }
  ASSERT_EQ(3, tensor_count); // input x2, output
  ASSERT_EQ(2, double_count); // alpha x2
}

TEST_F(TensorParserTest, TestModuleAddFloat) {
  test_module_add(float_loader_, ScalarType::Float, sizeof(float));
}

TEST_F(TensorParserTest, TestModuleAddHalf) {
  test_module_add(half_loader_, ScalarType::Half, sizeof(exec_aten::Half));
}

TEST_F(TensorParserTest, TestMutableState) {
  // Load the serialized ModuleSimpleTrain data.
  const char* path = std::getenv("ET_MODULE_SIMPLE_TRAIN_PATH");
  Result<FileDataLoader> train_loader = FileDataLoader::from(path);
  ASSERT_EQ(train_loader.error(), Error::Ok);

  Result<Program> program =
      Program::load(&train_loader.get(), Program::Verification::Minimal);
  EXPECT_EQ(program.error(), Error::Ok);

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  ManagedMemoryManager mmm_copy(
      kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);

  const executorch_flatbuffer::Program* internal_program =
      ProgramTestFriend::GetInternalProgram(&program.get());
  executorch_flatbuffer::ExecutionPlan* execution_plan =
      internal_program->execution_plan()->GetMutableObject(0);
  auto flatbuffer_values = execution_plan->values();

  size_t num_mutable_tensors = 0;
  for (size_t i = 0; i < flatbuffer_values->size(); ++i) {
    auto serialization_value = flatbuffer_values->Get(i);
    if (serialization_value->val_type() ==
            executorch_flatbuffer::KernelTypes::Tensor &&
        serialization_value->val_as_Tensor()->allocation_info() != nullptr &&
        serialization_value->val_as_Tensor()->data_buffer_idx() > 0) {
      num_mutable_tensors++;
      Result<torch::executor::Tensor> tensor = parseTensor(
          &program.get(), &mmm.get(), serialization_value->val_as_Tensor());
      torch::executor::Tensor t = tensor.get();
      float loaded_value = t.const_data_ptr<float>()[0];
      ASSERT_NE(nullptr, t.const_data_ptr());
      ASSERT_NE(t.mutable_data_ptr<float>()[0], 0.5);
      t.mutable_data_ptr<float>()[0] = 0.5;
      ASSERT_EQ(
          t.mutable_data_ptr<float>()[0],
          0.5); // 0.5 can be represented perfectly by float so EQ and NE work
                // fine here. Any power of 2 rational can be perfectly
                // represented. See dyadic rationals for more info.

      // Load the same tensor using the same mem manager and show the value is
      // updated again.
      Result<torch::executor::Tensor> tensor1_alias = parseTensor(
          &program.get(), &mmm.get(), serialization_value->val_as_Tensor());
      torch::executor::Tensor t2 = tensor.get();
      ASSERT_NE(t2.mutable_data_ptr<float>()[0], 0.5);

      // Show the tensors are equivalent
      ASSERT_EQ(t.const_data_ptr(), t2.const_data_ptr());
      // Set mutable tensor value back to 0.5 since it got overwritten by second
      // parse.
      t.mutable_data_ptr<float>()[0] = 0.5;

      // Load the same tensor using a different mem manager and show the value
      // is not the same as t.
      Result<torch::executor::Tensor> tensor_new = parseTensor(
          &program.get(),
          &mmm_copy.get(),
          serialization_value->val_as_Tensor());
      torch::executor::Tensor t3 = tensor_new.get();
      ASSERT_NE(t3.mutable_data_ptr<float>()[0], 0.5);
      ASSERT_NE(t3.const_data_ptr(), t.const_data_ptr());
      ASSERT_EQ(loaded_value, t3.const_data_ptr<float>()[0]);
    }
  }
  ASSERT_EQ(num_mutable_tensors, 2);
}
