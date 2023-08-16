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
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/util.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using torch::executor::Error;
using torch::executor::EValue;
using torch::executor::Method;
using torch::executor::Program;
using torch::executor::Result;
using torch::executor::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

class MethodTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a loader for the serialized ModuleAdd program.
    const char* path = std::getenv("ET_MODULE_ADD_PATH");
    Result<FileDataLoader> loader = FileDataLoader::From(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    add_loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));

    // Use it to load the program.
    Result<Program> program = Program::Load(
        add_loader_.get(), Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);
    add_program_ = std::make_unique<Program>(std::move(program.get()));

    // Create a loader for the serialized ModuleIndex program.
    const char* index_path = std::getenv("ET_MODULE_INDEX_PATH");
    Result<FileDataLoader> index_loader = FileDataLoader::From(index_path);
    ASSERT_EQ(index_loader.error(), Error::Ok);
    index_loader_ =
        std::make_unique<FileDataLoader>(std::move(index_loader.get()));

    // Use it to load the program.
    Result<Program> index_program = Program::Load(
        index_loader_.get(), Program::Verification::InternalConsistency);
    ASSERT_EQ(index_program.error(), Error::Ok);
    index_program_ = std::make_unique<Program>(std::move(index_program.get()));
  }

 private:
  // Must outlive program_, but tests shouldn't need to touch it.
  std::unique_ptr<FileDataLoader> add_loader_;
  std::unique_ptr<FileDataLoader> index_loader_;

 protected:
  std::unique_ptr<Program> add_program_;
  std::unique_ptr<Program> index_program_;
};

TEST_F(MethodTest, MoveTest) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = add_program_->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  // Can execute the method.
  exec_aten::ArrayRef<void*> inputs =
      torch::executor::util::PrepareInputTensors(*method);
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

  torch::executor::util::FreeInputs(inputs);
}

// TODO(T161163608): Test is disabled due to a resize bug in tensor_index_out of
// the portable op lib

// TEST_F(MethodTest, OptionalTensorListDeserialization) {
//   ManagedMemoryManager mmm(kDefaultNonConstMemBytes,
//   kDefaultRuntimeMemBytes); Result<Method> method =
//   index_program_->load_method("forward", &mmm.get());
//   ASSERT_EQ(method.error(), Error::Ok);

//   // Can execute the method.
//   exec_aten::ArrayRef<void*> inputs =
//       torch::executor::util::PrepareInputTensors(*method);
//   Error err = method->execute();
//   ASSERT_EQ(err, Error::Ok);

//   EXPECT_EQ(method->inputs_size(), 1);

//   auto outputs = method->get_output(0);
//   EXPECT_EQ(outputs.toTensor().dim(), 3);
//   EXPECT_EQ(outputs.toTensor().size(0), 5);
//   EXPECT_EQ(outputs.toTensor().size(1), 2);
//   EXPECT_EQ(outputs.toTensor().size(2), 10);

//   torch::executor::util::FreeInputs(inputs);
// }
