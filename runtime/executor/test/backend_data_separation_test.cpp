
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/exir/backend/test/demos/rpc/ExecutorBackend.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::FlatTensorDataMap;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

class BackendDataSeparationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Make sure that the backend has been registered. Safe to call multiple
    // times. Doing this at runtime ensures that it's only registered if these
    // tests are run.
    ASSERT_EQ(example::register_executor_backend(), Error::Ok);

    // Create data loaders.
    Result<FileDataLoader> linear_program_loader = FileDataLoader::from(
        std::getenv("ET_MODULE_LINEAR_DELEGATE_PROGRAM_PATH"));
    ASSERT_EQ(linear_program_loader.error(), Error::Ok);
    linear_program_loader_ = std::make_unique<FileDataLoader>(
        std::move(linear_program_loader.get()));

    Result<FileDataLoader> linear_data_loader =
        FileDataLoader::from(std::getenv("ET_MODULE_LINEAR_DATA_PATH"));
    ASSERT_EQ(linear_data_loader.error(), Error::Ok);
    linear_data_loader_ =
        std::make_unique<FileDataLoader>(std::move(linear_data_loader.get()));

    // Create programs.
    Result<Program> linear_program = Program::load(
        linear_program_loader_.get(),
        Program::Verification::InternalConsistency);
    ASSERT_EQ(linear_program.error(), Error::Ok);
    linear_program_ =
        std::make_unique<Program>(std::move(linear_program.get()));

    Result<FlatTensorDataMap> linear_data_map =
        FlatTensorDataMap::load(linear_data_loader_.get());
    EXPECT_EQ(linear_data_map.error(), Error::Ok);
    linear_data_map_ =
        std::make_unique<FlatTensorDataMap>(std::move(linear_data_map.get()));

    ET_LOG(
        Info,
        "setup done, named_data_map_ = %lu",
        linear_data_map_->get_num_keys().get());
  }

 private:
  std::unique_ptr<FileDataLoader> linear_program_loader_;
  std::unique_ptr<FileDataLoader> linear_data_loader_;

 protected:
  std::unique_ptr<Program> linear_program_;
  std::unique_ptr<FlatTensorDataMap> linear_data_map_;
};

TEST_F(BackendDataSeparationTest, TestSeparation) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = linear_program_->load_method(
      "forward",
      &mmm.get(),
      /*event_tracer=*/nullptr,
      /*named_data_map=*/linear_data_map_.get());
  ASSERT_EQ(method.error(), Error::Ok);

  // Set a dummy input.
  int32_t sizes[1] = {3};
  uint8_t dim_order[1] = {0};
  int32_t strides[1] = {1};
  executorch::aten::TensorImpl impl(
      executorch::aten::ScalarType::Float,
      1,
      sizes,
      nullptr,
      dim_order,
      strides);
  auto input_err = method->set_input(
      executorch::runtime::EValue(executorch::aten::Tensor(&impl)), 0);
  ASSERT_EQ(input_err, Error::Ok);

  // Can execute the method.
  Error err = method->execute();
  ASSERT_EQ(err, Error::Ok);
}
