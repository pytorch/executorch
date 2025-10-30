/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

class DataSeparationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Create data loaders.
    Result<FileDataLoader> linear_program_loader =
        FileDataLoader::from(std::getenv("ET_MODULE_LINEAR_XNN_PROGRAM_PATH"));
    ASSERT_EQ(linear_program_loader.error(), Error::Ok);
    linear_program_loader_ = std::make_unique<FileDataLoader>(
        std::move(linear_program_loader.get()));

    Result<FileDataLoader> linear_data_loader =
        FileDataLoader::from(std::getenv("ET_MODULE_LINEAR_XNN_DATA_PATH"));
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
  }

 private:
  std::unique_ptr<FileDataLoader> linear_program_loader_;
  std::unique_ptr<FileDataLoader> linear_data_loader_;

 protected:
  std::unique_ptr<Program> linear_program_;
  std::unique_ptr<FlatTensorDataMap> linear_data_map_;
};

TEST_F(DataSeparationTest, TestExternalData) {
  FlatTensorDataMap* data_map = linear_data_map_.get();
  EXPECT_EQ(data_map->get_num_keys().get(), 2);

  Result<const char*> key0 = data_map->get_key(0);
  EXPECT_EQ(key0.error(), Error::Ok);
  Result<const char*> key1 = data_map->get_key(1);
  EXPECT_EQ(key1.error(), Error::Ok);

  // Check that accessing keys out of bounds fails.
  EXPECT_EQ(data_map->get_key(2).error(), Error::InvalidArgument);

  // Linear.weight
  Result<FreeableBuffer> data0 = data_map->get_data(key0.get());
  EXPECT_EQ(data0.error(), Error::Ok);
  EXPECT_EQ(data0.get().size(), 36); // 3*3*4 (3*3 matrix, 4 bytes per float)

  // Linear.bias
  Result<FreeableBuffer> data1 = data_map->get_data(key1.get());
  EXPECT_EQ(data1.error(), Error::Ok);
  EXPECT_EQ(data1.get().size(), 12); // 3*4 (3 vector, 4 bytes per float)

  // Check that accessing non-existent data fails.
  Result<FreeableBuffer> data2 = data_map->get_data("nonexistent");
  EXPECT_EQ(data2.error(), Error::NotFound);
}

TEST_F(DataSeparationTest, TestE2E) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = linear_program_->load_method(
      "forward", &mmm.get(), nullptr, linear_data_map_.get());
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
