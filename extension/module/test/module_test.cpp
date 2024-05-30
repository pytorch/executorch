/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/extension/module/module.h>
#include <array>

using namespace ::testing;

namespace torch::executor {

class ModuleTest : public ::testing::Test {};

TEST_F(ModuleTest, TestLoad) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  EXPECT_FALSE(module.is_loaded());
  const auto error = module.load();
  EXPECT_EQ(error, Error::Ok);
  EXPECT_TRUE(module.is_loaded());
}

TEST_F(ModuleTest, TestLoadNonExistent) {
  Module module("/path/to/nonexistent/file.pte");
  const auto error = module.load();

  EXPECT_NE(error, Error::Ok);
  EXPECT_FALSE(module.is_loaded());
}

TEST_F(ModuleTest, TestLoadCorruptedFile) {
  Module module("/dev/null");
  const auto error = module.load();

  EXPECT_NE(error, Error::Ok);
  EXPECT_FALSE(module.is_loaded());
}

TEST_F(ModuleTest, TestMethodNames) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto method_names = module.method_names();
  EXPECT_TRUE(method_names.ok());
  EXPECT_EQ(method_names.get(), std::unordered_set<std::string>{"forward"});
}

TEST_F(ModuleTest, TestNonExistentMethodNames) {
  Module module("/path/to/nonexistent/file.pte");

  const auto method_names = module.method_names();
  EXPECT_FALSE(method_names.ok());
}

TEST_F(ModuleTest, TestLoadMethod) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  EXPECT_FALSE(module.is_method_loaded("forward"));
  const auto error = module.load_method("forward");
  EXPECT_EQ(error, Error::Ok);
  EXPECT_TRUE(module.is_method_loaded("forward"));
  EXPECT_TRUE(module.is_loaded());
}

TEST_F(ModuleTest, TestLoadNonExistentMethod) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto error = module.load_method("backward");
  EXPECT_NE(error, Error::Ok);
  EXPECT_FALSE(module.is_method_loaded("backward"));
  EXPECT_TRUE(module.is_loaded());
}

TEST_F(ModuleTest, TestMethodMeta) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto meta = module.method_meta("forward");
  EXPECT_TRUE(meta.ok());
  EXPECT_STREQ(meta->name(), "forward");
  EXPECT_EQ(meta->num_inputs(), 1);
  EXPECT_EQ(*(meta->input_tag(0)), Tag::Tensor);
  EXPECT_EQ(meta->num_outputs(), 1);
  EXPECT_EQ(*(meta->output_tag(0)), Tag::Tensor);

  const auto input_meta = meta->input_tensor_meta(0);
  EXPECT_TRUE(input_meta.ok());
  EXPECT_EQ(input_meta->scalar_type(), ScalarType::Float);
  EXPECT_EQ(input_meta->sizes().size(), 2);
  EXPECT_EQ(input_meta->sizes()[0], 1);
  EXPECT_EQ(input_meta->sizes()[1], 2);

  const auto output_meta = meta->output_tensor_meta(0);
  EXPECT_TRUE(output_meta.ok());
  EXPECT_EQ(output_meta->scalar_type(), ScalarType::Float);
  EXPECT_EQ(output_meta->sizes().size(), 1);
  EXPECT_EQ(output_meta->sizes()[0], 1);
}

TEST_F(ModuleTest, TestNonExistentMethodMeta) {
  Module module("/path/to/nonexistent/file.pte");

  const auto meta = module.method_meta("forward");
  EXPECT_FALSE(meta.ok());
}

TEST_F(ModuleTest, TestExecute) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(module.is_loaded());
  EXPECT_TRUE(module.is_method_loaded("forward"));

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, TestExecutePreload) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto error = module.load();
  EXPECT_EQ(error, Error::Ok);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, TestExecutePreload_method) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto error = module.load_method("forward");
  EXPECT_EQ(error, Error::Ok);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, TestExecutePreloadProgramAndMethod) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto load_error = module.load();
  EXPECT_EQ(load_error, Error::Ok);

  const auto load_method_error = module.load_method("forward");
  EXPECT_EQ(load_method_error, Error::Ok);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, TestExecuteOnNonExistent) {
  Module module("/path/to/nonexistent/file.pte");

  const auto result = module.execute("forward");

  EXPECT_FALSE(result.ok());
}

TEST_F(ModuleTest, TestExecuteOnCurrupted) {
  Module module("/dev/null");

  const auto result = module.execute("forward");

  EXPECT_FALSE(result.ok());
}

TEST_F(ModuleTest, TestForward) {
  auto module = std::make_unique<Module>(
      std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());
  const auto result = module->forward({EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);

  std::array<float, 2> input2{2, 3};
  TensorImpl tensor2(
      ScalarType::Float, sizes.size(), sizes.data(), input2.data());
  const auto result2 = module->forward({EValue(Tensor(&tensor2))});
  EXPECT_TRUE(result2.ok());

  const auto data2 = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data2[0], 2.5, 1e-5);
}

TEST_F(ModuleTest, TestForwardWithInvalidInputs) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto result = module.forward({EValue()});

  EXPECT_FALSE(result.ok());
}

} // namespace torch::executor
