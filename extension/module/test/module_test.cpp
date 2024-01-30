/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/extension/module/module.h>

using namespace ::testing;

namespace torch::executor {

class ModuleTest : public ::testing::Test {};

TEST_F(ModuleTest, testLoad) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  EXPECT_FALSE(module.isLoaded());
  const auto error = module.load();
  EXPECT_EQ(error, Error::Ok);
  EXPECT_TRUE(module.isLoaded());
}

TEST_F(ModuleTest, testLoadNonExistent) {
  Module module("/path/to/nonexistent/file.pte");
  const auto error = module.load();

  EXPECT_NE(error, Error::Ok);
  EXPECT_FALSE(module.isLoaded());
}

TEST_F(ModuleTest, testLoadCorruptedFile) {
  Module module("/dev/null");
  const auto error = module.load();

  EXPECT_NE(error, Error::Ok);
  EXPECT_FALSE(module.isLoaded());
}

TEST_F(ModuleTest, testMethodNames) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto methodNames = module.methodNames();
  EXPECT_TRUE(methodNames.ok());
  EXPECT_EQ(methodNames.get(), std::unordered_set<std::string>{"forward"});
}

TEST_F(ModuleTest, testNonExistentMethodNames) {
  Module module("/path/to/nonexistent/file.pte");

  const auto methodNames = module.methodNames();
  EXPECT_FALSE(methodNames.ok());
}

TEST_F(ModuleTest, testLoadMethod) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  EXPECT_FALSE(module.isMethodLoaded("forward"));
  const auto error = module.loadMethod("forward");
  EXPECT_EQ(error, Error::Ok);
  EXPECT_TRUE(module.isMethodLoaded("forward"));
  EXPECT_TRUE(module.isLoaded());
}

TEST_F(ModuleTest, testLoadNonExistentMethod) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto error = module.loadMethod("backward");
  EXPECT_NE(error, Error::Ok);
  EXPECT_FALSE(module.isMethodLoaded("backward"));
  EXPECT_TRUE(module.isLoaded());
}

TEST_F(ModuleTest, testMethodMeta) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto meta = module.methodMeta("forward");
  EXPECT_TRUE(meta.ok());
  EXPECT_STREQ(meta->name(), "forward");
  EXPECT_EQ(meta->num_inputs(), 1);
  EXPECT_EQ(*(meta->input_tag(0)), Tag::Tensor);
  EXPECT_EQ(meta->num_outputs(), 1);
  EXPECT_EQ(*(meta->output_tag(0)), Tag::Tensor);

  const auto inputMeta = meta->input_tensor_meta(0);
  EXPECT_TRUE(inputMeta.ok());
  EXPECT_EQ(inputMeta->scalar_type(), ScalarType::Float);
  EXPECT_EQ(inputMeta->sizes().size(), 2);
  EXPECT_EQ(inputMeta->sizes()[0], 1);
  EXPECT_EQ(inputMeta->sizes()[1], 2);

  const auto outputMeta = meta->output_tensor_meta(0);
  EXPECT_TRUE(outputMeta.ok());
  EXPECT_EQ(outputMeta->scalar_type(), ScalarType::Float);
  EXPECT_EQ(outputMeta->sizes().size(), 1);
  EXPECT_EQ(outputMeta->sizes()[0], 1);
}

TEST_F(ModuleTest, testNonExistentMethodMeta) {
  Module module("/path/to/nonexistent/file.pte");

  const auto meta = module.methodMeta("forward");
  EXPECT_FALSE(meta.ok());
}

TEST_F(ModuleTest, testExecute) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(module.isLoaded());
  EXPECT_TRUE(module.isMethodLoaded("forward"));

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, testExecutePreload) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto loadError = module.load();
  EXPECT_EQ(loadError, Error::Ok);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, testExecutePreloadMethod) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto loadMethodError = module.loadMethod("forward");
  EXPECT_EQ(loadMethodError, Error::Ok);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, testExecutePreloadProgramAndMethod) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto loadError = module.load();
  EXPECT_EQ(loadError, Error::Ok);

  const auto loadMethodError = module.loadMethod("forward");
  EXPECT_EQ(loadMethodError, Error::Ok);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, testExecuteOnNonExistent) {
  Module module("/path/to/nonexistent/file.pte");

  const auto result = module.execute("forward");

  EXPECT_FALSE(result.ok());
}

TEST_F(ModuleTest, testExecuteOnCurrupted) {
  Module module("/dev/null");

  const auto result = module.execute("forward");

  EXPECT_FALSE(result.ok());
}

TEST_F(ModuleTest, testForward) {
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

TEST_F(ModuleTest, testForwardWithInvalidInputs) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto result = module.forward({EValue()});

  EXPECT_FALSE(result.ok());
}

} // namespace torch::executor
