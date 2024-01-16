/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/extension/runner/module/module.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;

namespace torch::executor {

class ModuleTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

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
  EXPECT_EQ(methodNames.get(), std::vector<std::string>{"forward"});
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
}

TEST_F(ModuleTest, testLoadNonExistentMethod) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto error = module.loadMethod("backward");
  EXPECT_NE(error, Error::Ok);
  EXPECT_FALSE(module.isMethodLoaded("backward"));
}

TEST_F(ModuleTest, testMethodMeta) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto meta = module.methodMeta("forward");
  EXPECT_TRUE(meta.ok());
  EXPECT_STREQ(meta->name(), "forward");
  EXPECT_EQ(meta->num_inputs(), 1);
  EXPECT_EQ(meta->num_outputs(), 1);

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

TEST_F(ModuleTest, testRun) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  float input[] = {1, 2};
  int32_t sizes[] = {1, 2};
  TensorImpl tensorImpl(ScalarType::Float, std::size(sizes), sizes, input);
  std::vector<EValue> inputs = {EValue(Tensor(&tensorImpl))};
  std::vector<EValue> outputs;

  const auto error = module.run("forward", inputs, outputs);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_TRUE(module.isLoaded());
  EXPECT_TRUE(module.isMethodLoaded("forward"));

  const auto outputTensor = outputs[0].toTensor();
  const auto data = outputTensor.const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, testRunPreload) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto loadError = module.load();
  EXPECT_EQ(loadError, Error::Ok);

  float input[] = {1, 2};
  int32_t sizes[] = {1, 2};
  TensorImpl tensorImpl(ScalarType::Float, std::size(sizes), sizes, input);
  std::vector<EValue> inputs = {EValue(Tensor(&tensorImpl))};
  std::vector<EValue> outputs;

  const auto error = module.run("forward", inputs, outputs);
  EXPECT_EQ(error, Error::Ok);

  const auto outputTensor = outputs[0].toTensor();
  const auto data = outputTensor.const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, testRunPreloadMethod) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto loadMethodError = module.loadMethod("forward");
  EXPECT_EQ(loadMethodError, Error::Ok);

  float input[] = {1, 2};
  int32_t sizes[] = {1, 2};
  TensorImpl tensorImpl(ScalarType::Float, std::size(sizes), sizes, input);
  std::vector<EValue> inputs = {EValue(Tensor(&tensorImpl))};
  std::vector<EValue> outputs;

  const auto error = module.run("forward", inputs, outputs);
  EXPECT_EQ(error, Error::Ok);

  const auto outputTensor = outputs[0].toTensor();
  const auto data = outputTensor.const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, testRunPreloadProgramAndMethod) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  const auto loadError = module.load();
  EXPECT_EQ(loadError, Error::Ok);

  const auto loadMethodError = module.loadMethod("forward");
  EXPECT_EQ(loadMethodError, Error::Ok);

  float input[] = {1, 2};
  int32_t sizes[] = {1, 2};
  TensorImpl tensorImpl(ScalarType::Float, std::size(sizes), sizes, input);
  std::vector<EValue> inputs = {EValue(Tensor(&tensorImpl))};
  std::vector<EValue> outputs;

  const auto error = module.run("forward", inputs, outputs);
  EXPECT_EQ(error, Error::Ok);

  const auto outputTensor = outputs[0].toTensor();
  const auto data = outputTensor.const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, testRunOnNonExistent) {
  Module module("/path/to/nonexistent/file.pte");
  std::vector<EValue> inputs;
  std::vector<EValue> outputs;

  const auto error = module.run("forward", inputs, outputs);

  EXPECT_NE(error, Error::Ok);
}

TEST_F(ModuleTest, testRunOnCurrupted) {
  Module module("/dev/null");
  std::vector<EValue> inputs;
  std::vector<EValue> outputs;

  const auto error = module.run("forward", inputs, outputs);

  EXPECT_NE(error, Error::Ok);
}

TEST_F(ModuleTest, testForward) {
  auto module = std::make_unique<Module>(
      std::getenv("RESOURCES_PATH") + std::string("/model.pte"));

  std::vector<float> input{1, 2};
  int32_t sizes[] = {1, 2};
  TensorImpl tensorImpl(
      ScalarType::Float, std::size(sizes), sizes, input.data());
  std::vector<EValue> inputs = {EValue(Tensor(&tensorImpl))};
  std::vector<EValue> outputs;

  auto error = module->forward(inputs, outputs);
  EXPECT_EQ(error, Error::Ok);

  auto outputTensor = outputs[0].toTensor();
  auto data = outputTensor.const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);

  input = {2, 3};
  TensorImpl tensorImpl2(
      ScalarType::Float, std::size(sizes), sizes, input.data());
  inputs = {EValue(Tensor(&tensorImpl2))};

  error = module->forward(inputs, outputs);
  EXPECT_EQ(error, Error::Ok);

  outputTensor = outputs[0].toTensor();
  data = outputTensor.const_data_ptr<float>();

  EXPECT_NEAR(data[0], 2.5, 1e-5);
}

TEST_F(ModuleTest, testForwardWithInvalidInputs) {
  Module module(std::getenv("RESOURCES_PATH") + std::string("/model.pte"));
  std::vector<EValue> inputs = {EValue()};
  std::vector<EValue> outputs;

  const auto error = module.forward(inputs, outputs);
  EXPECT_NE(error, Error::Ok);
}

} // namespace torch::executor
