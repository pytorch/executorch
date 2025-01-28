/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/training/module/training_module.h>

#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

// @lint-ignore-every CLANGTIDY facebook-hte-CArray

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::Error;
using torch::executor::Span;
using torch::executor::testing::TensorFactory;

class TrainingModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(TrainingModuleTest, JointGraphTest) {
  // Create a loader for the serialized ModuleAdd program.
  const char* path = std::getenv("ET_MODULE_SIMPLE_TRAIN_PATH");
  executorch::runtime::Result<torch::executor::util::FileDataLoader>
      loader_res = torch::executor::util::FileDataLoader::from(path);
  ASSERT_EQ(loader_res.error(), Error::Ok);
  auto loader = std::make_unique<torch::executor::util::FileDataLoader>(
      std::move(loader_res.get()));

  auto mod = executorch::extension::training::TrainingModule(std::move(loader));

  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.make({3}, {1.0, 1.0, 1.0});
  Tensor label = tf.make({3}, {1.0, 0.0, 0.0});

  std::vector<executorch::runtime::EValue> inputs;
  inputs.push_back(input);
  inputs.push_back(label);

  auto res = mod.execute_forward_backward("forward", inputs);
  ASSERT_EQ(res.error(), Error::Ok);
  ASSERT_EQ(res.get().size(), 1);

  // Test Gradients
  auto grad_res = mod.named_gradients("forward");
  ASSERT_EQ(grad_res.error(), Error::Ok);
  auto& grad = grad_res.get();
  ASSERT_EQ(grad.size(), 2);
  ASSERT_NE(grad.find("linear.weight"), grad.end());
  ASSERT_NE(grad.find("linear.bias"), grad.end());

  ASSERT_EQ(grad.find("linear.weight")->second.sizes()[0], 3);
  ASSERT_EQ(grad.find("linear.weight")->second.sizes()[1], 3);
  ASSERT_EQ(grad.find("linear.weight")->second.dim(), 2);
  ASSERT_EQ(grad.find("linear.bias")->second.sizes()[0], 3);
  ASSERT_EQ(grad.find("linear.bias")->second.dim(), 1);

  // Test Parameters
  auto param_res = mod.named_parameters("forward");
  ASSERT_EQ(param_res.error(), Error::Ok);
  auto& param = grad_res.get();
  ASSERT_EQ(param.size(), 2);
  ASSERT_NE(param.find("linear.weight"), grad.end());
  ASSERT_NE(param.find("linear.bias"), grad.end());

  ASSERT_EQ(param.find("linear.weight")->second.sizes()[0], 3);
  ASSERT_EQ(param.find("linear.weight")->second.sizes()[1], 3);
  ASSERT_EQ(param.find("linear.weight")->second.dim(), 2);
  ASSERT_EQ(param.find("linear.bias")->second.sizes()[0], 3);
  ASSERT_EQ(param.find("linear.bias")->second.dim(), 1);
}

TEST_F(TrainingModuleTest, NonTrainingModuleTest) {
  // Create a loader for the serialized ModuleAdd program.
  const char* path = std::getenv("ET_MODULE_ADD_PATH");
  executorch::runtime::Result<torch::executor::util::FileDataLoader>
      loader_res = torch::executor::util::FileDataLoader::from(path);
  ASSERT_EQ(loader_res.error(), Error::Ok);
  auto loader = std::make_unique<torch::executor::util::FileDataLoader>(
      std::move(loader_res.get()));

  auto mod = executorch::extension::training::TrainingModule(std::move(loader));

  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  Tensor input2 = tf.make({2, 2}, {1.0, 0.0, 0.0, 0.0});

  std::vector<executorch::runtime::EValue> inputs;
  inputs.push_back(input);
  inputs.push_back(input2);

  // Non-training module should fail to execute forward/backward as it cant find
  // the gradients or params.
  auto res = mod.execute_forward_backward("forward", inputs);
  ASSERT_EQ(res.error(), Error::InvalidArgument);
}
