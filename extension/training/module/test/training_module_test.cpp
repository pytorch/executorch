/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/training/module/training_module.h>

#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

// @lint-ignore-every CLANGTIDY facebook-hte-CArray

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::FlatTensorDataMap;
using executorch::extension::FlatTensorHeader;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::TensorLayout;
using torch::executor::Error;
using torch::executor::Span;
using torch::executor::testing::TensorFactory;
using torch::executor::util::FileDataLoader;

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

TEST_F(TrainingModuleTest, SeperateDataTest) {
  // Load data map.
  // The eager linear model is defined at:
  // //executorch/test/models/linear_model.py
  const char* ptd_path = std::getenv("ET_MODULE_TRAIN_DATA_PATH");
  Result<FileDataLoader> data_map_loader_res = FileDataLoader::from(ptd_path);
  ASSERT_EQ(data_map_loader_res.error(), Error::Ok);

  auto data_map_loader =
      std::make_unique<torch::executor::util::FileDataLoader>(
          std::move(data_map_loader_res.get()));

  const char* pte_path = std::getenv("ET_MODULE_TRAIN_PROGRAM_PATH");
  Result<FileDataLoader> pte_loader_res = FileDataLoader::from(pte_path);
  ASSERT_EQ(pte_loader_res.error(), Error::Ok);

  auto pte_loader = std::make_unique<torch::executor::util::FileDataLoader>(
      std::move(pte_loader_res.get()));

  auto mod = executorch::extension::training::TrainingModule(
      std::move(pte_loader),
      nullptr,
      nullptr,
      nullptr,
      std::move(data_map_loader));

  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.make({3}, {1.0, 1.0, 1.0});
  Tensor label = tf.make({3}, {1.0, 0.0, 0.0});

  std::vector<executorch::runtime::EValue> inputs;
  inputs.push_back(input);
  inputs.push_back(label);

  auto res = mod.execute_forward_backward("forward", inputs);
  ASSERT_EQ(res.error(), Error::Ok);
  ASSERT_EQ(res.get().size(), 1);
}
