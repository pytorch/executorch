/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/extension/training/optimizer/sgd.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>
#include <iostream>

// @lint-ignore-every CLANGTIDY facebook-hte-CArray

using namespace ::testing;
using namespace executorch::extension::training::optimizer;
using namespace torch::executor::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using namespace torch::executor;
using torch::executor::util::FileDataLoader;

class TrainingLoopTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(TrainingLoopTest, OptimizerSteps) {
  const char* path = std::getenv("ET_MODULE_SIMPLE_TRAIN_PATH");
  executorch::runtime::Result<torch::executor::util::FileDataLoader>
      loader_res = torch::executor::util::FileDataLoader::from(path);
  ASSERT_EQ(loader_res.error(), Error::Ok);
  auto loader = std::make_unique<torch::executor::util::FileDataLoader>(
      std::move(loader_res.get()));

  auto mod = executorch::extension::training::TrainingModule(std::move(loader));

  // Create inputs.
  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.make({3}, {1.0, 1.0, 1.0});
  Tensor label = tf.make({3}, {1.0, 0.0, 0.0});

  auto res = mod.execute_forward_backward("forward", {input, label});
  ASSERT_TRUE(res.ok());

  // Set up optimizer.
  // Get the params and names
  auto param_res = mod.named_parameters("forward");
  ASSERT_EQ(param_res.error(), Error::Ok);

  float orig_data = param_res.get().at("linear.weight").data_ptr<float>()[0];

  SGDOptions options{0.1};
  SGD optimizer(param_res.get(), options);

  // Get the gradients
  auto grad_res = mod.named_gradients("forward");
  ASSERT_EQ(grad_res.error(), Error::Ok);
  auto& grad = grad_res.get();
  ASSERT_EQ(grad.size(), 2);
  ASSERT_NE(grad.find("linear.weight"), grad.end());
  ASSERT_NE(grad.find("linear.bias"), grad.end());

  // Step
  auto opt_err = optimizer.step(grad_res.get());
  ASSERT_EQ(opt_err, Error::Ok);

  // Check that the data has changed.
  ASSERT_NE(
      param_res.get().at("linear.weight").data_ptr<float>()[0], orig_data);
}
