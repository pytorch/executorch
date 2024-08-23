/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
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

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024;

class TrainingLoopTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a loader for the serialized ModuleAdd program.
    const char* path = std::getenv("ET_MODULE_SIMPLE_TRAIN_PATH");
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));

    // Use it to load the program.
    Result<Program> program = Program::load(
        loader_.get(), Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);
    program_ = std::make_unique<Program>(std::move(program.get()));
  }

  // Must outlive program_, but tests shouldn't need to touch it.
  std::unique_ptr<FileDataLoader> loader_;

  std::unique_ptr<Program> program_;
};

TEST_F(TrainingLoopTest, OptimizerSteps) {
  // Execute model with constants stored in segment.
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = program_->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  // Create inputs.
  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.make({3}, {1.0, 1.0, 1.0});
  Tensor label = tf.make({3}, {1.0, 0.0, 0.0});

  Error e = method->set_input(input, 0);
  e = method->set_input(label, 1);

  // Set up optimizer.
  const char* param_name[2] = {"mod.linear1.weight", "mod.linear2.bias"};
  Span<const char*> param_names(param_name, 2);

  Tensor param_data[2] = {
      method.get().get_output(3).toTensor(), // mod.linear1.weight
      method.get().get_output(4).toTensor()}; // mod.linear1.bias
  Span<Tensor> param_data_span(param_data, 2);

  auto orig_data = param_data[0].data_ptr<float>()[0];

  Tensor grad_data[2] = {
      method.get().get_output(1).toTensor(), // mod.linear1.weight.grad
      method.get().get_output(2).toTensor()}; // mod.linear1.bias.grad
  ;
  Span<Tensor> grad_data_span(grad_data, 2);

  SGDOptions options{0.1};
  SGD optimizer(param_names, param_data_span, options);

  // Execute the method. (Forward and Backward)
  Error err = method->execute();
  ASSERT_EQ(err, Error::Ok);

  // Step
  auto opt_err = optimizer.step(param_names, grad_data_span);
  ASSERT_EQ(opt_err, Error::Ok);

  // Check that the data has changed.
  ASSERT_NE(param_data[0].data_ptr<float>()[0], orig_data);
}
