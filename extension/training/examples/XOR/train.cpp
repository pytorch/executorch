/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/extension/training/optimizer/sgd.h>
#include <gflags/gflags.h>
#include <random>

#pragma clang diagnostic ignored \
    "-Wbraced-scalar-init" // {0} below upsets clang.

using executorch::extension::FileDataLoader;
using executorch::extension::training::optimizer::SGD;
using executorch::extension::training::optimizer::SGDOptions;
using executorch::runtime::Error;
using executorch::runtime::Result;
DEFINE_string(model_path, "xor.pte", "Model serialized in flatbuffer format.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args: ";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  // Load the model file.
  executorch::runtime::Result<executorch::extension::FileDataLoader>
      loader_res =
          executorch::extension::FileDataLoader::from(FLAGS_model_path.c_str());
  if (loader_res.error() != Error::Ok) {
    ET_LOG(Error, "Failed to open model file: %s", FLAGS_model_path.c_str());
    return 1;
  }
  auto loader = std::make_unique<executorch::extension::FileDataLoader>(
      std::move(loader_res.get()));

  auto mod = executorch::extension::training::TrainingModule(std::move(loader));

  // Create full data set of input and labels.
  std::vector<std::pair<
      executorch::extension::TensorPtr,
      executorch::extension::TensorPtr>>
      data_set;
  data_set.push_back( // XOR(1, 1) = 0
      {executorch::extension::make_tensor_ptr<float>({1, 2}, {1, 1}),
       executorch::extension::make_tensor_ptr<int64_t>({1}, {0})});
  data_set.push_back( // XOR(0, 0) = 0
      {executorch::extension::make_tensor_ptr<float>({1, 2}, {0, 0}),
       executorch::extension::make_tensor_ptr<int64_t>({1}, {0})});
  data_set.push_back( // XOR(1, 0) = 1
      {executorch::extension::make_tensor_ptr<float>({1, 2}, {1, 0}),
       executorch::extension::make_tensor_ptr<int64_t>({1}, {1})});
  data_set.push_back( // XOR(0, 1) = 1
      {executorch::extension::make_tensor_ptr<float>({1, 2}, {0, 1}),
       executorch::extension::make_tensor_ptr<int64_t>({1}, {1})});

  // Create optimizer.
  // Get the params and names
  auto param_res = mod.named_parameters("forward");
  if (param_res.error() != Error::Ok) {
    ET_LOG(Error, "Failed to get named parameters");
    return 1;
  }

  SGDOptions options{0.1};
  SGD optimizer(param_res.get(), options);

  // Randomness to sample the data set.
  std::default_random_engine URBG{std::random_device{}()};
  std::uniform_int_distribution<int> dist{
      0, static_cast<int>(data_set.size()) - 1};

  // Train the model.
  size_t num_epochs = 5000;
  for (int i = 0; i < num_epochs; i++) {
    int index = dist(URBG);
    auto& data = data_set[index];
    const auto& results =
        mod.execute_forward_backward("forward", {*data.first, *data.second});
    if (results.error() != Error::Ok) {
      ET_LOG(Error, "Failed to execute forward_backward");
      return 1;
    }
    if (i % 500 == 0 || i == num_epochs - 1) {
      ET_LOG(
          Info,
          "Step %d, Loss %f, Input [%.0f, %.0f], Prediction %ld, Label %ld",
          i,
          results.get()[0].toTensor().const_data_ptr<float>()[0],
          data.first->const_data_ptr<float>()[0],
          data.first->const_data_ptr<float>()[1],
          results.get()[1].toTensor().const_data_ptr<int64_t>()[0],
          data.second->const_data_ptr<int64_t>()[0]);
    }
    optimizer.step(mod.named_gradients("forward").get());
  }
}
