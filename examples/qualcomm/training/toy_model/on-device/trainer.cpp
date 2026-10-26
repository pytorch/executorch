/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// On-device C++ trainer runs the QNN/HTP-delegated
// "forward_backward" method directly via executorch::extension::Module and
// applies SGD.
//
// Usage: qnn_toy_model_trainer <config.json> <manifest.json>
// The .pte and init_weights.safetensors are expected alongside manifest.json
// (export_toy_model.py's layout). A single (x, target) batch is generated
// on-device via N(0,1) once, seeded from config.json's "seed" if present, and
// reused every step -- the model memorizes it, so loss should decrease
// monotonically.

#include "common.h"

#include <executorch/runtime/platform/log.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;

namespace qnn_toy_model_training {

static void train(
    Module& module,
    const nlohmann::json& config,
    const nlohmann::json& manifest,
    const std::string& safetensors_path,
    const std::string& model_weights_path) {
  const size_t num_weights = manifest.at("param_names").size();

  auto meta_result = module.method_meta("forward_backward");
  ET_CHECK_MSG(
      meta_result.ok(), "Failed to load forward_backward method metadata.");
  const auto& meta = *meta_result;
  ET_CHECK_MSG(
      meta.num_outputs() == num_weights + 2,
      "Expected %zu outputs (loss, pred, %zu grads), got %zu.",
      num_weights + 2,
      num_weights,
      meta.num_outputs());

  std::vector<TensorPtr> weights = load_weights(manifest, safetensors_path);
  ET_CHECK_MSG(
      weights.size() == num_weights,
      "Loaded %zu weights from safetensors, expected %zu.",
      weights.size(),
      num_weights);

  const auto x_shape = shape_from_json(manifest.at("x_shape"));
  const auto target_shape = shape_from_json(manifest.at("target_shape"));
  std::mt19937 rng(
      config.contains("seed") ? config.at("seed").get<unsigned>()
                              : std::random_device{}());

  // export_toy_model.py's tangent is always a constant torch.ones([1]).
  auto tangent = executorch::extension::make_tensor_ptr(
      std::vector<executorch::aten::SizesType>{1}, std::vector<float>{1.0f});

  const int steps = config.value("steps", kDefaultSteps);
  const float lr = config.value("lr", kDefaultLr);
  const int log_every = config.value("log_every", kDefaultLogEvery);

  TensorPtr x = make_random_tensor(x_shape, rng);
  TensorPtr target = make_random_tensor(target_shape, rng);

  for (int step = 0; step < steps; step++) {
    std::vector<EValue> inputs;
    inputs.reserve(num_weights + 3);
    for (auto& w : weights) {
      inputs.push_back(*w);
    }
    inputs.push_back(*x);
    inputs.push_back(*target);
    inputs.push_back(*tangent);

    auto outputs = module.execute("forward_backward", inputs);
    ET_CHECK_MSG(
        outputs.ok(), "forward_backward execution failed at step %d.", step);

    for (size_t i = 0; i < num_weights; i++) {
      const auto& grad = outputs->at(2 + i).toTensor();
      float* w = weights[i]->mutable_data_ptr<float>();
      const float* g = grad.const_data_ptr<float>();
      const size_t n = weights[i]->numel();
      for (size_t j = 0; j < n; j++) {
        w[j] -= lr * g[j];
      }
    }

    if (step % log_every == 0 || step == steps - 1) {
      const auto& loss = outputs->at(0).toTensor();
      ET_LOG(
          Info, "step %5d  loss %.6f", step, loss.const_data_ptr<float>()[0]);
    }
  }

  // Save trained weights so inference.cpp can load them.
  std::vector<std::string> param_names;
  for (const auto& name : manifest.at("param_names")) {
    param_names.push_back(name.get<std::string>());
  }
  save_safetensors(model_weights_path, param_names, weights);
  ET_LOG(Info, "Saved trained weights to %s", model_weights_path.c_str());
}

} // namespace qnn_toy_model_training

int main(int argc, char** argv) {
  if (argc != 3) {
    ET_LOG(Error, "Usage: %s <config.json> <manifest.json>", argv[0]);
    return 1;
  }
  const std::string config_path = argv[1];
  const std::string manifest_path = argv[2];

  const nlohmann::json config = qnn_toy_model_training::load_json(config_path);
  const nlohmann::json manifest =
      qnn_toy_model_training::load_json(manifest_path);

  const std::string dir = qnn_toy_model_training::manifest_dir(manifest_path);
  const std::string pte_path =
      dir + "/" + manifest.at("pte_name").get<std::string>();
  const std::string safetensors_path = dir + "/init_weights.safetensors";
  const std::string model_weights_path = dir + "/trained_model.safetensors";

  executorch::extension::Module module(pte_path);
  qnn_toy_model_training::train(
      module, config, manifest, safetensors_path, model_weights_path);
  return 0;
}
