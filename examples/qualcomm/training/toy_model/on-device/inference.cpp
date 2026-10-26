/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// On-device inference runner for the QNN/HTP-delegated ToyModel.
//
// Loads trained weights from trained_model.safetensors (written by trainer
// after the training loop), and feeds (*weights, x) into the "forward" method.

// Usage: qnn_toy_model_inference <config.json> <manifest.json>
// trained_model.safetensors is expected alongside manifest.json. Each batch's
// x is generated on-device via N(0,1), seeded from config.json's "seed" if
// present.

#include "common.h"

#include <executorch/runtime/platform/log.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;

namespace qnn_toy_model_training {

static void run_inference(
    Module& module,
    const nlohmann::json& config,
    const nlohmann::json& manifest,
    const std::string& trained_weights_path) {
  const int log_every = config.value("log_every", kDefaultLogEvery);

  std::vector<TensorPtr> weights = load_weights(manifest, trained_weights_path);

  const auto x_shape = shape_from_json(manifest.at("x_shape"));
  std::mt19937 rng(
      config.contains("seed") ? config.at("seed").get<unsigned>()
                              : std::random_device{}());

  const int num_batches = config.value("steps", kDefaultSteps);

  for (int i = 0; i < num_batches; i++) {
    TensorPtr x = make_random_tensor(x_shape, rng);

    std::vector<EValue> inputs;
    inputs.reserve(weights.size() + 1);
    for (auto& w : weights) {
      inputs.push_back(*w);
    }
    inputs.push_back(*x);

    auto outputs = module.execute("forward", inputs);
    ET_CHECK_MSG(outputs.ok(), "forward execution failed at batch %d.", i);

    if (i % log_every == 0 || i == num_batches - 1) {
      const auto& pred = outputs->at(0).toTensor();
      ET_LOG(
          Info, "batch %5d  pred[0] %.6f", i, pred.const_data_ptr<float>()[0]);
    }
  }
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
  // no_ws: fwd_pte_name is the forward-only .pte; ws: pte_name holds both
  // methods.
  const std::string pte_name = manifest.count("fwd_pte_name")
      ? manifest.at("fwd_pte_name").get<std::string>()
      : manifest.at("pte_name").get<std::string>();
  const std::string pte_path = dir + "/" + pte_name;
  const std::string trained_weights_path = dir + "/trained_model.safetensors";

  executorch::extension::Module module(pte_path);
  qnn_toy_model_training::run_inference(
      module, config, manifest, trained_weights_path);
  return 0;
}
