/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <nlohmann/json.hpp>

namespace qnn_toy_model_training {

// Fallbacks for fields missing from config.json -- mirror config.json's values.
constexpr int kDefaultSteps = 500;
constexpr float kDefaultLr = 0.1f;
constexpr int kDefaultLogEvery = 50;

struct Batch {
  executorch::extension::TensorPtr x;
  executorch::extension::TensorPtr target;
};

// Loads and parses a JSON file (config.json or manifest.json).
nlohmann::json load_json(const std::string& path);

// Parses a safetensors file into owned F32 tensors keyed by name.
std::unordered_map<std::string, executorch::extension::TensorPtr>
load_safetensors(const std::string& path);

// Loads init_weights.safetensors and reorders tensors to match manifest's
// "param_names" order (the joint graph's primals_N placeholder order).
std::vector<executorch::extension::TensorPtr> load_weights(
    const nlohmann::json& manifest,
    const std::string& safetensors_path);

// Converts a JSON int array (e.g. manifest's "x_shape") to tensor sizes.
std::vector<executorch::aten::SizesType> shape_from_json(
    const nlohmann::json& shape_array);

// Generates one random F32 tensor of `shape` via N(0,1), matching the
// torch.randn semantics the Python export scripts used.
executorch::extension::TensorPtr make_random_tensor(
    const std::vector<executorch::aten::SizesType>& shape,
    std::mt19937& rng);

// Writes tensors (keyed by name, all F32) to a safetensors file.
void save_safetensors(
    const std::string& path,
    const std::vector<std::string>& names,
    const std::vector<executorch::extension::TensorPtr>& tensors);

// Resolves manifest_dir from manifest_path (directory part).
std::string manifest_dir(const std::string& manifest_path);

} // namespace qnn_toy_model_training
