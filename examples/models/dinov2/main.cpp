/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * DINOv2 image classification runner for ExecuTorch.
 *
 * Usage:
 *   ./dinov2_runner --model_path model.pte --data_path aoti_cuda_blob.ptd
 *   ./dinov2_runner --model_path model.pte --data_path aoti_cuda_blob.ptd
 *                   --input_path image.raw
 *
 * The input file should be a raw binary file containing float32 values
 * for a pre-processed image tensor of shape (1, 3, 224, 224).
 * If no input_path is given, random input is used for testing.
 */

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

DEFINE_string(model_path, "model.pte", "Path to DINOv2 model (.pte).");
DEFINE_string(
    data_path,
    "",
    "Path to data file (.ptd) for delegate data (optional, required for CUDA).");
DEFINE_string(
    input_path,
    "",
    "Path to raw input file (float32 binary, shape 1x3x224x224). "
    "If empty, uses random input for testing.");
DEFINE_int32(img_size, 224, "Input image size (default: 224).");
DEFINE_int32(top_k, 5, "Number of top predictions to display (default: 5).");
DEFINE_bool(
    bf16,
    true,
    "Use bfloat16 input (default: true, matching export dtype).");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace {

/**
 * Load a raw float32 binary file into a vector.
 */
std::vector<float> load_raw_input(
    const std::string& path,
    size_t expected_size) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    ET_LOG(Error, "Failed to open input file: %s", path.c_str());
    return {};
  }

  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  size_t expected_bytes = expected_size * sizeof(float);
  if (file_size != expected_bytes) {
    ET_LOG(
        Error,
        "Input file size mismatch: got %zu bytes, expected %zu bytes",
        file_size,
        expected_bytes);
    return {};
  }

  std::vector<float> data(expected_size);
  file.read(reinterpret_cast<char*>(data.data()), file_size);
  return data;
}

/**
 * Generate random input data for testing.
 */
std::vector<float> generate_random_input(size_t size) {
  std::vector<float> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
  }
  return data;
}

/**
 * Print top-k predictions from logits.
 */
void print_top_k(const float* logits, int num_classes, int k) {
  std::vector<int> indices(num_classes);
  std::iota(indices.begin(), indices.end(), 0);

  std::partial_sort(
      indices.begin(),
      indices.begin() + k,
      indices.end(),
      [logits](int a, int b) { return logits[a] > logits[b]; });

  std::cout << "\nTop-" << k << " predictions:" << std::endl;
  for (int i = 0; i < k && i < num_classes; ++i) {
    int idx = indices[i];
    std::cout << "  Class " << idx << ": " << logits[idx] << std::endl;
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Load model
  std::unique_ptr<Module> model;
  if (!FLAGS_data_path.empty()) {
    ET_LOG(
        Info,
        "Loading model from %s with data from %s",
        FLAGS_model_path.c_str(),
        FLAGS_data_path.c_str());
    model = std::make_unique<Module>(
        FLAGS_model_path, FLAGS_data_path, Module::LoadMode::Mmap);
  } else {
    ET_LOG(Info, "Loading model from %s", FLAGS_model_path.c_str());
    model = std::make_unique<Module>(FLAGS_model_path, Module::LoadMode::Mmap);
  }

  // Prepare input tensor
  const int img_size = FLAGS_img_size;
  const size_t input_size = 1 * 3 * img_size * img_size;

  std::vector<float> input_data;
  if (!FLAGS_input_path.empty()) {
    ET_LOG(Info, "Loading input from %s", FLAGS_input_path.c_str());
    input_data = load_raw_input(FLAGS_input_path, input_size);
    if (input_data.empty()) {
      ET_LOG(Error, "Failed to load input data");
      return 1;
    }
  } else {
    ET_LOG(Info, "Using random input for testing");
    input_data = generate_random_input(input_size);
  }

  // Create input tensor: shape (1, 3, img_size, img_size)
  std::vector<int32_t> input_shape = {1, 3, img_size, img_size};

  // Convert to bf16 if needed (model is exported with bf16 by default)
  std::vector<executorch::aten::BFloat16> bf16_data;
  executorch::extension::TensorPtr input_tensor;
  if (FLAGS_bf16) {
    bf16_data.resize(input_size);
    for (size_t i = 0; i < input_size; ++i) {
      bf16_data[i] = executorch::aten::BFloat16(input_data[i]);
    }
    input_tensor = from_blob(
        bf16_data.data(),
        {input_shape.begin(), input_shape.end()},
        executorch::aten::ScalarType::BFloat16);
  } else {
    input_tensor = from_blob(
        input_data.data(),
        {input_shape.begin(), input_shape.end()},
        executorch::aten::ScalarType::Float);
  }

  // Run inference
  ET_LOG(Info, "Running inference...");
  std::vector<executorch::runtime::EValue> inputs;
  inputs.push_back(*input_tensor);
  auto result = model->execute("forward", inputs);

  if (!result.ok()) {
    ET_LOG(Error, "Inference failed with error: %d", (int)result.error());
    return 1;
  }

  // Process output
  auto& outputs = result.get();
  if (outputs.empty()) {
    ET_LOG(Error, "No outputs from model");
    return 1;
  }

  auto& output_evalue = outputs[0];
  if (!output_evalue.isTensor()) {
    ET_LOG(Error, "Output is not a tensor");
    return 1;
  }

  auto output_tensor = output_evalue.toTensor();
  int num_classes = output_tensor.size(output_tensor.dim() - 1);

  std::cout << "Output shape: (" << output_tensor.size(0) << ", "
            << num_classes << ")" << std::endl;

  // Convert output to float for top-k processing
  std::vector<float> logits_float(num_classes);
  if (output_tensor.scalar_type() == executorch::aten::ScalarType::BFloat16) {
    const auto* bf16_ptr =
        output_tensor.template const_data_ptr<executorch::aten::BFloat16>();
    for (int i = 0; i < num_classes; ++i) {
      logits_float[i] = static_cast<float>(bf16_ptr[i]);
    }
  } else {
    const float* float_ptr =
        output_tensor.template const_data_ptr<float>();
    for (int i = 0; i < num_classes; ++i) {
      logits_float[i] = float_ptr[i];
    }
  }

  // Print top-k predictions
  print_top_k(logits_float.data(), num_classes, FLAGS_top_k);

  return 0;
}
