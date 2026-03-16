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
 *   ./dinov2_runner --model_path model.pte --data_path aoti_cuda_blob.ptd \
 *                   --image_path image.jpg
 */

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

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
    "Path to data file (.ptd) for CUDA delegate data.");
DEFINE_string(
    image_path,
    "",
    "Path to input image file (.jpg, .png, .bmp). "
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

// ImageNet normalization constants
constexpr float kImageNetMean[] = {0.485f, 0.456f, 0.406f};
constexpr float kImageNetStd[] = {0.229f, 0.224f, 0.225f};

/**
 * Load an image file, resize to target_size x target_size, and apply
 * ImageNet normalization. Returns CHW float data.
 */
std::vector<float> load_image(const std::string& path, int target_size) {
  int width, height, channels;
  unsigned char* raw = stbi_load(path.c_str(), &width, &height, &channels, 3);
  if (!raw) {
    ET_LOG(Error, "Failed to load image: %s", path.c_str());
    return {};
  }

  // Resize to target_size x target_size
  std::vector<unsigned char> resized(target_size * target_size * 3);
  stbir_resize_uint8(
      raw, width, height, 0, resized.data(), target_size, target_size, 0, 3);
  stbi_image_free(raw);

  // Convert to CHW float with ImageNet normalization
  size_t spatial = target_size * target_size;
  std::vector<float> chw_data(3 * spatial);
  for (int h = 0; h < target_size; ++h) {
    for (int w = 0; w < target_size; ++w) {
      int hwc_idx = (h * target_size + w) * 3;
      for (int c = 0; c < 3; ++c) {
        float pixel = static_cast<float>(resized[hwc_idx + c]) / 255.0f;
        chw_data[c * spatial + h * target_size + w] =
            (pixel - kImageNetMean[c]) / kImageNetStd[c];
      }
    }
  }
  return chw_data;
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
 * ImageNet-1k class labels (subset for display).
 */
const char* get_imagenet_label(int class_id) {
  static const std::unordered_map<int, const char*> labels = {
      {0, "tench"},
      {1, "goldfish"},
      {2, "great white shark"},
      {6, "stingray"},
      {15, "robin"},
      {65, "sea snake"},
      {99, "goose"},
      {207, "golden retriever"},
      {208, "Labrador retriever"},
      {229, "Old English sheepdog"},
      {232, "Border collie"},
      {243, "bull mastiff"},
      {258, "Samoyed"},
      {281, "tabby cat"},
      {282, "tiger cat"},
      {283, "Persian cat"},
      {285, "Egyptian cat"},
      {291, "lion"},
      {292, "tiger"},
      {340, "zebra"},
      {355, "llama"},
      {360, "otter"},
      {386, "African elephant"},
      {388, "giant panda"},
      {463, "bucket"},
      {508, "computer keyboard"},
      {530, "digital clock"},
      {543, "drum"},
      {620, "laptop"},
      {717, "pickup truck"},
      {751, "racket"},
      {779, "school bus"},
      {817, "sports car"},
      {849, "teapot"},
      {852, "tennis ball"},
      {864, "tow truck"},
      {895, "warplane"},
      {920, "traffic light"},
      {948, "Granny Smith"},
      {950, "orange"},
      {954, "banana"},
      {963, "pizza"},
  };
  auto it = labels.find(class_id);
  return it != labels.end() ? it->second : nullptr;
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
    const char* label = get_imagenet_label(idx);
    if (label) {
      std::cout << "  Class " << idx << " (" << label << "): " << logits[idx]
                << std::endl;
    } else {
      std::cout << "  Class " << idx << ": " << logits[idx] << std::endl;
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Load model
  std::unique_ptr<Module> model;
  if (!FLAGS_data_path.empty()) {
    model = std::make_unique<Module>(
        FLAGS_model_path, FLAGS_data_path, Module::LoadMode::Mmap);
  } else {
    model = std::make_unique<Module>(FLAGS_model_path, Module::LoadMode::Mmap);
  }

  // Prepare input tensor
  const int img_size = FLAGS_img_size;
  const size_t input_size = 1 * 3 * img_size * img_size;

  std::vector<float> input_data;
  if (!FLAGS_image_path.empty()) {
    input_data = load_image(FLAGS_image_path, img_size);
    if (input_data.empty()) {
      ET_LOG(Error, "Failed to load image");
      return 1;
    }
  } else {
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

  std::cout << "Output shape: (" << output_tensor.size(0) << ", " << num_classes
            << ")" << std::endl;

  // Convert output to float for top-k processing
  std::vector<float> logits_float(num_classes);
  if (output_tensor.scalar_type() == executorch::aten::ScalarType::BFloat16) {
    const auto* bf16_ptr =
        output_tensor.template const_data_ptr<executorch::aten::BFloat16>();
    for (int i = 0; i < num_classes; ++i) {
      logits_float[i] = static_cast<float>(bf16_ptr[i]);
    }
  } else {
    const float* float_ptr = output_tensor.template const_data_ptr<float>();
    for (int i = 0; i < num_classes; ++i) {
      logits_float[i] = float_ptr[i];
    }
  }

  // Print top-k predictions
  print_top_k(logits_float.data(), num_classes, FLAGS_top_k);

  return 0;
}
