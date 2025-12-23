/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstring>
#include <fstream>

#include <gflags/gflags.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(
    model_path,
    "multimodal.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(data_path, "", "Path to data file.");
DEFINE_string(tokenizer_path, "tokenizer.json", "Tokenizer stuff.");

DEFINE_string(prompt, "What is in this image?", "Text prompt.");

DEFINE_string(image_path, "", "Path to input image file.");

DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

DEFINE_bool(warmup, false, "Whether to run a warmup run.");

namespace {

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::llm::Image;
using ::executorch::extension::llm::make_image_input;
using ::executorch::extension::llm::make_text_input;
using ::executorch::extension::llm::MultimodalInput;
using ::executorch::runtime::EValue;

bool ends_with(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
      str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/**
 * @brief Loads an image from a file and resizes it to 896x896
 *
 * This function loads an image using stb_image and resizes it to the expected
 * input size for Gemma3 (896x896). The image is converted to CHW (Channel,
 * Height, Width) format which is expected by the model.
 *
 * @param image_path Path to the image file (.jpg, .png, etc.)
 * @return MultimodalInput containing the loaded and processed image data
 * @throws std::runtime_error if image loading fails
 */
MultimodalInput loadImage(const std::string& image_path) {
  if (!ends_with(image_path, ".jpg") && !ends_with(image_path, ".jpeg") &&
      !ends_with(image_path, ".png") && !ends_with(image_path, ".bmp")) {
    ET_LOG(
        Error,
        "Unsupported image file format: %s (only .jpg, .jpeg, .png, .bmp are supported)",
        image_path.c_str());
    throw std::runtime_error("Unsupported image file format");
  }

  int width, height, channels;
  unsigned char* data =
      stbi_load(image_path.c_str(), &width, &height, &channels, 0);
  if (!data) {
    ET_LOG(Error, "Failed to load image: %s", image_path.c_str());
    throw std::runtime_error("Failed to load image");
  }

  ET_LOG(
      Info,
      "Loaded image: %s, original size: %dx%d, channels: %d",
      image_path.c_str(),
      width,
      height,
      channels);

  // Resize to 896x896 (Gemma3 vision encoder input size)
  const int target_size = 896;
  std::vector<uint8_t> resized_data(target_size * target_size * channels);

  int resize_result = stbir_resize_uint8(
      data,
      width,
      height,
      0,
      resized_data.data(),
      target_size,
      target_size,
      0,
      channels);

  if (!resize_result) {
    stbi_image_free(data);
    ET_LOG(Error, "Failed to resize image");
    throw std::runtime_error("Failed to resize image");
  }

  // Convert from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
  // and normalize uint8 [0, 255] to float32 [0.0, 1.0]
  std::vector<float> chw_data(channels * target_size * target_size);
  for (int h = 0; h < target_size; ++h) {
    for (int w = 0; w < target_size; ++w) {
      for (int c = 0; c < channels; ++c) {
        uint8_t pixel_value =
            resized_data[h * target_size * channels + w * channels + c];
        chw_data[c * target_size * target_size + h * target_size + w] =
            static_cast<float>(pixel_value) / 255.0f;
      }
    }
  }

  ET_LOG(
      Info,
      "Resized and converted image to CHW format (float32): %dx%d, channels: %d",
      target_size,
      target_size,
      channels);

  Image image(std::move(chw_data), target_size, target_size, channels);
  stbi_image_free(data);

  return make_image_input(std::move(image));
}

} // namespace

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char* model_path = FLAGS_model_path.c_str();
  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();
  const char* prompt = FLAGS_prompt.c_str();
  const char* image_path = FLAGS_image_path.c_str();
  const char* data_path = FLAGS_data_path.c_str();
  float temperature = FLAGS_temperature;
  int32_t cpu_threads = FLAGS_cpu_threads;
  bool warmup = FLAGS_warmup;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif

  std::unique_ptr<::tokenizers::Tokenizer> tokenizer =
      ::executorch::extension::llm::load_tokenizer(tokenizer_path);
  if (tokenizer == nullptr) {
    ET_LOG(Error, "Failed to load tokenizer from: %s", tokenizer_path);
    return 1;
  }

  // Create multimodal runner
  std::unique_ptr<::executorch::extension::llm::MultimodalRunner> runner =
      ::executorch::extension::llm::create_multimodal_runner(
          model_path, std::move(tokenizer), data_path);

  if (runner == nullptr) {
    ET_LOG(Error, "Failed to create multimodal runner");
    return 1;
  }

  // Load runner
  auto load_error = runner->load();
  if (load_error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load multimodal runner");
    return 1;
  }

  // Prepare inputs
  std::vector<MultimodalInput> inputs = {
      make_text_input("<start_of_turn>user\n<start_of_image>"),
      loadImage(image_path),
      make_text_input(
          std::string(prompt) + "<end_of_turn>\n<start_of_turn>model\n"),
  };

  ::executorch::extension::llm::GenerationConfig config;
  config.max_new_tokens = 100;
  config.temperature = temperature;

  // Run warmup if requested
  if (warmup) {
    ET_LOG(Info, "Running warmup...");
    auto warmup_error = runner->generate(inputs, config);
    if (warmup_error != ::executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to run warmup");
      return 1;
    }
    runner->reset();
  }

  auto error = runner->generate(inputs, config);

  if (error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to generate with multimodal runner\n");
    return 1;
  }
  ET_LOG(Info, "Generated successfully");

  return 0;
}
