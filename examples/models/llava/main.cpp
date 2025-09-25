/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <gflags/gflags.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
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
    "llava.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");

DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");

DEFINE_string(image_path, "", "The path to a .jpg file.");

DEFINE_double(
    temperature,
    0.8f,
    "Temperature; Default is 0.8f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    seq_len,
    1024,
    "Total number of tokens to generate (prompt + output). Defaults to max_seq_len. If the number of input tokens + seq_len > max_seq_len, the output will be truncated to max_seq_len tokens.");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

using ::executorch::extension::llm::Image;
using ::executorch::extension::llm::make_image_input;
using ::executorch::extension::llm::make_text_input;
using ::executorch::extension::llm::MultimodalInput;

void load_image(const std::string& image_path, Image& image) {
  int width, height, channels;
  unsigned char* data =
      stbi_load(image_path.c_str(), &width, &height, &channels, 0);
  if (!data) {
    ET_LOG(Fatal, "Failed to load image: %s", image_path.c_str());
    exit(1);
  }
  // resize the longest edge to 336
  int new_width = width;
  int new_height = height;
  if (width > height) {
    new_width = 336;
    new_height = static_cast<int>(height * 336.0 / width);
  } else {
    new_height = 336;
    new_width = static_cast<int>(width * 336.0 / height);
  }
  std::vector<uint8_t> resized_data(new_width * new_height * channels);
  stbir_resize_uint8(
      data,
      width,
      height,
      0,
      resized_data.data(),
      new_width,
      new_height,
      0,
      channels);
  std::vector<uint8_t> chw_data(channels * new_width * new_height);
  for (int i = 0; i < new_width * new_height; ++i) {
    for (int c = 0; c < channels; ++c) {
      chw_data[c * new_width * new_height + i] = resized_data[i * channels + c];
    }
  }
  image = Image(std::move(chw_data), new_width, new_height, channels);
  // convert to tensor
  ET_LOG(
      Info,
      "image Channels: %" PRId32 ", Height: %" PRId32 ", Width: %" PRId32,
      image.channels(),
      image.height(),
      image.width());
  stbi_image_free(data);
}

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point32_t to data that's already in memory,
  // and users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  const char* prompt = FLAGS_prompt.c_str();

  std::string image_path = FLAGS_image_path;

  double temperature = FLAGS_temperature;

  int32_t seq_len = FLAGS_seq_len;

  int32_t cpu_threads = FLAGS_cpu_threads;

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
  // Load tokenizer
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer =
      ::executorch::extension::llm::load_tokenizer(tokenizer_path);
  if (tokenizer == nullptr) {
    ET_LOG(Error, "Failed to load tokenizer from: %s", tokenizer_path);
    return 1;
  }

  // Create multimodal runner
  std::unique_ptr<::executorch::extension::llm::MultimodalRunner> runner =
      ::executorch::extension::llm::create_multimodal_runner(
          model_path, std::move(tokenizer));
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
  static const char* kPresetPrompt =
      "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ";
  Image image;
  load_image(image_path, image);
  std::vector<MultimodalInput> inputs = {
      make_text_input(std::string(kPresetPrompt)),
      make_image_input(image),
      make_text_input(std::string(prompt)),
  };

  ::executorch::extension::llm::GenerationConfig config;
  config.temperature = temperature;
  config.echo = true;

  // Generate
  ET_LOG(Info, "Starting generation...");
  auto error = runner->generate(inputs, config);
  if (error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to generate with multimodal runner");
    return 1;
  }

  printf("\n");
  return 0;
}
