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
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#ifdef GEMMA4_ENABLE_VISION
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>
#endif

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(model_path, "gemma4.pte", "Model serialized in flatbuffer format.");
DEFINE_string(data_path, "", "Path to data file (.ptd for CUDA).");
DEFINE_string(tokenizer_path, "tokenizer.json", "Tokenizer file.");
DEFINE_string(prompt, "Hello, how are you?", "Text prompt.");
DEFINE_string(image_path, "", "Path to input image file (optional for multimodal).");
DEFINE_double(temperature, 0.0f, "Sampling temperature. 0 = greedy argmax.");
DEFINE_int32(cpu_threads, -1, "Number of CPU threads (-1 = auto).");
DEFINE_int32(seq_len, 512, "Maximum new tokens to generate.");
DEFINE_int32(target_size, 896, "Target image size for resizing.");
DEFINE_bool(warmup, false, "Run a warmup pass before generation.");
DEFINE_bool(
    raw_prompt,
    false,
    "If set, --prompt is sent verbatim (no chat-template wrapping). Use with "
    "render_chat.py output for system prompts, tools, or reasoning mode.");

namespace {

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::llm::make_text_input;
using ::executorch::extension::llm::MultimodalInput;
using ::executorch::runtime::EValue;

#ifdef GEMMA4_ENABLE_VISION
using ::executorch::extension::llm::Image;
using ::executorch::extension::llm::make_image_input;

MultimodalInput loadImage(const std::string& image_path) {
  int width, height, channels;
  unsigned char* data =
      stbi_load(image_path.c_str(), &width, &height, &channels, 0);
  if (!data) {
    ET_LOG(Error, "Failed to load image: %s", image_path.c_str());
    throw std::runtime_error("Failed to load image");
  }

  ET_LOG(
      Info, "Loaded image: %s (%dx%d, %d ch)",
      image_path.c_str(), width, height, channels);

  const int target_size = FLAGS_target_size;
  std::vector<uint8_t> resized_data(target_size * target_size * channels);
  stbir_resize_uint8(
      data, width, height, 0,
      resized_data.data(), target_size, target_size, 0, channels);

  std::vector<float> chw_data(channels * target_size * target_size);
  for (int h = 0; h < target_size; ++h) {
    for (int w = 0; w < target_size; ++w) {
      for (int c = 0; c < channels; ++c) {
        uint8_t pixel = resized_data[h * target_size * channels + w * channels + c];
        chw_data[c * target_size * target_size + h * target_size + w] =
            static_cast<float>(pixel) / 255.0f;
      }
    }
  }

  Image image(std::move(chw_data), target_size, target_size, channels);
  stbi_image_free(data);
  return make_image_input(std::move(image));
}
#endif

} // namespace

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char* model_path = FLAGS_model_path.c_str();
  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();
  const char* data_path = FLAGS_data_path.c_str();
  float temperature = FLAGS_temperature;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_cores = FLAGS_cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(FLAGS_cpu_threads);
  ET_LOG(Info, "Using %d CPU threads", num_cores);
  if (num_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_cores);
  }
#endif

  auto tokenizer = ::executorch::extension::llm::load_tokenizer(tokenizer_path);
  if (!tokenizer) {
    ET_LOG(Error, "Failed to load tokenizer from: %s", tokenizer_path);
    return 1;
  }

  // Use the standard text LLM runner (single `forward` method).
  fprintf(stderr, "Creating text runner...\n");
  std::optional<const std::string> opt_data_path =
      strlen(data_path) > 0 ? std::optional(std::string(data_path)) : std::nullopt;
  auto runner = ::executorch::extension::llm::create_text_llm_runner(
      model_path, std::move(tokenizer), opt_data_path);
  if (!runner) {
    ET_LOG(Error, "Failed to create runner");
    return 1;
  }

  fprintf(stderr, "Loading model (%s)...\n", model_path);
  auto load_error = runner->load();
  if (load_error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load model: %d", static_cast<int>(load_error));
    return 1;
  }
  fprintf(stderr, "Model loaded.\n");

  // Gemma4 chat template uses <|turn> ... <turn|> markers (not Gemma3's
  // <start_of_turn>/<end_of_turn>). BOS is prepended automatically by the
  // runner. Use --raw_prompt with `render_chat.py` output for system prompts,
  // tools, or reasoning mode (the official vLLM template).
  std::string full_prompt = FLAGS_raw_prompt
      ? FLAGS_prompt
      : "<|turn>user\n" + FLAGS_prompt + "<turn|>\n<|turn>model\n";

  ::executorch::extension::llm::GenerationConfig config;
  config.max_new_tokens = FLAGS_seq_len;
  config.temperature = temperature;

  auto error = runner->generate(full_prompt, config);
  if (error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Generation failed");
    return 1;
  }

  printf("\n");
  return 0;
}
