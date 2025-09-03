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

#include <executorch/extension/llm/runner/audio.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(
    model_path,
    "multimodal.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(tokenizer_path, "tekken.json", "Tokenizer stuff.");

DEFINE_string(prompt, "What is happening in this audio?", "Text prompt.");

DEFINE_string(audio_path, "", "Path to input audio file.");

DEFINE_double(
    temperature,
    0.8f,
    "Temperature; Default is 0.8f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

DEFINE_bool(warmup, false, "Whether to run a warmup run.");

namespace {

using ::executorch::extension::llm::Image;
using ::executorch::extension::llm::make_image_input;
using ::executorch::extension::llm::make_text_input;
using ::executorch::extension::llm::MultimodalInput;

bool ends_with(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
      str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/**
 * @brief Loads preprocessed audio data from a binary file
 *
 * Reads mel spectrogram features that have been pre-computed and saved as a
 * binary file. The audio data is expected to be stored as float values in
 * binary format, typically saved using:
 *   with open("tensor.bin", "wb") as f:
 *       f.write(t.numpy().tobytes())
 *
 * @param audio_path Path to the binary audio file (.bin)
 * @return MultimodalInput containing the loaded audio data
 */
MultimodalInput loadPreprocessedAudio(const std::string& audio_path) {
  std::ifstream f(audio_path, std::ios::binary | std::ios::ate);
  int32_t n_bins = 128;
  int32_t n_frames = 3000;
  std::size_t n_floats =
      f.tellg() / sizeof(float); // Number of floats in the audio file.
  f.seekg(0, std::ios::beg);
  int32_t batch_size = ceil(
      n_floats /
      (n_bins * n_frames)); // Batch in increments of n_frames, rounding up.
  std::vector<float> audio_data(batch_size * n_bins * n_frames);
  f.read(
      reinterpret_cast<char*>(audio_data.data()),
      audio_data.size() * sizeof(float));

  ET_LOG(Info, "audio_data len = %d", audio_data.size());

  auto audio = std::make_unique<::executorch::extension::llm::Audio>();
  audio->batch_size = batch_size;
  audio->n_bins = n_bins;
  audio->n_frames = n_frames;
  audio->data.resize(audio_data.size() * sizeof(float));
  std::memcpy(
      audio->data.data(), audio_data.data(), audio_data.size() * sizeof(float));
  return ::executorch::extension::llm::make_audio_input(std::move(*audio));
}

/**
 * @brief Processes audio files for multimodal input
 *
 * Dispatches audio file processing based on file extension:
 * - .bin files: Loads preprocessed mel spectrogram features directly
 * - .wav/.mp3 files: Currently unsupported, throws runtime_error
 *
 * This function provides a interface for different audio input formats
 * and can be extended to support raw audio processing in the future.
 *
 * @param audio_path Path to the audio file
 * @return MultimodalInput containing the processed audio data
 * @throws std::runtime_error if file format is unsupported or processing fails
 */
MultimodalInput processAudioFile(const std::string& audio_path) {
  if (ends_with(audio_path, ".bin")) {
    // Current behavior - load preprocessed audio stored as a binary file.
    return loadPreprocessedAudio(audio_path);
  } else if (ends_with(audio_path, ".wav") || ends_with(audio_path, ".mp3")) {
    // New: Process raw audio files - unsupported for now
    ET_LOG(Error, "Raw audio file processing (.wav/.mp3) is not yet supported");
    throw std::runtime_error("Raw audio file processing not supported");
  } else {
    ET_LOG(Error, "Unsupported audio file format: %s", audio_path.c_str());
    throw std::runtime_error("Unsupported audio file format");
  }
}

} // namespace

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char* model_path = FLAGS_model_path.c_str();

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();
  const char* prompt = FLAGS_prompt.c_str();
  const char* audio_path = FLAGS_audio_path.c_str();
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
  std::vector<MultimodalInput> inputs;

  // 1. Add start bos-related text inputs and modality start token.
  inputs.emplace_back(make_text_input("<s>[INST][BEGIN_AUDIO]"));

  // 2. Add audio input
  inputs.emplace_back(processAudioFile(audio_path));

  // 3. Add text input (the actual user-submitted prompt)
  inputs.emplace_back(make_text_input(std::string(prompt) + "[/INST]"));

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
