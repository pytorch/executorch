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

DEFINE_string(
    processor_path,
    "",
    "Path to processor .pte file for raw audio processing.");

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
  if (!f.is_open()) {
    ET_LOG(Error, "Failed to open audio file: %s", audio_path.c_str());
    throw std::runtime_error("Failed to open audio file");
  }

  std::size_t n_floats = f.tellg() / sizeof(float);
  f.seekg(0, std::ios::beg);

  int32_t n_bins = 128;
  int32_t n_frames = 3000;

  int32_t batch_size = ceil(
      n_floats /
      (n_bins * n_frames)); // Batch in increments of n_frames, rounding up.

  ET_LOG(Info, "audio_data len = %zu", n_floats);

  // Create Audio multimodal input
  auto audio = std::make_unique<::executorch::extension::llm::Audio>();
  audio->batch_size = batch_size;
  audio->n_bins = n_bins;
  audio->n_frames = n_frames;
  audio->data.resize(n_floats * sizeof(float));
  f.read(reinterpret_cast<char*>(audio->data.data()), n_floats * sizeof(float));
  f.close();
  return ::executorch::extension::llm::make_audio_input(std::move(*audio));
}

/**
 * @brief Loads a .bin file into a tensor and processes it using a .pte
 * processor
 *
 * This function loads raw audio data from a .bin file (similar to
 * loadPreprocessedAudio), creates a tensor from it, and then passes it through
 * a processor module loaded from a .pte file to generate processed audio
 * features.
 *
 * @param audio_path Path to the .bin audio file
 * @param processor_path Path to the .pte processor file
 * @return MultimodalInput containing the processed audio data
 * @throws std::runtime_error if file loading or processing fails
 */
MultimodalInput processRawAudioFile(
    const std::string& audio_path,
    const std::string& processor_path) {
  if (processor_path.empty()) {
    ET_LOG(Error, "Processor path is required for raw audio processing");
    throw std::runtime_error(
        "Processor path is required for raw audio processing");
  }

  // Load the audio processor .pte.
  std::unique_ptr<Module> processor_module;
  try {
    processor_module =
        std::make_unique<Module>(processor_path, Module::LoadMode::File);
    auto load_error = processor_module->load();
    if (load_error != ::executorch::runtime::Error::Ok) {
      ET_LOG(
          Error,
          "Failed to load processor module from: %s",
          processor_path.c_str());
      throw std::runtime_error("Failed to load processor module");
    }
  } catch (const std::exception& e) {
    ET_LOG(Error, "Exception while loading processor module: %s", e.what());
    throw std::runtime_error("Exception while loading processor module");
  }

  // Load the audio data from file.
  std::ifstream f(audio_path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    ET_LOG(Error, "Failed to open audio file: %s", audio_path.c_str());
    throw std::runtime_error("Failed to open audio file");
  }

  std::size_t n_floats = f.tellg() / sizeof(float);
  f.seekg(0, std::ios::beg);

  std::vector<float> audio_data(n_floats);
  f.read(
      reinterpret_cast<char*>(audio_data.data()),
      audio_data.size() * sizeof(float));
  f.close();

  ET_LOG(
      Info, "Loaded .bin file: %s, %zu floats", audio_path.c_str(), n_floats);

  // Execute the processor
  std::vector<executorch::aten::SizesType> tensor_shape = {
      static_cast<executorch::aten::SizesType>(audio_data.size())};
  auto input_tensor = from_blob(
      audio_data.data(), tensor_shape, ::executorch::aten::ScalarType::Float);

  ET_LOG(Info, "Processing audio through processor module...");
  auto result = processor_module->execute("forward", input_tensor);
  if (!result.ok()) {
    ET_LOG(Error, "Failed to execute processor's forward method");
    throw std::runtime_error("Failed to execute processor forward method");
  }

  auto outputs = result.get();
  if (outputs.empty()) {
    ET_LOG(Error, "Processor returned no outputs");
    throw std::runtime_error("Processor returned no outputs");
  }

  // Extract processed audio features
  const auto& processed_tensor = outputs[0].toTensor();
  const float* processed_data = processed_tensor.const_data_ptr<float>();
  const auto& sizes = processed_tensor.sizes();

  ET_LOG(
      Info,
      "Processed audio tensor shape: [%d, %d, %d]",
      static_cast<int>(sizes[0]),
      static_cast<int>(sizes[1]),
      static_cast<int>(sizes[2]));

  // Create Audio multimodal input from processed features
  auto processed_audio =
      std::make_unique<::executorch::extension::llm::Audio>();
  processed_audio->batch_size =
      static_cast<int32_t>(sizes[0]); // Note: batching for s > 30 doesn't work
                                      // yet, so this will just be = 1.
  processed_audio->n_bins = static_cast<int32_t>(sizes[1]);
  processed_audio->n_frames =
      static_cast<int32_t>(sizes[2]); // And this will just be = 3000.

  size_t total_elements = processed_audio->batch_size *
      processed_audio->n_bins * processed_audio->n_frames;
  processed_audio->data.resize(total_elements * sizeof(float));
  std::memcpy(
      processed_audio->data.data(),
      processed_data,
      total_elements * sizeof(float));

  ET_LOG(
      Info,
      "Created processed Audio: batch_size=%d, n_bins=%d, n_frames=%d",
      processed_audio->batch_size,
      processed_audio->n_bins,
      processed_audio->n_frames);

  return ::executorch::extension::llm::make_audio_input(
      std::move(*processed_audio));
}

/**
 * @brief Processes audio files for multimodal input
 *
 * Dispatches audio file processing based on file extension and processor
 * availability:
 * - .bin files with processor: Loads raw audio from .bin and processes through
 * processor
 * - .bin files without processor: Loads preprocessed mel spectrogram features
 * directly
 *
 * @param audio_path Path to the audio file (.bin)
 * @param processor_path Path to the processor .pte file (optional)
 * @return MultimodalInput containing the processed audio data
 * @throws std::runtime_error if file format is unsupported or processing fails
 */
MultimodalInput processAudioFile(
    const std::string& audio_path,
    const std::string& processor_path = "") {
  if (ends_with(audio_path, ".bin")) {
    if (!processor_path.empty()) {
      // Process raw audio from .bin file through the processor
      return processRawAudioFile(audio_path, processor_path);
    } else {
      // Load preprocessed audio stored as a binary file (existing behavior)
      return loadPreprocessedAudio(audio_path);
    }
  } else {
    ET_LOG(
        Error,
        "Unsupported audio file format: %s (only .bin files are supported)",
        audio_path.c_str());
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
  const char* processor_path = FLAGS_processor_path.c_str();
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
  std::vector<MultimodalInput> inputs = {
      make_text_input("<s>[INST][BEGIN_AUDIO]"),
      processAudioFile(audio_path, processor_path),
      make_text_input(std::string(prompt) + "[/INST]"),
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
