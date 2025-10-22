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
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#if defined(__has_include)
#if __has_include(<cuda_runtime_api.h>)
#define ET_GEMMA3_HAS_CUDA_RUNTIME 1
#include <cuda_runtime_api.h>
#else
#define ET_GEMMA3_HAS_CUDA_RUNTIME 0
#endif
#else
#define ET_GEMMA3_HAS_CUDA_RUNTIME 0
#endif

#include <executorch/examples/models/whisper/runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

DEFINE_string(model_path, "model.pte", "Path to Whisper model (.pte).");
DEFINE_string(data_path, "", "Optional path to Whisper weights (.ptd).");
DEFINE_string(
    tokenizer_path,
    ".",
    "Path to tokenizer directory containing tokenizer.json, tokenizer_config.json, and special_tokens_map.json.");
DEFINE_string(
    preprocessor_path,
    "",
    "Path to preprocessor .pte for converting raw audio.");
DEFINE_string(
    audio_path,
    "",
    "Path to input audio file. Accepts .wav or raw float .bin.");
DEFINE_double(
    temperature,
    0.0,
    "Sampling temperature. 0.0 performs greedy decoding.");
DEFINE_int32(max_new_tokens, 128, "Maximum number of tokens to generate.");

namespace {

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;

#if ET_GEMMA3_HAS_CUDA_RUNTIME
class CudaMemoryTracker {
 public:
  CudaMemoryTracker() {
    if (!query(&last_free_bytes_, &total_bytes_)) {
      return;
    }
    available_ = true;
    min_free_bytes_ = last_free_bytes_;
    log_state("startup", last_free_bytes_, total_bytes_);
  }

  void log_sample(const char* tag) {
    if (!available_) {
      return;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!query(&free_bytes, &total_bytes)) {
      return;
    }
    min_free_bytes_ = std::min(min_free_bytes_, free_bytes);
    total_bytes_ = total_bytes;
    last_free_bytes_ = free_bytes;
    log_state(tag, free_bytes, total_bytes);
  }

  ~CudaMemoryTracker() {
    if (!available_) {
      return;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!query(&free_bytes, &total_bytes)) {
      return;
    }
    min_free_bytes_ = std::min(min_free_bytes_, free_bytes);
    total_bytes_ = total_bytes;
    last_free_bytes_ = free_bytes;
    const double peak_mb =
        static_cast<double>(total_bytes_ - min_free_bytes_) / (1024.0 * 1024.0);
    const double total_mb =
        static_cast<double>(total_bytes_) / (1024.0 * 1024.0);
    std::cout << "CUDA memory peak usage: " << peak_mb
              << " MB, total: " << total_mb << " MB" << std::endl;
  }

 private:
  bool query(size_t* free_bytes, size_t* total_bytes) {
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    if (err != cudaSuccess) {
      if (!error_logged_) {
        error_logged_ = true;
        std::cerr << "Warning: cudaMemGetInfo failed with error: "
                  << cudaGetErrorString(err) << std::endl;
      }
      available_ = false;
      return false;
    }
    return true;
  }

  void log_state(const char* tag, size_t free_bytes, size_t total_bytes) const {
    const double used_mb =
        static_cast<double>(total_bytes - free_bytes) / (1024.0 * 1024.0);
    const double free_mb = static_cast<double>(free_bytes) / (1024.0 * 1024.0);
    const double total_mb =
        static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    std::cout << "CUDA memory (" << tag << "): used " << used_mb << " MB, free "
              << free_mb << " MB, total " << total_mb << " MB" << std::endl;
  }

  bool available_{false};
  bool error_logged_{false};
  size_t last_free_bytes_{0};
  size_t total_bytes_{0};
  size_t min_free_bytes_{std::numeric_limits<size_t>::max()};
};
#else
class CudaMemoryTracker {
 public:
  CudaMemoryTracker() = default;
  void log_sample(const char* tag) {
    (void)tag;
  }
};
#endif

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CudaMemoryTracker cuda_memory_tracker;
  ::executorch::extension::TensorPtr features;
  std::vector<float> audio_data;
  std::unique_ptr<Module> processor;

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "audio_path flag must be provided.");
    return 1;
  }

  audio_data =
      executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
  ET_LOG(
      Info,
      "First 2 values of audio data: %f, %f",
      audio_data[0],
      audio_data[1]);

  processor =
      std::make_unique<Module>(FLAGS_preprocessor_path, Module::LoadMode::Mmap);
  auto load_error = processor->load();
  if (load_error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load preprocessor module.");
    return 1;
  }

  auto audio_tensor = from_blob(
      audio_data.data(),
      {static_cast<::executorch::aten::SizesType>(audio_data.size())},
      ::executorch::aten::ScalarType::Float);

  auto processed_result = processor->execute("forward", audio_tensor);
  if (processed_result.error() != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Audio preprocessing failed.");
    return 1;
  }
  auto outputs = std::move(processed_result.get());
  if (outputs.empty() || !outputs[0].isTensor()) {
    ET_LOG(Error, "Preprocessor returned unexpected outputs.");
    return 1;
  }
  auto tensor = outputs[0].toTensor();
  ET_LOG(
      Info,
      "Result scalar_type: %s, first value %f",
      ::executorch::runtime::toString(tensor.scalar_type()),
      tensor.mutable_data_ptr<float>()[0]);
  features = std::make_shared<::executorch::aten::Tensor>(std::move(tensor));

  example::WhisperRunner runner(
      FLAGS_model_path, FLAGS_data_path, FLAGS_tokenizer_path);
  auto load_err = runner.load();
  if (load_err != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load Whisper model.");
    return 1;
  }
  cuda_memory_tracker.log_sample("post-runner-load");

  example::WhisperTranscribeConfig config;
  config.max_new_tokens = FLAGS_max_new_tokens;
  config.temperature = static_cast<float>(FLAGS_temperature);

  std::string transcript;
  auto result =
      runner.transcribe(features, config, [&](const std::string& piece) {
        ::executorch::extension::llm::safe_printf(piece.c_str());
        fflush(stdout);
      });
  cuda_memory_tracker.log_sample("post-transcribe");

  if (!result.ok()) {
    ET_LOG(Error, "Transcription failed.");
    return 1;
  }

  return 0;
}
