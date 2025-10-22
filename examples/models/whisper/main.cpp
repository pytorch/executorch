/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>

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
DEFINE_string(
    preprocessed_audio_path,
    "",
    "Path to preprocessed audio features file (.bin). If provided, skips preprocessing.");
DEFINE_double(
    temperature,
    0.0,
    "Sampling temperature. 0.0 performs greedy decoding.");
DEFINE_int32(max_new_tokens, 128, "Maximum number of tokens to generate.");

namespace {

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;

bool ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
      value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<float> load_preprocessed_audio(
    const std::string& preprocessed_audio_path) {
  ET_LOG(
      Info,
      "Loading preprocessed audio from: %s",
      preprocessed_audio_path.c_str());

  std::ifstream stream(
      preprocessed_audio_path, std::ios::binary | std::ios::ate);
  if (!stream.is_open()) {
    ET_LOG(
        Error,
        "Failed to open preprocessed audio file: %s",
        preprocessed_audio_path.c_str());
    throw std::runtime_error("Failed to open preprocessed audio file");
  }

  std::size_t byte_size = static_cast<std::size_t>(stream.tellg());
  stream.seekg(0, std::ios::beg);

  const int64_t batch_size = 1;
  const int64_t feature_dim = 128;
  const int64_t time_steps = 3000;
  const int64_t expected_elements = batch_size * feature_dim * time_steps;
  const std::size_t expected_bytes = expected_elements * sizeof(float);

  if (byte_size != expected_bytes) {
    ET_LOG(
        Error,
        "Preprocessed audio file size mismatch. Expected %zu bytes, got %zu bytes",
        expected_bytes,
        byte_size);
    throw std::runtime_error("Preprocessed audio file size mismatch");
  }

  std::vector<float> feature_data(expected_elements);
  stream.read(reinterpret_cast<char*>(feature_data.data()), byte_size);
  stream.close();

  return feature_data;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ::executorch::extension::TensorPtr features;
  std::vector<float> audio_data;
  std::unique_ptr<Module> processor;

  if (!FLAGS_preprocessed_audio_path.empty()) {
    audio_data = load_preprocessed_audio(FLAGS_preprocessed_audio_path);

    const int64_t batch_size = 1;
    const int64_t feature_dim = 128;
    const int64_t time_steps = 3000;
    features = from_blob(
        audio_data.data(),
        /*sizes=*/{batch_size, feature_dim, time_steps},
        /*strides=*/{feature_dim * time_steps, feature_dim, 1},
        ::executorch::aten::ScalarType::Float);
  } else {
    // Original preprocessing path
    if (FLAGS_audio_path.empty()) {
      ET_LOG(
          Error,
          "Either audio_path or preprocessed_audio_path flag must be provided.");
      return 1;
    }

    audio_data =
        executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
    ET_LOG(
        Info,
        "First 2 values of audio data: %f, %f",
        audio_data[0],
        audio_data[1]);
    // Preprocess audio
    processor = std::make_unique<Module>(
        FLAGS_preprocessor_path, Module::LoadMode::Mmap);
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
  }

  example::WhisperRunner runner(
      FLAGS_model_path, FLAGS_data_path, FLAGS_tokenizer_path);
  auto load_err = runner.load();
  if (load_err != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load Whisper model.");
    return 1;
  }

  example::WhisperTranscribeConfig config;
  config.max_new_tokens = FLAGS_max_new_tokens;
  config.temperature = static_cast<float>(FLAGS_temperature);

  std::string transcript;
  auto result =
      runner.transcribe(features, config, [&](const std::string& piece) {
        ::executorch::extension::llm::safe_printf(piece.c_str());
        fflush(stdout);
      });

  if (!result.ok()) {
    ET_LOG(Error, "Transcription failed.");
    return 1;
  }

  return 0;
}
