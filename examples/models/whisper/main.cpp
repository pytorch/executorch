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

#include <executorch/extension/asr/runner/runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(model_path, "model.pte", "Path to Whisper model (.pte).");
DEFINE_string(data_path, "", "Optional path to Whisper weights (.ptd).");
DEFINE_string(
    tokenizer_path,
    ".",
    "Path to tokenizer directory containing tokenizer.json, tokenizer_config.json, and special_tokens_map.json.");
DEFINE_string(
    processor_path,
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
DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::executorch::extension::TensorPtr features;
  std::vector<float> audio_data;
  std::unique_ptr<Module> processor;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = FLAGS_cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(FLAGS_cpu_threads);
  std::cerr << "Using CPU threads: " << num_performant_cores << std::endl;
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif

  if (FLAGS_audio_path.empty()) {
    std::cerr << "ERROR: audio_path flag must be provided." << std::endl;
    ET_LOG(Error, "audio_path flag must be provided.");
    return 1;
  }
  std::cerr << "Loading audio from: " << FLAGS_audio_path << std::endl;

  std::cerr << "Calling load_wav_audio_data..." << std::endl;
  audio_data =
      executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
  std::cerr << "Audio data loaded, size: " << audio_data.size() << std::endl;
  ET_LOG(
      Info,
      "First 2 values of audio data: %f, %f",
      audio_data[0],
      audio_data[1]);

  processor =
      std::make_unique<Module>(FLAGS_processor_path, Module::LoadMode::Mmap);
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

  executorch::extension::asr::AsrRunner runner(
      FLAGS_model_path, FLAGS_data_path, FLAGS_tokenizer_path);
  auto load_err = runner.load();
  if (load_err != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load Whisper model.");
    return 1;
  }

  executorch::extension::asr::AsrTranscribeConfig config;
  config.max_new_tokens = FLAGS_max_new_tokens;
  config.temperature = static_cast<float>(FLAGS_temperature);

  // All Whisper models from HuggingFace now use the v3 tokenizer format
  // where token 50257 = <|endoftext|> and token 50258 = <|startoftranscript|>
  config.decoder_start_token_id = 50258;
  ET_LOG(Info, "Using decoder_start_token_id=50258");

  auto result =
      runner.transcribe(features, config, [&](const std::string& piece) {
        ::executorch::extension::llm::safe_printf(piece.c_str());
        fflush(stdout);
      });

  if (!result.ok()) {
    ET_LOG(Error, "Transcription failed.");
    std::cerr << "Transcription failed with error code: "
              << static_cast<int>(result.error()) << std::endl;
    return 1;
  }

  return 0;
}
