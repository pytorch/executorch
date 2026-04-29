/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <gflags/gflags.h>

#include "timestamp_utils.h"
#include "tokenizer_utils.h"
#include "types.h"

#include <executorch/extension/asr/runner/transducer_runner.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>
#ifdef ET_BUILD_METAL
#include <executorch/backends/apple/metal/runtime/stats.h>
#endif

DEFINE_string(model_path, "parakeet.pte", "Path to Parakeet model (.pte).");
DEFINE_string(audio_path, "", "Path to input audio file (.wav).");
DEFINE_string(
    tokenizer_path,
    "tokenizer.model",
    "Path to SentencePiece tokenizer model file.");
DEFINE_string(
    data_path,
    "",
    "Path to data file (.ptd) for delegate data (optional, required for CUDA).");
DEFINE_string(
    timestamps,
    "segment",
    "Timestamp output mode: none|token|word|segment|all");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using ::parakeet::TextWithOffsets;
using ::parakeet::TokenWithTextInfo;

namespace {
// TDT duration values for Parakeet models
const std::vector<int> DURATIONS = {0, 1, 2, 3, 4};

struct TimestampOutputMode {
  bool token = false;
  bool word = false;
  bool segment = false;

  bool enabled() const {
    return token || word || segment;
  }
};

std::string to_lower_ascii(std::string s) {
  for (char& ch : s) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return s;
}

TimestampOutputMode parse_timestamp_output_mode(const std::string& raw_arg) {
  if (raw_arg.empty()) {
    throw std::invalid_argument(
        "Invalid --timestamps value (empty). Expected: token, word, segment, all.");
  }
  const std::string mode = to_lower_ascii(raw_arg);
  if (mode == "none") {
    return {false, false, false};
  }
  if (mode == "token") {
    return {true, false, false};
  }
  if (mode == "word") {
    return {false, true, false};
  }
  if (mode == "segment") {
    return {false, false, true};
  }
  if (mode == "all") {
    return {true, true, true};
  }
  throw std::invalid_argument(
      "Invalid --timestamps value '" + raw_arg +
      "'. Expected: token, word, segment, all.");
}
} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  TimestampOutputMode timestamp_mode;
  try {
    timestamp_mode = parse_timestamp_output_mode(FLAGS_timestamps);
  } catch (const std::invalid_argument& e) {
    ET_LOG(Error, "%s", e.what());
    return 1;
  }

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "audio_path flag must be provided.");
    return 1;
  }

  // --- Build config and runner ---
  executorch::extension::asr::TransducerConfig config;
  config.durations = DURATIONS;

  std::optional<std::string> data_path_opt;
  if (!FLAGS_data_path.empty()) {
    data_path_opt = FLAGS_data_path;
  }

  executorch::extension::asr::TransducerRunner runner(
      FLAGS_model_path, FLAGS_tokenizer_path, config, data_path_opt);

  auto load_err = runner.load();
  if (load_err != Error::Ok) {
    ET_LOG(Error, "Failed to load model.");
    return 1;
  }

  // --- Load and preprocess audio ---
  ET_LOG(Info, "Loading audio from: %s", FLAGS_audio_path.c_str());
  std::vector<float> audio_data =
      ::executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
  ET_LOG(Info, "Loaded %zu audio samples", audio_data.size());

  auto audio_tensor = from_blob(
      audio_data.data(),
      {static_cast<::executorch::aten::SizesType>(audio_data.size())},
      ::executorch::aten::ScalarType::Float);

  ET_LOG(Info, "Running preprocessor...");
  auto preprocess_result = runner.preprocess(audio_tensor);
  if (!preprocess_result.ok()) {
    ET_LOG(Error, "Preprocessing failed.");
    return 1;
  }
  auto mel_features = preprocess_result.get();

  // --- Transcribe ---
  ET_LOG(Info, "Running TDT greedy decode...");
  auto result = runner.transcribe(mel_features, [](const std::string& piece) {
    std::cout << piece << std::flush;
  });

  if (!result.ok()) {
    ET_LOG(Error, "Transcription failed.");
    return 1;
  }

  auto& decoded_tokens = result.get();
  ET_LOG(Info, "Decoded %zu tokens", decoded_tokens.size());

  // Use the runner's tokenizer for text decoding and timestamps
  const auto* tokenizer = runner.tokenizer();
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer not available.");
    return 1;
  }

  // Print full transcribed text
  std::string text = parakeet::tokenizer_utils::decode_token_sequence(
      decoded_tokens, *tokenizer);
  std::cout << "\nTranscribed text: " << text << std::endl;

#ifdef ET_BUILD_METAL
  executorch::backends::metal::print_metal_backend_stats();
#endif // ET_BUILD_METAL

  if (!timestamp_mode.enabled()) {
    return 0;
  }

  // --- Timestamps ---
  // Query timestamp-related metadata from the model.
  // These are Parakeet-specific constants, not part of TransducerRunner.
  std::unique_ptr<Module> meta_module;
  if (data_path_opt) {
    meta_module = std::make_unique<Module>(
        FLAGS_model_path, *data_path_opt, Module::LoadMode::Mmap);
  } else {
    meta_module =
        std::make_unique<Module>(FLAGS_model_path, Module::LoadMode::Mmap);
  }
  auto meta_load_err = meta_module->load();
  if (meta_load_err != Error::Ok) {
    ET_LOG(Error, "Failed to load model for timestamp metadata.");
    return 1;
  }

  std::vector<::executorch::runtime::EValue> empty_inputs;
  auto window_stride_result =
      meta_module->execute("window_stride", empty_inputs);
  auto encoder_subsampling_factor_result =
      meta_module->execute("encoder_subsampling_factor", empty_inputs);

  if (!window_stride_result.ok() || !encoder_subsampling_factor_result.ok()) {
    ET_LOG(
        Error,
        "Failed to query timestamp metadata (window_stride, encoder_subsampling_factor).");
    return 1;
  }

  double window_stride = window_stride_result.get()[0].toDouble();
  int64_t encoder_subsampling_factor =
      encoder_subsampling_factor_result.get()[0].toInt();
  meta_module.reset();

  ET_LOG(Info, "Computing timestamps...");
  std::unordered_set<std::string> supported_punctuation =
      parakeet::tokenizer_utils::derive_supported_punctuation(*tokenizer);

  std::vector<TokenWithTextInfo> tokens_with_text_info;
  try {
    tokens_with_text_info =
        parakeet::timestamp_utils::get_tokens_with_text_info(
            decoded_tokens, *tokenizer, supported_punctuation);
  } catch (const std::exception& e) {
    ET_LOG(Error, "Failed to get tokens with text info: %s", e.what());
    return 1;
  }
  const auto word_offsets = parakeet::timestamp_utils::get_words_offsets(
      tokens_with_text_info, *tokenizer, supported_punctuation);
  const auto segment_offsets =
      parakeet::timestamp_utils::get_segment_offsets(word_offsets);

  const double frame_to_seconds =
      window_stride * static_cast<double>(encoder_subsampling_factor);

  if (timestamp_mode.segment) {
    std::cout << "\nSegment timestamps:" << std::endl;
    for (const auto& segment : segment_offsets) {
      const double start = segment.start_offset * frame_to_seconds;
      const double end = segment.end_offset * frame_to_seconds;
      std::cout << start << "s - " << end << "s : " << segment.text
                << std::endl;
    }
  }

  if (timestamp_mode.word) {
    std::cout << "\nWord timestamps:" << std::endl;
    for (const auto& word : word_offsets) {
      const double start = word.start_offset * frame_to_seconds;
      const double end = word.end_offset * frame_to_seconds;
      std::cout << start << "s - " << end << "s : " << word.text << std::endl;
    }
  }

  if (timestamp_mode.token) {
    std::cout << "\nToken timestamps:" << std::endl;
    for (const auto& token : tokens_with_text_info) {
      const double start = token.start_offset * frame_to_seconds;
      const double end = token.end_offset * frame_to_seconds;
      std::cout << start << "s - " << end << "s : " << token.decoded_text
                << std::endl;
    }
  }

  return 0;
}
