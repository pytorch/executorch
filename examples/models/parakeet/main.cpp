/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <exception>
#include <iostream>
#include <string>

#include "parakeet_transcriber.h"

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
DEFINE_bool(
    runtime_profile,
    false,
    "Print a detailed runtime profile for preprocessor, encoder, and decode-loop execution.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  parakeet::TimestampOutputMode timestamp_mode;
  try {
    timestamp_mode = parakeet::parse_timestamp_output_mode(FLAGS_timestamps);
  } catch (const std::invalid_argument& e) {
    ET_LOG(Error, "%s", e.what());
    return 1;
  }

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "audio_path flag must be provided.");
    return 1;
  }

  try {
    parakeet::ParakeetTranscriber transcriber(
        FLAGS_model_path, FLAGS_tokenizer_path, FLAGS_data_path);
    const auto result = transcriber.transcribe_wav_path(
        FLAGS_audio_path,
        parakeet::TranscribeConfig{timestamp_mode, FLAGS_runtime_profile});

    std::cout << "Transcribed text: " << result.text << std::endl;
    if (!result.stats_json.empty()) {
      std::cout << "PyTorchObserver " << result.stats_json << std::endl;
    }
    if (result.runtime_profile_report.has_value()) {
      std::cout << *result.runtime_profile_report;
    }

#ifdef ET_BUILD_METAL
    executorch::backends::metal::print_metal_backend_stats();
#endif

    if (timestamp_mode.segment) {
      std::cout << "\nSegment timestamps:" << std::endl;
      for (const auto& segment : result.segment_offsets) {
        const double start = segment.start_offset * result.frame_to_seconds;
        const double end = segment.end_offset * result.frame_to_seconds;
        std::cout << start << "s - " << end << "s : " << segment.text
                  << std::endl;
      }
    }

    if (timestamp_mode.word) {
      std::cout << "\nWord timestamps:" << std::endl;
      for (const auto& word : result.word_offsets) {
        const double start = word.start_offset * result.frame_to_seconds;
        const double end = word.end_offset * result.frame_to_seconds;
        std::cout << start << "s - " << end << "s : " << word.text << std::endl;
      }
    }

    if (timestamp_mode.token) {
      std::cout << "\nToken timestamps:" << std::endl;
      for (const auto& token : result.token_offsets) {
        const double start = token.start_offset * result.frame_to_seconds;
        const double end = token.end_offset * result.frame_to_seconds;
        std::cout << start << "s - " << end << "s : " << token.decoded_text
                  << std::endl;
      }
    }

    return 0;
  } catch (const std::exception& e) {
    ET_LOG(Error, "%s", e.what());
    return 1;
  }
}
