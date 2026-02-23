/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CLI entry point for the Sortformer diarizer.
//
// Loads a .pte model and a 16kHz mono WAV file, runs the three-stage
// diarization pipeline (preprocessor → pre_encode → streaming encode), and
// prints "start end speaker_N" segments. See README.md for usage.

#include <iomanip>
#include <iostream>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/runtime/platform/log.h>

#include "sortformer_runner.h"

DEFINE_string(model_path, "sortformer.pte", "Path to Sortformer model (.pte).");
DEFINE_string(audio_path, "", "Path to input audio file (.wav).");
DEFINE_double(threshold, 0.5, "Speaker activity threshold (0.0 - 1.0).");
DEFINE_int32(chunk_len, 124, "Streaming chunk length in 80ms frames.");
DEFINE_int32(fifo_len, 124, "FIFO buffer length in 80ms frames.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "audio_path flag must be provided.");
    return 1;
  }

  ::executorch::extension::llm::Stats stats;
  stats.model_load_start_ms = ::executorch::extension::llm::time_in_ms();

  sortformer::SortformerRunner runner(FLAGS_model_path);

  stats.model_load_end_ms = ::executorch::extension::llm::time_in_ms();
  stats.inference_start_ms = ::executorch::extension::llm::time_in_ms();

  ET_LOG(Info, "Loading audio from: %s", FLAGS_audio_path.c_str());
  auto audio_data =
      ::executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);

  sortformer::StreamingConfig config;
  config.chunk_len = static_cast<int64_t>(FLAGS_chunk_len);
  config.fifo_len = static_cast<int64_t>(FLAGS_fifo_len);

  std::cout << std::fixed << std::setprecision(3);

  auto result = runner.diarize(
      audio_data.data(),
      static_cast<int64_t>(audio_data.size()),
      static_cast<float>(FLAGS_threshold),
      config,
      [](const sortformer::Segment& seg) {
        std::cout << "  " << seg.start << " " << seg.end << " speaker_"
                  << seg.speaker << std::endl;
      });

  stats.inference_end_ms = ::executorch::extension::llm::time_in_ms();

  // Summary
  std::cout << "\n"
            << result.num_segments << " segments, " << result.num_frames
            << " frames, " << std::setprecision(1)
            << result.num_frames * runner.frame_duration() << "s\n";

  for (int spk = 0; spk < static_cast<int>(runner.max_spks()); spk++) {
    auto active = result.speaker_active_frames[static_cast<size_t>(spk)];
    if (active > 0) {
      double pct = 100.0 * active / static_cast<double>(result.num_frames);
      std::cout << "Speaker " << spk << ": " << active << "/"
                << result.num_frames << " frames active ("
                << std::setprecision(1) << pct << "%)\n";
    }
  }

  ::executorch::extension::llm::print_report(stats);

  return 0;
}
