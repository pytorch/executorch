/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CLI entry point for Silero VAD.
//
// Loads a .pte model and a 16kHz mono WAV file, runs voice activity detection,
// and prints speech segments as "start end speech". See README.md for usage.

#include <iomanip>
#include <iostream>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/runtime/platform/log.h>

#include "silero_vad_runner.h"

DEFINE_string(model_path, "silero_vad.pte", "Path to Silero VAD model (.pte).");
DEFINE_string(audio_path, "", "Path to input audio file (.wav).");
DEFINE_double(threshold, 0.5, "Speech probability threshold (0.0 - 1.0).");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "audio_path flag must be provided.");
    return 1;
  }

  ::executorch::extension::llm::Stats stats;
  stats.model_load_start_ms = ::executorch::extension::llm::time_in_ms();

  silero_vad::SileroVadRunner runner(FLAGS_model_path);

  stats.model_load_end_ms = ::executorch::extension::llm::time_in_ms();
  stats.inference_start_ms = ::executorch::extension::llm::time_in_ms();

  ET_LOG(Info, "Loading audio from: %s", FLAGS_audio_path.c_str());
  auto audio_data =
      ::executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);

  std::cout << std::fixed << std::setprecision(3);

  auto result = runner.detect(
      audio_data.data(),
      static_cast<int64_t>(audio_data.size()),
      static_cast<float>(FLAGS_threshold),
      [](const silero_vad::Segment& seg) {
        std::cout << "  " << seg.start << " " << seg.end << " speech"
                  << std::endl;
      });

  stats.inference_end_ms = ::executorch::extension::llm::time_in_ms();

  // Summary
  double total_duration = result.num_frames * runner.frame_duration();
  double speech_pct =
      100.0 * result.speech_frames / static_cast<double>(result.num_frames);
  std::cout << "\n"
            << result.num_segments << " segments, " << result.num_frames
            << " frames, " << std::setprecision(1) << total_duration << "s\n"
            << "Speech: " << result.speech_frames << "/" << result.num_frames
            << " frames (" << std::setprecision(1) << speech_pct << "%)\n";

  ::executorch::extension::llm::print_report(stats);

  return 0;
}
