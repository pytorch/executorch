/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "runner.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <gflags/gflags.h>
#include <nlohmann/json.hpp>

#ifdef EXECUTORCH_BUILD_MLX
#include <mlx/memory.h>
#endif

#include <executorch/extension/llm/runner/wav_loader.h>

DEFINE_string(
    model_path,
    "nemotron_exports/nemotron3_diarization.pte",
    "Exported model");
DEFINE_string(data_path, "", "External CUDA data file (aoti_cuda_blob.ptd)");
DEFINE_string(audio_path, "", "Mono 16 kHz WAV file (PCM16 or float32)");
DEFINE_string(preset, "offline", "offline, low, very_low, or ultra_low");
DEFINE_int32(feed_samples, 4096, "Samples per streaming feed");
DEFINE_double(threshold, 0.5, "Speaker activity threshold");
#ifdef EXECUTORCH_BUILD_MLX
DEFINE_int32(mlx_cache_limit_mb, 128, "Maximum unused MLX buffer cache in MiB");
DEFINE_int32(mlx_memory_limit_mb, 4096, "MLX working memory guideline in MiB");
#endif
DEFINE_string(output, "", "Optional JSON output file");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    if (FLAGS_audio_path.empty() || FLAGS_feed_samples <= 0 ||
        !std::isfinite(FLAGS_threshold) || FLAGS_threshold < 0 ||
        FLAGS_threshold > 1) {
      throw std::invalid_argument(
          "Provide --audio_path, positive --feed_samples, and --threshold in [0,1]");
    }
#ifdef EXECUTORCH_BUILD_MLX
    if (FLAGS_mlx_cache_limit_mb < 0 || FLAGS_mlx_memory_limit_mb <= 0) {
      throw std::invalid_argument(
          "Provide nonnegative --mlx_cache_limit_mb and positive --mlx_memory_limit_mb");
    }
    mlx::core::set_cache_limit(
        static_cast<size_t>(FLAGS_mlx_cache_limit_mb) << 20);
    mlx::core::set_memory_limit(
        static_cast<size_t>(FLAGS_mlx_memory_limit_mb) << 20);
#endif
    nemotron3::Runner runner(
        FLAGS_model_path,
        nemotron3::StreamingConfig::from_preset(FLAGS_preset),
        FLAGS_data_path);
    auto header = executorch::extension::llm::load_wav_header(FLAGS_audio_path);
    if (!header || header->NumOfChan != 1 ||
        header->SamplesPerSec != runner.sample_rate()) {
      throw std::invalid_argument("Audio must be a mono 16 kHz WAV file");
    }
    auto audio =
        executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
    if (audio.empty()) {
      throw std::invalid_argument(
          "WAV file contains no supported audio samples");
    }
    std::vector<float> probabilities;
    for (size_t offset = 0; offset < audio.size();) {
      const size_t count =
          std::min<size_t>(FLAGS_feed_samples, audio.size() - offset);
      auto next = runner.feed(
          audio.data() + offset, count, offset + count == audio.size());
      probabilities.insert(probabilities.end(), next.begin(), next.end());
      offset += count;
    }
    const int64_t speakers = runner.num_speakers();
    const int64_t frames = probabilities.size() / speakers;
    const double dt = runner.frame_duration();
    struct Segment {
      int64_t start, end, speaker;
    };
    std::vector<Segment> segments;
    for (int64_t speaker = 0; speaker < speakers; ++speaker) {
      int64_t begin = -1;
      for (int64_t t = 0; t <= frames; ++t) {
        const bool active = t < frames &&
            probabilities[t * speakers + speaker] > FLAGS_threshold;
        if (active && begin < 0) {
          begin = t;
        } else if (!active && begin >= 0) {
          segments.push_back({begin, t, speaker});
          begin = -1;
        }
      }
    }
    std::sort(
        segments.begin(), segments.end(), [](const auto& a, const auto& b) {
          return a.start == b.start ? a.speaker < b.speaker : a.start < b.start;
        });
    nlohmann::json result = {
        {"preset", FLAGS_preset},
        {"num_frames", frames},
        {"num_speakers", speakers},
        {"frame_duration", dt},
        {"audio_seconds",
         static_cast<double>(audio.size()) / runner.sample_rate()},
        {"segments", nlohmann::json::array()}};
    for (const auto& segment : segments) {
      result["segments"].push_back(
          {{"start", segment.start * dt},
           {"end", segment.end * dt},
           {"speaker", segment.speaker}});
    }
    if (!FLAGS_output.empty()) {
      std::ofstream output(FLAGS_output);
      output << result.dump(2) << '\n';
      if (!output) {
        throw std::runtime_error("Could not write JSON output");
      }
    }
    std::cout << result.dump(2) << '\n';
  } catch (const std::exception& error) {
    std::cerr << "Nemotron 3 Diarization: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
