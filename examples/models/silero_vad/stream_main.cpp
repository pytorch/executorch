/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Streaming CLI entry point for Silero VAD.
//
// Reads 16kHz mono float32 PCM from stdin, runs the model frame-by-frame, and
// writes a simple line protocol to stdout:
//   READY
//   PROB <time_seconds> <probability>

#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/runtime/platform/log.h>

#include "silero_vad_runner.h"

DEFINE_string(model_path, "silero_vad.pte", "Path to Silero VAD model (.pte).");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  silero_vad::SileroVadRunner runner(FLAGS_model_path);
  runner.reset_stream();

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "READY" << std::endl;

  std::array<char, 4096> read_buf{};
  std::vector<char> pending_bytes;
  std::vector<float> pending_samples;
  pending_bytes.reserve(4096);
  pending_samples.reserve(static_cast<size_t>(runner.window_size() * 2));

  while (true) {
    std::cin.read(read_buf.data(), read_buf.size());
    std::streamsize read_count = std::cin.gcount();
    if (read_count <= 0) {
      break;
    }

    pending_bytes.insert(
        pending_bytes.end(), read_buf.begin(), read_buf.begin() + read_count);

    size_t float_bytes = pending_bytes.size() / sizeof(float) * sizeof(float);
    size_t float_count = float_bytes / sizeof(float);
    size_t prior_sample_count = pending_samples.size();
    pending_samples.resize(prior_sample_count + float_count);

    std::memcpy(
        pending_samples.data() + prior_sample_count,
        pending_bytes.data(),
        float_bytes);
    pending_bytes.erase(pending_bytes.begin(), pending_bytes.begin() + float_bytes);

    while (pending_samples.size() >=
           static_cast<size_t>(runner.window_size())) {
      float prob =
          runner.process_frame(pending_samples.data(), runner.window_size());
      double timestamp =
          static_cast<double>(runner.stream_frame_index()) *
          runner.frame_duration();
      std::cout << "PROB " << timestamp << " " << prob << std::endl;
      pending_samples.erase(
          pending_samples.begin(),
          pending_samples.begin() + runner.window_size());
    }
  }

  return 0;
}
