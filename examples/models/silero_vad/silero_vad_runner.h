/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Silero VAD runner.
//
// Runs the Silero VAD 16kHz model exported as a single-method .pte:
//   forward: (1, 576) audio + (2, 1, 128) LSTM state → (1, 1) probability +
//   (2, 1, 128) new state
//
// Audio is processed in 512-sample chunks (32ms at 16kHz). Each chunk is
// prepended with 64 samples of context from the previous chunk. The model
// outputs a single speech probability per chunk. See export_silero_vad.py
// for architecture details.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>

namespace silero_vad {

struct Segment {
  double start;
  double end;
};

using SegmentCallback = std::function<void(const Segment&)>;

class SileroVadRunner {
 public:
  explicit SileroVadRunner(const std::string& model_path);

  struct Result {
    int64_t num_frames;
    int num_segments;
    int64_t speech_frames;
  };

  Result detect(
      const float* audio_data,
      int64_t num_samples,
      float threshold,
      SegmentCallback segment_cb);

  double frame_duration() const {
    return frame_duration_;
  }

 private:
  std::unique_ptr<::executorch::extension::Module> model_;

  int64_t sample_rate_;
  int64_t window_size_;
  int64_t context_size_;
  int64_t input_size_;
  double frame_duration_;

  static constexpr int64_t kHiddenDim = 128;
};

} // namespace silero_vad
