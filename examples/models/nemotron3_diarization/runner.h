/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>

namespace nemotron3 {

struct StreamingConfig {
  int64_t chunk = 340;
  int64_t right_context = 40;
  int64_t fifo = 40;
  int64_t update_period = 300;

  static StreamingConfig from_preset(const std::string& preset);
};

class Runner {
 public:
  explicit Runner(const std::string& model_path, StreamingConfig config = {});

  // Returns new 10 ms frames, with num_speakers() probabilities per frame.
  // Audio is mono at sample_rate(). final=true flushes the remaining lookahead.
  // Reset after an execution failure before starting another stream.
  std::vector<float> feed(const float* audio, size_t count, bool final = false);
  void reset();

  int64_t sample_rate() const {
    return sample_rate_;
  }
  int64_t num_speakers() const {
    return num_speakers_;
  }
  int64_t frames_processed() const {
    return frames_processed_;
  }
  double frame_duration() const {
    return static_cast<double>(hop_) / sample_rate_;
  }

 private:
  std::vector<float> features(int64_t count) const;
  std::vector<float>
  step(int64_t feature_frames, int64_t valid, int64_t central);
  void compress_cache();

  executorch::extension::Module model_;
  StreamingConfig config_;
  int64_t sample_rate_, hop_, n_fft_, num_mels_, d_model_, num_speakers_,
      factor_;
  int64_t pad_to_, cache_capacity_, silence_frames_, max_encoder_,
      max_features_;
  float preemphasis_, score_threshold_, latest_boost_, strong_boost_,
      weak_boost_;
  float min_positive_;
  std::vector<float> window_, mel_filters_, silence_;

  std::vector<float> audio_, cache_, cache_probs_, fifo_;
  int64_t samples_received_ = 0;
  int64_t sample_offset_ = 0;
  int64_t frames_processed_ = 0;
  bool compressed_ = false;
  bool finished_ = false;
};

} // namespace nemotron3
