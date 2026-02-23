/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Streaming Sortformer diarizer runner.
//
// Runs a three-stage pipeline exported from NeMo's SortformerEncLabelModel:
//   1. preprocessor:  raw audio → mel spectrogram (128 bins, 10ms stride)
//   2. pre_encode:    mel → 512-dim embeddings (8x convolutional downsampling)
//   3. encode:        embeddings → per-frame speaker probabilities (up to 4
//   spks)
//
// The .pte model is stateless — all streaming state (speaker cache, FIFO) is
// managed here. Each output frame covers 80ms of audio (10ms stride × 8x
// subsampling). See model.md for architecture details.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>

namespace sortformer {

// Streaming chunk/FIFO sizes in 80ms frames. Defaults match the "High" config.
// See README.md § Streaming Configurations for all presets.
struct StreamingConfig {
  int64_t chunk_len = 124;
  int64_t fifo_len = 124;
};

// A diarization output segment: [start, end) in seconds for a speaker slot.
// Speaker indices are fixed output slots (0–3), not cluster IDs — identity is
// maintained across chunks because the speaker cache feeds
// speaker-discriminative embeddings from earlier audio into each encode call,
// giving the model context to assign the same slot to the same physical
// speaker.
struct Segment {
  double start;
  double end;
  int speaker;
};

using SegmentCallback = std::function<void(const Segment&)>;

class SortformerRunner {
 public:
  explicit SortformerRunner(const std::string& model_path);

  struct Result {
    int64_t num_frames;
    int num_segments;
    std::vector<int64_t> speaker_active_frames;
  };

  Result diarize(
      const float* audio_data,
      int64_t num_samples,
      float threshold,
      const StreamingConfig& config,
      SegmentCallback segment_cb);

  double frame_duration() const {
    return frame_duration_;
  }
  int64_t max_spks() const {
    return max_spks_;
  }

 private:
  std::unique_ptr<::executorch::extension::Module> model_;

  // Model parameters. window_stride_, subsampling_factor_, spkcache_len_, and
  // max_spks_ are read from .pte constant_methods at load time. d_model_ is
  // inferred from the first pre_encode output. frame_duration_ is computed.
  double window_stride_;
  int64_t subsampling_factor_;
  int64_t spkcache_len_; // Max speaker cache frames (188)
  int64_t max_spks_; // Fixed speaker output slots (4)
  int64_t d_model_ = 0; // Conformer hidden dim (512), set on first pre_encode
  double frame_duration_; // window_stride * subsampling_factor = 80ms

  // Stage 1: audio → mel spectrogram.
  // Output is transposed from the model's (1, 128, T) channels-first to
  // (T, 128) channels-last, which is what pre_encode expects.
  std::pair<std::vector<float>, int64_t> run_preprocessor(
      const float* audio,
      int64_t num_samples);

  // Stage 2: mel → 512-dim embeddings via 8x convolutional downsampling.
  // Input is padded to 4000 mel frames (static shape required by the exported
  // model — see model.md § torch.export Fixes #5).
  int64_t run_pre_encode(
      const float* mel_transposed,
      int64_t valid_mel_len,
      std::vector<float>& all_embs);

  // Stage 3: streaming encode with cache/FIFO state management.
  // Processes embeddings in chunks, concatenating [cache | FIFO | chunk] for
  // each encode call. Only the current chunk's predictions are kept as output.
  // The speaker cache provides long-term context (most speaker-discriminative
  // frames), the FIFO provides short-term sliding window context.
  Result run_streaming_encode(
      const std::vector<float>& all_embs,
      int64_t total_emb_len,
      float threshold,
      const StreamingConfig& config,
      SegmentCallback segment_cb);
};

} // namespace sortformer
