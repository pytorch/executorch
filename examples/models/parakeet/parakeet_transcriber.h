/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

#include "timestamp_utils.h"
#include "tokenizer_utils.h"
#include "types.h"

namespace parakeet {

struct TimestampOutputMode {
  bool token = false;
  bool word = false;
  bool segment = false;

  bool enabled() const {
    return token || word || segment;
  }
};

TimestampOutputMode parse_timestamp_output_mode(const std::string& raw_arg);

struct TranscribeConfig {
  TimestampOutputMode timestamp_output_mode;
  bool runtime_profile = false;
};

struct TranscribeResult {
  std::string text;
  std::string stats_json;
  std::optional<std::string> runtime_profile_report;
  double frame_to_seconds = 0.0;
  std::vector<TokenWithTextInfo> token_offsets;
  std::vector<TextWithOffsets> word_offsets;
  std::vector<TextWithOffsets> segment_offsets;
};

using StatusCallback = std::function<void(const std::string&)>;

class ParakeetTranscriber {
 public:
  ParakeetTranscriber(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& data_path = "");

  TranscribeResult transcribe_audio(
      const float* audio_data,
      int64_t num_samples,
      const TranscribeConfig& config,
      StatusCallback status_callback = {});

  TranscribeResult transcribe_wav_path(
      const std::string& audio_path,
      const TranscribeConfig& config,
      StatusCallback status_callback = {});

 private:
  std::unique_ptr<::executorch::extension::Module> model_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

  int64_t vocab_size_ = 0;
  int64_t blank_id_ = 0;
  int64_t num_rnn_layers_ = 0;
  int64_t pred_hidden_ = 0;
  int64_t sample_rate_ = 0;
  double window_stride_ = 0.0;
  int64_t encoder_subsampling_factor_ = 0;
  double frame_to_seconds_ = 0.0;

  long model_load_start_ms_ = 0;
  long model_load_end_ms_ = 0;

  std::unordered_set<std::string> supported_punctuation_;
};

std::optional<std::string> extract_runtime_profile_line(
    const std::optional<std::string>& report);

} // namespace parakeet
