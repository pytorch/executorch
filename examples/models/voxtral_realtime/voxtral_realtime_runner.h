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
#include <string>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace voxtral_realtime {

// Custom runner rather than MultimodalRunner because Voxtral Realtime sums
// audio and text embeddings at each position (element-wise add), while
// MultimodalRunner concatenates modality segments sequentially.

struct TranscribeConfig {
  int max_new_tokens = 500;
  float temperature = 0.0f; // 0 = greedy
};

using TokenCallback = std::function<void(const std::string&)>;

class VoxtralRealtimeRunner {
 public:
  VoxtralRealtimeRunner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& preprocessor_path = "");

  // Transcribe audio. Returns the number of generated tokens.
  int transcribe(
      const float* audio_data,
      int64_t num_samples,
      const TranscribeConfig& config,
      TokenCallback token_cb);

  int64_t max_seq_len() const {
    return max_seq_len_;
  }
  int64_t vocab_size() const {
    return vocab_size_;
  }

 private:
  std::unique_ptr<::executorch::extension::Module> model_;
  std::unique_ptr<::executorch::extension::Module> preprocessor_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

  // From model metadata (constant_methods)
  int64_t max_seq_len_ = 4096;
  int64_t vocab_size_ = 131072;
  int64_t dim_ = 3072;

  // Tokenizer special tokens
  uint64_t bos_id_ = 1;
  uint64_t eos_id_ = 2;

  // Run preprocessor.pte on raw audio -> mel spectrogram tensor.
  ::executorch::extension::TensorPtr run_preprocessor(
      const float* audio,
      int64_t num_samples);
};

} // namespace voxtral_realtime
