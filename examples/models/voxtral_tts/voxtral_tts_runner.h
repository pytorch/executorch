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
#include <random>
#include <string>
#include <vector>

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace voxtral_tts {

struct GenerateConfig {
  int max_tokens = 2048;
  float temperature = 0.0f;
  uint64_t seed = 42;
};

// Callback invoked for each generated token (for progress display).
using TokenCallback = std::function<void(const std::string&)>;

class VoxtralTTSRunner {
 public:
  VoxtralTTSRunner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& codec_path = "",
      bool warmup = true);

  // Load voice embedding from a raw binary file (int64 n_frames, int64 dim,
  // then n_frames*dim float32 values). Converts to model dtype internally.
  void load_voice(const std::string& voice_bin_path);

  // Generate audio from text. Returns float32 PCM samples at codec sample rate.
  // If no codec is loaded, returns empty and audio_codes_out is populated.
  std::vector<float> generate(
      const std::string& prompt,
      const GenerateConfig& config,
      TokenCallback token_cb = nullptr,
      std::vector<int64_t>* audio_codes_out = nullptr);

  int64_t max_seq_len() const {
    return max_seq_len_;
  }
  int64_t vocab_size() const {
    return vocab_size_;
  }
  int64_t sample_rate() const {
    return sample_rate_;
  }

 private:
  std::unique_ptr<::executorch::extension::Module> model_;
  std::unique_ptr<::executorch::extension::Module> codec_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

  // Model metadata (from constant_methods)
  int64_t max_seq_len_ = 4096;
  int64_t vocab_size_ = 131072;
  int64_t dim_ = 3072;
  int64_t n_acoustic_codebook_ = 36;
  int64_t n_codebooks_ = 37; // 1 semantic + n_acoustic
  int64_t codec_chunk_size_ = 375;
  int64_t sample_rate_ = 24000;

  // Model dtype
  ::executorch::aten::ScalarType model_dtype_ =
      ::executorch::aten::ScalarType::Float;

  // Tokenizer special tokens
  uint64_t bos_id_ = 1;
  uint64_t eos_id_ = 2;
  uint64_t audio_tok_id_ = 0;

  // Voice embedding (loaded via load_voice())
  std::vector<float> voice_data_; // fp32, (n_voice_frames * dim)
  int64_t n_voice_frames_ = 0;

  // Decode accumulated audio codes to waveform via codec.pte
  std::vector<float> decode_audio(
      const std::vector<int64_t>& codes,
      int64_t n_frames);
};

// Write float32 PCM samples to a WAV file (mono, 16-bit PCM).
bool write_wav(
    const std::string& path,
    const float* samples,
    int64_t num_samples,
    int32_t sample_rate);

} // namespace voxtral_tts
