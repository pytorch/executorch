/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <filesystem>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace voxtral_tts {

using AudioChunkCallback =
    std::function<void(const float* samples, std::size_t count)>;

class VoxtralTTSRunner {
 public:
  VoxtralTTSRunner(
      const std::string& model_path,
      const std::string& codec_path,
      const std::string& tokenizer_path);

  void set_trace_output_path(const std::string& trace_output_path);
  void set_seed(uint32_t seed);

  void synthesize_offline(
      const std::string& text,
      const std::string& voice_path,
      const std::string& output_path,
      float temperature = 0.0f,
      int max_new_tokens = 2048);

  void synthesize_streaming(
      const std::string& text,
      const std::string& voice_path,
      const std::string& output_path,
      AudioChunkCallback callback = nullptr,
      float temperature = 0.0f,
      int max_new_tokens = 2048);

 private:
  void load_metadata();
  void reload_stateful_model();
  void warmup();

  std::vector<int64_t> tokenize(const std::string& text);

  void decode_codes_to_wav(
      const std::vector<std::vector<int64_t>>& frame_codes,
      const std::string& output_path,
      std::vector<float>* out_samples = nullptr);

  void decode_code_window(
      const std::vector<std::vector<int64_t>>& frame_codes,
      int64_t start_frame,
      int64_t end_frame,
      std::vector<float>& out_samples);

  void build_prompt(
      const std::string& text,
      std::vector<int64_t>& token_ids,
      int& voice_start,
      int& voice_len);

  std::filesystem::path resolve_voice_path(const std::string& voice_path) const;
  void load_voice_embedding(const std::string& voice_path);

  int64_t sample_semantic_code(
      const float* logits,
      int64_t vocab_size,
      float temperature);

  std::unique_ptr<::executorch::extension::Module> model_;
  std::unique_ptr<::executorch::extension::Module> codec_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::mt19937 rng_;
  uint32_t seed_ = 42;

  // Voice embedding loaded from .pt or raw .bin assets.
  std::vector<float> voice_embed_data_;

  // Config from metadata
  int64_t sample_rate_ = 24000;
  int64_t n_decoding_steps_ = 7;
  float cfg_alpha_ = 1.2f;
  int64_t n_acoustic_codebook_ = 36;
  int64_t acoustic_levels_ = 21;
  int64_t n_special_tokens_ = 2;
  int64_t vocab_size_ = 131072;
  int64_t max_seq_len_ = 4096;
  int64_t dim_ = 3072;
  int64_t downsample_factor_ = 1920;
  int64_t n_codebooks_ = 37;
  int64_t end_audio_code_ = 1;
  int64_t empty_audio_code_ = 0;
  int64_t max_codec_frames_ = 256;
  bool codec_supports_exact_frames_ = false;
  int64_t audio_token_id_ = 24;
  int64_t begin_audio_token_id_ = 25;
  int64_t text_to_audio_token_id_ = 36;
  int64_t repeat_audio_text_token_id_ = 35;
  int64_t voice_embed_len_ = 147;
  int64_t runtime_voice_embed_len_ = 0;

  // Streaming
  bool is_streaming_ = false;
  int64_t streaming_chunk_frames_ = 25;
  int64_t streaming_initial_chunk_ = 5;
  int64_t streaming_left_context_ = 25;

  std::string trace_output_path_;
  std::filesystem::path asset_root_dir_;
  std::string model_path_;
};

} // namespace voxtral_tts
