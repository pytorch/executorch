/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/models/gemma4/runner/gemma4_stats.h>
#include <executorch/examples/models/gemma4/runner/generation_config.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace executorch::examples::gemma4 {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

/**
 * Orchestrates the Gemma 4 E2B/E4B multimodal pipeline using a
 * single PTE with up to 4 named methods:
 *   speech_transform, audio_encoder, vision_encoder, text_decoder
 *
 * Supports audio transcription, image understanding, and text-only generation.
 */
class Gemma4Runner {
 public:
  Gemma4Runner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      bool enable_workspace_sharing = true);

  Error load();
  bool is_loaded() const;

  /**
   * Run the full audio transcription pipeline.
   *
   * @param waveform 1D tensor of raw audio samples (16kHz, float32)
   * @param actual_samples Number of real audio samples before padding
   * @param prompt Transcription prompt
   * @param config Generation configuration
   * @param token_callback Optional callback for streaming tokens
   * @param stats Optional stats for performance tracking
   * @return Generated text
   */
  Result<std::string> generate(
      const TensorPtr& waveform,
      int64_t actual_samples,
      const std::string& prompt,
      const GenerationConfig& config,
      const std::function<void(const std::string&)>& token_callback = {},
      Gemma4Stats* stats = nullptr);

  Result<std::string> generate(
      const TensorPtr& waveform,
      int64_t actual_samples,
      const std::string& prompt,
      int32_t max_new_tokens = 100,
      float temperature = 0.0f,
      const std::function<void(const std::string&)>& token_callback = {},
      Gemma4Stats* stats = nullptr);

  /**
   * Run text-only generation (no audio or vision).
   */
  Result<std::string> generate_text(
      const std::string& prompt,
      const GenerationConfig& config,
      const std::function<void(const std::string&)>& token_callback = {},
      Gemma4Stats* stats = nullptr);

  Result<std::string> generate_text(
      const std::string& prompt,
      int32_t max_new_tokens = 100,
      float temperature = 0.0f,
      const std::function<void(const std::string&)>& token_callback = {},
      Gemma4Stats* stats = nullptr);

  /**
   * Run the vision understanding pipeline.
   *
   * @param pixel_values Pre-patchified image [1, num_patches, 768]
   * @param pixel_position_ids 2D patch positions [1, num_patches, 2]
   * @param prompt Query prompt for the image
   * @param config Generation configuration
   * @param token_callback Optional callback for streaming tokens
   * @param stats Optional stats for performance tracking
   * @return Generated text
   */
  Result<std::string> generate_vision(
      const TensorPtr& pixel_values,
      const TensorPtr& pixel_position_ids,
      const std::string& prompt,
      const GenerationConfig& config,
      const std::function<void(const std::string&)>& token_callback = {},
      Gemma4Stats* stats = nullptr);

  Result<std::string> generate_vision(
      const TensorPtr& pixel_values,
      const TensorPtr& pixel_position_ids,
      const std::string& prompt,
      int32_t max_new_tokens = 100,
      float temperature = 0.0f,
      const std::function<void(const std::string&)>& token_callback = {},
      Gemma4Stats* stats = nullptr);

  /**
   * Reset model state for next inference.
   * Reloads the text_decoder method to clear KV cache.
   */
  void reset();

 private:
  static int64_t round_to_valid_frames(int64_t num_frames);
  static int64_t compute_audio_num_tokens(int64_t num_samples);
  static float* get_last_logits_as_float(
      const Tensor& logits,
      std::vector<float>& buf,
      int32_t vocab_size);

  std::vector<int64_t> build_input_ids(
      const std::string& prompt,
      int64_t num_audio_tokens);

  std::vector<int64_t> build_text_input_ids(const std::string& prompt);

  std::vector<int64_t> build_vision_input_ids(
      const std::string& prompt,
      int64_t num_vision_tokens);

  TensorPtr build_inputs_embeds(
      const std::vector<int64_t>& input_ids,
      const Tensor& media_embeddings,
      int64_t num_media_tokens,
      int64_t placeholder_token_id);

  Result<std::string> decode_loop(
      const Tensor& prefill_logits,
      int64_t seq_len,
      const GenerationConfig& config,
      const std::function<void(const std::string&)>& token_callback,
      Gemma4Stats* stats);

  static constexpr int64_t kBosId = 2;
  static constexpr int64_t kEosId = 1;
  static constexpr int64_t kTurnStartId = 105;
  static constexpr int64_t kTurnEndId = 106;
  static constexpr int64_t kAudioTokenId = 258881;
  static constexpr int64_t kImageTokenId = 258880;
  static constexpr int64_t kBoiTokenId = 255999;
  static constexpr int64_t kEoiTokenId = 258882;
  static constexpr int64_t kDefaultHiddenSizeE2B = 1536;
  static constexpr int64_t kDefaultHiddenSizeE4B = 2560;
  static constexpr int64_t kMaxSamples = 480000;
  static constexpr int32_t kSampleRate = 16000;
  static constexpr int64_t kFrameLength =
      320; // sample_rate * frame_length_ms / 1000
  static constexpr int64_t kHopLength =
      160; // sample_rate * hop_length_ms / 1000
  static constexpr int64_t kMaxAudioTokens =
      750; // audio_seq_length (30s * 1000 / 40ms)

  std::unique_ptr<Module> module_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::string model_path_;
  std::string tokenizer_path_;
  bool enable_workspace_sharing_;
  Error load_audio_methods();
  Error load_vision_methods();

  ScalarType embeds_dtype_ = ScalarType::Float;
  int64_t hidden_size_ = kDefaultHiddenSizeE2B;
  bool loaded_ = false;
  bool audio_loaded_ = false;
  bool vision_loaded_ = false;
};

} // namespace executorch::examples::gemma4
