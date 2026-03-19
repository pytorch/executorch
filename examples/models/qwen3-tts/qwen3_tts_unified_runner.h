/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>

namespace qwen3_tts {

struct SynthesizeConfig {
  int max_new_tokens = 200;
  float temperature = 1.0f;
  int top_k = -1;
  float top_p = -1.0f;
  float repetition_penalty = -1.0f;
};

typedef void (*audio_callback_t)(
    const float* samples,
    int64_t num_samples,
    void* user_data);

class Qwen3TTSUnifiedRunner {
 public:
  Qwen3TTSUnifiedRunner(
      const std::string& model_path,
      const std::string& tokenizer_path);

  int output_sample_rate() const { return output_sample_rate_; }
  int max_seq_len() const { return max_seq_len_; }
  int num_code_groups() const { return num_code_groups_; }
  bool is_loaded() const { return module_ != nullptr; }

  // Full text-to-audio pipeline.
  bool synthesize(
      const std::string& text,
      const std::string& language,
      const SynthesizeConfig& config,
      std::vector<float>* waveform);

  // Decode precomputed codes (backward compat).
  bool decode_codes_file(
      const std::string& codes_path,
      std::vector<float>* waveform);

  // Pre-load and warm up decode_audio method (XNNPACK init).
  void warmup_decode();

  bool write_wav_file(
      const std::string& output_wav_path,
      const std::vector<float>& waveform) const;

 private:
  // Pipeline stages.
  bool run_encode_text(
      const std::vector<int64_t>& token_ids,
      std::vector<float>* projected);

  bool run_talker(
      const std::vector<float>& embeds,
      int32_t seq_len,
      const std::vector<int64_t>& input_pos,
      std::vector<float>* logits,
      std::vector<float>* hidden);

  bool run_codec_embed(
      int64_t token_id,
      int64_t group_idx,
      std::vector<float>* embed);

  bool run_code_predictor(
      const std::vector<float>& embeds,
      int32_t seq_len,
      const std::vector<int64_t>& input_pos,
      std::vector<float>* hidden);

  bool run_cp_head(
      const std::vector<float>& hidden,
      int64_t head_idx,
      std::vector<float>* logits);

  bool run_decode_audio(
      const std::vector<int64_t>& codes,
      int32_t codes_len,
      int32_t num_quantizers,
      std::vector<float>* waveform);

  bool read_codes_file(
      const std::string& codes_path,
      std::vector<int64_t>* codes,
      int32_t* codes_len,
      int32_t* num_quantizers) const;

  int64_t sample_token(
      const std::vector<float>& logits,
      int vocab_size,
      float temperature,
      int top_k);

  void load_metadata();
  void load_methods();
  bool ensure_method(const std::string& method_name);

  std::unique_ptr<::executorch::extension::Module> module_;

  int output_sample_rate_ = 24000;
  int max_seq_len_ = 256;
  int talker_vocab_size_ = 3072;
  int talker_dim_ = 1024;
  int num_code_groups_ = 16;
  int num_quantizers_ = 16;
  int codebook_size_ = 2048;
};

} // namespace qwen3_tts
