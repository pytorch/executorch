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

#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace qwen3_tts {

using AudioChunkCallback =
    std::function<void(const std::vector<float>& chunk, bool is_final)>;

struct SynthesizeConfig {
  int max_new_tokens = 200;
  float temperature = 0.9f;
  int top_k = 50;
  float top_p = 1.0f;
  float repetition_penalty = 1.05f;
  uint64_t seed = 0;
  bool use_fused_cp_generate = true;
  std::string instruct;
  bool non_streaming_mode = false;
  float streaming_interval_sec = 2.0f;
  int streaming_chunk_steps = 0;
  int streaming_chunk_size = 300;
  int streaming_left_context_size = 25;
  bool disable_streaming_decoder_surface = false;
  bool force_streaming_decoder_surface = false;
  bool use_legacy_cumulative_streaming_decode = false;
};

struct SynthesisTiming {
  int prompt_token_count = 0;
  int generated_codec_steps = 0;
  int text_tokens_consumed = 0;
  double prompt_prep_ms = 0.0;
  double talker_prefill_ms = 0.0;
  double codegen_ms = 0.0;
  double first_audio_ms = 0.0;
  double chunk_decode_ms = 0.0;
  double final_decode_ms = 0.0;
  double decode_audio_ms = 0.0;
  double total_generation_ms = 0.0;
};

class SynthesisSession;

class Qwen3TTSUnifiedRunner {
 public:
  Qwen3TTSUnifiedRunner(
      const std::string& model_path,
      const std::string& tokenizer_path);

  int output_sample_rate() const { return output_sample_rate_; }
  int max_seq_len() const { return max_seq_len_; }
  int num_code_groups() const { return num_code_groups_; }
  bool is_loaded() const { return module_ != nullptr; }
  bool has_tokenizer() const { return tokenizer_ != nullptr; }

  // Full text-to-audio pipeline.
  bool synthesize(
      const std::string& text,
      const std::string& language,
      const SynthesizeConfig& config,
      std::vector<float>* waveform);

  bool synthesize(
      const std::string& text,
      const std::string& language,
      const SynthesizeConfig& config,
      std::vector<float>* waveform,
      SynthesisTiming* timing);

  std::unique_ptr<SynthesisSession> create_synthesis_session(
      const SynthesizeConfig& config);

  // Decode precomputed codes (backward compat).
  bool decode_codes_file(
      const std::string& codes_path,
      std::vector<float>* waveform);

  // Pre-load and warm up methods.
  void warmup_decode();
  void warmup_all();

  bool write_wav_file(
      const std::string& output_wav_path,
      const std::vector<float>& waveform) const;

 private:
  friend class SynthesisSession;

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

  // Fused 15-step code predictor (replaces 15x code_predictor + cp_head calls).
  bool run_cp_generate(
      const std::vector<float>& talker_hidden,
      const std::vector<float>& code_0_embed,
      float temperature,
      const std::vector<float>& sample_uniforms,
      std::vector<int64_t>* sampled_subcodes,
      std::vector<float>* embed_sum);

  bool run_decode_audio(
      const std::vector<int64_t>& codes,
      int32_t codes_len,
      int32_t num_quantizers,
      std::vector<float>* waveform);

  bool run_decode_audio_stream(
      const std::vector<int64_t>& padded_codes,
      int32_t padded_codes_len,
      int32_t num_quantizers,
      std::vector<float>* waveform);

  bool read_codes_file(
      const std::string& codes_path,
      std::vector<int64_t>* codes,
      int32_t* codes_len,
      int32_t* num_quantizers) const;

  // Embedding helpers for synthesize().
  bool get_text_embed(int64_t token_id, std::vector<float>* embed);
  void vec_add(std::vector<float>& dst, const std::vector<float>& src);
  void vec_zero(std::vector<float>& v);

  int64_t sample_token(
      const std::vector<float>& logits,
      int vocab_size,
      float temperature,
      int top_k,
      float top_p,
      std::mt19937* gen);

  int64_t sample_token(
      const std::vector<float>& logits,
      int vocab_size,
      float temperature,
      int top_k,
      float top_p,
      float repetition_penalty,
      const std::vector<int64_t>* generated_tokens,
      const std::vector<int64_t>* suppress_tokens,
      int64_t eos_token_id,
      std::mt19937* gen);

  void load_metadata();
  void load_methods();
  bool ensure_method(const std::string& method_name);
  bool has_streaming_decode_method();
  int effective_streaming_interval_steps(const SynthesizeConfig& config) const;
  bool decode_code_step_range(
      const std::vector<std::vector<int64_t>>& all_codes,
      int start_step,
      int end_step,
      int left_context_steps,
      bool allow_streaming_surface,
      std::vector<float>* waveform);
  bool decode_codes_chunked(
      const std::vector<std::vector<int64_t>>& all_codes,
      int chunk_size_steps,
      int left_context_steps,
      bool allow_streaming_surface,
      std::vector<float>* waveform,
      double* decode_ms,
      double* first_audio_ms);

  std::unique_ptr<::executorch::extension::Module> module_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

  // Model metadata (from constant_methods).
  int output_sample_rate_ = 24000;
  int decode_upsample_rate_ = 1920;
  int max_seq_len_ = 256;
  int talker_vocab_size_ = 3072;
  int talker_dim_ = 1024;
  int num_code_groups_ = 16;
  int num_quantizers_ = 16;
  int codebook_size_ = 2048;
  int text_prompt_min_token_count_ = 9;
  int text_prompt_prefill_token_count_ = 8;
  int text_prompt_prefill_token_count_with_language_ = 9;
  int text_prompt_trailing_template_token_count_ = 5;
  int cp_generate_contract_version_ = 1;
  int cp_generate_fast_top_k_ = 50;
  int generation_backend_code_ = 0;
  int decoder_backend_code_ = 0;
  int streaming_decoder_contract_version_ = 0;
  int streaming_decoder_chunk_size_ = 0;
  int streaming_decoder_left_context_size_ = 0;
  int streaming_decoder_max_codes_ = 0;
  int prefer_streaming_decoder_surface_ = 1;
  bool checked_streaming_decode_method_ = false;
  bool has_streaming_decode_method_ = false;

  // Special token IDs.
  int64_t tts_pad_id_ = 151671;
  int64_t tts_bos_id_ = 151672;
  int64_t tts_eod_id_ = 151673;
  int64_t codec_pad_id_ = 2148;
  int64_t codec_bos_id_ = 2149;
  int64_t codec_eos_id_ = 2150;
  int64_t codec_think_id_ = 2154;
  int64_t codec_language_english_id_ = 2050;
  int64_t codec_nothink_id_ = 2155;
  int64_t codec_think_bos_id_ = 2156;
  int64_t codec_think_eos_id_ = 2157;
  int64_t im_start_id_ = 151644;
  int64_t assistant_id_ = 77091;
  int64_t newline_id_ = 198;
};

class SynthesisSession {
 public:
  bool synthesize(
      const std::string& text,
      const std::string& language,
      std::vector<float>* waveform,
      SynthesisTiming* timing = nullptr);

  bool synthesize(
      const std::string& text,
      const std::string& language,
      std::vector<float>* waveform,
      SynthesisTiming* timing,
      AudioChunkCallback on_audio_chunk);

 private:
  friend class Qwen3TTSUnifiedRunner;
  SynthesisSession(
      Qwen3TTSUnifiedRunner* runner,
      const SynthesizeConfig& config);

  bool synthesize_impl(
      const std::string& text,
      const std::string& language,
      std::vector<float>* waveform,
      SynthesisTiming* timing,
      AudioChunkCallback on_audio_chunk);

  Qwen3TTSUnifiedRunner* runner_;
  SynthesizeConfig config_;
  std::mt19937 rng_;
};

} // namespace qwen3_tts
