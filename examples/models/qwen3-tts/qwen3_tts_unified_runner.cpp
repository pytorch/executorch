/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generated with assistance from Claude.

#include "qwen3_tts_unified_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

namespace qwen3_tts {
namespace {

using ::executorch::extension::from_blob;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

constexpr int kAssistantRoleTokenCount = 3;
constexpr int kFirstTextTokenCount = 1;
constexpr int kTextOnlyCodecPrefixTokenCount = 5;
constexpr int kTextOnlyCombinedPrefixTokenCount =
    kTextOnlyCodecPrefixTokenCount - 1;

template <typename T>
float to_float(T value) {
  return static_cast<float>(value);
}

template <>
float to_float<::executorch::aten::Half>(::executorch::aten::Half value) {
  return static_cast<float>(value);
}

template <>
float to_float<::executorch::aten::BFloat16>(
    ::executorch::aten::BFloat16 value) {
  return static_cast<float>(value);
}

void extract_float_tensor(
    const ::executorch::aten::Tensor& tensor,
    std::vector<float>* out) {
  int64_t numel = tensor.numel();
  out->resize(static_cast<size_t>(numel));

  if (tensor.scalar_type() == ::executorch::aten::ScalarType::Float) {
    const float* src = tensor.const_data_ptr<float>();
    std::copy(src, src + numel, out->begin());
  } else if (tensor.scalar_type() == ::executorch::aten::ScalarType::Half) {
    const auto* src = tensor.const_data_ptr<::executorch::aten::Half>();
    for (int64_t i = 0; i < numel; ++i) {
      (*out)[i] = to_float(src[i]);
    }
  } else if (
      tensor.scalar_type() == ::executorch::aten::ScalarType::BFloat16) {
    const auto* src = tensor.const_data_ptr<::executorch::aten::BFloat16>();
    for (int64_t i = 0; i < numel; ++i) {
      (*out)[i] = to_float(src[i]);
    }
  }
}

const char* backend_code_name(int code) {
  switch (code) {
    case 1:
      return "xnnpack";
    case 2:
      return "metal";
    default:
      return "portable";
  }
}

std::string build_assistant_prompt_text(const std::string& text) {
  return std::string("<|im_start|>assistant\n") + text +
      "<|im_end|>\n<|im_start|>assistant\n";
}

std::string build_instruct_prefix(const std::string& instruct) {
  return std::string("<|im_start|>user\n") + instruct + "<|im_end|>\n";
}

void copy_token_slice(
    const std::vector<float>& flat_embeds,
    int token_start,
    int token_count,
    int dim,
    std::vector<float>* out) {
  const size_t start = static_cast<size_t>(token_start) * dim;
  const size_t end = start + static_cast<size_t>(token_count) * dim;
  out->assign(flat_embeds.begin() + start, flat_embeds.begin() + end);
}

void extract_last_token_slice(
    const std::vector<float>& flat_values,
    int token_count,
    int stride,
    std::vector<float>* out) {
  const size_t start =
      static_cast<size_t>(token_count - 1) * static_cast<size_t>(stride);
  out->assign(
      flat_values.begin() + start,
      flat_values.begin() + start + static_cast<size_t>(stride));
}

struct PreparedPromptState {
  int prompt_token_count = 0;
  int prefill_len = 0;
  int trailing_prompt_token_count = 0;
  std::vector<float> prefill_embeds;
  std::vector<std::vector<float>> trailing_text_embeds;
  std::vector<float> tts_pad_embed;
};

} // namespace

SynthesisSession::SynthesisSession(
    Qwen3TTSUnifiedRunner* runner,
    const SynthesizeConfig& config)
    : runner_(runner),
      config_(config),
      rng_(config.seed == 0 ? std::random_device{}() : config.seed) {}

Qwen3TTSUnifiedRunner::Qwen3TTSUnifiedRunner(
    const std::string& model_path,
    const std::string& tokenizer_path) {
  ET_LOG(Info, "Loading unified model from: %s", model_path.c_str());
  module_ = std::make_unique<::executorch::extension::Module>(
      model_path, ::executorch::extension::Module::LoadMode::Mmap);

  auto load_error = module_->load();
  ET_CHECK_MSG(
      load_error == Error::Ok, "Failed to load qwen3-tts unified model.");

  load_metadata();
  load_methods();

  if (!tokenizer_path.empty()) {
    ET_LOG(Info, "Loading tokenizer from: %s", tokenizer_path.c_str());
    tokenizer_ =
        ::executorch::extension::llm::load_tokenizer(tokenizer_path);
    if (tokenizer_ == nullptr) {
      ET_LOG(Error, "Failed to load tokenizer: %s", tokenizer_path.c_str());
    }
  }

  ET_LOG(
      Info,
      "Unified runner: sample_rate=%d max_seq_len=%d talker_dim=%d "
      "num_code_groups=%d text_prompt_prefill=%d tokenizer=%s "
      "generation_backend=%s decoder_backend=%s prefer_stream_surface=%s",
      output_sample_rate_,
      max_seq_len_,
      talker_dim_,
      num_code_groups_,
      text_prompt_prefill_token_count_,
      tokenizer_ ? "loaded" : "none",
      backend_code_name(generation_backend_code_),
      backend_code_name(decoder_backend_code_),
      prefer_streaming_decoder_surface_ > 0 ? "true" : "false");
}

std::unique_ptr<SynthesisSession>
Qwen3TTSUnifiedRunner::create_synthesis_session(
    const SynthesizeConfig& config) {
  return std::unique_ptr<SynthesisSession>(
      new SynthesisSession(this, config));
}

void Qwen3TTSUnifiedRunner::load_metadata() {
  std::vector<EValue> empty;
  auto try_int = [&](const char* name, int* out) {
    auto result = module_->execute(name, empty);
    if (result.ok()) {
      *out = static_cast<int>(result.get()[0].toInt());
    }
  };
  try_int("output_sample_rate", &output_sample_rate_);
  try_int("decode_upsample_rate", &decode_upsample_rate_);
  try_int("max_seq_len", &max_seq_len_);
  try_int("talker_vocab_size", &talker_vocab_size_);
  try_int("talker_dim", &talker_dim_);
  try_int("num_code_groups", &num_code_groups_);
  try_int("num_quantizers", &num_quantizers_);
  try_int("codebook_size", &codebook_size_);
  try_int("text_prompt_min_token_count", &text_prompt_min_token_count_);
  try_int("text_prompt_prefill_token_count", &text_prompt_prefill_token_count_);
  try_int(
      "text_prompt_prefill_token_count_with_language",
      &text_prompt_prefill_token_count_with_language_);
  try_int(
      "text_prompt_trailing_template_token_count",
      &text_prompt_trailing_template_token_count_);
  try_int("cp_generate_contract_version", &cp_generate_contract_version_);
  try_int("cp_generate_fast_top_k", &cp_generate_fast_top_k_);
  try_int("generation_backend_code", &generation_backend_code_);
  try_int("decoder_backend_code", &decoder_backend_code_);
  try_int(
      "prefer_streaming_decoder_surface",
      &prefer_streaming_decoder_surface_);
  try_int(
      "streaming_decoder_contract_version",
      &streaming_decoder_contract_version_);
  try_int("streaming_decoder_chunk_size", &streaming_decoder_chunk_size_);
  try_int(
      "streaming_decoder_left_context_size",
      &streaming_decoder_left_context_size_);
  try_int("streaming_decoder_max_codes", &streaming_decoder_max_codes_);

  auto try_int64 = [&](const char* name, int64_t* out) {
    auto result = module_->execute(name, empty);
    if (result.ok()) {
      *out = result.get()[0].toInt();
    }
  };
  try_int64("tts_pad_token_id", &tts_pad_id_);
  try_int64("tts_bos_token_id", &tts_bos_id_);
  try_int64("tts_eod_token_id", &tts_eod_id_);
  try_int64("codec_pad_id", &codec_pad_id_);
  try_int64("codec_bos_id", &codec_bos_id_);
  try_int64("codec_eos_id", &codec_eos_id_);
  try_int64("codec_think_id", &codec_think_id_);
  try_int64("codec_language_english_id", &codec_language_english_id_);
  try_int64("codec_nothink_id", &codec_nothink_id_);
  try_int64("codec_think_bos_id", &codec_think_bos_id_);
  try_int64("codec_think_eos_id", &codec_think_eos_id_);
  try_int64("im_start_token_id", &im_start_id_);
  try_int64("assistant_token_id", &assistant_id_);
  try_int64("newline_token_id", &newline_id_);
}

void Qwen3TTSUnifiedRunner::load_methods() {
  // Don't eagerly load all methods — they allocate KV caches and execution
  // plans that consume memory. Instead, load on first use via ensure_method().
}

bool Qwen3TTSUnifiedRunner::ensure_method(const std::string& method_name) {
  if (module_->is_method_loaded(method_name)) {
    return true;
  }
  ET_LOG(Info, "Lazy-loading method: %s", method_name.c_str());
  auto err = module_->load_method(method_name);
  if (err != Error::Ok) {
    ET_LOG(Error, "Failed to load method: %s", method_name.c_str());
    return false;
  }
  // Run a warmup call to trigger XNNPACK delegate initialization.
  // Without this, the first real call pays a multi-second init penalty.
  if (method_name == "decode_audio" || method_name == "decode_audio_stream") {
    ET_LOG(Info, "Warming up %s (XNNPACK init)...", method_name.c_str());
    const int warmup_codes_len =
        method_name == "decode_audio_stream" && streaming_decoder_max_codes_ > 0
        ? streaming_decoder_max_codes_
        : 1;
    std::vector<int64_t> warmup_codes(
        static_cast<size_t>(warmup_codes_len) * num_quantizers_, -1);
    for (int q = 0; q < num_quantizers_; ++q) {
      warmup_codes[q] = 0;
    }
    if (method_name == "decode_audio_stream") {
      run_decode_audio_stream(
          warmup_codes, warmup_codes_len, num_quantizers_, nullptr);
    } else {
      run_decode_audio(warmup_codes, 1, num_quantizers_, nullptr);
    }
  }
  return true;
}

bool Qwen3TTSUnifiedRunner::has_streaming_decode_method() {
  if (checked_streaming_decode_method_) {
    return has_streaming_decode_method_;
  }
  checked_streaming_decode_method_ = true;
  if (streaming_decoder_contract_version_ <= 0 ||
      streaming_decoder_max_codes_ <= 0) {
    has_streaming_decode_method_ = false;
    return false;
  }
  has_streaming_decode_method_ = ensure_method("decode_audio_stream");
  return has_streaming_decode_method_;
}

int Qwen3TTSUnifiedRunner::effective_streaming_interval_steps(
    const SynthesizeConfig& config) const {
  if (config.streaming_chunk_steps > 0) {
    return config.streaming_chunk_steps;
  }
  if (config.streaming_interval_sec <= 0.0f) {
    return 0;
  }
  const double codec_steps_per_second =
      static_cast<double>(output_sample_rate_) /
      static_cast<double>(decode_upsample_rate_);
  const int interval_steps = static_cast<int>(std::lround(
      static_cast<double>(config.streaming_interval_sec) *
      codec_steps_per_second));
  return std::max(1, interval_steps);
}

// ---------------------------------------------------------------------------
// Pipeline stage implementations
// ---------------------------------------------------------------------------

bool Qwen3TTSUnifiedRunner::run_encode_text(
    const std::vector<int64_t>& token_ids,
    std::vector<float>* projected) {
  if (!ensure_method("encode_text")) return false;
  int32_t seq_len = static_cast<int32_t>(token_ids.size());
  auto ids_tensor = from_blob(
      const_cast<int64_t*>(token_ids.data()),
      {1, seq_len},
      ::executorch::aten::ScalarType::Long);

  std::vector<EValue> inputs_et = {EValue(*ids_tensor)};
  auto result = module_->execute("encode_text", inputs_et);
  if (!result.ok()) {
    ET_LOG(Error, "encode_text execution failed.");
    return false;
  }
  extract_float_tensor(result.get()[0].toTensor(), projected);
  return true;
}

bool Qwen3TTSUnifiedRunner::run_talker(
    const std::vector<float>& embeds,
    int32_t seq_len,
    const std::vector<int64_t>& input_pos,
    std::vector<float>* logits,
    std::vector<float>* hidden) {
  if (!ensure_method("talker")) return false;
  auto embeds_tensor = from_blob(
      const_cast<float*>(embeds.data()),
      {1, seq_len, talker_dim_},
      ::executorch::aten::ScalarType::Float);
  auto pos_tensor = from_blob(
      const_cast<int64_t*>(input_pos.data()),
      {seq_len},
      ::executorch::aten::ScalarType::Long);

  std::vector<EValue> inputs_talker = {
      EValue(*embeds_tensor), EValue(*pos_tensor)};
  auto result = module_->execute("talker", inputs_talker);
  if (!result.ok()) {
    ET_LOG(Error, "talker execution failed.");
    return false;
  }
  auto outputs = result.get();
  extract_float_tensor(outputs[0].toTensor(), logits);
  extract_float_tensor(outputs[1].toTensor(), hidden);
  return true;
}

bool Qwen3TTSUnifiedRunner::run_codec_embed(
    int64_t token_id,
    int64_t group_idx,
    std::vector<float>* embed) {
  if (!ensure_method("codec_embed")) return false;
  auto tid_tensor = from_blob(
      &token_id, {1}, ::executorch::aten::ScalarType::Long);
  auto gidx_tensor = from_blob(
      &group_idx, {1}, ::executorch::aten::ScalarType::Long);

  std::vector<EValue> inputs_ce = {EValue(*tid_tensor), EValue(*gidx_tensor)};
  auto result = module_->execute("codec_embed", inputs_ce);
  if (!result.ok()) {
    ET_LOG(Error, "codec_embed execution failed.");
    return false;
  }
  extract_float_tensor(result.get()[0].toTensor(), embed);
  return true;
}

bool Qwen3TTSUnifiedRunner::run_code_predictor(
    const std::vector<float>& embeds,
    int32_t seq_len,
    const std::vector<int64_t>& input_pos,
    std::vector<float>* hidden) {
  if (!ensure_method("code_predictor")) return false;
  auto embeds_tensor = from_blob(
      const_cast<float*>(embeds.data()),
      {1, seq_len, talker_dim_},
      ::executorch::aten::ScalarType::Float);
  auto pos_tensor = from_blob(
      const_cast<int64_t*>(input_pos.data()),
      {seq_len},
      ::executorch::aten::ScalarType::Long);

  std::vector<EValue> inputs_cp = {
      EValue(*embeds_tensor), EValue(*pos_tensor)};
  auto result =
      module_->execute("code_predictor", inputs_cp);
  if (!result.ok()) {
    ET_LOG(Error, "code_predictor execution failed.");
    return false;
  }
  extract_float_tensor(result.get()[0].toTensor(), hidden);
  return true;
}

bool Qwen3TTSUnifiedRunner::run_cp_head(
    const std::vector<float>& hidden,
    int64_t head_idx,
    std::vector<float>* logits) {
  if (!ensure_method("cp_head")) return false;
  auto hidden_tensor = from_blob(
      const_cast<float*>(hidden.data()),
      {1, talker_dim_},
      ::executorch::aten::ScalarType::Float);
  auto hidx_tensor = from_blob(
      &head_idx, {1}, ::executorch::aten::ScalarType::Long);

  std::vector<EValue> inputs_head = {
      EValue(*hidden_tensor), EValue(*hidx_tensor)};
  auto result = module_->execute("cp_head", inputs_head);
  if (!result.ok()) {
    ET_LOG(Error, "cp_head execution failed.");
    return false;
  }
  extract_float_tensor(result.get()[0].toTensor(), logits);
  return true;
}

bool Qwen3TTSUnifiedRunner::run_cp_generate(
    const std::vector<float>& talker_hidden,
    const std::vector<float>& code_0_embed,
    float temperature,
    const std::vector<float>& sample_uniforms,
    std::vector<int64_t>* sampled_subcodes,
    std::vector<float>* embed_sum) {
  if (!ensure_method("cp_generate")) return false;
  auto hidden_tensor = from_blob(
      const_cast<float*>(talker_hidden.data()),
      {1, 1, talker_dim_},
      ::executorch::aten::ScalarType::Float);
  auto embed_tensor = from_blob(
      const_cast<float*>(code_0_embed.data()),
      {1, 1, talker_dim_},
      ::executorch::aten::ScalarType::Float);
  auto temperature_tensor = from_blob(
      &temperature, {1}, ::executorch::aten::ScalarType::Float);
  auto uniform_tensor = from_blob(
      const_cast<float*>(sample_uniforms.data()),
      {num_code_groups_ - 1},
      ::executorch::aten::ScalarType::Float);

  std::vector<EValue> inputs = {
      EValue(*hidden_tensor),
      EValue(*embed_tensor),
      EValue(*temperature_tensor),
      EValue(*uniform_tensor)};
  auto result = module_->execute("cp_generate", inputs);
  if (!result.ok()) {
    ET_LOG(Error, "cp_generate execution failed.");
    return false;
  }
  auto outputs = result.get();
  auto sampled_tensor = outputs[0].toTensor();
  sampled_subcodes->resize(static_cast<size_t>(sampled_tensor.numel()));
  const int64_t* sampled_ptr = sampled_tensor.const_data_ptr<int64_t>();
  std::copy(
      sampled_ptr,
      sampled_ptr + sampled_tensor.numel(),
      sampled_subcodes->begin());
  extract_float_tensor(outputs[1].toTensor(), embed_sum);
  return true;
}

bool Qwen3TTSUnifiedRunner::run_decode_audio(
    const std::vector<int64_t>& codes,
    int32_t codes_len,
    int32_t num_quantizers,
    std::vector<float>* waveform) {
  if (!ensure_method("decode_audio")) return false;
  auto codes_tensor = from_blob(
      const_cast<int64_t*>(codes.data()),
      {1, codes_len, num_quantizers},
      ::executorch::aten::ScalarType::Long);

  std::vector<EValue> inputs_da = {EValue(*codes_tensor)};
  auto result = module_->execute("decode_audio", inputs_da);
  if (!result.ok()) {
    ET_LOG(Error, "decode_audio execution failed.");
    return false;
  }
  if (waveform == nullptr) {
    return true; // Warmup call — discard output.
  }
  auto outputs = result.get();
  auto wav_tensor = outputs[0].toTensor();
  auto len_tensor = outputs[1].toTensor();
  int64_t wav_len = len_tensor.const_data_ptr<int64_t>()[0];
  int64_t total = wav_tensor.numel();
  int64_t used = std::min(wav_len, total);

  waveform->resize(static_cast<size_t>(used));
  if (wav_tensor.scalar_type() == ::executorch::aten::ScalarType::Float) {
    const float* src = wav_tensor.const_data_ptr<float>();
    std::copy(src, src + used, waveform->begin());
  } else {
    extract_float_tensor(wav_tensor, waveform);
    waveform->resize(static_cast<size_t>(used));
  }
  return true;
}

bool Qwen3TTSUnifiedRunner::run_decode_audio_stream(
    const std::vector<int64_t>& padded_codes,
    int32_t padded_codes_len,
    int32_t num_quantizers,
    std::vector<float>* waveform) {
  if (!ensure_method("decode_audio_stream")) {
    return false;
  }
  auto codes_tensor = from_blob(
      const_cast<int64_t*>(padded_codes.data()),
      {1, padded_codes_len, num_quantizers},
      ::executorch::aten::ScalarType::Long);

  std::vector<EValue> inputs_da = {EValue(*codes_tensor)};
  auto result = module_->execute("decode_audio_stream", inputs_da);
  if (!result.ok()) {
    ET_LOG(Error, "decode_audio_stream execution failed.");
    return false;
  }
  if (waveform == nullptr) {
    return true;
  }
  auto outputs = result.get();
  auto wav_tensor = outputs[0].toTensor();
  auto len_tensor = outputs[1].toTensor();
  int64_t wav_len = len_tensor.const_data_ptr<int64_t>()[0];
  int64_t total = wav_tensor.numel();
  int64_t used = std::min(wav_len, total);

  waveform->resize(static_cast<size_t>(used));
  if (wav_tensor.scalar_type() == ::executorch::aten::ScalarType::Float) {
    const float* src = wav_tensor.const_data_ptr<float>();
    std::copy(src, src + used, waveform->begin());
  } else {
    extract_float_tensor(wav_tensor, waveform);
    waveform->resize(static_cast<size_t>(used));
  }
  return true;
}

bool Qwen3TTSUnifiedRunner::decode_code_step_range(
    const std::vector<std::vector<int64_t>>& all_codes,
    int start_step,
    int end_step,
    int left_context_steps,
    bool allow_streaming_surface,
    std::vector<float>* waveform) {
  if (start_step < 0 || end_step < start_step ||
      end_step > static_cast<int>(all_codes.size())) {
    ET_LOG(
        Error,
        "Invalid decode range [%d, %d) for %zu codec steps.",
        start_step,
        end_step,
        all_codes.size());
    return false;
  }
  const int context_steps = std::min(left_context_steps, start_step);
  const int window_start = start_step - context_steps;
  const int window_steps = end_step - window_start;
  std::vector<int64_t> window_codes(
      static_cast<size_t>(window_steps) * num_code_groups_);
  for (int t = 0; t < window_steps; ++t) {
    const auto& step_codes = all_codes[window_start + t];
    for (int g = 0; g < num_code_groups_; ++g) {
      window_codes[t * num_code_groups_ + g] = step_codes[g];
    }
  }

  std::vector<float> decoded_window;
  const bool use_streaming_surface =
      allow_streaming_surface &&
      has_streaming_decode_method() &&
      window_steps <= streaming_decoder_max_codes_;
  if (use_streaming_surface) {
    std::vector<int64_t> padded_codes(
        static_cast<size_t>(streaming_decoder_max_codes_) * num_code_groups_,
        -1);
    std::copy(window_codes.begin(), window_codes.end(), padded_codes.begin());
    if (!run_decode_audio_stream(
            padded_codes,
            streaming_decoder_max_codes_,
            num_code_groups_,
            &decoded_window)) {
      return false;
    }
  } else if (!run_decode_audio(
                 window_codes, window_steps, num_code_groups_, &decoded_window)) {
    return false;
  }

  const size_t trim_samples =
      static_cast<size_t>(context_steps) *
      static_cast<size_t>(decode_upsample_rate_);
  if (trim_samples >= decoded_window.size()) {
    waveform->clear();
    return true;
  }
  waveform->assign(
      decoded_window.begin() + static_cast<std::ptrdiff_t>(trim_samples),
      decoded_window.end());
  return true;
}

bool Qwen3TTSUnifiedRunner::decode_codes_chunked(
    const std::vector<std::vector<int64_t>>& all_codes,
    int chunk_size_steps,
    int left_context_steps,
    bool allow_streaming_surface,
    std::vector<float>* waveform,
    double* decode_ms,
    double* first_audio_ms) {
  using Clock = std::chrono::steady_clock;
  const auto t_decode = Clock::now();
  const auto ms_since = [&](const Clock::time_point& begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin)
        .count();
  };
  chunk_size_steps = std::max(1, chunk_size_steps);

  waveform->clear();
  bool saw_first_audio = false;
  for (int start = 0; start < static_cast<int>(all_codes.size());
       start += chunk_size_steps) {
    const int end =
        std::min(start + chunk_size_steps, static_cast<int>(all_codes.size()));
    std::vector<float> chunk_wav;
    if (!decode_code_step_range(
            all_codes,
            start,
            end,
            left_context_steps,
            allow_streaming_surface,
            &chunk_wav)) {
      return false;
    }
    if (!saw_first_audio && !chunk_wav.empty() && first_audio_ms != nullptr) {
      *first_audio_ms = ms_since(t_decode);
      saw_first_audio = true;
    }
    waveform->insert(waveform->end(), chunk_wav.begin(), chunk_wav.end());
  }
  if (decode_ms != nullptr) {
    *decode_ms = ms_since(t_decode);
  }
  return true;
}

// ---------------------------------------------------------------------------
// Token sampling
// ---------------------------------------------------------------------------

int64_t Qwen3TTSUnifiedRunner::sample_token(
    const std::vector<float>& logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    std::mt19937* gen) {
  return sample_token(
      logits,
      vocab_size,
      temperature,
      top_k,
      top_p,
      1.0f,
      nullptr,
      nullptr,
      -1,
      gen);
}

int64_t Qwen3TTSUnifiedRunner::sample_token(
    const std::vector<float>& logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    const std::vector<int64_t>* generated_tokens,
    const std::vector<int64_t>* suppress_tokens,
    int64_t eos_token_id,
    std::mt19937* gen) {
  std::vector<float> adjusted(logits.begin(), logits.begin() + vocab_size);

  if (generated_tokens != nullptr && repetition_penalty > 1.0f) {
    std::vector<int64_t> unique_tokens = *generated_tokens;
    std::sort(unique_tokens.begin(), unique_tokens.end());
    unique_tokens.erase(
        std::unique(unique_tokens.begin(), unique_tokens.end()),
        unique_tokens.end());
    for (int64_t token : unique_tokens) {
      if (token < 0 || token >= vocab_size) {
        continue;
      }
      float& value = adjusted[static_cast<size_t>(token)];
      value = value < 0.0f ? value * repetition_penalty
                           : value / repetition_penalty;
    }
  }

  if (suppress_tokens != nullptr) {
    const float kNegInf = -std::numeric_limits<float>::infinity();
    for (int64_t token : *suppress_tokens) {
      if (token < 0 || token >= vocab_size) {
        continue;
      }
      adjusted[static_cast<size_t>(token)] = kNegInf;
    }
  }

  const bool preserve_eos = eos_token_id >= 0 && eos_token_id < vocab_size;
  const float preserved_eos_logit =
      preserve_eos ? adjusted[static_cast<size_t>(eos_token_id)]
                   : -std::numeric_limits<float>::infinity();

  if (temperature <= 0.0f || temperature < 1e-6f) {
    return static_cast<int64_t>(
        std::max_element(adjusted.begin(), adjusted.end()) - adjusted.begin());
  }

  if (top_k > 0 && top_k < vocab_size) {
    std::vector<std::pair<float, int>> indexed;
    indexed.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
      float value = adjusted[static_cast<size_t>(i)];
      if (!std::isfinite(value)) {
      continue;
    }
      indexed.push_back({value, i});
    }
    if (static_cast<int>(indexed.size()) > top_k) {
      std::partial_sort(
          indexed.begin(),
          indexed.begin() + top_k,
          indexed.end(),
          [](const auto& a, const auto& b) { return a.first > b.first; });
      std::vector<char> keep(vocab_size, 0);
      for (int i = 0; i < top_k; ++i) {
        keep[static_cast<size_t>(indexed[i].second)] = 1;
      }
      for (int i = 0; i < vocab_size; ++i) {
        if (!keep[static_cast<size_t>(i)]) {
          adjusted[static_cast<size_t>(i)] =
              -std::numeric_limits<float>::infinity();
        }
      }
    }
  }

  if (top_p > 0.0f && top_p < 1.0f) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab_size; ++i) {
      float value = adjusted[static_cast<size_t>(i)];
      if (std::isfinite(value)) {
        max_val = std::max(max_val, value);
      }
    }
    if (std::isfinite(max_val)) {
      std::vector<std::pair<float, int>> indexed;
      indexed.reserve(vocab_size);
      float total = 0.0f;
      for (int i = 0; i < vocab_size; ++i) {
        float value = adjusted[static_cast<size_t>(i)];
        if (!std::isfinite(value)) {
          continue;
        }
        const float prob = std::exp(value - max_val);
        indexed.push_back({prob, i});
        total += prob;
      }
      if (total > 0.0f) {
        for (auto& [prob, idx] : indexed) {
          prob /= total;
        }
        std::sort(
            indexed.begin(),
            indexed.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        std::vector<char> keep(vocab_size, 0);
        float cumulative = 0.0f;
        for (const auto& [prob, idx] : indexed) {
          if (prob <= 0.0f) {
            continue;
          }
          keep[static_cast<size_t>(idx)] = 1;
          cumulative += prob;
          if (cumulative >= top_p) {
            break;
          }
        }
        for (int i = 0; i < vocab_size; ++i) {
          if (!keep[static_cast<size_t>(i)]) {
            adjusted[static_cast<size_t>(i)] =
                -std::numeric_limits<float>::infinity();
          }
        }
      }
    }
  }

  if (preserve_eos && std::isfinite(preserved_eos_logit)) {
    adjusted[static_cast<size_t>(eos_token_id)] = preserved_eos_logit;
  }

  std::vector<float> scaled(adjusted.begin(), adjusted.end());
  for (auto& v : scaled) {
    if (std::isfinite(v)) {
      v /= temperature;
    }
  }

  float max_val = -std::numeric_limits<float>::infinity();
  for (float value : scaled) {
    if (std::isfinite(value)) {
      max_val = std::max(max_val, value);
    }
  }

  std::vector<float> probs(vocab_size, 0.0f);
  float sum = 0.0f;
  if (std::isfinite(max_val)) {
    for (int i = 0; i < vocab_size; ++i) {
      float value = scaled[static_cast<size_t>(i)];
      if (!std::isfinite(value)) {
        continue;
      }
      const float prob = std::exp(value - max_val);
      probs[static_cast<size_t>(i)] = prob;
      sum += prob;
    }
  }
  if (sum <= 0.0f) {
    return static_cast<int64_t>(
        std::max_element(adjusted.begin(), adjusted.end()) - adjusted.begin());
  }
  for (auto& prob : probs) {
    prob /= sum;
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float sample = std::max(
      0.0f,
      std::min(std::nextafter(1.0f, 0.0f), dist(*gen)));
  float cumulative = 0.0f;
  for (int i = 0; i < vocab_size; ++i) {
    cumulative += probs[static_cast<size_t>(i)];
    if (sample <= cumulative) {
      return i;
    }
  }
  return static_cast<int64_t>(
      std::max_element(adjusted.begin(), adjusted.end()) - adjusted.begin());
}

// ---------------------------------------------------------------------------
// Decode codes file (backward compat)
// ---------------------------------------------------------------------------

bool Qwen3TTSUnifiedRunner::read_codes_file(
    const std::string& codes_path,
    std::vector<int64_t>* codes,
    int32_t* codes_len,
    int32_t* num_quantizers) const {
  std::ifstream in(codes_path, std::ios::binary);
  if (!in.good()) {
    ET_LOG(Error, "Could not open codes file: %s", codes_path.c_str());
    return false;
  }

  int32_t t_len = 0;
  int32_t n_q = 0;
  in.read(reinterpret_cast<char*>(&t_len), sizeof(int32_t));
  in.read(reinterpret_cast<char*>(&n_q), sizeof(int32_t));
  if (!in.good() || t_len <= 0 || n_q <= 0) {
    ET_LOG(Error, "Invalid codes header in: %s", codes_path.c_str());
    return false;
  }

  std::vector<int32_t> values(
      static_cast<size_t>(t_len) * static_cast<size_t>(n_q));
  in.read(
      reinterpret_cast<char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(int32_t)));
  if (!in.good()) {
    ET_LOG(Error, "Failed to read codes payload from: %s", codes_path.c_str());
    return false;
  }

  codes->resize(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    (*codes)[i] = static_cast<int64_t>(values[i]);
  }
  *codes_len = t_len;
  *num_quantizers = n_q;
  return true;
}

void Qwen3TTSUnifiedRunner::warmup_decode() {
  if (!ensure_method("decode_audio")) return;
  has_streaming_decode_method();
}

void Qwen3TTSUnifiedRunner::warmup_all() {
  ensure_method("encode_text");
  ensure_method("talker");
  ensure_method("codec_embed");
  ensure_method("code_predictor");
  ensure_method("cp_head");
  ensure_method("cp_generate");
  ensure_method("decode_audio");
  has_streaming_decode_method();

  ET_LOG(Info, "Warming up full text synthesis path...");

  std::vector<float> projected;
  if (!run_encode_text({assistant_id_}, &projected)) {
    return;
  }

  std::vector<float> codec_bos_embed;
  if (!run_codec_embed(codec_bos_id_, 0, &codec_bos_embed)) {
    return;
  }

  std::vector<float> talker_logits;
  std::vector<float> talker_hidden;
  if (!run_talker(projected, 1, {0}, &talker_logits, &talker_hidden)) {
    return;
  }

  std::vector<float> cp_prefill(static_cast<size_t>(talker_dim_) * 2, 0.0f);
  std::copy(talker_hidden.begin(), talker_hidden.end(), cp_prefill.begin());
  std::copy(
      codec_bos_embed.begin(),
      codec_bos_embed.end(),
      cp_prefill.begin() + talker_dim_);
  std::vector<float> cp_hidden;
  std::vector<float> cp_logits;
  if (run_code_predictor(cp_prefill, 2, {0, 1}, &cp_hidden)) {
    run_cp_head(cp_hidden, 0, &cp_logits);
  }

  std::vector<int64_t> fused_codes;
  std::vector<float> fused_embed_sum;
  std::vector<float> sample_uniforms(
      static_cast<size_t>(num_code_groups_ - 1), 0.5f);
  if (cp_generate_contract_version_ >= 2) {
    run_cp_generate(
        talker_hidden,
        codec_bos_embed,
        1.0f,
        sample_uniforms,
        &fused_codes,
        &fused_embed_sum);
  }

  std::vector<int64_t> warmup_codes(1 * num_quantizers_, 0);
  run_decode_audio(warmup_codes, 1, num_quantizers_, nullptr);
  if (has_streaming_decode_method()) {
    std::vector<int64_t> padded_stream_codes(
        static_cast<size_t>(streaming_decoder_max_codes_) * num_quantizers_, -1);
    for (int q = 0; q < num_quantizers_; ++q) {
      padded_stream_codes[q] = 0;
    }
    run_decode_audio_stream(
        padded_stream_codes,
        streaming_decoder_max_codes_,
        num_quantizers_,
        nullptr);
  }
}

bool Qwen3TTSUnifiedRunner::decode_codes_file(
    const std::string& codes_path,
    std::vector<float>* waveform) {
  std::vector<int64_t> flat_codes;
  int32_t codes_len = 0;
  int32_t num_quantizers = 0;
  if (!read_codes_file(codes_path, &flat_codes, &codes_len, &num_quantizers)) {
    return false;
  }
  ET_LOG(
      Info,
      "Decoding codes: codes_len=%d num_quantizers=%d",
      codes_len,
      num_quantizers);
  if (num_quantizers != num_code_groups_) {
    return run_decode_audio(flat_codes, codes_len, num_quantizers, waveform);
  }
  std::vector<std::vector<int64_t>> all_codes(
      static_cast<size_t>(codes_len),
      std::vector<int64_t>(static_cast<size_t>(num_quantizers), 0));
  for (int t = 0; t < codes_len; ++t) {
    for (int g = 0; g < num_quantizers; ++g) {
      all_codes[static_cast<size_t>(t)][static_cast<size_t>(g)] =
          flat_codes[static_cast<size_t>(t * num_quantizers + g)];
    }
  }
  double decode_ms = 0.0;
  double first_audio_ms = 0.0;
  return decode_codes_chunked(
      all_codes,
      streaming_decoder_chunk_size_ > 0 ? streaming_decoder_chunk_size_ : 300,
      streaming_decoder_left_context_size_ > 0
          ? streaming_decoder_left_context_size_
          : 25,
      prefer_streaming_decoder_surface_ > 0,
      waveform,
      &decode_ms,
      &first_audio_ms);
}

// ---------------------------------------------------------------------------
// Full text-to-audio pipeline (placeholder for tokenizer integration)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Embedding helpers
// ---------------------------------------------------------------------------

bool Qwen3TTSUnifiedRunner::get_text_embed(
    int64_t token_id,
    std::vector<float>* embed) {
  std::vector<int64_t> ids = {token_id};
  std::vector<float> projected;
  if (!run_encode_text(ids, &projected)) {
    return false;
  }
  *embed = std::move(projected);
  return true;
}

void Qwen3TTSUnifiedRunner::vec_add(
    std::vector<float>& dst,
    const std::vector<float>& src) {
  for (size_t i = 0; i < dst.size() && i < src.size(); ++i) {
    dst[i] += src[i];
  }
}

void Qwen3TTSUnifiedRunner::vec_zero(std::vector<float>& v) {
  std::fill(v.begin(), v.end(), 0.0f);
}

// ---------------------------------------------------------------------------
// Full text-to-audio pipeline
// ---------------------------------------------------------------------------

bool Qwen3TTSUnifiedRunner::synthesize(
    const std::string& text,
    const std::string& language,
    const SynthesizeConfig& config,
    std::vector<float>* waveform) {
  return synthesize(text, language, config, waveform, nullptr);
}

bool Qwen3TTSUnifiedRunner::synthesize(
    const std::string& text,
    const std::string& language,
    const SynthesizeConfig& config,
    std::vector<float>* waveform,
    SynthesisTiming* timing) {
  auto session = create_synthesis_session(config);
  return session->synthesize(text, language, waveform, timing);
}

bool SynthesisSession::synthesize(
    const std::string& text,
    const std::string& language,
    std::vector<float>* waveform,
    SynthesisTiming* timing) {
  return synthesize_impl(text, language, waveform, timing, nullptr);
}

bool SynthesisSession::synthesize(
    const std::string& text,
    const std::string& language,
    std::vector<float>* waveform,
    SynthesisTiming* timing,
    AudioChunkCallback on_audio_chunk) {
  return synthesize_impl(
      text, language, waveform, timing, std::move(on_audio_chunk));
}

bool SynthesisSession::synthesize_impl(
    const std::string& text,
    const std::string& language,
    std::vector<float>* waveform,
    SynthesisTiming* timing,
    AudioChunkCallback on_audio_chunk) {
  auto* runner = runner_;
  if (!runner->tokenizer_) {
    ET_LOG(
        Error,
        "Tokenizer not loaded. Provide --tokenizer_path for text synthesis.");
    return false;
  }

  using Clock = std::chrono::steady_clock;
  const auto t_start = Clock::now();
  const auto ms_since = [](const Clock::time_point& begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin)
        .count();
  };

  std::string language_lower = language;
  std::transform(
      language_lower.begin(),
      language_lower.end(),
      language_lower.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  const bool use_language_prefix = language_lower == "english";
  if (!language.empty() && language_lower != "auto" && !use_language_prefix) {
    ET_LOG(
        Info,
        "Language '%s' is not implemented in the unified text-only path yet; "
        "continuing with the default text-only contract.",
        language.c_str());
  } else if (use_language_prefix) {
    ET_LOG(
        Info,
        "Using English language-conditioned codec prefix (language_id=%lld).",
        static_cast<long long>(runner->codec_language_english_id_));
  }

  const auto t_prompt = Clock::now();

  // 0. VoiceDesign instruct: tokenize and project the instruct prefix.
  std::vector<float> instruct_embeds_flat;
  int instruct_token_count = 0;
  if (!config_.instruct.empty()) {
    auto instruct_text = build_instruct_prefix(config_.instruct);
    auto instruct_enc =
        runner->tokenizer_->encode(instruct_text, /*bos=*/0, /*eos=*/0);
    if (!instruct_enc.ok()) {
      ET_LOG(Error, "Failed to tokenize VoiceDesign instruct.");
      return false;
    }
    auto instruct_ids_raw = instruct_enc.get();
    std::vector<int64_t> instruct_ids(
        instruct_ids_raw.begin(), instruct_ids_raw.end());
    instruct_token_count = static_cast<int>(instruct_ids.size());
    if (!runner->run_encode_text(instruct_ids, &instruct_embeds_flat)) {
      return false;
    }
    ET_LOG(
        Info,
        "VoiceDesign instruct: %d tokens prepended.",
        instruct_token_count);
  }

  // 1. Tokenize the assistant-wrapped prompt. This mirrors the upstream helper
  // and the mlx-audio reference path for text-only prompting.
  auto prompt_text = build_assistant_prompt_text(text);
  auto encode_result = runner->tokenizer_->encode(prompt_text, /*bos=*/0, /*eos=*/0);
  if (!encode_result.ok()) {
    ET_LOG(Error, "Failed to tokenize assistant prompt text.");
    return false;
  }
  auto prompt_token_ids_raw = encode_result.get();
  std::vector<int64_t> prompt_token_ids(
      prompt_token_ids_raw.begin(), prompt_token_ids_raw.end());
  const int prompt_token_count = static_cast<int>(prompt_token_ids.size());
  ET_LOG(Info, "Tokenized assistant prompt: %d tokens", prompt_token_count);
  if (prompt_token_count < runner->text_prompt_min_token_count_) {
    ET_LOG(
        Error,
        "Assistant prompt is too short (%d tokens, need at least %d).",
        prompt_token_count,
        runner->text_prompt_min_token_count_);
    return false;
  }

  std::vector<float> prompt_embeds_flat;
  if (!runner->run_encode_text(prompt_token_ids, &prompt_embeds_flat)) {
    return false;
  }
  if (static_cast<int>(prompt_embeds_flat.size()) !=
      prompt_token_count * runner->talker_dim_) {
    ET_LOG(
        Error,
        "encode_text returned unexpected size: got %zu, expected %d.",
        prompt_embeds_flat.size(),
        prompt_token_count * runner->talker_dim_);
    return false;
  }

  // 2. Split prompt embeddings following the text-only contract:
  //    role = first 3 tokens, first_text = token 3, trailing = tokens 4:-5 + tts_eos.
  std::vector<float> role_embed;
  copy_token_slice(
      prompt_embeds_flat,
      0,
      kAssistantRoleTokenCount,
      runner->talker_dim_,
      &role_embed);

  std::vector<float> first_text_embed;
  copy_token_slice(
      prompt_embeds_flat,
      kAssistantRoleTokenCount,
      kFirstTextTokenCount,
      runner->talker_dim_,
      &first_text_embed);

  // 3. Get text-side special embeddings in one batch.
  std::vector<int64_t> tts_special_ids = {
      runner->tts_bos_id_, runner->tts_eod_id_, runner->tts_pad_id_};
  std::vector<float> tts_special_flat;
  if (!runner->run_encode_text(tts_special_ids, &tts_special_flat)) {
    return false;
  }
  std::vector<float> tts_bos_embed;
  copy_token_slice(
      tts_special_flat, 0, 1, runner->talker_dim_, &tts_bos_embed);
  std::vector<float> tts_eos_embed;
  copy_token_slice(
      tts_special_flat, 1, 1, runner->talker_dim_, &tts_eos_embed);
  std::vector<float> tts_pad_embed;
  copy_token_slice(
      tts_special_flat, 2, 1, runner->talker_dim_, &tts_pad_embed);

  const int trailing_prompt_token_count =
      prompt_token_count - kAssistantRoleTokenCount - kFirstTextTokenCount -
      runner->text_prompt_trailing_template_token_count_ + 1;
  std::vector<std::vector<float>> trailing_text_embeds;
  trailing_text_embeds.reserve(static_cast<size_t>(trailing_prompt_token_count));
  for (int i = kAssistantRoleTokenCount + kFirstTextTokenCount;
       i < prompt_token_count - runner->text_prompt_trailing_template_token_count_;
       ++i) {
    std::vector<float> token_embed;
    copy_token_slice(
        prompt_embeds_flat, i, 1, runner->talker_dim_, &token_embed);
    trailing_text_embeds.push_back(std::move(token_embed));
  }
  trailing_text_embeds.push_back(tts_eos_embed);

  // 4. Get codec control embeddings for the text-only prefix.
  std::vector<float> codec_nothink_embed, codec_think_embed, codec_think_bos_embed;
  std::vector<float> codec_language_embed, codec_think_eos_embed;
  std::vector<float> codec_pad_embed, codec_bos_embed;
  if (use_language_prefix) {
    if (!runner->run_codec_embed(
            runner->codec_think_id_, 0, &codec_think_embed)) {
      return false;
    }
    if (!runner->run_codec_embed(
            runner->codec_language_english_id_, 0, &codec_language_embed)) {
      return false;
    }
  } else if (!runner->run_codec_embed(
                 runner->codec_nothink_id_, 0, &codec_nothink_embed)) {
    return false;
  }
  if (!runner->run_codec_embed(
          runner->codec_think_bos_id_, 0, &codec_think_bos_embed))
    return false;
  if (!runner->run_codec_embed(
          runner->codec_think_eos_id_, 0, &codec_think_eos_embed))
    return false;
  if (!runner->run_codec_embed(runner->codec_pad_id_, 0, &codec_pad_embed)) {
    return false;
  }
  if (!runner->run_codec_embed(runner->codec_bos_id_, 0, &codec_bos_embed)) {
    return false;
  }

  const int base_prefill_len = use_language_prefix
      ? runner->text_prompt_prefill_token_count_with_language_
      : runner->text_prompt_prefill_token_count_;
  const int prefill_len = base_prefill_len + instruct_token_count;
  if (static_cast<int>(trailing_text_embeds.size()) != trailing_prompt_token_count) {
    ET_LOG(
        Error,
        "Trailing prompt split mismatch: expected=%d got=%zu.",
        trailing_prompt_token_count,
        trailing_text_embeds.size());
    return false;
  }
  if (config_.max_new_tokens < trailing_prompt_token_count) {
    ET_LOG(
        Error,
        "max_new_tokens=%d is too small to consume the trailing prompt budget=%d.",
        config_.max_new_tokens,
        trailing_prompt_token_count);
    return false;
  }
  if (prefill_len + config_.max_new_tokens > runner->max_seq_len_) {
    ET_LOG(
        Error,
        "Prompt budget exceeds talker max_seq_len: prefill=%d max_new_tokens=%d "
        "max_seq_len=%d.",
        prefill_len,
        config_.max_new_tokens,
        runner->max_seq_len_);
    return false;
  }

  // 5. Build composite prefill embeddings.
  //    VoiceDesign: instruct tokens are prepended before the role tokens.
  //    Text-only schedule (after instruct offset):
  //    pos 0-2: role tokens from the assistant-wrapped prompt
  //    auto:    pos 3-5 = tts_pad + codec_nothink/think_bos/think_eos,
  //             pos 6 = tts_bos + codec_pad, pos 7 = first_text + codec_bos
  //    English: pos 3-6 = tts_pad + codec_think/think_bos/lang/think_eos,
  //             pos 7 = tts_bos + codec_pad, pos 8 = first_text + codec_bos
  int dim = runner->talker_dim_;
  const int off = instruct_token_count;

  std::vector<float> prefill_embeds(prefill_len * dim, 0.0f);
  auto set_pos = [&](int pos, const std::vector<float>& v) {
    std::copy(v.begin(), v.begin() + dim, prefill_embeds.begin() + pos * dim);
  };
  auto add_pos = [&](int pos, const std::vector<float>& v) {
    for (int i = 0; i < dim; ++i) {
      prefill_embeds[pos * dim + i] += v[i];
    }
  };

  // Instruct tokens (VoiceDesign prefix).
  for (int i = 0; i < instruct_token_count; ++i) {
    std::vector<float> token_embed;
    copy_token_slice(instruct_embeds_flat, i, 1, dim, &token_embed);
    set_pos(i, token_embed);
  }

  // Role tokens.
  for (int i = 0; i < kAssistantRoleTokenCount; ++i) {
    std::vector<float> token_embed;
    copy_token_slice(role_embed, i, 1, dim, &token_embed);
    set_pos(off + i, token_embed);
  }

  // Combined codec/text prefix.
  if (use_language_prefix) {
    set_pos(off + 3, tts_pad_embed);
    add_pos(off + 3, codec_think_embed);
    set_pos(off + 4, tts_pad_embed);
    add_pos(off + 4, codec_think_bos_embed);
    set_pos(off + 5, tts_pad_embed);
    add_pos(off + 5, codec_language_embed);
    set_pos(off + 6, tts_pad_embed);
    add_pos(off + 6, codec_think_eos_embed);
    set_pos(off + 7, tts_bos_embed);
    add_pos(off + 7, codec_pad_embed);
    set_pos(off + 8, first_text_embed);
    add_pos(off + 8, codec_bos_embed);
  } else {
    set_pos(off + 3, tts_pad_embed);
    add_pos(off + 3, codec_nothink_embed);
    set_pos(off + 4, tts_pad_embed);
    add_pos(off + 4, codec_think_bos_embed);
    set_pos(off + 5, tts_pad_embed);
    add_pos(off + 5, codec_think_eos_embed);
    set_pos(off + 6, tts_bos_embed);
    add_pos(off + 6, codec_pad_embed);
    set_pos(off + 7, first_text_embed);
    add_pos(off + 7, codec_bos_embed);
  }

  const auto t_prompt_prep_end = Clock::now();

  // 6. Run talker prefill.
  std::vector<int64_t> prefill_pos(prefill_len);
  std::iota(prefill_pos.begin(), prefill_pos.end(), 0);

  std::vector<float> logits, hidden;
  if (!runner->run_talker(
          prefill_embeds, prefill_len, prefill_pos, &logits, &hidden)) {
    return false;
  }
  ET_LOG(Info, "Talker prefill done (seq_len=%d)", prefill_len);
  const auto t_prefill_end = Clock::now();
  const double prompt_prep_ms =
      std::chrono::duration<double, std::milli>(t_prompt_prep_end - t_prompt)
          .count();
  const double talker_prefill_ms =
      std::chrono::duration<double, std::milli>(t_prefill_end - t_prompt_prep_end)
          .count();

  // 7. Autoregressive generation loop.
  std::vector<std::vector<int64_t>> all_codes;
  std::vector<float> streamed_waveform;
  std::vector<int64_t> generated_code_0_tokens;
  std::vector<int64_t> suppress_tokens;
  suppress_tokens.reserve(1024);
  for (int token_id = runner->talker_vocab_size_ - 1024;
       token_id < runner->talker_vocab_size_;
       ++token_id) {
    if (token_id != runner->codec_eos_id_) {
      suppress_tokens.push_back(token_id);
    }
  }
  int talker_pos = prefill_len;
  int trailing_idx = 0;
  const bool use_fused_cp_generate =
      config_.use_fused_cp_generate &&
      runner->cp_generate_contract_version_ >= 2 &&
      config_.temperature >= 1e-6f &&
      config_.top_k == runner->cp_generate_fast_top_k_ &&
      (config_.top_p <= 0.0f || config_.top_p >= 1.0f);
  if (!use_fused_cp_generate) {
    ET_LOG(
        Info,
        "Falling back to legacy code predictor loop "
        "(fast path requires cp_generate v2, temperature>0, matching top_k, "
        "and top_p disabled).");
  }
  double codegen_ms = 0.0;
  auto t_codegen_cursor = Clock::now();
  const int streaming_interval_steps =
      runner->effective_streaming_interval_steps(config_);
  const bool enable_streaming_decode =
      on_audio_chunk != nullptr && !config_.non_streaming_mode &&
      streaming_interval_steps > 0;
  const bool use_streaming_decoder_surface =
      !config_.disable_streaming_decoder_surface &&
      (config_.force_streaming_decoder_surface ||
       runner->prefer_streaming_decoder_surface_ > 0);
  if (enable_streaming_decode) {
    ET_LOG(
        Info,
        "Streaming decode policy: %s (generation_backend=%s decoder_backend=%s)",
        use_streaming_decoder_surface ? "fixed_surface" : "overlap_window",
        backend_code_name(runner->generation_backend_code_),
        backend_code_name(runner->decoder_backend_code_));
  }
  std::vector<float> cumulative_stream_waveform;
  int decoded_steps = 0;
  double chunk_decode_ms = 0.0;
  double first_audio_ms = 0.0;
  bool saw_first_audio = false;

  for (int step = 0; step < config_.max_new_tokens; ++step) {
    int64_t code_0 = runner->sample_token(
        logits,
        runner->talker_vocab_size_,
        config_.temperature,
        config_.top_k,
        config_.top_p,
        config_.repetition_penalty,
        &generated_code_0_tokens,
        &suppress_tokens,
        runner->codec_eos_id_,
        &rng_);

    if (code_0 == runner->codec_eos_id_) {
      ET_LOG(Info, "EOS at step %d", step);
      break;
    }
    if (code_0 < 0 || code_0 >= runner->codebook_size_) {
      ET_LOG(
          Error,
          "Talker produced invalid primary codec id %lld at step %d",
          static_cast<long long>(code_0),
          step);
      return false;
    }
    generated_code_0_tokens.push_back(code_0);

    std::vector<float> main_embed;
    if (!runner->run_codec_embed(code_0, 0, &main_embed)) {
      return false;
    }

    std::vector<int64_t> step_codes(runner->num_code_groups_);
    step_codes[0] = code_0;
    std::vector<float> next_input_embed = main_embed;

    if (use_fused_cp_generate) {
      std::uniform_real_distribution<float> uniform(1e-6f, 1.0f - 1e-6f);
      std::vector<float> sample_uniforms(
          static_cast<size_t>(runner->num_code_groups_ - 1));
      for (float& value : sample_uniforms) {
        value = uniform(rng_);
      }
      std::vector<int64_t> fused_subcodes;
      std::vector<float> fused_embed_sum;
      if (!runner->run_cp_generate(
              hidden,
              main_embed,
              config_.temperature,
              sample_uniforms,
              &fused_subcodes,
              &fused_embed_sum)) {
        return false;
      }
      if (static_cast<int>(fused_subcodes.size()) != runner->num_code_groups_ - 1) {
        ET_LOG(
            Error,
            "cp_generate returned %zu subcodes, expected %d.",
            fused_subcodes.size(),
            runner->num_code_groups_ - 1);
        return false;
      }
      for (size_t i = 0; i < fused_subcodes.size(); ++i) {
        const int64_t code = fused_subcodes[i];
        if (code < 0 || code >= runner->codebook_size_) {
          ET_LOG(
              Error,
              "cp_generate produced invalid codec id %lld at step %d group %zu",
              static_cast<long long>(code),
              step,
              i + 1);
          return false;
        }
        step_codes[i + 1] = code;
      }
      next_input_embed = std::move(fused_embed_sum);
    } else {
      std::vector<float> cp_prefill(static_cast<size_t>(runner->talker_dim_) * 2);
      std::copy(hidden.begin(), hidden.end(), cp_prefill.begin());
      std::copy(
          main_embed.begin(),
          main_embed.end(),
          cp_prefill.begin() + runner->talker_dim_);
      std::vector<int64_t> cp_pos = {0, 1};
      std::vector<float> cp_hidden;
      if (!runner->run_code_predictor(cp_prefill, 2, cp_pos, &cp_hidden)) {
        return false;
      }

      for (int g = 0; g < runner->num_code_groups_ - 1; ++g) {
        std::vector<float> cp_logits;
        if (!runner->run_cp_head(cp_hidden, g, &cp_logits)) {
          return false;
        }
        int64_t code = runner->sample_token(
            cp_logits,
            runner->codebook_size_,
            config_.temperature,
            config_.top_k,
            config_.top_p,
            &rng_);
        if (code < 0 || code >= runner->codebook_size_) {
          ET_LOG(
              Error,
              "Code predictor produced invalid codec id %lld at step %d group %d",
              static_cast<long long>(code),
              step,
              g + 1);
          return false;
        }
        step_codes[g + 1] = code;

        std::vector<float> code_embed;
        if (!runner->run_codec_embed(code, g + 1, &code_embed)) {
          return false;
        }
        runner->vec_add(next_input_embed, code_embed);

        if (g + 1 < runner->num_code_groups_ - 1) {
          std::vector<int64_t> cp_step_pos = {static_cast<int64_t>(g + 2)};
          if (!runner->run_code_predictor(code_embed, 1, cp_step_pos, &cp_hidden)) {
            return false;
          }
        }
      }
    }

    all_codes.push_back(step_codes);

    if (enable_streaming_decode) {
      const int n_accumulated = static_cast<int>(all_codes.size());
      if (n_accumulated - decoded_steps >= streaming_interval_steps) {
        codegen_ms += ms_since(t_codegen_cursor);
        const auto t_chunk_decode = Clock::now();
        std::vector<float> chunk_wav;
        if (config_.use_legacy_cumulative_streaming_decode) {
          std::vector<int64_t> chunk_flat(
              static_cast<size_t>(n_accumulated) * runner->num_code_groups_);
          for (int t = 0; t < n_accumulated; ++t) {
            for (int g = 0; g < runner->num_code_groups_; ++g) {
              chunk_flat[t * runner->num_code_groups_ + g] = all_codes[t][g];
            }
          }
          if (!runner->run_decode_audio(
                  chunk_flat, n_accumulated, runner->num_code_groups_, &chunk_wav)) {
            return false;
          }
        } else if (!runner->decode_code_step_range(
                       all_codes,
                       decoded_steps,
                       n_accumulated,
                       config_.streaming_left_context_size,
                       use_streaming_decoder_surface,
                       &chunk_wav)) {
          return false;
        }
        chunk_decode_ms +=
            std::chrono::duration<double, std::milli>(
                Clock::now() - t_chunk_decode)
                .count();
        if (!saw_first_audio && !chunk_wav.empty()) {
          first_audio_ms = ms_since(t_start);
          saw_first_audio = true;
        }
        on_audio_chunk(chunk_wav, false);
        if (config_.use_legacy_cumulative_streaming_decode) {
          cumulative_stream_waveform = chunk_wav;
          ET_LOG(
              Info,
              "Streamed cumulative audio through step %d (%zu samples)",
              step + 1,
              chunk_wav.size());
        } else {
          streamed_waveform.insert(
              streamed_waveform.end(), chunk_wav.begin(), chunk_wav.end());
          decoded_steps = n_accumulated;
          ET_LOG(
              Info,
              "Streamed delta audio through step %d (%zu samples)",
              step + 1,
              chunk_wav.size());
        }
        if (config_.use_legacy_cumulative_streaming_decode) {
          decoded_steps = n_accumulated;
        }
        t_codegen_cursor = Clock::now();
      }
    }

    if (trailing_idx < static_cast<int>(trailing_text_embeds.size())) {
      runner->vec_add(next_input_embed, trailing_text_embeds[trailing_idx]);
      ++trailing_idx;
    } else {
      runner->vec_add(next_input_embed, tts_pad_embed);
    }

    std::vector<int64_t> step_pos = {static_cast<int64_t>(talker_pos)};
    if (!runner->run_talker(next_input_embed, 1, step_pos, &logits, &hidden)) {
      return false;
    }
    ++talker_pos;

    if ((step + 1) % 10 == 0) {
      ET_LOG(Info, "  Step %d/%d (pos=%d)", step + 1, config_.max_new_tokens,
             talker_pos);
    }
  }
  codegen_ms += ms_since(t_codegen_cursor);

  int n_codes = static_cast<int>(all_codes.size());
  ET_LOG(
      Info,
      "Generated %d codec steps (%d text tokens consumed)",
      n_codes,
      trailing_idx + kFirstTextTokenCount);

  if (n_codes == 0) {
    ET_LOG(Error, "No codes generated.");
    return false;
  }

  for (int t = 0; t < n_codes; ++t) {
    for (int g = 0; g < runner->num_code_groups_; ++g) {
      int64_t code = all_codes[t][g];
      if (code < 0 || code >= runner->codebook_size_) {
        ET_LOG(
            Error,
            "Invalid decoder code %lld at frame %d group %d",
            static_cast<long long>(code),
            t,
            g);
        return false;
      }
    }
  }

  double final_decode_ms = 0.0;
  if (enable_streaming_decode) {
    if (decoded_steps < n_codes) {
      const auto t_chunk_decode = Clock::now();
      std::vector<float> final_chunk;
      if (config_.use_legacy_cumulative_streaming_decode) {
        std::vector<int64_t> flat_codes(
            static_cast<size_t>(n_codes) * runner->num_code_groups_);
        for (int t = 0; t < n_codes; ++t) {
          for (int g = 0; g < runner->num_code_groups_; ++g) {
            flat_codes[t * runner->num_code_groups_ + g] = all_codes[t][g];
          }
        }
        if (!runner->run_decode_audio(
                flat_codes, n_codes, runner->num_code_groups_, &final_chunk)) {
          return false;
        }
      } else if (!runner->decode_code_step_range(
                     all_codes,
                     decoded_steps,
                     n_codes,
                     config_.streaming_left_context_size,
                     use_streaming_decoder_surface,
                     &final_chunk)) {
        return false;
      }
      chunk_decode_ms +=
          std::chrono::duration<double, std::milli>(
              Clock::now() - t_chunk_decode)
              .count();
      if (!saw_first_audio && !final_chunk.empty()) {
        first_audio_ms = ms_since(t_start);
        saw_first_audio = true;
      }
      on_audio_chunk(final_chunk, true);
      streamed_waveform.insert(
          streamed_waveform.end(), final_chunk.begin(), final_chunk.end());
      cumulative_stream_waveform = final_chunk;
    } else {
      on_audio_chunk({}, true);
    }
    *waveform = config_.use_legacy_cumulative_streaming_decode
        ? std::move(cumulative_stream_waveform)
        : std::move(streamed_waveform);
  } else {
    ET_LOG(Info, "Decoding %d codes to audio...", n_codes);
    const auto t_final_decode_start = Clock::now();
    double first_audio_from_decode_ms = 0.0;
    if (!runner->decode_codes_chunked(
            all_codes,
            config_.streaming_chunk_size,
            config_.streaming_left_context_size,
            use_streaming_decoder_surface,
            waveform,
            &final_decode_ms,
            &first_audio_from_decode_ms)) {
      return false;
    }
    if (first_audio_from_decode_ms > 0.0) {
      first_audio_ms =
          std::chrono::duration<double, std::milli>(
              t_final_decode_start - t_start)
              .count() +
          first_audio_from_decode_ms;
    }
  }
  const double decode_audio_ms = chunk_decode_ms + final_decode_ms;

  if (timing != nullptr) {
    timing->prompt_token_count = prompt_token_count;
    timing->generated_codec_steps = n_codes;
    timing->text_tokens_consumed = trailing_idx + kFirstTextTokenCount;
    timing->prompt_prep_ms = prompt_prep_ms;
    timing->talker_prefill_ms = talker_prefill_ms;
    timing->codegen_ms = codegen_ms;
    timing->first_audio_ms = first_audio_ms;
    timing->chunk_decode_ms = chunk_decode_ms;
    timing->final_decode_ms = final_decode_ms;
    timing->decode_audio_ms = decode_audio_ms;
    timing->total_generation_ms = ms_since(t_start);
  }
  return true;
}

// ---------------------------------------------------------------------------
// WAV writing
// ---------------------------------------------------------------------------

bool Qwen3TTSUnifiedRunner::write_wav_file(
    const std::string& output_wav_path,
    const std::vector<float>& waveform) const {
  std::ofstream out(output_wav_path, std::ios::binary);
  if (!out.good()) {
    ET_LOG(
        Error, "Could not open output wav path: %s", output_wav_path.c_str());
    return false;
  }

  const uint16_t num_channels = 1;
  const uint16_t bits_per_sample = 16;
  const uint32_t sample_rate = static_cast<uint32_t>(output_sample_rate_);
  const uint32_t byte_rate =
      sample_rate * num_channels * (bits_per_sample / 8U);
  const uint16_t block_align = num_channels * (bits_per_sample / 8U);
  const uint32_t data_bytes =
      static_cast<uint32_t>(waveform.size() * sizeof(int16_t));

  out.write("RIFF", 4);
  const uint32_t riff_chunk_size = 36U + data_bytes;
  out.write(reinterpret_cast<const char*>(&riff_chunk_size), 4);
  out.write("WAVE", 4);

  out.write("fmt ", 4);
  const uint32_t fmt_chunk_size = 16;
  out.write(reinterpret_cast<const char*>(&fmt_chunk_size), 4);
  const uint16_t audio_format = 1;
  out.write(reinterpret_cast<const char*>(&audio_format), 2);
  out.write(reinterpret_cast<const char*>(&num_channels), 2);
  out.write(reinterpret_cast<const char*>(&sample_rate), 4);
  out.write(reinterpret_cast<const char*>(&byte_rate), 4);
  out.write(reinterpret_cast<const char*>(&block_align), 2);
  out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

  out.write("data", 4);
  out.write(reinterpret_cast<const char*>(&data_bytes), 4);
  for (float sample : waveform) {
    const float clipped = std::max(-1.0f, std::min(1.0f, sample));
    const int16_t pcm = static_cast<int16_t>(std::lrint(clipped * 32767.0f));
    out.write(reinterpret_cast<const char*>(&pcm), sizeof(int16_t));
  }

  return out.good();
}

} // namespace qwen3_tts
