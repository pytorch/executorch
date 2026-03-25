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

std::string build_assistant_prompt_text(const std::string& text) {
  return std::string("<|im_start|>assistant\n") + text +
      "<|im_end|>\n<|im_start|>assistant\n";
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
      "num_code_groups=%d text_prompt_prefill=%d tokenizer=%s",
      output_sample_rate_,
      max_seq_len_,
      talker_dim_,
      num_code_groups_,
      text_prompt_prefill_token_count_,
      tokenizer_ ? "loaded" : "none");
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
  if (method_name == "decode_audio") {
    ET_LOG(Info, "Warming up decode_audio (XNNPACK init)...");
    std::vector<int64_t> warmup_codes(1 * 1 * num_quantizers_, 0);
    run_decode_audio(warmup_codes, 1, num_quantizers_, nullptr);
  }
  return true;
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
}

void Qwen3TTSUnifiedRunner::warmup_all() {
  ensure_method("encode_text");
  ensure_method("talker");
  ensure_method("codec_embed");
  ensure_method("code_predictor");
  ensure_method("cp_head");
  ensure_method("cp_generate");
  ensure_method("decode_audio");

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
  return run_decode_audio(flat_codes, codes_len, num_quantizers, waveform);
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

  const int prefill_len = use_language_prefix
      ? runner->text_prompt_prefill_token_count_with_language_
      : runner->text_prompt_prefill_token_count_;
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
  //    Text-only schedule:
  //    pos 0-2: role tokens from the assistant-wrapped prompt
  //    auto:    pos 3-5 = tts_pad + codec_nothink/think_bos/think_eos,
  //             pos 6 = tts_bos + codec_pad, pos 7 = first_text + codec_bos
  //    English: pos 3-6 = tts_pad + codec_think/think_bos/lang/think_eos,
  //             pos 7 = tts_bos + codec_pad, pos 8 = first_text + codec_bos
  int dim = runner->talker_dim_;

  std::vector<float> prefill_embeds(prefill_len * dim, 0.0f);
  auto set_pos = [&](int pos, const std::vector<float>& v) {
    std::copy(v.begin(), v.begin() + dim, prefill_embeds.begin() + pos * dim);
  };
  auto add_pos = [&](int pos, const std::vector<float>& v) {
    for (int i = 0; i < dim; ++i) {
      prefill_embeds[pos * dim + i] += v[i];
    }
  };

  // Role tokens.
  for (int i = 0; i < kAssistantRoleTokenCount; ++i) {
    std::vector<float> token_embed;
    copy_token_slice(role_embed, i, 1, dim, &token_embed);
    set_pos(i, token_embed);
  }

  // Combined codec/text prefix.
  if (use_language_prefix) {
    set_pos(3, tts_pad_embed);
    add_pos(3, codec_think_embed);
    set_pos(4, tts_pad_embed);
    add_pos(4, codec_think_bos_embed);
    set_pos(5, tts_pad_embed);
    add_pos(5, codec_language_embed);
    set_pos(6, tts_pad_embed);
    add_pos(6, codec_think_eos_embed);
    set_pos(7, tts_bos_embed);
    add_pos(7, codec_pad_embed);
    set_pos(8, first_text_embed);
    add_pos(8, codec_bos_embed);
  } else {
    set_pos(3, tts_pad_embed);
    add_pos(3, codec_nothink_embed);
    set_pos(4, tts_pad_embed);
    add_pos(4, codec_think_bos_embed);
    set_pos(5, tts_pad_embed);
    add_pos(5, codec_think_eos_embed);
    set_pos(6, tts_bos_embed);
    add_pos(6, codec_pad_embed);
    set_pos(7, first_text_embed);
    add_pos(7, codec_bos_embed);
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
  const auto t_codegen = Clock::now();

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
  const double codegen_ms = ms_since(t_codegen);

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

  // 8. Flatten codes to [n_codes, num_code_groups] and decode audio.
  std::vector<int64_t> flat_codes(
      static_cast<size_t>(n_codes) * runner->num_code_groups_);
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
      flat_codes[t * runner->num_code_groups_ + g] = code;
    }
  }

  ET_LOG(Info, "Decoding %d codes to audio...", n_codes);
  const auto t_decode = Clock::now();
  if (!runner->run_decode_audio(
          flat_codes, n_codes, runner->num_code_groups_, waveform)) {
    return false;
  }
  const double decode_audio_ms = ms_since(t_decode);

  if (timing != nullptr) {
    timing->prompt_token_count = prompt_token_count;
    timing->generated_codec_steps = n_codes;
    timing->text_tokens_consumed = trailing_idx + kFirstTextTokenCount;
    timing->prompt_prep_ms = prompt_prep_ms;
    timing->talker_prefill_ms = talker_prefill_ms;
    timing->codegen_ms = codegen_ms;
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
