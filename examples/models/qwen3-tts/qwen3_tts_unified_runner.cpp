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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

namespace qwen3_tts {
namespace {

using ::executorch::extension::from_blob;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

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

} // namespace

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

  ET_LOG(
      Info,
      "Unified runner: sample_rate=%d max_seq_len=%d talker_dim=%d "
      "num_code_groups=%d",
      output_sample_rate_,
      max_seq_len_,
      talker_dim_,
      num_code_groups_);
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
    int top_k) {
  if (temperature <= 0.0f || temperature < 1e-6f) {
    // Greedy: argmax.
    return static_cast<int64_t>(
        std::max_element(logits.begin(), logits.begin() + vocab_size) -
        logits.begin());
  }

  // Apply temperature.
  std::vector<float> scaled(logits.begin(), logits.begin() + vocab_size);
  for (auto& v : scaled) {
    v /= temperature;
  }

  // Softmax.
  float max_val = *std::max_element(scaled.begin(), scaled.end());
  float sum = 0.0f;
  for (auto& v : scaled) {
    v = std::exp(v - max_val);
    sum += v;
  }
  for (auto& v : scaled) {
    v /= sum;
  }

  // Top-k filtering.
  if (top_k > 0 && top_k < vocab_size) {
    std::vector<std::pair<float, int>> indexed(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
      indexed[i] = {scaled[i], i};
    }
    std::partial_sort(
        indexed.begin(),
        indexed.begin() + top_k,
        indexed.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<float> topk_probs(top_k);
    std::vector<int> topk_indices(top_k);
    float topk_sum = 0.0f;
    for (int i = 0; i < top_k; ++i) {
      topk_probs[i] = indexed[i].first;
      topk_indices[i] = indexed[i].second;
      topk_sum += topk_probs[i];
    }
    for (auto& p : topk_probs) {
      p /= topk_sum;
    }

    // Sample.
    static std::mt19937 gen(42);
    std::discrete_distribution<int> dist(topk_probs.begin(), topk_probs.end());
    return static_cast<int64_t>(topk_indices[dist(gen)]);
  }

  // Sample from full distribution.
  static std::mt19937 gen(42);
  std::discrete_distribution<int> dist(scaled.begin(), scaled.end());
  return static_cast<int64_t>(dist(gen));
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

bool Qwen3TTSUnifiedRunner::synthesize(
    const std::string& text,
    const std::string& language,
    const SynthesizeConfig& config,
    std::vector<float>* waveform) {
  // TODO: Integrate tiktoken tokenizer for text tokenization.
  // For now, this method demonstrates the pipeline using the .pte methods.
  // The full pipeline requires:
  // 1. Tokenize text (tiktoken C++)
  // 2. encode_text(token_ids) -> projected text embeddings
  // 3. Assemble composite prefill (codec control tokens + projected text)
  // 4. talker(prefill) -> logits, hidden
  // 5. Autoregressive loop:
  //    a. Sample code_0 from logits
  //    b. codec_embed(code_0, group=0) -> main embed
  //    c. code_predictor(prefill=[hidden, main_embed])
  //    d. For i in 1..15: cp_head -> sample code_i, codec_embed -> embed,
  //       code_predictor(step)
  //    e. Sum all 16 embeddings + next text embed -> next input
  //    f. talker(decode_step) -> next logits, hidden
  // 6. decode_audio(accumulated codes) -> waveform

  ET_LOG(
      Error,
      "Full text-to-audio synthesis not yet implemented. "
      "Use --codes_path with precomputed codes for now.");
  return false;
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
