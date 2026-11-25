/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/lfm3_speech_generator.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

using executorch::extension::from_blob;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;

namespace example {

namespace {
constexpr std::array<int, 3> kMaskGitSchedule = {4, 4, 4};

Result<int32_t> sampling_int(
    executorch::extension::Module& module,
    const char* name) {
  ET_ASSIGN_OR_RETURN(value, module.get(name));
  ET_CHECK_OR_RETURN_ERROR(
      value.isInt() && value.toInt() >= 0 &&
          value.toInt() <= std::numeric_limits<int32_t>::max(),
      InvalidArgument,
      "Invalid speech sampling metadata %s",
      name);
  return static_cast<int32_t>(value.toInt());
}

Result<float> sampling_float(
    executorch::extension::Module& module,
    const char* name) {
  ET_ASSIGN_OR_RETURN(value, module.get(name));
  ET_CHECK_OR_RETURN_ERROR(
      value.isDouble() && std::isfinite(value.toDouble()),
      InvalidArgument,
      "Invalid speech sampling metadata %s",
      name);
  return static_cast<float>(value.toDouble());
}

Error load_sampling_policy(
    executorch::extension::Module& module,
    const std::string& prefix,
    speech::SamplingPolicy& policy,
    size_t& window) {
  ET_ASSIGN_OR_RETURN(
      temperature, sampling_float(module, (prefix + "temperature").c_str()));
  ET_ASSIGN_OR_RETURN(top_k, sampling_int(module, (prefix + "top_k").c_str()));
  ET_ASSIGN_OR_RETURN(
      penalty, sampling_float(module, (prefix + "repetition_penalty").c_str()));
  ET_ASSIGN_OR_RETURN(
      history_window,
      sampling_int(module, (prefix + "repetition_window").c_str()));
  policy.temperature = temperature;
  policy.top_k = top_k;
  policy.repetition_penalty = penalty;
  window = history_window;
  return Error::Ok;
}
} // namespace

Lfm3SpeechTokenGenerator::Lfm3SpeechTokenGenerator(
    tokenizers::Tokenizer* tokenizer,
    TokenEmbeddingProcessor* tok_embedding_runner,
    DecoderRunner* decoder_runner,
    KVManager* kv_manager,
    const std::string& method_name,
    std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
    Metadata metadata,
    executorch::llm::Stats* stats,
    std::unique_ptr<executorch::extension::MethodMeta> method_meta,
    executorch::extension::Module* depth,
    executorch::extension::Module* embed_frame,
    SpeechConfig speech_config)
    : MultimodalTokenGenerator(
          tokenizer,
          tok_embedding_runner,
          decoder_runner,
          kv_manager,
          method_name,
          std::move(eos_ids),
          metadata,
          stats,
          std::move(method_meta)),
      depth_(depth),
      embed_frame_(embed_frame),
      speech_config_(speech_config),
      rng_(speech_config.seed),
      hidden_f_(metadata.hidden_dim) {
  ET_CHECK_MSG(
      metadata.hidden_dim > 0, "Speech needs the decoder hidden state");
  ET_CHECK_MSG(metadata.ar_len == 1, "Speech generation expects an AR-1 graph");
}

Result<uint64_t> Lfm3SpeechTokenGenerator::special_token(const char* spelling) {
  auto encoded = tokenizer_->encode(spelling, 0, 0);
  ET_CHECK_TK_OK_OR_RETURN_ERROR(
      encoded.error(), "Failed to encode speech token %s", spelling);
  ET_CHECK_OR_RETURN_ERROR(
      encoded.get().size() == 1 && encoded.get()[0] < metadata_.vocab_size,
      InvalidArgument,
      "Speech token %s must be one in-vocabulary token",
      spelling);
  return encoded.get()[0];
}

Error Lfm3SpeechTokenGenerator::load_speech() {
  ET_CHECK_OR_RETURN_ERROR(
      speech_config_.codebook_size > 1 && speech_config_.audio_eos_code >= 0 &&
          speech_config_.audio_eos_code < speech_config_.codebook_size &&
          speech_config_.hidden_zero_point >= 0 &&
          speech_config_.hidden_zero_point <=
              std::numeric_limits<uint16_t>::max() &&
          std::isfinite(speech_config_.hidden_scale) &&
          speech_config_.hidden_scale > 0 &&
          std::isfinite(speech_config_.logits_scale) &&
          speech_config_.logits_scale > 0 &&
          speech_config_.logits_zero_point >= 0 &&
          speech_config_.logits_zero_point <=
              std::numeric_limits<uint16_t>::max(),
      InvalidArgument,
      "Invalid speech sampling or codebook contract");
  ET_ASSIGN_OR_RETURN(
      version, sampling_int(*depth_, "get_speech_sampling_version"));
  ET_CHECK_OR_RETURN_ERROR(
      version == 1, NotSupported, "Re-export speech sampling metadata");
  ET_CHECK_OK_OR_RETURN_ERROR(load_sampling_policy(
      *depth_, "get_text_", text_policy_, text_repetition_window_));
  ET_CHECK_OK_OR_RETURN_ERROR(load_sampling_policy(
      *depth_, "get_audio_", audio_policy_, audio_repetition_window_));
  ET_ASSIGN_OR_RETURN(top_p, sampling_float(*depth_, "get_text_top_p"));
  ET_ASSIGN_OR_RETURN(min_p, sampling_float(*depth_, "get_text_min_p"));
  text_policy_.top_p = top_p;
  text_policy_.min_p = min_p;
  audio_policy_ = speech::audio_overrides(
      audio_policy_,
      speech_config_.audio_temperature,
      speech_config_.audio_repetition_penalty);
  ET_CHECK_OR_RETURN_ERROR(
      text_policy_.valid() && audio_policy_.valid(),
      InvalidArgument,
      "Invalid text or audio sampling policy");
  ET_ASSIGN_OR_RETURN(audio_start, special_token("<|audio_start|>"));
  ET_ASSIGN_OR_RETURN(text_end, special_token("<|text_end|>"));
  ET_ASSIGN_OR_RETURN(mixed_start, special_token("<|mixed_start|>"));
  ET_ASSIGN_OR_RETURN(audio_out, special_token("<|audio_out|>"));
  audio_start_ = audio_start;
  text_end_ = text_end;
  mixed_start_ = mixed_start;
  audio_out_ = audio_out;
  const std::array<const char*, 4> spellings = {
      "<|video|>", "<|image|>", "<|audio|>", "<|audio_out|>"};
  for (size_t i = 0; i < spellings.size(); ++i) {
    ET_ASSIGN_OR_RETURN(token, special_token(spellings[i]));
    suppressed_[i] = token;
  }
  return Error::Ok;
}

void Lfm3SpeechTokenGenerator::set_prompt_outputs(
    const uint16_t* hidden,
    const uint16_t* logits) {
  dequantize_hidden(hidden);
  std::memcpy(logits_.data, logits, metadata_.vocab_size * sizeof(uint16_t));
}

void Lfm3SpeechTokenGenerator::dequantize_hidden(const uint16_t* hidden) {
  for (size_t i = 0; i < hidden_f_.size(); ++i) {
    hidden_f_[i] =
        (static_cast<int32_t>(hidden[i]) - speech_config_.hidden_zero_point) *
        speech_config_.hidden_scale;
  }
}

int32_t Lfm3SpeechTokenGenerator::sample_code(
    const float* logits,
    int codebook) {
  std::vector<float> scores(logits, logits + speech_config_.codebook_size);
  std::unordered_set<int32_t> seen;
  for (const Frame& frame : history_) {
    seen.insert(frame[codebook]);
  }
  speech::penalize(scores, seen, audio_policy_.repetition_penalty);
  return speech::sample_scores(std::move(scores), audio_policy_, rng_);
}

Result<Lfm3SpeechTokenGenerator::Frame>
Lfm3SpeechTokenGenerator::sample_frame() {
  Frame codes{};
  std::array<float, kNumCodebooks> masked;
  masked.fill(1.0f);
  std::array<int, kNumCodebooks> order;
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng_);

  auto hidden = from_blob(
      hidden_f_.data(),
      {1, static_cast<int>(hidden_f_.size())},
      executorch::aten::ScalarType::Float);
  auto codes_t = from_blob(
      codes.data(), {kNumCodebooks}, executorch::aten::ScalarType::Int);
  auto masked_t = from_blob(
      masked.data(), {kNumCodebooks}, executorch::aten::ScalarType::Float);
  size_t next = 0;
  for (int count : kMaskGitSchedule) {
    auto out =
        depth_->forward({EValue(hidden), EValue(codes_t), EValue(masked_t)});
    ET_CHECK_OK_OR_RETURN_ERROR(out.error());
    ET_CHECK_OR_RETURN_ERROR(
        out.get().size() == 1 && out.get()[0].isTensor(),
        InvalidArgument,
        "Depth decoder must return one tensor");
    const auto& depth_logits = out.get()[0].toTensor();
    ET_CHECK_OR_RETURN_ERROR(
        depth_logits.scalar_type() == executorch::aten::ScalarType::Float &&
            depth_logits.dim() == 2 && depth_logits.size(0) == kNumCodebooks &&
            depth_logits.size(1) == speech_config_.codebook_size,
        InvalidArgument,
        "Depth logits must be float32 [num_codebooks, codebook_size]");
    const float* logits = depth_logits.const_data_ptr<float>();
    ET_CHECK_OR_RETURN_ERROR(
        std::all_of(
            logits,
            logits + depth_logits.numel(),
            [](float value) { return std::isfinite(value); }),
        InvalidArgument,
        "Depth logits contain non-finite values");
    for (int j = 0; j < count; ++j, ++next) {
      const int codebook = order[next];
      codes[codebook] = sample_code(
          logits + codebook * speech_config_.codebook_size, codebook);
      ET_CHECK_OR_RETURN_ERROR(
          codes[codebook] >= 0 &&
              codes[codebook] < speech_config_.codebook_size,
          InvalidArgument,
          "Invalid code sampling distribution");
    }
    for (size_t j = next - count; j < next; ++j) {
      masked[order[j]] = 0.0f;
    }
  }
  return codes;
}

uint64_t Lfm3SpeechTokenGenerator::sample_text() {
  ET_CHECK_MSG(
      logits_.dtype == executorch::aten::ScalarType::UInt16,
      "Expected uint16 logits");
  const int32_t token = speech::sample_text_logits(
      reinterpret_cast<const uint16_t*>(logits_.data),
      metadata_.vocab_size,
      speech_config_.logits_scale,
      speech_config_.logits_zero_point,
      text_policy_,
      text_history_,
      text_repetition_window_,
      suppressed_,
      rng_);
  ET_CHECK_MSG(token >= 0, "Invalid text sampling distribution");
  return token;
}

Error Lfm3SpeechTokenGenerator::embed_frame(const Frame& frame) {
  Frame codes = frame;
  auto codes_t = from_blob(
      codes.data(), {kNumCodebooks}, executorch::aten::ScalarType::Int);
  auto out = embed_frame_->forward({EValue(codes_t)});
  ET_CHECK_OK_OR_RETURN_ERROR(out.error());
  ET_CHECK_OR_RETURN_ERROR(
      out.get().size() == 1 && out.get()[0].isTensor(),
      InvalidArgument,
      "Frame embedding must return one tensor");
  const auto& embedding = out.get()[0].toTensor();
  ET_CHECK_OR_RETURN_ERROR(
      embedding.scalar_type() == executorch::aten::ScalarType::Float &&
          embedding.dim() == 3 && embedding.size(0) == 1 &&
          embedding.size(1) == 1 &&
          embedding.size(2) == metadata_.embedding_dim &&
          embedding.nbytes() == input_embedding_.size,
      InvalidArgument,
      "Frame embedding must match the float32 AR-1 decoder input");
  std::memcpy(
      input_embedding_.data,
      embedding.const_data_ptr<float>(),
      input_embedding_.size);
  return Error::Ok;
}

void Lfm3SpeechTokenGenerator::embed_token(uint64_t token) {
  tok_embedding_runner_->prefill({token});
  const TensorStruct<float>& embeddings =
      tok_embedding_runner_->get_prompt_embeddings();
  std::memcpy(
      input_embedding_.data,
      embeddings.data,
      metadata_.embedding_dim * sizeof(float));
}

Error Lfm3SpeechTokenGenerator::step(int64_t pos) {
  *input_pos_.data = static_cast<int32_t>(pos);
  ET_CHECK_OK_OR_RETURN_ERROR(
      decoder_runner_->step(method_name_, inputs_).error());
  kv_manager_->update_cache(metadata_.ar_len, pos, metadata_.ar_len, {});
  if (conv_state_manager_ != nullptr) {
    conv_state_manager_->update(metadata_.ar_len);
  }
  kv_manager_->update_attention_mask(
      attention_mask_.data, metadata_.ar_len, pos, metadata_.ar_len);
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    kv_manager_->update_attention_mask(
        window_attention_mask_.data,
        metadata_.ar_len,
        pos,
        metadata_.ar_len,
        metadata_.sliding_window);
  }
  dequantize_hidden(hidden());
  return Error::Ok;
}

Result<int64_t> Lfm3SpeechTokenGenerator::generate(
    std::vector<uint64_t> tokens,
    int64_t start_pos,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    bool /*dump_logits*/,
    AttentionSinkRopeRunner* attention_sink_rope_runner) {
  ET_CHECK_MSG(
      attention_sink_rope_runner == nullptr && tokens.size() >= 2,
      "Speech generation needs the prompt and the prefill token");
  frames_.clear();
  history_.clear();
  text_history_.clear();
  int64_t pos = start_pos;

  kv_manager_->rearrange_cache(metadata_.ar_len);
  std::vector<int32_t> attention_map(metadata_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, metadata_.ar_len, pos);
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    kv_manager_->init_attention_mask(
        window_attention_mask_.data,
        attention_map,
        metadata_.ar_len,
        pos,
        metadata_.sliding_window);
  }
  ET_CHECK_MSG(
      decoder_runner_->set_outputs(method_name_, output_tensors_) == Error::Ok,
      "Failed to set output tensor for module %s",
      method_name_.c_str());

  const uint64_t last_prompt_token = tokens[tokens.size() - 2];
  speech::ModeState state;
  state.mode =
      (last_prompt_token == audio_start_ || last_prompt_token == audio_out_)
      ? speech::Mode::kAudio
      : last_prompt_token == mixed_start_ ? speech::Mode::kInterleaved
                                          : speech::Mode::kText;
  uint64_t prev_token = last_prompt_token;

  while (pos < seq_len - 1) {
    const bool interleaving = state.mode == speech::Mode::kInterleaved;
    bool finished = false;
    if (state.audio_slot()) {
      auto sampled = sample_frame();
      ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
      Frame codes = sampled.get();
      if (speech::normalize_audio_eos(codes, speech_config_.audio_eos_code)) {
        state.mode = speech::Mode::kText;
      } else {
        frames_.push_back(codes);
        speech::append_history(history_, codes, audio_repetition_window_);
      }
      ET_CHECK_OK_OR_RETURN_ERROR(embed_frame(codes));
    } else {
      const uint64_t token = sample_text();
      auto piece = tokenizer_->decode(prev_token, token);
      if (piece.ok() && token_callback) {
        token_callback(piece.get());
      }
      prev_token = token;
      finished = state.text_token(
          token,
          eos_ids_->count(token) > 0,
          text_end_,
          audio_start_,
          mixed_start_);
      embed_token(token);
    }
    state.advance(interleaving);
    if (finished) {
      break;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(step(pos));
    ++pos;
  }
  ET_LOG(Info, "Generated %zu speech frames", frames_.size());
  return pos - start_pos;
}

} // namespace example
