/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

#include <executorch/examples/models/gemma4/runner/gemma4_runner.h>

#include <executorch/backends/xnnpack/runtime/XNNPACKBackend.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <chrono>
#include <cstring>

namespace executorch::examples::gemma4 {

using ::executorch::extension::from_blob;
using ::executorch::extension::zeros;
using ::executorch::runtime::EValue;

Gemma4Runner::Gemma4Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    bool enable_workspace_sharing)
    : model_path_(model_path),
      tokenizer_path_(tokenizer_path),
      enable_workspace_sharing_(enable_workspace_sharing) {}

Error Gemma4Runner::load() {
  // Set XNNPACK workspace sharing explicitly. The compile-time default
  // varies across build configurations, and set_option state is process-
  // global, so always set it here to get the intended mode regardless of
  // how the binary was built or what other code in the process did first.
  {
    auto mode = enable_workspace_sharing_
        ? ::executorch::backends::xnnpack::WorkspaceSharingMode::PerModel
        : ::executorch::backends::xnnpack::WorkspaceSharingMode::Disabled;
    ::executorch::runtime::BackendOptions<2> xnnpack_opts;
    xnnpack_opts.set_option(
        ::executorch::backends::xnnpack::weight_cache_option_key,
        enable_workspace_sharing_);
    xnnpack_opts.set_option(
        ::executorch::backends::xnnpack::workspace_sharing_mode_option_key,
        static_cast<int>(mode));
    auto opts_status = ::executorch::runtime::set_option(
        ::executorch::backends::xnnpack::xnnpack_backend_key,
        xnnpack_opts.view());
    if (opts_status != Error::Ok) {
      ET_LOG(Error, "Failed to set XNNPACK options");
    }
  }

  ET_LOG(Info, "Loading model: %s", model_path_.c_str());
  module_ = std::make_unique<Module>(model_path_, Module::LoadMode::Mmap);

  auto err = module_->load_method("text_decoder");
  if (err != Error::Ok) {
    ET_LOG(Error, "Failed to load text_decoder method");
    return err;
  }

  ET_LOG(Info, "Loading tokenizer: %s", tokenizer_path_.c_str());
  tokenizer_ = executorch::extension::llm::load_tokenizer(tokenizer_path_);
  if (!tokenizer_) {
    ET_LOG(Error, "Failed to load tokenizer");
    return Error::InvalidArgument;
  }

  // Auto-detect embeds dtype and hidden_size from text decoder
  auto method_meta = module_->method_meta("text_decoder");
  if (method_meta.ok()) {
    auto tensor_meta = method_meta->input_tensor_meta(2);
    if (tensor_meta.ok()) {
      embeds_dtype_ = tensor_meta->scalar_type();
      if (tensor_meta->sizes().size() >= 3) {
        hidden_size_ = tensor_meta->sizes()[2];
      }
      ET_LOG(
          Info,
          "Auto-detected embeds dtype: %d, hidden_size: %d",
          (int)embeds_dtype_,
          (int)hidden_size_);
    }
  }

  loaded_ = true;
  ET_LOG(Info, "Model loaded successfully");
  return Error::Ok;
}

bool Gemma4Runner::is_loaded() const {
  return loaded_;
}

Error Gemma4Runner::load_audio_methods() {
  if (audio_loaded_) {
    return Error::Ok;
  }
  auto err = module_->load_method("speech_transform");
  if (err != Error::Ok) {
    ET_LOG(Info, "No speech_transform method — audio disabled");
    return err;
  }
  err = module_->load_method("audio_encoder");
  if (err != Error::Ok) {
    ET_LOG(Error, "Failed to load audio_encoder method");
    return err;
  }
  ET_LOG(Info, "Audio methods loaded");
  audio_loaded_ = true;
  return Error::Ok;
}

Error Gemma4Runner::load_vision_methods() {
  if (vision_loaded_) {
    return Error::Ok;
  }
  auto err = module_->load_method("vision_encoder");
  if (err != Error::Ok) {
    ET_LOG(Info, "No vision_encoder method — vision disabled");
    return err;
  }
  ET_LOG(Info, "Vision encoder loaded");
  vision_loaded_ = true;
  return Error::Ok;
}

void Gemma4Runner::reset() {
  if (module_) {
    ET_LOG(Info, "Resetting model (reloading text_decoder method)");
    auto err = module_->load_method("text_decoder");
    (void)err;
  }
}

// ---- Static helpers ----

int64_t Gemma4Runner::round_to_valid_frames(int64_t num_frames) {
  int64_t k = (num_frames + 25 + 47) / 48;
  if (k < 2)
    k = 2;
  if (k > 63)
    k = 63;
  return 48 * k - 25;
}

int64_t Gemma4Runner::compute_audio_num_tokens(int64_t num_samples) {
  int64_t padded = num_samples + kFrameLength / 2;
  int64_t mel_frames = (padded - (kFrameLength + 1)) / kHopLength + 1;
  int64_t after_conv1 = (mel_frames + 2 - 3) / 2 + 1;
  int64_t after_conv2 = (after_conv1 + 2 - 3) / 2 + 1;
  return std::min(after_conv2, kMaxAudioTokens);
}

float* Gemma4Runner::get_last_logits_as_float(
    const Tensor& logits,
    std::vector<float>& buf,
    int32_t vocab_size) {
  int64_t offset = logits.numel() - vocab_size;
  if (logits.scalar_type() == ScalarType::BFloat16) {
    buf.resize(vocab_size);
    auto* bf16 = reinterpret_cast<const uint16_t*>(logits.const_data_ptr());
    for (int32_t i = 0; i < vocab_size; ++i) {
      uint32_t bits = static_cast<uint32_t>(bf16[offset + i]) << 16;
      std::memcpy(&buf[i], &bits, sizeof(float));
    }
    return buf.data();
  }
  return const_cast<float*>(
      reinterpret_cast<const float*>(logits.const_data_ptr()) + offset);
}

// ---- Input building ----

std::vector<int64_t> Gemma4Runner::build_input_ids(
    const std::string& prompt,
    int64_t num_audio_tokens) {
  auto user_res = tokenizer_->encode("user\n", /*add_bos=*/0, /*add_eos=*/0);
  auto prompt_res = tokenizer_->encode(prompt, /*add_bos=*/0, /*add_eos=*/0);
  auto newline_res = tokenizer_->encode("\n", /*add_bos=*/0, /*add_eos=*/0);
  auto model_res = tokenizer_->encode("model\n", /*add_bos=*/0, /*add_eos=*/0);

  const auto& user_tokens = user_res.get();
  const auto& prompt_tokens = prompt_res.get();
  const auto& newline_tokens = newline_res.get();
  const auto& model_tokens = model_res.get();

  std::vector<int64_t> ids;
  ids.push_back(kBosId);
  ids.push_back(kTurnStartId);
  for (auto t : user_tokens)
    ids.push_back(static_cast<int64_t>(t));
  for (int64_t i = 0; i < num_audio_tokens; ++i)
    ids.push_back(kAudioTokenId);
  for (auto t : prompt_tokens)
    ids.push_back(static_cast<int64_t>(t));
  ids.push_back(kTurnEndId);
  for (auto t : newline_tokens)
    ids.push_back(static_cast<int64_t>(t));
  ids.push_back(kTurnStartId);
  for (auto t : model_tokens)
    ids.push_back(static_cast<int64_t>(t));

  return ids;
}

std::vector<int64_t> Gemma4Runner::build_text_input_ids(
    const std::string& prompt) {
  return build_input_ids(prompt, /*num_audio_tokens=*/0);
}

std::vector<int64_t> Gemma4Runner::build_vision_input_ids(
    const std::string& prompt,
    int64_t num_vision_tokens) {
  auto user_res = tokenizer_->encode("user\n", /*add_bos=*/0, /*add_eos=*/0);
  auto prompt_res = tokenizer_->encode(prompt, /*add_bos=*/0, /*add_eos=*/0);
  auto newline_res = tokenizer_->encode("\n", /*add_bos=*/0, /*add_eos=*/0);
  auto model_res = tokenizer_->encode("model\n", /*add_bos=*/0, /*add_eos=*/0);

  std::vector<int64_t> ids;
  ids.push_back(kBosId);
  ids.push_back(kTurnStartId);
  for (auto t : user_res.get())
    ids.push_back(static_cast<int64_t>(t));
  ids.push_back(kBoiTokenId);
  for (int64_t i = 0; i < num_vision_tokens; ++i)
    ids.push_back(kImageTokenId);
  ids.push_back(kEoiTokenId);
  for (auto t : prompt_res.get())
    ids.push_back(static_cast<int64_t>(t));
  ids.push_back(kTurnEndId);
  for (auto t : newline_res.get())
    ids.push_back(static_cast<int64_t>(t));
  ids.push_back(kTurnStartId);
  for (auto t : model_res.get())
    ids.push_back(static_cast<int64_t>(t));

  return ids;
}

TensorPtr Gemma4Runner::build_inputs_embeds(
    const std::vector<int64_t>& input_ids,
    const Tensor& media_embeddings,
    int64_t num_media_tokens,
    int64_t placeholder_token_id) {
  int64_t seq_len = static_cast<int64_t>(input_ids.size());
  auto embeds = zeros(
      {1, static_cast<int32_t>(seq_len), static_cast<int32_t>(hidden_size_)},
      embeds_dtype_);

  float* media_data_ptr = const_cast<float*>(
      reinterpret_cast<const float*>(media_embeddings.const_data_ptr()));
  int64_t media_idx = 0;

  for (int64_t i = 0; i < seq_len; ++i) {
    if (input_ids[i] == placeholder_token_id && media_idx < num_media_tokens) {
      const float* src = media_data_ptr + media_idx * hidden_size_;
      if (embeds_dtype_ == ScalarType::BFloat16) {
        auto* dst = embeds->mutable_data_ptr<executorch::aten::BFloat16>() +
            i * hidden_size_;
        for (int64_t j = 0; j < hidden_size_; ++j) {
          dst[j] = executorch::aten::BFloat16(src[j]);
        }
      } else {
        std::memcpy(
            embeds->mutable_data_ptr<float>() + i * hidden_size_,
            src,
            hidden_size_ * sizeof(float));
      }
      ++media_idx;
    }
  }

  return embeds;
}

// ---- Decode loop ----

Result<std::string> Gemma4Runner::decode_loop(
    const Tensor& prefill_logits,
    int64_t seq_len,
    const GenerationConfig& config,
    const std::function<void(const std::string&)>& token_callback,
    Gemma4Stats* stats) {
  int32_t vocab_size =
      static_cast<int32_t>(prefill_logits.size(prefill_logits.dim() - 1));

  executorch::extension::llm::Sampler sampler(
      vocab_size,
      config.temperature,
      config.topp,
      std::chrono::system_clock::now().time_since_epoch().count());

  std::vector<float> logits_f32_buf;
  float* last_logits =
      get_last_logits_as_float(prefill_logits, logits_f32_buf, vocab_size);
  int64_t next_token = sampler.sample(last_logits);

  if (stats) {
    stats->on_generation_begin();
  }

  if (stats) {
    stats->rss_before_gen_kb = Gemma4Stats::read_rss_kb();
    stats->rss_peak_gen_kb = stats->rss_before_gen_kb;
  }

  int32_t num_generated = 0;
  std::string result_text;

  auto is_stop = [&config](int64_t t) {
    for (auto s : config.stop_tokens) {
      if (t == s)
        return true;
    }
    return false;
  };

  if (!is_stop(next_token)) {
    ++num_generated;
    auto decode_res = tokenizer_->decode(
        static_cast<uint64_t>(kBosId), static_cast<uint64_t>(next_token));
    if (decode_res.ok()) {
      result_text += decode_res.get();
      if (token_callback) {
        token_callback(decode_res.get());
      }
    }

    TensorPtr decode_embeds =
        zeros({1, 1, static_cast<int32_t>(hidden_size_)}, embeds_dtype_);

    for (int32_t step = 0; step < config.max_new_tokens - 1; ++step) {
      int64_t current_token_val = next_token;
      TensorPtr current_ids =
          from_blob(&current_token_val, {1, 1}, ScalarType::Long);

      int64_t current_pos_val = seq_len + step;
      TensorPtr current_pos =
          from_blob(&current_pos_val, {1}, ScalarType::Long);

      auto step_res_result = module_->execute(
          "text_decoder",
          {EValue(current_ids), EValue(current_pos), EValue(decode_embeds)});
      if (!step_res_result.ok()) {
        return step_res_result.error();
      }
      auto step_res = std::move(step_res_result.get());

      Tensor step_logits = step_res[0].toTensor();
      float* step_logits_ptr =
          get_last_logits_as_float(step_logits, logits_f32_buf, vocab_size);

      int64_t prev_token = next_token;
      next_token = sampler.sample(step_logits_ptr);

      if (stats) {
        int64_t cur_rss = Gemma4Stats::read_rss_kb();
        if (cur_rss > stats->rss_peak_gen_kb) {
          stats->rss_peak_gen_kb = cur_rss;
        }
      }

      if (is_stop(next_token)) {
        break;
      }

      ++num_generated;

      auto step_decode_res = tokenizer_->decode(
          static_cast<uint64_t>(prev_token), static_cast<uint64_t>(next_token));
      if (step_decode_res.ok()) {
        result_text += step_decode_res.get();
        if (token_callback) {
          token_callback(step_decode_res.get());
        }
      }
    }
  }

  if (stats) {
    stats->on_generation_end();
    stats->num_generated_tokens = num_generated;
    stats->rss_after_gen_kb = Gemma4Stats::read_rss_kb();
  }

  return result_text;
}

// ---- Public generate methods ----

Result<std::string> Gemma4Runner::generate(
    const TensorPtr& waveform,
    int64_t actual_samples,
    const std::string& prompt,
    const GenerationConfig& config,
    const std::function<void(const std::string&)>& token_callback,
    Gemma4Stats* stats) {
  if (!loaded_) {
    return Error::InvalidState;
  }

  auto err = load_audio_methods();
  if (err != Error::Ok) {
    ET_LOG(Error, "Audio methods not available");
    return err;
  }

  if (stats) {
    stats->audio_duration_ms =
        actual_samples / static_cast<double>(kSampleRate) * 1000.0;
  }

  // Step 1: Speech transform
  if (stats)
    stats->on_speech_transform_begin();
  auto transform_result =
      module_->execute("speech_transform", {EValue(waveform)});
  if (!transform_result.ok()) {
    ET_LOG(Error, "speech_transform failed");
    return transform_result.error();
  }
  auto transform_res = std::move(transform_result.get());
  if (stats)
    stats->on_speech_transform_end();

  Tensor mel_raw = transform_res[0].toTensor();
  int64_t num_frames_raw = mel_raw.size(0);
  int64_t n_mels = mel_raw.size(1);

  // Step 2: Pad mel frames to valid conformer count (48*k - 25)
  int64_t target_frames = round_to_valid_frames(num_frames_raw);
  auto mel_buf = std::make_unique<float[]>(1 * target_frames * n_mels);
  int64_t frames_to_copy = std::min(num_frames_raw, target_frames);
  float* mel_src = mel_raw.mutable_data_ptr<float>();

  std::memcpy(mel_buf.get(), mel_src, frames_to_copy * n_mels * sizeof(float));
  if (frames_to_copy < target_frames) {
    std::memset(
        mel_buf.get() + frames_to_copy * n_mels,
        0,
        (target_frames - frames_to_copy) * n_mels * sizeof(float));
  }

  TensorPtr mel_tensor = from_blob(
      mel_buf.get(),
      {1, static_cast<int32_t>(target_frames), static_cast<int32_t>(n_mels)},
      ScalarType::Float);

  // Step 3: Build mel mask matching HF's _extract_spectrogram frame-end logic
  const int64_t pad_left = kFrameLength / 2;
  const int64_t frame_size =
      kFrameLength + 1; // unfold window (extra sample for preemphasis)

  auto mask_buf = std::make_unique<bool[]>(1 * target_frames);
  const int64_t real_end = pad_left + actual_samples;
  for (int64_t i = 0; i < target_frames; ++i) {
    const int64_t frame_end = i * kHopLength + frame_size - 1;
    mask_buf[i] = (i < num_frames_raw) && (frame_end < real_end);
  }
  TensorPtr mel_mask = from_blob(
      mask_buf.get(),
      {1, static_cast<int32_t>(target_frames)},
      ScalarType::Bool);

  // Step 4: Audio encoder
  if (stats)
    stats->on_audio_encode_begin();
  auto encode_result =
      module_->execute("audio_encoder", {EValue(mel_tensor), EValue(mel_mask)});
  if (!encode_result.ok()) {
    ET_LOG(Error, "audio_encoder failed");
    return encode_result.error();
  }
  auto encode_res = std::move(encode_result.get());
  if (stats)
    stats->on_audio_encode_end();

  Tensor audio_embeddings = encode_res[0].toTensor();
  int64_t encoder_tokens = audio_embeddings.size(1);

  int64_t num_audio_tokens = 0;
  if (encode_res.size() > 1) {
    Tensor output_mask = encode_res[1].toTensor();
    auto* mask_data = output_mask.const_data_ptr<bool>();
    for (int64_t i = 0; i < encoder_tokens; ++i) {
      if (mask_data[i])
        ++num_audio_tokens;
    }
  } else {
    num_audio_tokens =
        std::min(compute_audio_num_tokens(actual_samples), encoder_tokens);
  }

  // Step 5: Build input_ids and inputs_embeds
  auto input_ids = build_input_ids(prompt, num_audio_tokens);
  auto inputs_embeds = build_inputs_embeds(
      input_ids, audio_embeddings, num_audio_tokens, kAudioTokenId);

  int64_t seq_len = static_cast<int64_t>(input_ids.size());
  if (stats)
    stats->num_prompt_tokens = static_cast<int32_t>(seq_len);

  // Step 6: Prefill
  TensorPtr input_ids_tensor = from_blob(
      input_ids.data(), {1, static_cast<int32_t>(seq_len)}, ScalarType::Long);

  std::vector<int64_t> positions(seq_len);
  for (int64_t i = 0; i < seq_len; ++i)
    positions[i] = i;
  TensorPtr input_pos = from_blob(
      positions.data(), {static_cast<int32_t>(seq_len)}, ScalarType::Long);

  if (stats)
    stats->on_prefill_begin();
  auto prefill_result = module_->execute(
      "text_decoder",
      {EValue(input_ids_tensor), EValue(input_pos), EValue(inputs_embeds)});
  if (!prefill_result.ok()) {
    ET_LOG(Error, "text_decoder prefill failed");
    return prefill_result.error();
  }
  auto prefill_res = std::move(prefill_result.get());
  if (stats)
    stats->on_prefill_end();

  return decode_loop(
      prefill_res[0].toTensor(), seq_len, config, token_callback, stats);
}

Result<std::string> Gemma4Runner::generate(
    const TensorPtr& waveform,
    int64_t actual_samples,
    const std::string& prompt,
    int32_t max_new_tokens,
    float temperature,
    const std::function<void(const std::string&)>& token_callback,
    Gemma4Stats* stats) {
  GenerationConfig config;
  config.max_new_tokens = max_new_tokens;
  config.temperature = temperature;
  return generate(
      waveform, actual_samples, prompt, config, token_callback, stats);
}

Result<std::string> Gemma4Runner::generate_text(
    const std::string& prompt,
    const GenerationConfig& config,
    const std::function<void(const std::string&)>& token_callback,
    Gemma4Stats* stats) {
  if (!loaded_) {
    return Error::InvalidState;
  }

  auto input_ids = build_text_input_ids(prompt);
  int64_t seq_len = static_cast<int64_t>(input_ids.size());
  if (stats)
    stats->num_prompt_tokens = static_cast<int32_t>(seq_len);

  TensorPtr input_ids_tensor = from_blob(
      input_ids.data(), {1, static_cast<int32_t>(seq_len)}, ScalarType::Long);

  std::vector<int64_t> positions(seq_len);
  for (int64_t i = 0; i < seq_len; ++i)
    positions[i] = i;
  TensorPtr input_pos = from_blob(
      positions.data(), {static_cast<int32_t>(seq_len)}, ScalarType::Long);

  auto inputs_embeds = zeros(
      {1, static_cast<int32_t>(seq_len), static_cast<int32_t>(hidden_size_)},
      embeds_dtype_);

  if (stats)
    stats->on_prefill_begin();
  auto prefill_result = module_->execute(
      "text_decoder",
      {EValue(input_ids_tensor), EValue(input_pos), EValue(inputs_embeds)});
  if (!prefill_result.ok()) {
    ET_LOG(Error, "text_decoder prefill failed");
    return prefill_result.error();
  }
  auto prefill_res = std::move(prefill_result.get());
  if (stats)
    stats->on_prefill_end();

  return decode_loop(
      prefill_res[0].toTensor(), seq_len, config, token_callback, stats);
}

Result<std::string> Gemma4Runner::generate_text(
    const std::string& prompt,
    int32_t max_new_tokens,
    float temperature,
    const std::function<void(const std::string&)>& token_callback,
    Gemma4Stats* stats) {
  GenerationConfig config;
  config.max_new_tokens = max_new_tokens;
  config.temperature = temperature;
  return generate_text(prompt, config, token_callback, stats);
}

Result<std::string> Gemma4Runner::generate_vision(
    const TensorPtr& pixel_values,
    const TensorPtr& pixel_position_ids,
    const std::string& prompt,
    const GenerationConfig& config,
    const std::function<void(const std::string&)>& token_callback,
    Gemma4Stats* stats) {
  if (!loaded_) {
    return Error::InvalidState;
  }

  auto err = load_vision_methods();
  if (err != Error::Ok) {
    ET_LOG(Error, "Vision methods not available");
    return err;
  }

  if (stats)
    stats->on_vision_encode_begin();
  auto ve_result = module_->execute(
      "vision_encoder", {EValue(pixel_values), EValue(pixel_position_ids)});
  if (!ve_result.ok()) {
    ET_LOG(Error, "vision_encoder execution failed");
    return ve_result.error();
  }
  auto ve_res = std::move(ve_result.get());
  if (stats)
    stats->on_vision_encode_end();

  Tensor vision_embeddings = ve_res[0].toTensor();
  int64_t encoder_tokens = vision_embeddings.size(1);

  int64_t num_vision_tokens = encoder_tokens;
  if (ve_res.size() > 1 && ve_res[1].isTensor()) {
    Tensor output_mask = ve_res[1].toTensor();
    const bool* mask_data = output_mask.const_data_ptr<bool>();
    num_vision_tokens = 0;
    for (int64_t i = 0; i < output_mask.numel(); ++i) {
      if (mask_data[i])
        ++num_vision_tokens;
    }
  }

  auto input_ids = build_vision_input_ids(prompt, num_vision_tokens);
  int64_t seq_len = static_cast<int64_t>(input_ids.size());
  if (stats)
    stats->num_prompt_tokens = static_cast<int32_t>(seq_len);

  TensorPtr input_ids_tensor = from_blob(
      input_ids.data(), {1, static_cast<int32_t>(seq_len)}, ScalarType::Long);

  std::vector<int64_t> positions(seq_len);
  for (int64_t i = 0; i < seq_len; ++i)
    positions[i] = i;
  TensorPtr input_pos = from_blob(
      positions.data(), {static_cast<int32_t>(seq_len)}, ScalarType::Long);

  auto inputs_embeds = build_inputs_embeds(
      input_ids, vision_embeddings, num_vision_tokens, kImageTokenId);

  if (stats)
    stats->on_prefill_begin();
  auto prefill_result = module_->execute(
      "text_decoder",
      {EValue(input_ids_tensor), EValue(input_pos), EValue(inputs_embeds)});
  if (!prefill_result.ok()) {
    ET_LOG(Error, "text_decoder prefill failed");
    return prefill_result.error();
  }
  auto prefill_res = std::move(prefill_result.get());
  if (stats)
    stats->on_prefill_end();

  return decode_loop(
      prefill_res[0].toTensor(), seq_len, config, token_callback, stats);
}

Result<std::string> Gemma4Runner::generate_vision(
    const TensorPtr& pixel_values,
    const TensorPtr& pixel_position_ids,
    const std::string& prompt,
    int32_t max_new_tokens,
    float temperature,
    const std::function<void(const std::string&)>& token_callback,
    Gemma4Stats* stats) {
  GenerationConfig config;
  config.max_new_tokens = max_new_tokens;
  config.temperature = temperature;
  return generate_vision(
      pixel_values, pixel_position_ids, prompt, config, token_callback, stats);
}

} // namespace executorch::examples::gemma4
