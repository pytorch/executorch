/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

DEFINE_string(model_path, "parakeet.pte", "Path to Parakeet model (.pte).");
DEFINE_string(audio_path, "", "Path to input audio file (.wav).");
DEFINE_string(
    tokenizer_path,
    "tokenizer.json",
    "Path to SentencePiece tokenizer model file.");
DEFINE_string(
    data_path,
    "",
    "Path to data file (.ptd) for delegate data (optional, required for CUDA).");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace {

// TDT duration values
const std::vector<int> DURATIONS = {0, 1, 2, 3, 4};

// Struct to hold decode timing results
struct DecodeTimingResult {
  std::vector<int64_t> hypothesis;
  long long decoder_ms;
  long long joint_ms;
  long long joint_project_encoder_ms;
  long long joint_project_decoder_ms;
};

DecodeTimingResult greedy_decode_executorch(
    Module& model,
    const ::executorch::aten::Tensor& encoder_output,
    int64_t encoder_len,
    int64_t blank_id,
    int64_t vocab_size,
    int64_t num_rnn_layers = 2,
    int64_t pred_hidden = 640,
    int64_t max_symbols_per_step = 10) {
  std::vector<int64_t> hypothesis;
  int64_t num_token_classes = vocab_size + 1;

  // Timing accumulators
  long long decoder_total_ms = 0;
  long long joint_total_ms = 0;
  long long joint_project_encoder_ms = 0;
  long long joint_project_decoder_ms = 0;

  // Transpose encoder output from [1, enc_dim, time] to [1, time, enc_dim]
  auto enc_sizes = encoder_output.sizes();
  int64_t batch = enc_sizes[0];
  int64_t enc_dim = enc_sizes[1];
  int64_t time_steps = enc_sizes[2];

  // Create transposed tensor
  std::vector<float> transposed_data(batch * time_steps * enc_dim);
  const float* src = encoder_output.const_data_ptr<float>();
  for (int64_t t = 0; t < time_steps; t++) {
    for (int64_t d = 0; d < enc_dim; d++) {
      transposed_data[t * enc_dim + d] = src[d * time_steps + t];
    }
  }

  auto transposed_tensor = from_blob(
      transposed_data.data(),
      {static_cast<::executorch::aten::SizesType>(batch),
       static_cast<::executorch::aten::SizesType>(time_steps),
       static_cast<::executorch::aten::SizesType>(enc_dim)},
      ::executorch::aten::ScalarType::Float);

  // Project encoder output
  auto proj_enc_start = std::chrono::high_resolution_clock::now();
  auto proj_enc_result = model.execute(
      "joint_project_encoder",
      std::vector<::executorch::runtime::EValue>{transposed_tensor});
  auto proj_enc_end = std::chrono::high_resolution_clock::now();
  if (!proj_enc_result.ok()) {
    ET_LOG(Error, "joint_project_encoder failed");
    return DecodeTimingResult{hypothesis, 0, 0, 0, 0};
  }
  auto f_proj = proj_enc_result.get()[0].toTensor();
  joint_project_encoder_ms +=
      std::chrono::duration_cast<std::chrono::milliseconds>(
          proj_enc_end - proj_enc_start)
          .count();

  // Initialize LSTM state
  std::vector<float> h_data(num_rnn_layers * 1 * pred_hidden, 0.0f);
  std::vector<float> c_data(num_rnn_layers * 1 * pred_hidden, 0.0f);

  auto h = from_blob(
      h_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      ::executorch::aten::ScalarType::Float);
  auto c = from_blob(
      c_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      ::executorch::aten::ScalarType::Float);

  // Initialize decoder state with zeros
  std::vector<float> sos_g_data(1 * 1 * pred_hidden, 0.0f);
  auto sos_g = from_blob(
      sos_g_data.data(),
      {1, 1, static_cast<::executorch::aten::SizesType>(pred_hidden)},
      ::executorch::aten::ScalarType::Float);

  auto proj_dec_init_start = std::chrono::high_resolution_clock::now();
  auto g_proj_result = model.execute(
      "joint_project_decoder",
      std::vector<::executorch::runtime::EValue>{sos_g});
  auto proj_dec_init_end = std::chrono::high_resolution_clock::now();
  if (!g_proj_result.ok()) {
    ET_LOG(Error, "joint_project_decoder failed");
    return DecodeTimingResult{hypothesis, 0, 0, 0, 0};
  }
  auto g_proj_tensor = g_proj_result.get()[0].toTensor();
  joint_project_decoder_ms +=
      std::chrono::duration_cast<std::chrono::milliseconds>(
          proj_dec_init_end - proj_dec_init_start)
          .count();

  // Copy g_proj data for reuse
  std::vector<float> g_proj_data(
      g_proj_tensor.const_data_ptr<float>(),
      g_proj_tensor.const_data_ptr<float>() + g_proj_tensor.numel());

  int64_t t = 0;
  int64_t symbols_on_frame = 0;

  // Scan over encoder output
  while (t < encoder_len) {
    // Get encoder frame at time t: f_proj[:, t:t+1, :]
    const float* f_proj_data = f_proj.const_data_ptr<float>();
    int64_t proj_dim = f_proj.sizes()[2];

    std::vector<float> f_t_data(1 * 1 * proj_dim);
    for (int64_t d = 0; d < proj_dim; d++) {
      f_t_data[d] = f_proj_data[t * proj_dim + d];
    }
    auto f_t = from_blob(
        f_t_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        ::executorch::aten::ScalarType::Float);

    auto g_proj = from_blob(
        g_proj_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        ::executorch::aten::ScalarType::Float);

    // Joint network
    auto joint_start = std::chrono::high_resolution_clock::now();
    auto joint_result = model.execute(
        "joint", std::vector<::executorch::runtime::EValue>{f_t, g_proj});
    auto joint_end = std::chrono::high_resolution_clock::now();
    if (!joint_result.ok()) {
      ET_LOG(Error, "joint failed at t=%lld", static_cast<long long>(t));
      return DecodeTimingResult{hypothesis, decoder_total_ms, joint_total_ms, joint_project_encoder_ms, joint_project_decoder_ms};
    }
    auto full_logits = joint_result.get()[0].toTensor();
    joint_total_ms +=
        std::chrono::duration_cast<std::chrono::milliseconds>(
            joint_end - joint_start)
            .count();

    // Split logits into token and duration
    const float* logits_data = full_logits.const_data_ptr<float>();

    // Find argmax for token logits
    int64_t k = 0;
    float max_token_logit = logits_data[0];
    for (int64_t i = 1; i < num_token_classes; i++) {
      if (logits_data[i] > max_token_logit) {
        max_token_logit = logits_data[i];
        k = i;
      }
    }

    // Find argmax for duration logits
    int64_t dur_idx = 0;
    float max_dur_logit = logits_data[num_token_classes];
    for (size_t i = 1; i < DURATIONS.size(); i++) {
      if (logits_data[num_token_classes + i] > max_dur_logit) {
        max_dur_logit = logits_data[num_token_classes + i];
        dur_idx = i;
      }
    }
    int64_t dur = DURATIONS[dur_idx];

    if (k == blank_id) {
      t += std::max(dur, (int64_t)1);
      symbols_on_frame = 0;
    } else {
      hypothesis.push_back(k);

      // Update decoder state
      std::vector<int64_t> token_data = {k};
      auto token = from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      auto decoder_start = std::chrono::high_resolution_clock::now();
      auto decoder_result = model.execute(
          "decoder_predict",
          std::vector<::executorch::runtime::EValue>{token, h, c});
      auto decoder_end = std::chrono::high_resolution_clock::now();
      if (!decoder_result.ok()) {
        ET_LOG(Error, "decoder_predict failed");
        return DecodeTimingResult{hypothesis, decoder_total_ms, joint_total_ms, joint_project_encoder_ms, joint_project_decoder_ms};
      }
      auto& outputs = decoder_result.get();
      auto g = outputs[0].toTensor();
      auto new_h = outputs[1].toTensor();
      auto new_c = outputs[2].toTensor();
      decoder_total_ms +=
          std::chrono::duration_cast<std::chrono::milliseconds>(
              decoder_end - decoder_start)
              .count();

      // Update h and c
      std::memcpy(
          h_data.data(),
          new_h.const_data_ptr<float>(),
          h_data.size() * sizeof(float));
      std::memcpy(
          c_data.data(),
          new_c.const_data_ptr<float>(),
          c_data.size() * sizeof(float));

      // Project decoder output
      auto proj_dec_start = std::chrono::high_resolution_clock::now();
      auto proj_dec_result = model.execute(
          "joint_project_decoder",
          std::vector<::executorch::runtime::EValue>{g});
      auto proj_dec_end = std::chrono::high_resolution_clock::now();
      if (!proj_dec_result.ok()) {
        ET_LOG(Error, "joint_project_decoder failed");
        return DecodeTimingResult{hypothesis, decoder_total_ms, joint_total_ms, joint_project_encoder_ms, joint_project_decoder_ms};
      }
      auto new_g_proj = proj_dec_result.get()[0].toTensor();
      std::memcpy(
          g_proj_data.data(),
          new_g_proj.const_data_ptr<float>(),
          g_proj_data.size() * sizeof(float));
      joint_project_decoder_ms +=
          std::chrono::duration_cast<std::chrono::milliseconds>(
              proj_dec_end - proj_dec_start)
              .count();

      t += dur;

      if (dur == 0) {
        symbols_on_frame++;
        if (symbols_on_frame >= max_symbols_per_step) {
          t++;
          symbols_on_frame = 0;
        }
      } else {
        symbols_on_frame = 0;
      }
    }
  }

  return DecodeTimingResult{
      hypothesis,
      decoder_total_ms,
      joint_total_ms,
      joint_project_encoder_ms,
      joint_project_decoder_ms};
}

std::string tokens_to_text(
    const std::vector<int64_t>& tokens,
    tokenizers::Tokenizer* tokenizer) {
  // Decode tokens to text one by one
  std::string result;
  uint64_t prev_token = 0;
  for (size_t i = 0; i < tokens.size(); i++) {
    uint64_t token = static_cast<uint64_t>(tokens[i]);
    auto decode_result = tokenizer->decode(prev_token, token);
    if (decode_result.ok()) {
      result += decode_result.get();
    }
    prev_token = token;
  }

  return result;
}

} // namespace

int main(int argc, char** argv) {
  auto binary_start = std::chrono::high_resolution_clock::now();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "audio_path flag must be provided.");
    return 1;
  }

  // Load model (which includes the bundled preprocessor)
  ET_LOG(Info, "Loading model from: %s", FLAGS_model_path.c_str());
  std::unique_ptr<Module> model;
  if (!FLAGS_data_path.empty()) {
    ET_LOG(Info, "Loading data from: %s", FLAGS_data_path.c_str());
    model = std::make_unique<Module>(
        FLAGS_model_path, FLAGS_data_path, Module::LoadMode::Mmap);
  } else {
    model = std::make_unique<Module>(FLAGS_model_path, Module::LoadMode::Mmap);
  }
  auto model_load_error = model->load();
  if (model_load_error != Error::Ok) {
    ET_LOG(Error, "Failed to load model.");
    return 1;
  }

  // Load audio
  ET_LOG(Info, "Loading audio from: %s", FLAGS_audio_path.c_str());
  std::vector<float> audio_data =
      ::executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
  ET_LOG(Info, "Loaded %zu audio samples", audio_data.size());

  auto audio_tensor = from_blob(
      audio_data.data(),
      {static_cast<::executorch::aten::SizesType>(audio_data.size())},
      ::executorch::aten::ScalarType::Float);
  std::vector<int64_t> audio_len_data = {
      static_cast<int64_t>(audio_data.size())};
  auto audio_len_tensor = from_blob(
      audio_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  ET_LOG(Info, "Running preprocessor...");
  auto preprocessor_start = std::chrono::high_resolution_clock::now();
  auto proc_result = model->execute(
      "preprocessor",
      std::vector<::executorch::runtime::EValue>{
          audio_tensor, audio_len_tensor});
  auto preprocessor_end = std::chrono::high_resolution_clock::now();
  if (!proc_result.ok()) {
    ET_LOG(Error, "Preprocessor forward failed.");
    return 1;
  }
  auto& proc_outputs = proc_result.get();
  auto mel = proc_outputs[0].toTensor();
  auto mel_len_tensor_out = proc_outputs[1].toTensor();
  int64_t mel_len_value = mel_len_tensor_out.const_data_ptr<int64_t>()[0];
  auto preprocessor_duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          preprocessor_end - preprocessor_start)
          .count();

  // Create mel_len tensor for encoder
  std::vector<int64_t> mel_len_data = {mel_len_value};
  auto mel_len =
      from_blob(mel_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  ET_LOG(
      Info,
      "Mel spectrogram shape: [%ld, %ld, %ld], mel_len: %lld",
      static_cast<long>(mel.sizes()[0]),
      static_cast<long>(mel.sizes()[1]),
      static_cast<long>(mel.sizes()[2]),
      static_cast<long long>(mel_len_value));

  // Run encoder
  ET_LOG(Info, "Running encoder...");
  auto encoder_start = std::chrono::high_resolution_clock::now();
  auto enc_result = model->execute(
      "encoder", std::vector<::executorch::runtime::EValue>{mel, mel_len});
  auto encoder_end = std::chrono::high_resolution_clock::now();
  if (!enc_result.ok()) {
    ET_LOG(Error, "Encoder forward failed.");
    return 1;
  }
  auto& enc_outputs = enc_result.get();
  auto encoded = enc_outputs[0].toTensor();
  int64_t encoded_len = enc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];
  auto encoder_duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          encoder_end - encoder_start)
          .count();

  ET_LOG(
      Info,
      "Encoder output shape: [%ld, %ld, %ld], len=%ld",
      static_cast<long>(encoded.sizes()[0]),
      static_cast<long>(encoded.sizes()[1]),
      static_cast<long>(encoded.sizes()[2]),
      static_cast<long>(encoded_len));

  // Query model metadata from constant_methods
  std::vector<::executorch::runtime::EValue> empty_inputs;
  auto num_rnn_layers_result = model->execute("num_rnn_layers", empty_inputs);
  auto pred_hidden_result = model->execute("pred_hidden", empty_inputs);
  auto vocab_size_result = model->execute("vocab_size", empty_inputs);
  auto blank_id_result = model->execute("blank_id", empty_inputs);
  auto sample_rate_result = model->execute("sample_rate", empty_inputs);

  if (!num_rnn_layers_result.ok() || !pred_hidden_result.ok() ||
      !vocab_size_result.ok() || !blank_id_result.ok() ||
      !sample_rate_result.ok()) {
    ET_LOG(
        Error,
        "Failed to query model metadata. Make sure the model was exported with constant_methods.");
    return 1;
  }

  int64_t vocab_size = vocab_size_result.get()[0].toInt();
  int64_t blank_id = blank_id_result.get()[0].toInt();
  int64_t num_rnn_layers = num_rnn_layers_result.get()[0].toInt();
  int64_t pred_hidden = pred_hidden_result.get()[0].toInt();
  int64_t sample_rate = sample_rate_result.get()[0].toInt();

  ET_LOG(
      Info,
      "Model metadata: vocab_size=%lld, blank_id=%lld, num_rnn_layers=%lld, pred_hidden=%lld, sample_rate=%lld",
      static_cast<long long>(vocab_size),
      static_cast<long long>(blank_id),
      static_cast<long long>(num_rnn_layers),
      static_cast<long long>(pred_hidden),
      static_cast<long long>(sample_rate));

  ET_LOG(Info, "Running TDT greedy decode...");
  auto decode_result = greedy_decode_executorch(
      *model,
      encoded,
      encoded_len,
      blank_id,
      vocab_size,
      num_rnn_layers,
      pred_hidden);
  auto& tokens = decode_result.hypothesis;

  ET_LOG(Info, "Decoded %zu tokens", tokens.size());

  // Load tokenizer
  ET_LOG(Info, "Loading tokenizer from: %s", FLAGS_tokenizer_path.c_str());
  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(FLAGS_tokenizer_path);
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(
        Error,
        "Failed to load tokenizer from: %s",
        FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // Convert tokens to text
  std::string text = tokens_to_text(tokens, tokenizer.get());
  std::cout << "Transcription tokens: " << text << std::endl;

  auto binary_end = std::chrono::high_resolution_clock::now();
  auto binary_duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          binary_end - binary_start)
          .count();

  // Print timing summary with percentages
  ET_LOG(Info, "=== Timing Summary ===");
  ET_LOG(
      Info,
      "Preprocessor: %lld ms (%.1f%%)",
      static_cast<long long>(preprocessor_duration_ms),
      100.0 * preprocessor_duration_ms / binary_duration_ms);
  ET_LOG(
      Info,
      "Encoder: %lld ms (%.1f%%)",
      static_cast<long long>(encoder_duration_ms),
      100.0 * encoder_duration_ms / binary_duration_ms);
  ET_LOG(
      Info,
      "Decoder: %lld ms (%.1f%%)",
      static_cast<long long>(decode_result.decoder_ms),
      100.0 * decode_result.decoder_ms / binary_duration_ms);
  ET_LOG(
      Info,
      "Joint: %lld ms (%.1f%%)",
      static_cast<long long>(decode_result.joint_ms),
      100.0 * decode_result.joint_ms / binary_duration_ms);
  ET_LOG(
      Info,
      "Joint project encoder: %lld ms (%.1f%%)",
      static_cast<long long>(decode_result.joint_project_encoder_ms),
      100.0 * decode_result.joint_project_encoder_ms / binary_duration_ms);
  ET_LOG(
      Info,
      "Joint project decoder: %lld ms (%.1f%%)",
      static_cast<long long>(decode_result.joint_project_decoder_ms),
      100.0 * decode_result.joint_project_decoder_ms / binary_duration_ms);
  ET_LOG(
      Info,
      "Total wall time: %lld ms",
      static_cast<long long>(binary_duration_ms));

  ET_LOG(Info, "Done!");
  return 0;
}
