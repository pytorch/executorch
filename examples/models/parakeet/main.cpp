/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
DEFINE_string(
    processor_path,
    "",
    "Path to preprocessor .pte for converting raw audio to mel spectrogram.");
DEFINE_string(audio_path, "", "Path to input audio file (.wav).");
DEFINE_string(
    tokenizer_path,
    "tokenizer.json",
    "Path to SentencePiece tokenizer model file.");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Error;

namespace {

// TDT duration values
const std::vector<int> DURATIONS = {0, 1, 2, 3, 4};

std::vector<int64_t> greedy_decode_executorch(
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

  // Transpose encoder output from [1, enc_dim, time] to [1, time, enc_dim]
  // The encoder output shape is [1, 1024, T], we need [1, T, 1024]
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
  auto proj_enc_result = model.execute(
      "joint_project_encoder",
      std::vector<::executorch::runtime::EValue>{transposed_tensor});
  if (!proj_enc_result.ok()) {
    ET_LOG(Error, "joint_project_encoder failed");
    return hypothesis;
  }
  auto f_proj = proj_enc_result.get()[0].toTensor();

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

  // Initialize decoder with SOS (zeros)
  std::vector<float> sos_g_data(1 * 1 * pred_hidden, 0.0f);
  auto sos_g = from_blob(
      sos_g_data.data(),
      {1, 1, static_cast<::executorch::aten::SizesType>(pred_hidden)},
      ::executorch::aten::ScalarType::Float);

  auto g_proj_result = model.execute(
      "joint_project_decoder",
      std::vector<::executorch::runtime::EValue>{sos_g});
  if (!g_proj_result.ok()) {
    ET_LOG(Error, "joint_project_decoder failed");
    return hypothesis;
  }
  auto g_proj_tensor = g_proj_result.get()[0].toTensor();

  // Copy g_proj data for reuse
  std::vector<float> g_proj_data(
      g_proj_tensor.const_data_ptr<float>(),
      g_proj_tensor.const_data_ptr<float>() + g_proj_tensor.numel());

  int64_t t = 0;
  int64_t symbols_on_frame = 0;

  // Debug: print first few tokens
  bool debug = true;
  int debug_count = 0;

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
    auto joint_result = model.execute(
        "joint", std::vector<::executorch::runtime::EValue>{f_t, g_proj});
    if (!joint_result.ok()) {
      ET_LOG(Error, "joint failed at t=%ld", t);
      return hypothesis;
    }
    auto full_logits = joint_result.get()[0].toTensor();

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

    // TDT decoding: joint network outputs both token logits and duration
    // logits.
    // - If blank: skip forward by predicted duration (min 1 frame)
    // - If token: emit it, update decoder state, advance by duration.
    //   Duration=0 means "emit another token on this frame" (up to
    //   max_symbols_per_step).
    if (k == blank_id) {
      t += std::max(dur, (int64_t)1);
      symbols_on_frame = 0;
    } else {
      if (debug && debug_count < 20) {
        ET_LOG(Info, "Token[%d]: t=%ld k=%ld dur=%ld", debug_count, t, k, dur);
        debug_count++;
      }
      hypothesis.push_back(k);

      // Update decoder state
      std::vector<int64_t> token_data = {k};
      auto token = from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      auto decoder_result = model.execute(
          "decoder_predict",
          std::vector<::executorch::runtime::EValue>{token, h, c});
      if (!decoder_result.ok()) {
        ET_LOG(Error, "decoder_predict failed");
        return hypothesis;
      }
      auto& outputs = decoder_result.get();
      auto g = outputs[0].toTensor();
      auto new_h = outputs[1].toTensor();
      auto new_c = outputs[2].toTensor();

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
      auto proj_dec_result = model.execute(
          "joint_project_decoder",
          std::vector<::executorch::runtime::EValue>{g});
      if (!proj_dec_result.ok()) {
        ET_LOG(Error, "joint_project_decoder failed");
        return hypothesis;
      }
      auto new_g_proj = proj_dec_result.get()[0].toTensor();
      std::memcpy(
          g_proj_data.data(),
          new_g_proj.const_data_ptr<float>(),
          g_proj_data.size() * sizeof(float));

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

  return hypothesis;
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
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "audio_path flag must be provided.");
    return 1;
  }

  if (FLAGS_processor_path.empty()) {
    ET_LOG(Error, "processor_path flag must be provided.");
    return 1;
  }

  // Load audio
  ET_LOG(Info, "Loading audio from: %s", FLAGS_audio_path.c_str());
  std::vector<float> audio_data =
      ::executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
  ET_LOG(Info, "Loaded %zu audio samples", audio_data.size());
  ET_LOG(
      Info,
      "First 5 audio samples: %f, %f, %f, %f, %f",
      audio_data[0],
      audio_data[1],
      audio_data[2],
      audio_data[3],
      audio_data[4]);

  // Load preprocessor
  ET_LOG(Info, "Loading preprocessor from: %s", FLAGS_processor_path.c_str());
  Module processor(FLAGS_processor_path, Module::LoadMode::Mmap);
  auto proc_load_error = processor.load();
  if (proc_load_error != Error::Ok) {
    ET_LOG(Error, "Failed to load preprocessor module.");
    return 1;
  }

  // Process audio to mel spectrogram
  auto audio_tensor = from_blob(
      audio_data.data(),
      {static_cast<::executorch::aten::SizesType>(audio_data.size())},
      ::executorch::aten::ScalarType::Float);

  auto proc_result = processor.execute(
      "forward", std::vector<::executorch::runtime::EValue>{audio_tensor});
  if (!proc_result.ok()) {
    ET_LOG(Error, "Preprocessor forward failed.");
    return 1;
  }
  auto& proc_outputs = proc_result.get();
  auto mel = proc_outputs[0].toTensor();

  // Compute mel_len from tensor shape
  std::vector<int64_t> mel_len_data = {
      static_cast<int64_t>(mel.sizes()[2])};
  auto mel_len = from_blob(
      mel_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  ET_LOG(
      Info,
      "Mel spectrogram shape: [%ld, %ld, %ld]",
      static_cast<long>(mel.sizes()[0]),
      static_cast<long>(mel.sizes()[1]),
      static_cast<long>(mel.sizes()[2]));

  // Load model
  ET_LOG(Info, "Loading model from: %s", FLAGS_model_path.c_str());
  Module model(FLAGS_model_path, Module::LoadMode::Mmap);
  auto model_load_error = model.load();
  if (model_load_error != Error::Ok) {
    ET_LOG(Error, "Failed to load model.");
    return 1;
  }

  // Run encoder
  ET_LOG(Info, "Running encoder...");
  auto enc_result = model.execute(
      "encoder", std::vector<::executorch::runtime::EValue>{mel, mel_len});
  if (!enc_result.ok()) {
    ET_LOG(Error, "Encoder forward failed.");
    return 1;
  }
  auto& enc_outputs = enc_result.get();
  auto encoded = enc_outputs[0].toTensor();
  int64_t encoded_len = enc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];

  ET_LOG(
      Info,
      "Encoder output shape: [%ld, %ld, %ld], len=%ld",
      static_cast<long>(encoded.sizes()[0]),
      static_cast<long>(encoded.sizes()[1]),
      static_cast<long>(encoded.sizes()[2]),
      static_cast<long>(encoded_len));

  // Greedy decode
  // Parakeet TDT uses vocab_size=8192, blank_id=8192
  int64_t vocab_size = 8192;
  int64_t blank_id = vocab_size;
  int64_t num_rnn_layers = 2;
  int64_t pred_hidden = 640;

  ET_LOG(Info, "Running TDT greedy decode...");
  auto tokens = greedy_decode_executorch(
      model,
      encoded,
      encoded_len,
      blank_id,
      vocab_size,
      num_rnn_layers,
      pred_hidden);

  ET_LOG(Info, "Decoded %zu tokens", tokens.size());

  // Load tokenizer using the LLM runner helper
  ET_LOG(Info, "Loading tokenizer from: %s", FLAGS_tokenizer_path.c_str());
  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(FLAGS_tokenizer_path);
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Failed to load tokenizer from: %s", FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // Convert tokens to text
  std::string text = tokens_to_text(tokens, tokenizer.get());
  std::cout << "Transcription tokens: " << text << std::endl;

  ET_LOG(Info, "Done!");
  return 0;
}
