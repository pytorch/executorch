/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <gflags/gflags.h>

#include "timestamp_utils.h"
#include "tokenizer_utils.h"
#include "types.h"

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/llm/tokenizers/third-party/llama.cpp-unicode/include/unicode.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>

DEFINE_string(model_path, "parakeet.pte", "Path to Parakeet model (.pte).");
DEFINE_string(audio_path, "", "Path to input audio file (.wav).");
DEFINE_string(
    tokenizer_path,
    "tokenizer.model",
    "Path to SentencePiece tokenizer model file.");
DEFINE_string(
    data_path,
    "",
    "Path to data file (.ptd) for delegate data (optional, required for CUDA).");
DEFINE_string(
    timestamps,
    "segment",
    "Timestamp output mode: none|token|word|segment|all");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using ::parakeet::TextWithOffsets;
using ::parakeet::Token;
using ::parakeet::TokenId;
using ::parakeet::TokenWithTextInfo;

namespace {
// TDT duration values
const std::vector<int> DURATIONS = {0, 1, 2, 3, 4};

struct TimestampOutputMode {
  bool token = false;
  bool word = false;
  bool segment = false;

  bool enabled() const {
    return token || word || segment;
  }
};

// Statistics for decode operations
struct DecodeStats {
  double joint_project_encoder_ms = 0.0;
  double decoder_predict_init_ms = 0.0;
  double joint_project_decoder_init_ms = 0.0;
  double joint_total_ms = 0.0;
  double decoder_predict_total_ms = 0.0;
  double joint_project_decoder_total_ms = 0.0;
  int64_t joint_calls = 0;
  int64_t decoder_predict_calls = 0;
  int64_t joint_project_decoder_calls = 0;
};

struct DecodeResult {
  std::vector<Token> tokens;
  DecodeStats stats;
};

std::string to_lower_ascii(std::string s) {
  for (char& ch : s) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return s;
}

TimestampOutputMode parse_timestamp_output_mode(const std::string& raw_arg) {
  if (raw_arg.empty()) {
    throw std::invalid_argument(
        "Invalid --timestamps value (empty). Expected: token, word, segment, all.");
  }
  const std::string mode = to_lower_ascii(raw_arg);
  if (mode == "none") {
    return {false, false, false};
  }
  if (mode == "token") {
    return {true, false, false};
  }
  if (mode == "word") {
    return {false, true, false};
  }
  if (mode == "segment") {
    return {false, false, true};
  }
  if (mode == "all") {
    return {true, true, true};
  }
  throw std::invalid_argument(
      "Invalid --timestamps value '" + raw_arg +
      "'. Expected: token, word, segment, all.");
}

DecodeResult greedy_decode_executorch(
    Module& model,
    const ::executorch::aten::Tensor& encoder_output,
    int64_t encoder_len,
    int64_t blank_id,
    int64_t vocab_size,
    int64_t num_rnn_layers = 2,
    int64_t pred_hidden = 640,
    int64_t max_symbols_per_step = 10) {
  DecodeResult result;
  std::vector<Token>& hypothesis = result.tokens;
  DecodeStats& stats = result.stats;
  int64_t num_token_classes = vocab_size + 1;

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
  if (!proj_enc_result.ok()) {
    ET_LOG(Error, "joint_project_encoder failed");
    return result;
  }
  auto proj_enc_end = std::chrono::high_resolution_clock::now();
  stats.joint_project_encoder_ms = std::chrono::duration<double, std::milli>(
                                       proj_enc_end - proj_enc_start)
                                       .count();
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

  // Prime the prediction network state with SOS (= blank_id) to match NeMo TDT
  // greedy label-looping decoding behavior:
  // - SOS is defined as blank:
  // https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c980b70cecc184fa8a083a9c3ddb87f905e/nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py#L250
  // - Predictor priming with SOS:
  // https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c980b70cecc184fa8a083a9c3ddb87f905e/nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py#L363-L368
  std::vector<int64_t> sos_token_data = {blank_id};
  auto sos_token = from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  auto decoder_init_start = std::chrono::high_resolution_clock::now();
  auto decoder_init_result = model.execute(
      "decoder_predict",
      std::vector<::executorch::runtime::EValue>{sos_token, h, c});
  if (!decoder_init_result.ok()) {
    ET_LOG(Error, "decoder_predict (SOS) failed");
    return result;
  }
  auto decoder_init_end = std::chrono::high_resolution_clock::now();
  stats.decoder_predict_init_ms = std::chrono::duration<double, std::milli>(
                                      decoder_init_end - decoder_init_start)
                                      .count();
  auto& init_outputs = decoder_init_result.get();
  auto g_init = init_outputs[0].toTensor();
  auto new_h_init = init_outputs[1].toTensor();
  auto new_c_init = init_outputs[2].toTensor();
  std::memcpy(
      h_data.data(),
      new_h_init.const_data_ptr<float>(),
      h_data.size() * sizeof(float));
  std::memcpy(
      c_data.data(),
      new_c_init.const_data_ptr<float>(),
      c_data.size() * sizeof(float));

  auto g_proj_init_start = std::chrono::high_resolution_clock::now();
  auto g_proj_result = model.execute(
      "joint_project_decoder",
      std::vector<::executorch::runtime::EValue>{g_init});
  if (!g_proj_result.ok()) {
    ET_LOG(Error, "joint_project_decoder failed");
    return result;
  }
  auto g_proj_init_end = std::chrono::high_resolution_clock::now();
  stats.joint_project_decoder_init_ms =
      std::chrono::duration<double, std::milli>(
          g_proj_init_end - g_proj_init_start)
          .count();
  auto g_proj_tensor = g_proj_result.get()[0].toTensor();

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
    if (!joint_result.ok()) {
      ET_LOG(Error, "joint failed at t=%lld", static_cast<long long>(t));
      return result;
    }
    auto joint_end = std::chrono::high_resolution_clock::now();
    stats.joint_total_ms +=
        std::chrono::duration<double, std::milli>(joint_end - joint_start)
            .count();
    stats.joint_calls++;
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

    if (k == blank_id) {
      t += std::max(dur, static_cast<int64_t>(1));
      symbols_on_frame = 0;
    } else {
      hypothesis.push_back({static_cast<TokenId>(k), t, dur});

      // Update decoder state
      std::vector<int64_t> token_data = {k};
      auto token = from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      auto decoder_start = std::chrono::high_resolution_clock::now();
      auto decoder_result = model.execute(
          "decoder_predict",
          std::vector<::executorch::runtime::EValue>{token, h, c});
      if (!decoder_result.ok()) {
        ET_LOG(Error, "decoder_predict failed");
        return result;
      }
      auto decoder_end = std::chrono::high_resolution_clock::now();
      stats.decoder_predict_total_ms +=
          std::chrono::duration<double, std::milli>(decoder_end - decoder_start)
              .count();
      stats.decoder_predict_calls++;
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
      auto proj_dec_start = std::chrono::high_resolution_clock::now();
      auto proj_dec_result = model.execute(
          "joint_project_decoder",
          std::vector<::executorch::runtime::EValue>{g});
      if (!proj_dec_result.ok()) {
        ET_LOG(Error, "joint_project_decoder failed");
        return result;
      }
      auto proj_dec_end = std::chrono::high_resolution_clock::now();
      stats.joint_project_decoder_total_ms +=
          std::chrono::duration<double, std::milli>(proj_dec_end - proj_dec_start)
              .count();
      stats.joint_project_decoder_calls++;
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

  return result;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  TimestampOutputMode timestamp_mode;
  try {
    timestamp_mode = parse_timestamp_output_mode(FLAGS_timestamps);
  } catch (const std::invalid_argument& e) {
    ET_LOG(Error, "%s", e.what());
    return 1;
  }

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
  auto preproc_start = std::chrono::high_resolution_clock::now();
  auto proc_result = model->execute(
      "preprocessor",
      std::vector<::executorch::runtime::EValue>{
          audio_tensor, audio_len_tensor});
  if (!proc_result.ok()) {
    ET_LOG(Error, "Preprocessor forward failed.");
    return 1;
  }
  auto preproc_end = std::chrono::high_resolution_clock::now();
  double preproc_ms = std::chrono::duration<double, std::milli>(
                          preproc_end - preproc_start)
                          .count();
  auto& proc_outputs = proc_result.get();
  auto mel = proc_outputs[0].toTensor();
  auto mel_len_tensor_out = proc_outputs[1].toTensor();
  int64_t mel_len_value = mel_len_tensor_out.const_data_ptr<int64_t>()[0];

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
  if (!enc_result.ok()) {
    ET_LOG(Error, "Encoder forward failed.");
    return 1;
  }
  auto encoder_end = std::chrono::high_resolution_clock::now();
  double encoder_ms =
      std::chrono::duration<double, std::milli>(encoder_end - encoder_start)
          .count();
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

  // Query model metadata from constant_methods
  std::vector<::executorch::runtime::EValue> empty_inputs;
  auto num_rnn_layers_result = model->execute("num_rnn_layers", empty_inputs);
  auto pred_hidden_result = model->execute("pred_hidden", empty_inputs);
  auto vocab_size_result = model->execute("vocab_size", empty_inputs);
  auto blank_id_result = model->execute("blank_id", empty_inputs);
  auto sample_rate_result = model->execute("sample_rate", empty_inputs);
  auto window_stride_result = model->execute("window_stride", empty_inputs);
  auto encoder_subsampling_factor_result =
      model->execute("encoder_subsampling_factor", empty_inputs);

  if (!num_rnn_layers_result.ok() || !pred_hidden_result.ok() ||
      !vocab_size_result.ok() || !blank_id_result.ok() ||
      !sample_rate_result.ok() || !window_stride_result.ok() ||
      !encoder_subsampling_factor_result.ok()) {
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
  double window_stride = window_stride_result.get()[0].toDouble();
  int64_t encoder_subsampling_factor =
      encoder_subsampling_factor_result.get()[0].toInt();

  ET_LOG(
      Info,
      "Model metadata: vocab_size=%lld, blank_id=%lld, num_rnn_layers=%lld, pred_hidden=%lld, sample_rate=%lld, window_stride=%.6f, encoder_subsampling_factor=%lld",
      static_cast<long long>(vocab_size),
      static_cast<long long>(blank_id),
      static_cast<long long>(num_rnn_layers),
      static_cast<long long>(pred_hidden),
      static_cast<long long>(sample_rate),
      window_stride,
      encoder_subsampling_factor);

  ET_LOG(Info, "Running TDT greedy decode...");
  auto decode_start = std::chrono::high_resolution_clock::now();
  auto decode_result = greedy_decode_executorch(
      *model,
      encoded,
      encoded_len,
      blank_id,
      vocab_size,
      num_rnn_layers,
      pred_hidden);
  auto decode_end = std::chrono::high_resolution_clock::now();
  double decode_total_ms =
      std::chrono::duration<double, std::milli>(decode_end - decode_start)
          .count();

  const auto& decoded_tokens = decode_result.tokens;
  const auto& decode_stats = decode_result.stats;

  ET_LOG(Info, "Decoded %zu tokens", decoded_tokens.size());

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
  std::string text = parakeet::tokenizer_utils::decode_token_sequence(
      decoded_tokens, *tokenizer);
  std::cout << "Transcribed text: " << text << std::endl;

  // Print performance statistics
  std::cout << "\n=== Performance Statistics ===" << std::endl;

  // Calculate audio duration in seconds
  double audio_duration_sec =
      static_cast<double>(audio_data.size()) / static_cast<double>(sample_rate);

  std::cout << "\nAudio duration: " << audio_duration_sec << " seconds"
            << std::endl;
  std::cout << "Tokens decoded: " << decoded_tokens.size() << std::endl;

  std::cout << "\n--- Encoding Phase ---" << std::endl;
  std::cout << "Preprocessor: " << preproc_ms << " ms" << std::endl;
  std::cout << "Encoder: " << encoder_ms << " ms" << std::endl;
  double encoding_total_ms = preproc_ms + encoder_ms;
  std::cout << "Encoding total: " << encoding_total_ms << " ms" << std::endl;

  std::cout << "\n--- Decoding Phase ---" << std::endl;
  std::cout << "joint_project_encoder (1 call): "
            << decode_stats.joint_project_encoder_ms << " ms" << std::endl;
  std::cout << "decoder_predict init (1 call): "
            << decode_stats.decoder_predict_init_ms << " ms" << std::endl;
  std::cout << "joint_project_decoder init (1 call): "
            << decode_stats.joint_project_decoder_init_ms << " ms" << std::endl;

  std::cout << "\njoint (" << decode_stats.joint_calls
            << " calls): " << decode_stats.joint_total_ms << " ms";
  if (decode_stats.joint_calls > 0) {
    std::cout << " (avg: "
              << decode_stats.joint_total_ms / decode_stats.joint_calls
              << " ms/call)";
  }
  std::cout << std::endl;

  std::cout << "decoder_predict (" << decode_stats.decoder_predict_calls
            << " calls): " << decode_stats.decoder_predict_total_ms << " ms";
  if (decode_stats.decoder_predict_calls > 0) {
    std::cout << " (avg: "
              << decode_stats.decoder_predict_total_ms /
                     decode_stats.decoder_predict_calls
              << " ms/call)";
  }
  std::cout << std::endl;

  std::cout << "joint_project_decoder ("
            << decode_stats.joint_project_decoder_calls
            << " calls): " << decode_stats.joint_project_decoder_total_ms
            << " ms";
  if (decode_stats.joint_project_decoder_calls > 0) {
    std::cout << " (avg: "
              << decode_stats.joint_project_decoder_total_ms /
                     decode_stats.joint_project_decoder_calls
              << " ms/call)";
  }
  std::cout << std::endl;

  std::cout << "\nDecoding total: " << decode_total_ms << " ms" << std::endl;

  std::cout << "\n--- Summary ---" << std::endl;
  double total_inference_ms = encoding_total_ms + decode_total_ms;
  std::cout << "Total inference time: " << total_inference_ms << " ms"
            << std::endl;

  double tokens_per_second = 0.0;
  if (decode_total_ms > 0 && !decoded_tokens.empty()) {
    tokens_per_second =
        static_cast<double>(decoded_tokens.size()) / (decode_total_ms / 1000.0);
  }
  std::cout << "Tokens/second (decode): " << tokens_per_second << std::endl;

  double real_time_factor = 0.0;
  if (audio_duration_sec > 0) {
    real_time_factor = (total_inference_ms / 1000.0) / audio_duration_sec;
  }
  std::cout << "Real-time factor: " << real_time_factor << "x" << std::endl;

  // Metal backend statistics
  std::cout << "\n--- Metal Backend ---" << std::endl;
  double metal_total_ms =
      executorch::backends::metal::get_metal_backend_execute_total_ms();
  int64_t metal_call_count =
      executorch::backends::metal::get_metal_backend_execute_call_count();
  std::cout << "Metal execute() total: " << metal_total_ms << " ms ("
            << metal_call_count << " calls)";
  if (metal_call_count > 0) {
    std::cout << " (avg: " << metal_total_ms / metal_call_count << " ms/call)";
  }
  std::cout << std::endl;

  std::cout << "==============================\n" << std::endl;

  if (!timestamp_mode.enabled()) {
    return 0;
  }

  ET_LOG(Info, "Computing timestamps...");
  std::unordered_set<std::string> supported_punctuation =
      parakeet::tokenizer_utils::derive_supported_punctuation(*tokenizer);
  ET_LOG(
      Info,
      "Derived supported_punctuation size=%zu",
      supported_punctuation.size());

  // for simplicity, compute all levels of timestamps regardless of mode
  std::vector<TokenWithTextInfo> tokens_with_text_info;
  try {
    tokens_with_text_info =
        parakeet::timestamp_utils::get_tokens_with_text_info(
            decoded_tokens, *tokenizer, supported_punctuation);
  } catch (const std::exception& e) {
    ET_LOG(Error, "Failed to get tokens with text info: %s", e.what());
    return 1;
  }
  const auto word_offsets = parakeet::timestamp_utils::get_words_offsets(
      tokens_with_text_info, *tokenizer, supported_punctuation);
  const auto segment_offsets =
      parakeet::timestamp_utils::get_segment_offsets(word_offsets);

  const double frame_to_seconds =
      window_stride * static_cast<double>(encoder_subsampling_factor);

  if (timestamp_mode.segment) {
    std::cout << "\nSegment timestamps:" << std::endl;
    for (const auto& segment : segment_offsets) {
      const double start = segment.start_offset * frame_to_seconds;
      const double end = segment.end_offset * frame_to_seconds;
      std::cout << start << "s - " << end << "s : " << segment.text
                << std::endl;
    }
  }

  if (timestamp_mode.word) {
    std::cout << "\nWord timestamps:" << std::endl;
    for (const auto& word : word_offsets) {
      const double start = word.start_offset * frame_to_seconds;
      const double end = word.end_offset * frame_to_seconds;
      std::cout << start << "s - " << end << "s : " << word.text << std::endl;
    }
  }

  if (timestamp_mode.token) {
    std::cout << "\nToken timestamps:" << std::endl;
    for (const auto& token : tokens_with_text_info) {
      const double start = token.start_offset * frame_to_seconds;
      const double end = token.end_offset * frame_to_seconds;
      std::cout << start << "s - " << end << "s : " << token.decoded_text
                << std::endl;
    }
  }

  return 0;
}
