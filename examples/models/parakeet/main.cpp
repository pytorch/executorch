/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
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

// For Metal synchronization debugging
#if defined(__APPLE__)
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#endif

// For AOTI tensor metadata cache cleanup
#include <executorch/backends/aoti/common_shims.h>

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

  // Debug counters
  int64_t total_blank_count = 0;
  int64_t total_iterations = 0;
  int64_t max_consecutive_non_blank = 0;
  int64_t current_consecutive_non_blank = 0;

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

  // Prime the prediction network state with SOS (= blank_id) to match NeMo TDT
  // greedy label-looping decoding behavior:
  // - SOS is defined as blank:
  // https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c980b70cecc184fa8a083a9c3ddb87f905e/nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py#L250
  // - Predictor priming with SOS:
  // https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c980b70cecc184fa8a083a9c3ddb87f905e/nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py#L363-L368
  std::vector<int64_t> sos_token_data = {blank_id};
  auto sos_token = from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  auto decoder_init_result = model.execute(
      "decoder_predict",
      std::vector<::executorch::runtime::EValue>{sos_token, h, c});
  if (!decoder_init_result.ok()) {
    ET_LOG(Error, "decoder_predict (SOS) failed");
    return hypothesis;
  }

  // Force GPU synchronization for SOS initialization
#if defined(__APPLE__)
  executorch::backends::metal::synchronize_metal_stream();
#endif

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

  // Clear AOTI tensor metadata cache after SOS initialization
  executorch::backends::aoti::cleanup_tensor_metadata();

  auto g_proj_result = model.execute(
      "joint_project_decoder",
      std::vector<::executorch::runtime::EValue>{g_init});
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

  // Scan over encoder output
  while (t < encoder_len) {
    // Get encoder frame at time t: f_proj[:, t:t+1, :]
    const float* f_proj_data = f_proj.const_data_ptr<float>();
    int64_t proj_dim = f_proj.sizes()[2];

    std::vector<float> f_t_data(1 * 1 * proj_dim);
    for (int64_t d = 0; d < proj_dim; d++) {
      f_t_data[d] = f_proj_data[t * proj_dim + d];
    }

    // Log encoder frame stats around critical time steps
    if (t >= 248 && t <= 260) {
      float f_sum = 0.0f, f_max = f_t_data[0], f_min = f_t_data[0];
      int nan_count = 0;
      for (size_t i = 0; i < f_t_data.size(); i++) {
        if (std::isnan(f_t_data[i]) || std::isinf(f_t_data[i])) {
          nan_count++;
        }
        f_sum += f_t_data[i];
        f_max = std::max(f_max, f_t_data[i]);
        f_min = std::min(f_min, f_t_data[i]);
      }
      ET_LOG(
          Info,
          "Encoder frame[t=%lld]: sum=%.4f, min=%.4f, max=%.4f, nan_inf=%d",
          static_cast<long long>(t),
          f_sum,
          f_min,
          f_max,
          nan_count);
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
      ET_LOG(Error, "joint failed at t=%lld", static_cast<long long>(t));
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

    // Update debug counters
    total_iterations++;

    // Debug logging for first 20 iterations and when issues appear
    bool should_log = hypothesis.size() < 20 || k >= vocab_size ||
        (hypothesis.size() >= 95 && hypothesis.size() <= 115);
    if (should_log) {
      ET_LOG(
          Info,
          "Decode[t=%lld, hyp_len=%zu]: k=%lld (blank=%lld), dur=%lld, "
          "token_logit=%.4f, dur_logit=%.4f, symbols_on_frame=%lld",
          static_cast<long long>(t),
          hypothesis.size(),
          static_cast<long long>(k),
          static_cast<long long>(blank_id),
          static_cast<long long>(dur),
          max_token_logit,
          max_dur_logit,
          static_cast<long long>(symbols_on_frame));

      // Also log the top 3 token logits to see distribution
      std::vector<std::pair<float, int64_t>> top_logits;
      for (int64_t i = 0; i < num_token_classes; i++) {
        top_logits.push_back({logits_data[i], i});
      }
      std::sort(top_logits.begin(), top_logits.end(), std::greater<>());
      ET_LOG(
          Info,
          "  Top3 tokens: [%lld]=%.3f, [%lld]=%.3f, [%lld]=%.3f, blank[%lld]=%.3f",
          static_cast<long long>(top_logits[0].second), top_logits[0].first,
          static_cast<long long>(top_logits[1].second), top_logits[1].first,
          static_cast<long long>(top_logits[2].second), top_logits[2].first,
          static_cast<long long>(blank_id), logits_data[blank_id]);
    }

    // Warn if token is out of range
    if (k > vocab_size) {
      ET_LOG(
          Error,
          "Invalid token id %lld (vocab_size=%lld) at t=%lld",
          static_cast<long long>(k),
          static_cast<long long>(vocab_size),
          static_cast<long long>(t));
    }

    if (k == blank_id) {
      t += std::max(dur, (int64_t)1);
      symbols_on_frame = 0;
      total_blank_count++;
      // Track max consecutive non-blank before reset
      if (current_consecutive_non_blank > max_consecutive_non_blank) {
        max_consecutive_non_blank = current_consecutive_non_blank;
      }
      current_consecutive_non_blank = 0;
    } else {
      hypothesis.push_back(k);
      current_consecutive_non_blank++;

      // Update decoder state
      std::vector<int64_t> token_data = {k};
      auto token = from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      // Log input state BEFORE decoder_predict for critical iterations
      if (hypothesis.size() >= 99 && hypothesis.size() <= 102) {
        float h_sum_before = 0.0f, c_sum_before = 0.0f;
        int nan_before = 0;
        for (size_t i = 0; i < h_data.size(); i++) {
          if (std::isnan(h_data[i]) || std::isinf(h_data[i])) nan_before++;
          h_sum_before += h_data[i];
        }
        for (size_t i = 0; i < c_data.size(); i++) {
          if (std::isnan(c_data[i]) || std::isinf(c_data[i])) nan_before++;
          c_sum_before += c_data[i];
        }
        ET_LOG(
            Info,
            "  BEFORE decoder_predict[hyp=%zu]: h_sum=%.4f, c_sum=%.4f, nan_inf=%d, token=%lld",
            hypothesis.size(),
            h_sum_before,
            c_sum_before,
            nan_before,
            static_cast<long long>(k));
      }

      auto decoder_result = model.execute(
          "decoder_predict",
          std::vector<::executorch::runtime::EValue>{token, h, c});
      if (!decoder_result.ok()) {
        ET_LOG(Error, "decoder_predict failed");
        return hypothesis;
      }

      // Force GPU synchronization to flush any internal MPSGraph state
      // This is a debugging workaround for potential MPSGraph caching issues
#if defined(__APPLE__)
      executorch::backends::metal::synchronize_metal_stream();
#endif

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

      // CRASH ON NAN for lldb debugging
      for (size_t i = 0; i < h_data.size(); i++) {
        if (std::isnan(h_data[i]) || std::isinf(h_data[i])) {
          ET_LOG(Error, "NaN/Inf detected in h_data[%zu] = %f at hyp=%zu",
                 i, h_data[i], hypothesis.size());
          __builtin_trap();  // Crash here for lldb
        }
      }
      for (size_t i = 0; i < c_data.size(); i++) {
        if (std::isnan(c_data[i]) || std::isinf(c_data[i])) {
          ET_LOG(Error, "NaN/Inf detected in c_data[%zu] = %f at hyp=%zu",
                 i, c_data[i], hypothesis.size());
          // __builtin_trap();  // Crash here for lldb
        }
      }

      // Recreate tensor wrappers to avoid AOTI caching issues with tensor identity
      h = from_blob(
          h_data.data(),
          {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
           1,
           static_cast<::executorch::aten::SizesType>(pred_hidden)},
          ::executorch::aten::ScalarType::Float);
      c = from_blob(
          c_data.data(),
          {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
           1,
           static_cast<::executorch::aten::SizesType>(pred_hidden)},
          ::executorch::aten::ScalarType::Float);

      // Clear AOTI tensor metadata cache to prevent stale pointer issues
      executorch::backends::aoti::cleanup_tensor_metadata();

      // Check for NaN/Inf in LSTM state (first 20 tokens, around failure point, or when issues occur)
      bool should_log_state = hypothesis.size() <= 20 || k >= vocab_size ||
          (hypothesis.size() >= 95 && hypothesis.size() <= 115);
      if (should_log_state) {
        // Log Metal buffer statistics to check for buffer accumulation
#if defined(__APPLE__)
        executorch::backends::metal::metal_log_buffer_stats();
#endif
        // Log pointer addresses to detect aliasing
        ET_LOG(
            Info,
            "  Buffer ptrs[hyp=%zu]: h_data=%p, c_data=%p, new_h=%p, new_c=%p",
            hypothesis.size(),
            static_cast<const void*>(h_data.data()),
            static_cast<const void*>(c_data.data()),
            static_cast<const void*>(new_h.const_data_ptr<float>()),
            static_cast<const void*>(new_c.const_data_ptr<float>()));
        float h_sum = 0.0f, c_sum = 0.0f;
        int nan_count = 0;
        for (size_t i = 0; i < h_data.size(); i++) {
          if (std::isnan(h_data[i]) || std::isinf(h_data[i])) {
            nan_count++;
          }
          h_sum += h_data[i];
        }
        for (size_t i = 0; i < c_data.size(); i++) {
          if (std::isnan(c_data[i]) || std::isinf(c_data[i])) {
            nan_count++;
          }
          c_sum += c_data[i];
        }
        // Always log since we're in should_log_state block
        ET_LOG(
            Info,
            "  LSTM state[hyp=%zu]: h_sum=%.4f, c_sum=%.4f, nan_inf_count=%d",
            hypothesis.size(),
            h_sum,
            c_sum,
            nan_count);
      }

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
          // Log when hitting the limit - this might indicate a problem
          ET_LOG(
              Info,
              "  Hit max_symbols_per_step at t=%lld, hyp_len=%zu, forcing advance",
              static_cast<long long>(t),
              hypothesis.size());
          t++;
          symbols_on_frame = 0;
        }
      } else {
        symbols_on_frame = 0;
      }
    }
  }

  // Final update for max consecutive
  if (current_consecutive_non_blank > max_consecutive_non_blank) {
    max_consecutive_non_blank = current_consecutive_non_blank;
  }

  // Summary statistics
  ET_LOG(
      Info,
      "Decode summary: total_iterations=%lld, tokens=%zu, blanks=%lld, "
      "max_consecutive_non_blank=%lld, encoder_len=%lld",
      static_cast<long long>(total_iterations),
      hypothesis.size(),
      static_cast<long long>(total_blank_count),
      static_cast<long long>(max_consecutive_non_blank),
      static_cast<long long>(encoder_len));

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
  auto proc_result = model->execute(
      "preprocessor",
      std::vector<::executorch::runtime::EValue>{
          audio_tensor, audio_len_tensor});
  if (!proc_result.ok()) {
    ET_LOG(Error, "Preprocessor forward failed.");
    return 1;
  }
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
  auto enc_result = model->execute(
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
  auto tokens = greedy_decode_executorch(
      *model,
      encoded,
      encoded_len,
      blank_id,
      vocab_size,
      num_rnn_layers,
      pred_hidden);

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

  ET_LOG(Info, "Done!");
  return 0;
}
