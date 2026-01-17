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
DEFINE_int32(repeat, 1, "Number of times to run inference (for benchmarking).");
DEFINE_int32(
    warmup_repeat,
    1,
    "Number of warmup iterations to initialize backends before timed runs.");

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

} // namespace

// ============================================================================
// ParakeetRunner: Encapsulates model loading and inference
// ============================================================================

struct ParakeetConfig {
  int64_t vocab_size;
  int64_t blank_id;
  int64_t num_rnn_layers;
  int64_t pred_hidden;
  int64_t sample_rate;
  double window_stride;
  int64_t encoder_subsampling_factor;
};

struct TranscriptionResult {
  std::string text;
  std::vector<Token> tokens;
  std::vector<TokenWithTextInfo> token_timestamps;
  std::vector<TextWithOffsets> word_timestamps;
  std::vector<TextWithOffsets> segment_timestamps;

  // Timing information (in milliseconds)
  struct Timing {
    int64_t preprocessor_ms = 0;
    int64_t encoder_ms = 0;
    int64_t decoder_total_ms = 0;
    int64_t decoder_step_ms = 0; // Time in decoder_step calls
    int64_t joint_ms = 0; // Time in joint calls

    // Computed: overhead from tensor creation, memcpy, loop control
    int64_t loop_overhead_ms() const {
      return decoder_total_ms - decoder_step_ms - joint_ms;
    }
  } timing;
};

class ParakeetRunner {
 public:
  // Factory method to load model, tokenizer, and cache metadata
  static std::unique_ptr<ParakeetRunner> load(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& data_path = "");

  // Run inference on audio samples
  TranscriptionResult transcribe(const std::vector<float>& audio_samples);

  // Run inference on audio file
  TranscriptionResult transcribe(const std::string& audio_path);

  // Accessors for timestamp computation
  const ParakeetConfig& config() const {
    return config_;
  }
  double frame_to_seconds() const {
    return config_.window_stride *
        static_cast<double>(config_.encoder_subsampling_factor);
  }

 private:
  ParakeetRunner() = default;

  // Greedy TDT decoding
  std::vector<Token> greedy_decode(
      const ::executorch::aten::Tensor& f_proj,
      int64_t encoder_len,
      TranscriptionResult::Timing& timing);

  std::unique_ptr<Module> model_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  ParakeetConfig config_;
  std::unordered_set<std::string> supported_punctuation_;
};

// ----------------------------------------------------------------------------
// ParakeetRunner Implementation
// ----------------------------------------------------------------------------

std::unique_ptr<ParakeetRunner> ParakeetRunner::load(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& data_path) {
  auto runner = std::unique_ptr<ParakeetRunner>(new ParakeetRunner());

  // Load model
  ET_LOG(Info, "Loading model from: %s", model_path.c_str());
  if (!data_path.empty()) {
    ET_LOG(Info, "Loading data from: %s", data_path.c_str());
    runner->model_ =
        std::make_unique<Module>(model_path, data_path, Module::LoadMode::Mmap);
  } else {
    runner->model_ =
        std::make_unique<Module>(model_path, Module::LoadMode::Mmap);
  }

  auto load_error = runner->model_->load();
  if (load_error != Error::Ok) {
    ET_LOG(Error, "Failed to load model.");
    return nullptr;
  }

  // Load tokenizer
  ET_LOG(Info, "Loading tokenizer from: %s", tokenizer_path.c_str());
  runner->tokenizer_ =
      ::executorch::extension::llm::load_tokenizer(tokenizer_path);
  if (!runner->tokenizer_ || !runner->tokenizer_->is_loaded()) {
    ET_LOG(Error, "Failed to load tokenizer from: %s", tokenizer_path.c_str());
    return nullptr;
  }

  // Query model metadata
  std::vector<EValue> empty_inputs;
  auto num_rnn_layers_result =
      runner->model_->execute("num_rnn_layers", empty_inputs);
  auto pred_hidden_result =
      runner->model_->execute("pred_hidden", empty_inputs);
  auto vocab_size_result = runner->model_->execute("vocab_size", empty_inputs);
  auto blank_id_result = runner->model_->execute("blank_id", empty_inputs);
  auto sample_rate_result =
      runner->model_->execute("sample_rate", empty_inputs);
  auto window_stride_result =
      runner->model_->execute("window_stride", empty_inputs);
  auto encoder_subsampling_factor_result =
      runner->model_->execute("encoder_subsampling_factor", empty_inputs);

  if (!num_rnn_layers_result.ok() || !pred_hidden_result.ok() ||
      !vocab_size_result.ok() || !blank_id_result.ok() ||
      !sample_rate_result.ok() || !window_stride_result.ok() ||
      !encoder_subsampling_factor_result.ok()) {
    ET_LOG(
        Error,
        "Failed to query model metadata. Make sure the model was exported with constant_methods.");
    return nullptr;
  }

  runner->config_.vocab_size = vocab_size_result.get()[0].toInt();
  runner->config_.blank_id = blank_id_result.get()[0].toInt();
  runner->config_.num_rnn_layers = num_rnn_layers_result.get()[0].toInt();
  runner->config_.pred_hidden = pred_hidden_result.get()[0].toInt();
  runner->config_.sample_rate = sample_rate_result.get()[0].toInt();
  runner->config_.window_stride = window_stride_result.get()[0].toDouble();
  runner->config_.encoder_subsampling_factor =
      encoder_subsampling_factor_result.get()[0].toInt();

  ET_LOG(
      Info,
      "Model metadata: vocab_size=%lld, blank_id=%lld, num_rnn_layers=%lld, "
      "pred_hidden=%lld, sample_rate=%lld, window_stride=%.6f, "
      "encoder_subsampling_factor=%lld",
      static_cast<long long>(runner->config_.vocab_size),
      static_cast<long long>(runner->config_.blank_id),
      static_cast<long long>(runner->config_.num_rnn_layers),
      static_cast<long long>(runner->config_.pred_hidden),
      static_cast<long long>(runner->config_.sample_rate),
      runner->config_.window_stride,
      static_cast<long long>(runner->config_.encoder_subsampling_factor));

  // Derive supported punctuation for timestamp computation
  runner->supported_punctuation_ =
      parakeet::tokenizer_utils::derive_supported_punctuation(
          *runner->tokenizer_);

  return runner;
}

TranscriptionResult ParakeetRunner::transcribe(
    const std::vector<float>& audio_samples) {
  TranscriptionResult result;

  // Create audio tensor (from_blob requires non-const but doesn't modify data)
  auto audio_tensor = from_blob(
      const_cast<float*>(audio_samples.data()),
      {static_cast<::executorch::aten::SizesType>(audio_samples.size())},
      ::executorch::aten::ScalarType::Float);
  std::vector<int64_t> audio_len_data = {
      static_cast<int64_t>(audio_samples.size())};
  auto audio_len_tensor = from_blob(
      audio_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  // Run preprocessor
  auto proc_start = std::chrono::high_resolution_clock::now();
  auto proc_result = model_->execute(
      "preprocessor", std::vector<EValue>{audio_tensor, audio_len_tensor});
  auto proc_end = std::chrono::high_resolution_clock::now();
  result.timing.preprocessor_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          proc_end - proc_start)
          .count();

  if (!proc_result.ok()) {
    ET_LOG(Error, "Preprocessor forward failed.");
    return result;
  }
  auto& proc_outputs = proc_result.get();
  auto mel = proc_outputs[0].toTensor();
  int64_t mel_len_value =
      proc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];

  // Run encoder
  std::vector<int64_t> mel_len_data = {mel_len_value};
  auto mel_len =
      from_blob(mel_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  auto enc_start = std::chrono::high_resolution_clock::now();
  auto enc_result =
      model_->execute("encoder", std::vector<EValue>{mel, mel_len});
  auto enc_end = std::chrono::high_resolution_clock::now();
  result.timing.encoder_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(enc_end - enc_start)
          .count();

  if (!enc_result.ok()) {
    ET_LOG(Error, "Encoder forward failed.");
    return result;
  }
  auto& enc_outputs = enc_result.get();
  auto f_proj = enc_outputs[0].toTensor();
  int64_t encoded_len = enc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];

  // Run greedy decode
  auto decode_start = std::chrono::high_resolution_clock::now();
  result.tokens = greedy_decode(f_proj, encoded_len, result.timing);
  auto decode_end = std::chrono::high_resolution_clock::now();
  result.timing.decoder_total_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          decode_end - decode_start)
          .count();

  // Convert tokens to text
  result.text = parakeet::tokenizer_utils::decode_token_sequence(
      result.tokens, *tokenizer_);

  // Compute timestamps
  try {
    result.token_timestamps =
        parakeet::timestamp_utils::get_tokens_with_text_info(
            result.tokens, *tokenizer_, supported_punctuation_);
    result.word_timestamps = parakeet::timestamp_utils::get_words_offsets(
        result.token_timestamps, *tokenizer_, supported_punctuation_);
    result.segment_timestamps =
        parakeet::timestamp_utils::get_segment_offsets(result.word_timestamps);
  } catch (const std::exception& e) {
    ET_LOG(Error, "Failed to compute timestamps: %s", e.what());
  }

  return result;
}

TranscriptionResult ParakeetRunner::transcribe(const std::string& audio_path) {
  ET_LOG(Info, "Loading audio from: %s", audio_path.c_str());
  std::vector<float> audio_data =
      ::executorch::extension::llm::load_wav_audio_data(audio_path);
  ET_LOG(Info, "Loaded %zu audio samples", audio_data.size());
  return transcribe(audio_data);
}

std::vector<Token> ParakeetRunner::greedy_decode(
    const ::executorch::aten::Tensor& f_proj,
    int64_t encoder_len,
    TranscriptionResult::Timing& timing) {
  std::vector<Token> hypothesis;

  const int64_t blank_id = config_.blank_id;
  const int64_t num_rnn_layers = config_.num_rnn_layers;
  const int64_t pred_hidden = config_.pred_hidden;
  const int64_t max_symbols_per_step = 10;

  // Shape: [1, time_steps, joint_hidden]
  auto f_proj_sizes = f_proj.sizes();
  int64_t proj_dim = f_proj_sizes[2];

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

  // Prime the decoder with SOS (= blank_id) to match NeMo TDT label-looping:
  // - SOS is defined as blank:
  //   https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L1063
  // - Predictor priming with SOS:
  //   https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L1122-L1127
  std::vector<int64_t> sos_token_data = {blank_id};
  auto sos_token = from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

  auto sos_start = std::chrono::high_resolution_clock::now();
  auto decoder_init_result =
      model_->execute("decoder_step", std::vector<EValue>{sos_token, h, c});
  auto sos_end = std::chrono::high_resolution_clock::now();
  timing.decoder_step_ms +=
      std::chrono::duration_cast<std::chrono::milliseconds>(sos_end - sos_start)
          .count();
  if (!decoder_init_result.ok()) {
    ET_LOG(Error, "decoder_step (SOS) failed");
    return hypothesis;
  }
  auto& init_outputs = decoder_init_result.get();
  auto g_proj_init = init_outputs[0].toTensor();
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

  // Copy g_proj data for reuse
  std::vector<float> g_proj_data(
      g_proj_init.const_data_ptr<float>(),
      g_proj_init.const_data_ptr<float>() + g_proj_init.numel());

  int64_t t = 0;
  int64_t symbols_on_frame = 0;

  // Scan over encoder output
  while (t < encoder_len) {
    const float* f_proj_ptr = f_proj.const_data_ptr<float>();

    std::vector<float> f_t_data(1 * 1 * proj_dim);
    for (int64_t d = 0; d < proj_dim; d++) {
      f_t_data[d] = f_proj_ptr[t * proj_dim + d];
    }
    auto f_t = from_blob(
        f_t_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        ::executorch::aten::ScalarType::Float);

    auto g_proj = from_blob(
        g_proj_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        ::executorch::aten::ScalarType::Float);

    auto joint_start = std::chrono::high_resolution_clock::now();
    auto joint_result =
        model_->execute("joint", std::vector<EValue>{f_t, g_proj});
    auto joint_end = std::chrono::high_resolution_clock::now();
    timing.joint_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
                           joint_end - joint_start)
                           .count();

    if (!joint_result.ok()) {
      ET_LOG(Error, "joint failed at t=%lld", static_cast<long long>(t));
      return hypothesis;
    }

    int64_t k = joint_result.get()[0].toTensor().const_data_ptr<int64_t>()[0];
    int64_t dur_idx =
        joint_result.get()[1].toTensor().const_data_ptr<int64_t>()[0];
    int64_t dur = DURATIONS[dur_idx];

    if (k == blank_id) {
      t += std::max(dur, static_cast<int64_t>(1));
      symbols_on_frame = 0;
    } else {
      hypothesis.push_back({static_cast<TokenId>(k), t, dur});

      std::vector<int64_t> token_data = {k};
      auto token = from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      auto dec_start = std::chrono::high_resolution_clock::now();
      auto decoder_result =
          model_->execute("decoder_step", std::vector<EValue>{token, h, c});
      auto dec_end = std::chrono::high_resolution_clock::now();
      timing.decoder_step_ms +=
          std::chrono::duration_cast<std::chrono::milliseconds>(
              dec_end - dec_start)
              .count();

      if (!decoder_result.ok()) {
        ET_LOG(Error, "decoder_step failed");
        return hypothesis;
      }
      auto& outputs = decoder_result.get();
      auto new_g_proj = outputs[0].toTensor();
      auto new_h = outputs[1].toTensor();
      auto new_c = outputs[2].toTensor();

      std::memcpy(
          h_data.data(),
          new_h.const_data_ptr<float>(),
          h_data.size() * sizeof(float));
      std::memcpy(
          c_data.data(),
          new_c.const_data_ptr<float>(),
          c_data.size() * sizeof(float));
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

// ============================================================================
// Main
// ============================================================================

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

  // Load model and tokenizer (done once)
  auto load_start = std::chrono::high_resolution_clock::now();
  auto runner = ParakeetRunner::load(
      FLAGS_model_path, FLAGS_tokenizer_path, FLAGS_data_path);
  auto load_end = std::chrono::high_resolution_clock::now();

  if (!runner) {
    ET_LOG(Error, "Failed to initialize ParakeetRunner.");
    return 1;
  }

  auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     load_end - load_start)
                     .count();
  ET_LOG(Info, "Load time: %lldms", static_cast<long long>(load_ms));

  // Load audio once (not included in inference timing)
  ET_LOG(Info, "Loading audio from: %s", FLAGS_audio_path.c_str());
  std::vector<float> audio_data =
      ::executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
  ET_LOG(Info, "Loaded %zu audio samples", audio_data.size());

  // Run inference
  TranscriptionResult result;
  const int num_iterations = std::max(1, FLAGS_repeat);
  const int num_warmup = std::max(1, FLAGS_warmup_repeat);

  // Warmup runs to initialize backends (e.g., Metal shader compilation).
  // The first iteration includes backend initialization overhead.
  // Note: Warmup can use any audio - it doesn't need to be the audio you
  // want to transcribe. The purpose is to warm up the GPU/accelerator.
  ET_LOG(Info, "Running %d warmup iteration(s)...", num_warmup);
  auto warmup_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_warmup; ++i) {
    result = runner->transcribe(audio_data);
  }
  auto warmup_end = std::chrono::high_resolution_clock::now();
  auto warmup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       warmup_end - warmup_start)
                       .count();
  ET_LOG(
      Info,
      "Warmup time: %lldms (%d iteration(s), includes backend initialization)",
      static_cast<long long>(warmup_ms),
      num_warmup);

  // Timed inference iterations
  ET_LOG(Info, "Running %d timed inference iteration(s)...", num_iterations);
  auto infer_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; ++i) {
    result = runner->transcribe(audio_data);
    if (num_iterations > 1 && (i + 1) % 10 == 0) {
      ET_LOG(Info, "Completed iteration %d/%d", i + 1, num_iterations);
    }
  }
  auto infer_end = std::chrono::high_resolution_clock::now();

  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      infer_end - infer_start)
                      .count();
  double avg_ms = static_cast<double>(total_ms) / num_iterations;

  if (num_iterations == 1) {
    ET_LOG(Info, "Inference time: %lldms", static_cast<long long>(total_ms));
  } else {
    ET_LOG(
        Info,
        "Inference time: %lldms total, %.2fms avg (%d iterations)",
        static_cast<long long>(total_ms),
        avg_ms,
        num_iterations);
  }

  // Log granular timing breakdown (from last iteration)
  ET_LOG(
      Info,
      "Granular timing - Preprocessor: %lldms, Encoder: %lldms, Decoder: %lldms "
      "(decoder_step: %lldms, joint: %lldms, loop_overhead: %lldms)",
      static_cast<long long>(result.timing.preprocessor_ms),
      static_cast<long long>(result.timing.encoder_ms),
      static_cast<long long>(result.timing.decoder_total_ms),
      static_cast<long long>(result.timing.decoder_step_ms),
      static_cast<long long>(result.timing.joint_ms),
      static_cast<long long>(result.timing.loop_overhead_ms()));

  // Output transcription
  std::cout << "Transcribed text: " << result.text << std::endl;

  if (!timestamp_mode.enabled()) {
    return 0;
  }

  const double frame_to_seconds = runner->frame_to_seconds();

  if (timestamp_mode.segment) {
    std::cout << "\nSegment timestamps:" << std::endl;
    for (const auto& segment : result.segment_timestamps) {
      const double start = segment.start_offset * frame_to_seconds;
      const double end = segment.end_offset * frame_to_seconds;
      std::cout << start << "s - " << end << "s : " << segment.text
                << std::endl;
    }
  }

  if (timestamp_mode.word) {
    std::cout << "\nWord timestamps:" << std::endl;
    for (const auto& word : result.word_timestamps) {
      const double start = word.start_offset * frame_to_seconds;
      const double end = word.end_offset * frame_to_seconds;
      std::cout << start << "s - " << end << "s : " << word.text << std::endl;
    }
  }

  if (timestamp_mode.token) {
    std::cout << "\nToken timestamps:" << std::endl;
    for (const auto& token : result.token_timestamps) {
      const double start = token.start_offset * frame_to_seconds;
      const double end = token.end_offset * frame_to_seconds;
      std::cout << start << "s - " << end << "s : " << token.decoded_text
                << std::endl;
    }
  }

  return 0;
}
