/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
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

#include "decode.h"
#include "timestamp_utils.h"
#include "tokenizer_utils.h"
#include "types.h"

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/llm/tokenizers/third-party/llama.cpp-unicode/include/unicode.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#ifdef ET_BUILD_METAL
#include <executorch/backends/apple/metal/runtime/stats.h>
#endif

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

// get_input_scalar_type and greedy_decode_executorch are in decode.h

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize stats for benchmarking
  ::executorch::extension::llm::Stats stats;
  stats.model_load_start_ms = ::executorch::extension::llm::time_in_ms();

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

  // Load all methods upfront so model_load_time captures the real cost.
  // With Mmap load mode, model->load() only sets up memory mappings;
  // the actual data is paged in lazily when methods are first loaded.
  const std::vector<std::string> required_methods = {
      "preprocessor", "encoder", "decoder_step", "joint"};
  for (const auto& method : required_methods) {
    auto method_load_error = model->load_method(method);
    if (method_load_error != Error::Ok) {
      ET_LOG(Error, "Failed to load method: %s", method.c_str());
      return 1;
    }
  }
  stats.model_load_end_ms = ::executorch::extension::llm::time_in_ms();
  stats.inference_start_ms = ::executorch::extension::llm::time_in_ms();

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

  // Check expected dtype for encoder input and convert if needed
  auto encoder_method_meta_result = model->method_meta("encoder");
  if (!encoder_method_meta_result.ok()) {
    ET_LOG(Error, "Failed to get encoder method metadata.");
    return 1;
  }
  auto encoder_method_meta = encoder_method_meta_result.get();

  ::executorch::aten::ScalarType expected_dtype =
      ::executorch::aten::ScalarType::Float;
  if (encoder_method_meta.num_inputs() > 0) {
    auto input_meta_result = encoder_method_meta.input_tensor_meta(0);
    if (input_meta_result.error() == Error::Ok) {
      expected_dtype = input_meta_result.get().scalar_type();
    }
  }

  ::executorch::extension::TensorPtr mel_input;
  if (mel.scalar_type() != expected_dtype) {
    auto mel_ptr = ::executorch::extension::clone_tensor_ptr(mel);
    if (expected_dtype == ::executorch::aten::ScalarType::Half) {
      ET_LOG(
          Info,
          "Converting mel from %s to Half",
          ::executorch::runtime::toString(mel.scalar_type()));
      auto convert_result =
          ::executorch::extension::llm::convert_to_half(mel_ptr);
      if (!convert_result.ok()) {
        ET_LOG(Error, "Failed to convert mel to Half.");
        return 1;
      }
      mel_input = std::move(convert_result.get());
    } else if (expected_dtype == ::executorch::aten::ScalarType::BFloat16) {
      ET_LOG(
          Info,
          "Converting mel from %s to BFloat16",
          ::executorch::runtime::toString(mel.scalar_type()));
      auto convert_result =
          ::executorch::extension::llm::convert_to_bfloat16(mel_ptr);
      if (!convert_result.ok()) {
        ET_LOG(Error, "Failed to convert mel to BFloat16.");
        return 1;
      }
      mel_input = std::move(convert_result.get());
    }
  }

  ET_LOG(Info, "Running encoder...");
  auto enc_result = mel_input
      ? model->execute(
            "encoder",
            std::vector<::executorch::runtime::EValue>{*mel_input, *mel_len})
      : model->execute(
            "encoder",
            std::vector<::executorch::runtime::EValue>{mel, *mel_len});
  if (!enc_result.ok()) {
    ET_LOG(Error, "Encoder forward failed.");
    return 1;
  }
  stats.prompt_eval_end_ms = ::executorch::extension::llm::time_in_ms();

  auto& enc_outputs = enc_result.get();
  auto f_proj = enc_outputs[0].toTensor(); // [B, T, joint_hidden]
  int64_t encoded_len = enc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];

  ET_LOG(
      Info,
      "Encoder output (f_proj) shape: [%ld, %ld, %ld], len=%ld",
      static_cast<long>(f_proj.sizes()[0]),
      static_cast<long>(f_proj.sizes()[1]),
      static_cast<long>(f_proj.sizes()[2]),
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
  auto decoded_tokens = parakeet::greedy_decode_executorch(
      *model,
      f_proj,
      encoded_len,
      blank_id,
      num_rnn_layers,
      pred_hidden,
      10,
      &stats);

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

  // Record inference end time and token counts
  stats.inference_end_ms = ::executorch::extension::llm::time_in_ms();
  stats.num_prompt_tokens =
      encoded_len; // Use encoder output length as "prompt" tokens
  stats.num_generated_tokens = static_cast<int64_t>(decoded_tokens.size());

  // Print PyTorchObserver stats for benchmarking
  ::executorch::extension::llm::print_report(stats);

#ifdef ET_BUILD_METAL
  executorch::backends::metal::print_metal_backend_stats();
#endif // ET_BUILD_METAL

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
