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

// Helper to get expected scalar type for a method input
::executorch::runtime::Result<::executorch::aten::ScalarType>
get_input_scalar_type(
    Module& model,
    const char* method_name,
    size_t input_index) {
  auto method_meta_result = model.method_meta(method_name);
  if (!method_meta_result.ok()) {
    ET_LOG(Error, "Failed to get method metadata for %s", method_name);
    return method_meta_result.error();
  }
  auto method_meta = method_meta_result.get();
  if (method_meta.num_inputs() <= input_index) {
    ET_LOG(
        Error,
        "Method %s has %zu inputs, but requested index %zu",
        method_name,
        method_meta.num_inputs(),
        input_index);
    return ::executorch::runtime::Error::InvalidArgument;
  }
  auto input_meta_result = method_meta.input_tensor_meta(input_index);
  if (input_meta_result.error() != ::executorch::runtime::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to get input tensor metadata for %s[%zu]",
        method_name,
        input_index);
    return input_meta_result.error();
  }
  return input_meta_result.get().scalar_type();
}

std::vector<Token> greedy_decode_executorch(
    Module& model,
    const ::executorch::aten::Tensor& f_proj,
    int64_t encoder_len,
    int64_t blank_id,
    int64_t num_rnn_layers = 2,
    int64_t pred_hidden = 640,
    int64_t max_symbols_per_step = 10) {
  std::vector<Token> hypothesis;

  // Shape: [1, T, joint_hidden]
  size_t proj_dim = static_cast<size_t>(f_proj.sizes()[2]);

  // Get expected dtype for decoder_step h and c inputs (indices 1 and 2)
  auto h_dtype_result = get_input_scalar_type(model, "decoder_step", 1);
  if (!h_dtype_result.ok()) {
    return hypothesis;
  }
  auto c_dtype_result = get_input_scalar_type(model, "decoder_step", 2);
  if (!c_dtype_result.ok()) {
    return hypothesis;
  }
  auto h_dtype = h_dtype_result.get();
  auto c_dtype = c_dtype_result.get();

  ET_LOG(
      Info,
      "Decoder h dtype: %s, c dtype: %s",
      ::executorch::runtime::toString(h_dtype),
      ::executorch::runtime::toString(c_dtype));

  // Calculate buffer sizes based on dtype
  size_t h_elem_size = ::executorch::runtime::elementSize(h_dtype);
  size_t c_elem_size = ::executorch::runtime::elementSize(c_dtype);
  size_t num_elements =
      static_cast<size_t>(num_rnn_layers) * static_cast<size_t>(pred_hidden);

  // Initialize LSTM state with zeros (using byte buffers for dtype flexibility)
  std::vector<uint8_t> h_data(num_elements * h_elem_size, 0);
  std::vector<uint8_t> c_data(num_elements * c_elem_size, 0);

  auto h = from_blob(
      h_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      h_dtype);
  auto c = from_blob(
      c_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      c_dtype);

  // Prime the decoder with SOS (= blank_id) to match NeMo TDT label-looping:
  // - SOS is defined as blank:
  //   https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L1063
  // - Predictor priming with SOS:
  //   https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L1122-L1127
  std::vector<int64_t> sos_token_data = {blank_id};
  auto sos_token = from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  auto decoder_init_result = model.execute(
      "decoder_step",
      std::vector<::executorch::runtime::EValue>{sos_token, h, c});
  if (!decoder_init_result.ok()) {
    ET_LOG(Error, "decoder_step (SOS) failed");
    return hypothesis;
  }
  auto& init_outputs = decoder_init_result.get();
  auto g_proj_init = init_outputs[0].toTensor();
  auto new_h_init = init_outputs[1].toTensor();
  auto new_c_init = init_outputs[2].toTensor();
  std::memcpy(h_data.data(), new_h_init.const_data_ptr(), h_data.size());
  std::memcpy(c_data.data(), new_c_init.const_data_ptr(), c_data.size());

  // Get expected dtype for joint inputs (f and g at indices 0 and 1)
  auto f_dtype_result = get_input_scalar_type(model, "joint", 0);
  if (!f_dtype_result.ok()) {
    return hypothesis;
  }
  auto g_dtype_result = get_input_scalar_type(model, "joint", 1);
  if (!g_dtype_result.ok()) {
    return hypothesis;
  }
  auto f_dtype = f_dtype_result.get();
  auto g_dtype = g_dtype_result.get();

  ET_LOG(
      Info,
      "Joint f dtype: %s, g dtype: %s",
      ::executorch::runtime::toString(f_dtype),
      ::executorch::runtime::toString(g_dtype));

  size_t f_elem_size = ::executorch::runtime::elementSize(f_dtype);
  size_t g_elem_size = ::executorch::runtime::elementSize(g_dtype);

  // Copy g_proj data for reuse (using byte buffer for dtype flexibility)
  size_t g_proj_num_bytes =
      static_cast<size_t>(g_proj_init.numel()) * g_elem_size;
  std::vector<uint8_t> g_proj_data(g_proj_num_bytes);
  std::memcpy(
      g_proj_data.data(), g_proj_init.const_data_ptr(), g_proj_num_bytes);

  int64_t t = 0;
  int64_t symbols_on_frame = 0;
  const uint8_t* f_proj_ptr =
      static_cast<const uint8_t*>(f_proj.const_data_ptr());
  size_t f_t_num_bytes = proj_dim * f_elem_size;

  // Scan over encoder output
  while (t < encoder_len) {
    // Get encoder frame at time t: f_proj[:, t:t+1, :]
    std::vector<uint8_t> f_t_data(f_t_num_bytes);
    std::memcpy(
        f_t_data.data(),
        f_proj_ptr + static_cast<size_t>(t) * f_t_num_bytes,
        f_t_num_bytes);

    auto f_t = from_blob(
        f_t_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        f_dtype);

    auto g_proj = from_blob(
        g_proj_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        g_dtype);

    auto joint_result = model.execute(
        "joint", std::vector<::executorch::runtime::EValue>{f_t, g_proj});
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

      auto decoder_result = model.execute(
          "decoder_step",
          std::vector<::executorch::runtime::EValue>{token, h, c});
      if (!decoder_result.ok()) {
        ET_LOG(Error, "decoder_step failed");
        return hypothesis;
      }
      auto& outputs = decoder_result.get();
      auto new_g_proj = outputs[0].toTensor();
      auto new_h = outputs[1].toTensor();
      auto new_c = outputs[2].toTensor();

      // Update h, c, and g_proj
      std::memcpy(h_data.data(), new_h.const_data_ptr(), h_data.size());
      std::memcpy(c_data.data(), new_c.const_data_ptr(), c_data.size());
      std::memcpy(
          g_proj_data.data(), new_g_proj.const_data_ptr(), g_proj_data.size());

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

  ET_LOG(Info, "Running encoder...");
  auto enc_result = model->execute(
      "encoder", std::vector<::executorch::runtime::EValue>{mel, mel_len});
  if (!enc_result.ok()) {
    ET_LOG(Error, "Encoder forward failed.");
    return 1;
  }
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

  stats.prompt_eval_end_ms = ::executorch::extension::llm::time_in_ms();

  ET_LOG(Info, "Running TDT greedy decode...");
  auto decoded_tokens = greedy_decode_executorch(
      *model, f_proj, encoded_len, blank_id, num_rnn_layers, pred_hidden);

  ET_LOG(Info, "Decoded %zu tokens", decoded_tokens.size());
  stats.first_token_ms = stats.prompt_eval_end_ms; // For ASR, first token is at end of encoding

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
  stats.num_prompt_tokens = encoded_len; // Use encoder output length as "prompt" tokens
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
