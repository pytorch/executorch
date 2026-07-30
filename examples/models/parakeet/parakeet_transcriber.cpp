/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "parakeet_transcriber.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

namespace parakeet {
namespace {

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::llm::Stats;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using SteadyClock = std::chrono::steady_clock;

const std::vector<int> kDurations = {0, 1, 2, 3, 4};

double elapsed_us(
    const SteadyClock::time_point& start,
    const SteadyClock::time_point& end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

struct MethodTiming {
  double total_us = 0.0;
  double max_us = 0.0;
  int64_t calls = 0;

  void add(double sample_us) {
    total_us += sample_us;
    max_us = std::max(max_us, sample_us);
    ++calls;
  }

  double total_ms() const {
    return total_us / 1000.0;
  }

  double avg_us() const {
    return calls == 0 ? 0.0 : total_us / static_cast<double>(calls);
  }
};

struct DecodeLoopProfile {
  double total_us = 0.0;
  double frame_copy_us = 0.0;
  double state_copy_us = 0.0;
  int64_t blank_steps = 0;
  int64_t emitted_tokens = 0;
  MethodTiming joint;
  MethodTiming decoder_step;

  double accounted_us() const {
    return joint.total_us + decoder_step.total_us + frame_copy_us +
        state_copy_us;
  }

  double host_overhead_us() const {
    return std::max(0.0, total_us - accounted_us());
  }
};

std::string format_method_profile(
    const char* name,
    const MethodTiming& timing,
    const std::string& indent = "  ") {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << indent << name << ": "
      << timing.total_ms() << " ms";
  if (timing.calls > 0) {
    oss << " (" << timing.calls << " calls, " << timing.avg_us() << " us avg, "
        << timing.max_us << " us max)";
  }
  return oss.str();
}

std::string build_runtime_profile_report(
    double preprocessor_us,
    double encoder_us,
    double metadata_us,
    const DecodeLoopProfile& decode_profile) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  oss << "\nRuntime profile:\n";
  oss << "  preprocessor: " << (preprocessor_us / 1000.0) << " ms\n";
  oss << "  encoder: " << (encoder_us / 1000.0) << " ms\n";
  oss << "  metadata: " << (metadata_us / 1000.0) << " ms\n";
  oss << "  decode_loop: " << (decode_profile.total_us / 1000.0) << " ms\n";
  oss << format_method_profile("joint", decode_profile.joint) << "\n";
  oss << format_method_profile("decoder_step", decode_profile.decoder_step)
      << "\n";
  oss << "  frame_copy: " << (decode_profile.frame_copy_us / 1000.0) << " ms\n";
  oss << "  state_copy: " << (decode_profile.state_copy_us / 1000.0) << " ms\n";
  oss << "  host_overhead: " << (decode_profile.host_overhead_us() / 1000.0)
      << " ms\n";
  oss << "  blank_steps: " << decode_profile.blank_steps << "\n";
  oss << "  emitted_tokens: " << decode_profile.emitted_tokens << "\n";
  oss << "RUNTIME_PROFILE" << " preprocessor_ms=" << (preprocessor_us / 1000.0)
      << " encoder_ms=" << (encoder_us / 1000.0)
      << " metadata_ms=" << (metadata_us / 1000.0)
      << " decode_loop_ms=" << (decode_profile.total_us / 1000.0)
      << " joint_ms=" << decode_profile.joint.total_ms()
      << " joint_calls=" << decode_profile.joint.calls
      << " joint_avg_us=" << decode_profile.joint.avg_us()
      << " decoder_step_ms=" << decode_profile.decoder_step.total_ms()
      << " decoder_step_calls=" << decode_profile.decoder_step.calls
      << " decoder_step_avg_us=" << decode_profile.decoder_step.avg_us()
      << " frame_copy_ms=" << (decode_profile.frame_copy_us / 1000.0)
      << " state_copy_ms=" << (decode_profile.state_copy_us / 1000.0)
      << " host_overhead_ms=" << (decode_profile.host_overhead_us() / 1000.0)
      << " blank_steps=" << decode_profile.blank_steps
      << " emitted_tokens=" << decode_profile.emitted_tokens << "\n";
  return oss.str();
}

std::string to_lower_ascii(std::string s) {
  for (char& ch : s) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return s;
}

[[noreturn]] void throw_runtime_error(const std::string& message) {
  ET_LOG(Error, "%s", message.c_str());
  throw std::runtime_error(message);
}

void emit_status(
    const StatusCallback& status_callback,
    const std::string& status) {
  if (status_callback) {
    status_callback(status);
  }
}

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

int64_t execute_int_constant(Module& model, const char* method_name) {
  std::vector<EValue> empty_inputs;
  auto result = model.execute(method_name, empty_inputs);
  if (!result.ok()) {
    throw_runtime_error(
        std::string("Failed to query model metadata method: ") + method_name);
  }
  return result.get()[0].toInt();
}

double execute_double_constant(Module& model, const char* method_name) {
  std::vector<EValue> empty_inputs;
  auto result = model.execute(method_name, empty_inputs);
  if (!result.ok()) {
    throw_runtime_error(
        std::string("Failed to query model metadata method: ") + method_name);
  }
  return result.get()[0].toDouble();
}

std::vector<Token> greedy_decode_executorch(
    Module& model,
    const ::executorch::aten::Tensor& f_proj,
    int64_t encoder_len,
    int64_t blank_id,
    int64_t num_rnn_layers,
    int64_t pred_hidden,
    int64_t max_symbols_per_step,
    Stats* stats = nullptr,
    DecodeLoopProfile* decode_profile = nullptr) {
  std::vector<Token> hypothesis;
  const auto decode_loop_start = SteadyClock::now();
  const auto finalize_profile = [&]() {
    if (!decode_profile) {
      return;
    }
    decode_profile->total_us =
        elapsed_us(decode_loop_start, SteadyClock::now());
    decode_profile->emitted_tokens = static_cast<int64_t>(hypothesis.size());
  };

  const size_t proj_dim = static_cast<size_t>(f_proj.sizes()[2]);

  auto h_dtype_result = get_input_scalar_type(model, "decoder_step", 1);
  if (!h_dtype_result.ok()) {
    finalize_profile();
    throw_runtime_error("Failed to inspect decoder_step hidden-state dtype.");
  }
  auto c_dtype_result = get_input_scalar_type(model, "decoder_step", 2);
  if (!c_dtype_result.ok()) {
    finalize_profile();
    throw_runtime_error("Failed to inspect decoder_step cell-state dtype.");
  }
  auto h_dtype = h_dtype_result.get();
  auto c_dtype = c_dtype_result.get();

  ET_LOG(
      Info,
      "Decoder h dtype: %s, c dtype: %s",
      ::executorch::runtime::toString(h_dtype),
      ::executorch::runtime::toString(c_dtype));

  const size_t h_elem_size = ::executorch::runtime::elementSize(h_dtype);
  const size_t c_elem_size = ::executorch::runtime::elementSize(c_dtype);
  const size_t num_elements =
      static_cast<size_t>(num_rnn_layers) * static_cast<size_t>(pred_hidden);

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

  std::vector<int64_t> sos_token_data = {blank_id};
  auto sos_token = from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  const auto decoder_init_start = SteadyClock::now();
  auto decoder_init_result =
      model.execute("decoder_step", std::vector<EValue>{sos_token, h, c});
  if (decode_profile) {
    decode_profile->decoder_step.add(
        elapsed_us(decoder_init_start, SteadyClock::now()));
  }
  if (!decoder_init_result.ok()) {
    finalize_profile();
    throw_runtime_error("decoder_step (SOS) failed");
  }
  auto& init_outputs = decoder_init_result.get();
  auto g_proj_init = init_outputs[0].toTensor();
  auto new_h_init = init_outputs[1].toTensor();
  auto new_c_init = init_outputs[2].toTensor();
  const auto init_state_copy_start = SteadyClock::now();
  std::memcpy(h_data.data(), new_h_init.const_data_ptr(), h_data.size());
  std::memcpy(c_data.data(), new_c_init.const_data_ptr(), c_data.size());
  if (decode_profile) {
    decode_profile->state_copy_us +=
        elapsed_us(init_state_copy_start, SteadyClock::now());
  }

  auto f_dtype_result = get_input_scalar_type(model, "joint", 0);
  if (!f_dtype_result.ok()) {
    finalize_profile();
    throw_runtime_error("Failed to inspect joint f dtype.");
  }
  auto g_dtype_result = get_input_scalar_type(model, "joint", 1);
  if (!g_dtype_result.ok()) {
    finalize_profile();
    throw_runtime_error("Failed to inspect joint g dtype.");
  }
  auto f_dtype = f_dtype_result.get();
  auto g_dtype = g_dtype_result.get();

  ET_LOG(
      Info,
      "Joint f dtype: %s, g dtype: %s",
      ::executorch::runtime::toString(f_dtype),
      ::executorch::runtime::toString(g_dtype));

  const size_t f_elem_size = ::executorch::runtime::elementSize(f_dtype);
  const size_t g_elem_size = ::executorch::runtime::elementSize(g_dtype);

  const size_t g_proj_num_bytes =
      static_cast<size_t>(g_proj_init.numel()) * g_elem_size;
  std::vector<uint8_t> g_proj_data(g_proj_num_bytes);
  std::memcpy(
      g_proj_data.data(), g_proj_init.const_data_ptr(), g_proj_num_bytes);

  int64_t t = 0;
  int64_t symbols_on_frame = 0;
  const uint8_t* f_proj_ptr =
      static_cast<const uint8_t*>(f_proj.const_data_ptr());
  const size_t f_t_num_bytes = proj_dim * f_elem_size;

  while (t < encoder_len) {
    std::vector<uint8_t> f_t_data(f_t_num_bytes);
    const auto frame_copy_start = SteadyClock::now();
    std::memcpy(
        f_t_data.data(),
        f_proj_ptr + static_cast<size_t>(t) * f_t_num_bytes,
        f_t_num_bytes);
    if (decode_profile) {
      decode_profile->frame_copy_us +=
          elapsed_us(frame_copy_start, SteadyClock::now());
    }

    auto f_t = from_blob(
        f_t_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        f_dtype);

    auto g_proj = from_blob(
        g_proj_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        g_dtype);

    const auto joint_start = SteadyClock::now();
    auto joint_result =
        model.execute("joint", std::vector<EValue>{f_t, g_proj});
    if (decode_profile) {
      decode_profile->joint.add(elapsed_us(joint_start, SteadyClock::now()));
    }
    if (!joint_result.ok()) {
      finalize_profile();
      throw_runtime_error(
          "joint failed at t=" + std::to_string(static_cast<long long>(t)));
    }

    const int64_t k =
        joint_result.get()[0].toTensor().const_data_ptr<int64_t>()[0];
    const int64_t dur_idx =
        joint_result.get()[1].toTensor().const_data_ptr<int64_t>()[0];
    const int64_t dur = kDurations[dur_idx];

    if (k == blank_id) {
      if (decode_profile) {
        ++decode_profile->blank_steps;
      }
      t += std::max(dur, static_cast<int64_t>(1));
      symbols_on_frame = 0;
    } else {
      if (hypothesis.empty() && stats) {
        stats->first_token_ms = ::executorch::extension::llm::time_in_ms();
      }
      hypothesis.push_back({static_cast<TokenId>(k), t, dur});

      std::vector<int64_t> token_data = {k};
      auto token = from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      const auto decoder_step_start = SteadyClock::now();
      auto decoder_result =
          model.execute("decoder_step", std::vector<EValue>{token, h, c});
      if (decode_profile) {
        decode_profile->decoder_step.add(
            elapsed_us(decoder_step_start, SteadyClock::now()));
      }
      if (!decoder_result.ok()) {
        finalize_profile();
        throw_runtime_error("decoder_step failed");
      }
      auto& outputs = decoder_result.get();
      auto new_g_proj = outputs[0].toTensor();
      auto new_h = outputs[1].toTensor();
      auto new_c = outputs[2].toTensor();

      const auto state_copy_start = SteadyClock::now();
      std::memcpy(h_data.data(), new_h.const_data_ptr(), h_data.size());
      std::memcpy(c_data.data(), new_c.const_data_ptr(), c_data.size());
      std::memcpy(
          g_proj_data.data(), new_g_proj.const_data_ptr(), g_proj_data.size());
      if (decode_profile) {
        decode_profile->state_copy_us +=
            elapsed_us(state_copy_start, SteadyClock::now());
      }

      t += dur;

      if (dur == 0) {
        ++symbols_on_frame;
        if (symbols_on_frame >= max_symbols_per_step) {
          ++t;
          symbols_on_frame = 0;
        }
      } else {
        symbols_on_frame = 0;
      }
    }
  }

  finalize_profile();
  return hypothesis;
}

} // namespace

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

ParakeetTranscriber::ParakeetTranscriber(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& data_path) {
  model_load_start_ms_ = ::executorch::extension::llm::time_in_ms();
  ET_LOG(Info, "Loading model from: %s", model_path.c_str());
  if (!data_path.empty()) {
    ET_LOG(Info, "Loading data from: %s", data_path.c_str());
    model_ =
        std::make_unique<Module>(model_path, data_path, Module::LoadMode::Mmap);
  } else {
    model_ = std::make_unique<Module>(model_path, Module::LoadMode::Mmap);
  }

  auto model_load_error = model_->load();
  if (model_load_error != Error::Ok) {
    throw_runtime_error("Failed to load model.");
  }

  const std::vector<std::string> required_methods = {
      "preprocessor", "encoder", "decoder_step", "joint"};
  for (const auto& method : required_methods) {
    auto method_load_error = model_->load_method(method);
    if (method_load_error != Error::Ok) {
      throw_runtime_error("Failed to load method: " + method);
    }
  }

  model_load_end_ms_ = ::executorch::extension::llm::time_in_ms();

  num_rnn_layers_ = execute_int_constant(*model_, "num_rnn_layers");
  pred_hidden_ = execute_int_constant(*model_, "pred_hidden");
  vocab_size_ = execute_int_constant(*model_, "vocab_size");
  blank_id_ = execute_int_constant(*model_, "blank_id");
  sample_rate_ = execute_int_constant(*model_, "sample_rate");
  window_stride_ = execute_double_constant(*model_, "window_stride");
  encoder_subsampling_factor_ =
      execute_int_constant(*model_, "encoder_subsampling_factor");
  frame_to_seconds_ =
      window_stride_ * static_cast<double>(encoder_subsampling_factor_);

  ET_LOG(
      Info,
      "Model metadata: vocab_size=%lld, blank_id=%lld, num_rnn_layers=%lld, pred_hidden=%lld, sample_rate=%lld, window_stride=%.6f, encoder_subsampling_factor=%lld",
      static_cast<long long>(vocab_size_),
      static_cast<long long>(blank_id_),
      static_cast<long long>(num_rnn_layers_),
      static_cast<long long>(pred_hidden_),
      static_cast<long long>(sample_rate_),
      window_stride_,
      static_cast<long long>(encoder_subsampling_factor_));

  ET_LOG(Info, "Loading tokenizer from: %s", tokenizer_path.c_str());
  tokenizer_ = ::executorch::extension::llm::load_tokenizer(tokenizer_path);
  if (!tokenizer_ || !tokenizer_->is_loaded()) {
    throw_runtime_error("Failed to load tokenizer from: " + tokenizer_path);
  }

  supported_punctuation_ =
      parakeet::tokenizer_utils::derive_supported_punctuation(*tokenizer_);
  ET_LOG(
      Info,
      "Derived supported_punctuation size=%zu",
      supported_punctuation_.size());
}

TranscribeResult ParakeetTranscriber::transcribe_wav_path(
    const std::string& audio_path,
    const TranscribeConfig& config,
    StatusCallback status_callback) {
  ET_LOG(Info, "Loading audio from: %s", audio_path.c_str());
  emit_status(status_callback, "Loading recording...");
  std::vector<float> audio_data =
      ::executorch::extension::llm::load_wav_audio_data(audio_path);
  ET_LOG(Info, "Loaded %zu audio samples", audio_data.size());
  return transcribe_audio(
      audio_data.data(),
      static_cast<int64_t>(audio_data.size()),
      config,
      std::move(status_callback));
}

TranscribeResult ParakeetTranscriber::transcribe_audio(
    const float* audio_data,
    int64_t num_samples,
    const TranscribeConfig& config,
    StatusCallback status_callback) {
  Stats stats;
  stats.model_load_start_ms = model_load_start_ms_;
  stats.model_load_end_ms = model_load_end_ms_;
  stats.inference_start_ms = ::executorch::extension::llm::time_in_ms();

  auto audio_tensor = from_blob(
      const_cast<float*>(audio_data),
      {static_cast<::executorch::aten::SizesType>(num_samples)},
      ::executorch::aten::ScalarType::Float);
  std::vector<int64_t> audio_len_data = {num_samples};
  auto audio_len_tensor = from_blob(
      audio_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  ET_LOG(Info, "Running preprocessor...");
  emit_status(status_callback, "Running preprocessor...");
  double preprocessor_us = 0.0;
  const auto preprocessor_start = SteadyClock::now();
  auto proc_result = model_->execute(
      "preprocessor", std::vector<EValue>{audio_tensor, audio_len_tensor});
  preprocessor_us = elapsed_us(preprocessor_start, SteadyClock::now());
  if (!proc_result.ok()) {
    throw_runtime_error("Preprocessor forward failed.");
  }
  auto& proc_outputs = proc_result.get();
  auto mel = proc_outputs[0].toTensor();
  auto mel_len_tensor_out = proc_outputs[1].toTensor();
  int64_t mel_len_value = mel_len_tensor_out.const_data_ptr<int64_t>()[0];

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
  emit_status(status_callback, "Running encoder...");
  double encoder_us = 0.0;
  const auto encoder_start = SteadyClock::now();
  auto enc_result =
      model_->execute("encoder", std::vector<EValue>{mel, mel_len});
  encoder_us = elapsed_us(encoder_start, SteadyClock::now());
  if (!enc_result.ok()) {
    throw_runtime_error("Encoder forward failed.");
  }
  stats.prompt_eval_end_ms = ::executorch::extension::llm::time_in_ms();

  auto& enc_outputs = enc_result.get();
  auto f_proj = enc_outputs[0].toTensor();
  const int64_t encoded_len =
      enc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];

  ET_LOG(
      Info,
      "Encoder output (f_proj) shape: [%ld, %ld, %ld], len=%ld",
      static_cast<long>(f_proj.sizes()[0]),
      static_cast<long>(f_proj.sizes()[1]),
      static_cast<long>(f_proj.sizes()[2]),
      static_cast<long>(encoded_len));

  ET_LOG(Info, "Running TDT greedy decode...");
  emit_status(status_callback, "Decoding final transcript...");
  DecodeLoopProfile decode_profile;
  auto decoded_tokens = greedy_decode_executorch(
      *model_,
      f_proj,
      encoded_len,
      blank_id_,
      num_rnn_layers_,
      pred_hidden_,
      10,
      &stats,
      config.runtime_profile ? &decode_profile : nullptr);

  ET_LOG(Info, "Decoded %zu tokens", decoded_tokens.size());

  const std::string text = parakeet::tokenizer_utils::decode_token_sequence(
      decoded_tokens, *tokenizer_);

  stats.inference_end_ms = ::executorch::extension::llm::time_in_ms();
  stats.num_prompt_tokens = encoded_len;
  stats.num_generated_tokens = static_cast<int64_t>(decoded_tokens.size());

  double metadata_us = 0.0;
  if (config.runtime_profile) {
    metadata_us = 0.0;
  }

  TranscribeResult result;
  result.text = text;
  result.stats_json = ::executorch::extension::llm::stats_to_json_string(stats);
  result.frame_to_seconds = frame_to_seconds_;
  if (config.runtime_profile) {
    result.runtime_profile_report = build_runtime_profile_report(
        preprocessor_us, encoder_us, metadata_us, decode_profile);
  }

  if (!config.timestamp_output_mode.enabled()) {
    return result;
  }

  ET_LOG(Info, "Computing timestamps...");
  emit_status(status_callback, "Computing timestamps...");
  auto tokens_with_text_info =
      parakeet::timestamp_utils::get_tokens_with_text_info(
          decoded_tokens, *tokenizer_, supported_punctuation_);
  auto word_offsets = parakeet::timestamp_utils::get_words_offsets(
      tokens_with_text_info, *tokenizer_, supported_punctuation_);
  auto segment_offsets =
      parakeet::timestamp_utils::get_segment_offsets(word_offsets);

  result.token_offsets = std::move(tokens_with_text_info);
  result.word_offsets = std::move(word_offsets);
  result.segment_offsets = std::move(segment_offsets);
  return result;
}

std::optional<std::string> extract_runtime_profile_line(
    const std::optional<std::string>& report) {
  if (!report.has_value()) {
    return std::nullopt;
  }

  std::istringstream stream(*report);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.rfind("RUNTIME_PROFILE", 0) == 0) {
      return line;
    }
  }
  return std::nullopt;
}

} // namespace parakeet
