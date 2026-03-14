/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "qwen3_tts_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

namespace qwen3_tts {
namespace {

using ::executorch::extension::from_blob;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\"'\"'";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

void append_optional_arg(
    std::ostringstream* oss,
    const std::string& key,
    const std::string& value) {
  if (!value.empty()) {
    (*oss) << " " << key << " " << shell_quote(value);
  }
}

template <typename T>
float to_float(T value) {
  return static_cast<float>(value);
}

template <>
float to_float<::executorch::aten::Half>(::executorch::aten::Half value) {
  return static_cast<float>(value);
}

template <>
float to_float<::executorch::aten::BFloat16>(::executorch::aten::BFloat16 value) {
  return static_cast<float>(value);
}

} // namespace

Qwen3TTSRunner::Qwen3TTSRunner(
    const std::string& model_path,
    const std::string& data_path) {
  ET_LOG(Info, "Loading model from: %s", model_path.c_str());
  if (!data_path.empty()) {
    ET_LOG(Info, "Loading data from: %s", data_path.c_str());
    module_ = std::make_unique<::executorch::extension::Module>(
        model_path, data_path, ::executorch::extension::Module::LoadMode::Mmap);
  } else {
    module_ = std::make_unique<::executorch::extension::Module>(
        model_path, ::executorch::extension::Module::LoadMode::Mmap);
  }

  auto load_error = module_->load();
  ET_CHECK_MSG(load_error == Error::Ok, "Failed to load qwen3-tts model.");

  std::vector<EValue> empty;
  auto sample_rate_result = module_->execute("output_sample_rate", empty);
  if (sample_rate_result.ok()) {
    output_sample_rate_ = static_cast<int>(sample_rate_result.get()[0].toInt());
  }
  auto fixed_len_result = module_->execute("fixed_codes_len", empty);
  if (fixed_len_result.ok()) {
    fixed_codes_len_ = static_cast<int>(fixed_len_result.get()[0].toInt());
  }

  ET_LOG(
      Info,
      "Runner output_sample_rate=%d fixed_codes_len=%d",
      output_sample_rate_,
      fixed_codes_len_);
}

bool Qwen3TTSRunner::run_code_generation(const CodeGenerationArgs& args) const {
  std::ostringstream cmd;
  cmd << shell_quote(args.python_executable) << " "
      << shell_quote(args.helper_script) << " --model-id-or-path "
      << shell_quote(args.model_id_or_path) << " --text "
      << shell_quote(args.text) << " --language " << shell_quote(args.language)
      << " --output-codes " << shell_quote(args.output_codes_path);

  append_optional_arg(&cmd, "--ref-audio", args.ref_audio_path);
  append_optional_arg(&cmd, "--ref-text", args.ref_text);
  if (args.x_vector_only_mode) {
    cmd << " --x-vector-only-mode";
  }
  if (args.non_streaming_mode) {
    cmd << " --non-streaming-mode";
  }
  if (args.max_new_tokens > 0) {
    cmd << " --max-new-tokens " << args.max_new_tokens;
  }
  if (args.top_k > 0) {
    cmd << " --top-k " << args.top_k;
  }
  if (args.top_p > 0.0f) {
    cmd << " --top-p " << args.top_p;
  }
  if (args.temperature > 0.0f) {
    cmd << " --temperature " << args.temperature;
  }
  if (args.repetition_penalty > 0.0f) {
    cmd << " --repetition-penalty " << args.repetition_penalty;
  }

  ET_LOG(Info, "Running code generation helper...");
  int rc = std::system(cmd.str().c_str());
  if (rc != 0) {
    ET_LOG(Error, "Code generation helper failed with rc=%d", rc);
    return false;
  }
  return true;
}

bool Qwen3TTSRunner::read_codes_file(
    const std::string& codes_path,
    std::vector<int64_t>* codes,
    int32_t* codes_len,
    int32_t* num_quantizers) const {
  std::ifstream in(codes_path, std::ios::binary);
  if (!in.good()) {
    ET_LOG(Error, "Could not open codes file: %s", codes_path.c_str());
    return false;
  }

  int32_t t_len = 0;
  int32_t n_q = 0;
  in.read(reinterpret_cast<char*>(&t_len), sizeof(int32_t));
  in.read(reinterpret_cast<char*>(&n_q), sizeof(int32_t));
  if (!in.good() || t_len <= 0 || n_q <= 0) {
    ET_LOG(Error, "Invalid codes header in: %s", codes_path.c_str());
    return false;
  }

  std::vector<int32_t> values(static_cast<size_t>(t_len) * static_cast<size_t>(n_q));
  in.read(
      reinterpret_cast<char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(int32_t)));
  if (!in.good()) {
    ET_LOG(Error, "Failed to read codes payload from: %s", codes_path.c_str());
    return false;
  }

  codes->resize(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    (*codes)[i] = static_cast<int64_t>(values[i]);
  }
  *codes_len = t_len;
  *num_quantizers = n_q;
  return true;
}

bool Qwen3TTSRunner::decode_codes(
    const std::vector<int64_t>& codes,
    int32_t codes_len,
    int32_t num_quantizers,
    std::vector<float>* waveform) const {
  int32_t effective_len = codes_len;
  std::vector<int64_t> effective_codes = codes;
  if (fixed_codes_len_ > 0) {
    if (codes_len > fixed_codes_len_) {
      ET_LOG(
          Error,
          "codes_len (%d) exceeds fixed export length (%d). Re-export with larger --fixed-codes-len.",
          static_cast<int>(codes_len),
          fixed_codes_len_);
      return false;
    }
    if (codes_len < fixed_codes_len_) {
      effective_len = fixed_codes_len_;
      effective_codes.resize(
          static_cast<size_t>(fixed_codes_len_) * static_cast<size_t>(num_quantizers),
          static_cast<int64_t>(-1));
    }
  }

  auto codes_tensor = from_blob(
      effective_codes.data(),
      {1, effective_len, num_quantizers},
      ::executorch::aten::ScalarType::Long);

  auto result =
      module_->execute("decode_codes", std::vector<EValue>{*codes_tensor});
  if (!result.ok()) {
    ET_LOG(Error, "decode_codes execution failed.");
    return false;
  }
  auto outputs = result.get();
  if (outputs.size() < 2 || !outputs[0].isTensor() || !outputs[1].isTensor()) {
    ET_LOG(Error, "Unexpected decode_codes outputs.");
    return false;
  }

  auto wav_tensor = outputs[0].toTensor();
  auto len_tensor = outputs[1].toTensor();
  int64_t wav_len = len_tensor.const_data_ptr<int64_t>()[0];
  if (wav_len <= 0) {
    ET_LOG(Error, "Decoded waveform length is non-positive.");
    return false;
  }

  const int64_t total_samples = wav_tensor.size(wav_tensor.dim() - 1);
  const int64_t used_samples = std::min(wav_len, total_samples);
  waveform->resize(static_cast<size_t>(used_samples));

  if (wav_tensor.scalar_type() == ::executorch::aten::ScalarType::Float) {
    const float* src = wav_tensor.const_data_ptr<float>();
    std::copy(src, src + used_samples, waveform->begin());
  } else if (wav_tensor.scalar_type() == ::executorch::aten::ScalarType::Half) {
    const auto* src = wav_tensor.const_data_ptr<::executorch::aten::Half>();
    for (int64_t i = 0; i < used_samples; ++i) {
      (*waveform)[static_cast<size_t>(i)] = to_float(src[i]);
    }
  } else if (
      wav_tensor.scalar_type() == ::executorch::aten::ScalarType::BFloat16) {
    const auto* src = wav_tensor.const_data_ptr<::executorch::aten::BFloat16>();
    for (int64_t i = 0; i < used_samples; ++i) {
      (*waveform)[static_cast<size_t>(i)] = to_float(src[i]);
    }
  } else {
    ET_LOG(
        Error,
        "Unsupported waveform dtype: %d",
        static_cast<int>(wav_tensor.scalar_type()));
    return false;
  }

  return true;
}

bool Qwen3TTSRunner::decode_codes_file(
    const std::string& codes_path,
    std::vector<float>* waveform) const {
  std::vector<int64_t> flat_codes;
  int32_t codes_len = 0;
  int32_t num_quantizers = 0;
  if (!read_codes_file(codes_path, &flat_codes, &codes_len, &num_quantizers)) {
    return false;
  }
  return decode_codes(flat_codes, codes_len, num_quantizers, waveform);
}

bool Qwen3TTSRunner::write_wav_file(
    const std::string& output_wav_path,
    const std::vector<float>& waveform) const {
  std::ofstream out(output_wav_path, std::ios::binary);
  if (!out.good()) {
    ET_LOG(Error, "Could not open output wav path: %s", output_wav_path.c_str());
    return false;
  }

  const uint16_t num_channels = 1;
  const uint16_t bits_per_sample = 16;
  const uint32_t sample_rate = static_cast<uint32_t>(output_sample_rate_);
  const uint32_t byte_rate =
      sample_rate * num_channels * (bits_per_sample / 8U);
  const uint16_t block_align = num_channels * (bits_per_sample / 8U);
  const uint32_t data_bytes =
      static_cast<uint32_t>(waveform.size() * sizeof(int16_t));

  out.write("RIFF", 4);
  const uint32_t riff_chunk_size = 36U + data_bytes;
  out.write(reinterpret_cast<const char*>(&riff_chunk_size), 4);
  out.write("WAVE", 4);

  out.write("fmt ", 4);
  const uint32_t fmt_chunk_size = 16;
  out.write(reinterpret_cast<const char*>(&fmt_chunk_size), 4);
  const uint16_t audio_format = 1;
  out.write(reinterpret_cast<const char*>(&audio_format), 2);
  out.write(reinterpret_cast<const char*>(&num_channels), 2);
  out.write(reinterpret_cast<const char*>(&sample_rate), 4);
  out.write(reinterpret_cast<const char*>(&byte_rate), 4);
  out.write(reinterpret_cast<const char*>(&block_align), 2);
  out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

  out.write("data", 4);
  out.write(reinterpret_cast<const char*>(&data_bytes), 4);
  for (float sample : waveform) {
    const float clipped = std::max(-1.0f, std::min(1.0f, sample));
    const int16_t pcm = static_cast<int16_t>(std::lrint(clipped * 32767.0f));
    out.write(reinterpret_cast<const char*>(&pcm), sizeof(int16_t));
  }

  return out.good();
}

} // namespace qwen3_tts
