/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>

namespace qwen3_tts {

struct CodeGenerationArgs {
  std::string python_executable;
  std::string helper_script;
  std::string model_id_or_path;
  std::string text;
  std::string language;
  std::string output_codes_path;
  std::string ref_audio_path;
  std::string ref_text;
  bool x_vector_only_mode = false;
  bool non_streaming_mode = false;
  int max_new_tokens = -1;
  int top_k = -1;
  float top_p = -1.0f;
  float temperature = -1.0f;
  float repetition_penalty = -1.0f;
};

class Qwen3TTSRunner {
 public:
  Qwen3TTSRunner(
      const std::string& model_path,
      const std::string& data_path = "");

  int output_sample_rate() const {
    return output_sample_rate_;
  }

  int fixed_codes_len() const {
    return fixed_codes_len_;
  }

  bool run_code_generation(const CodeGenerationArgs& args) const;

  bool read_codes_file(
      const std::string& codes_path,
      std::vector<int64_t>* codes,
      int32_t* codes_len,
      int32_t* num_quantizers) const;

  bool decode_codes(
      const std::vector<int64_t>& codes,
      int32_t codes_len,
      int32_t num_quantizers,
      std::vector<float>* waveform) const;

  bool decode_codes_file(
      const std::string& codes_path,
      std::vector<float>* waveform) const;

  bool write_wav_file(
      const std::string& output_wav_path,
      const std::vector<float>& waveform) const;

 private:
  std::unique_ptr<::executorch::extension::Module> module_;
  int output_sample_rate_ = 24000;
  int fixed_codes_len_ = -1;
};

} // namespace qwen3_tts
