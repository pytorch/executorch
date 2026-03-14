/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <filesystem>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/runtime/platform/log.h>

#include "qwen3_tts_runner.h"

DEFINE_string(model_path, "model.pte", "Path to qwen3-tts decoder model (.pte).");
DEFINE_string(
    data_path,
    "",
    "Path to optional data file (.ptd) for delegate data.");
DEFINE_string(
    codes_path,
    "",
    "Path to pre-generated codec ids (.bin). If omitted, helper script is used.");
DEFINE_string(output_wav, "output.wav", "Path to output wav file.");

DEFINE_string(
    text,
    "",
    "Text for synthesis. Required when --codes_path is not provided.");
DEFINE_string(language, "English", "Language used for generation helper.");
DEFINE_string(
    model_id_or_path,
    "",
    "Model id/path used by the generation helper (required when --codes_path is not provided).");
DEFINE_string(
    helper_script,
    "examples/models/qwen3-tts/generate_codes.py",
    "Path to Python helper script that generates codec ids.");
DEFINE_string(
    python_executable,
    "python",
    "Python executable used to run the helper script.");
DEFINE_string(ref_audio, "", "Optional reference audio for voice cloning.");
DEFINE_string(ref_text, "", "Optional reference text for voice cloning.");
DEFINE_bool(x_vector_only_mode, false, "Use x-vector-only mode for voice clone.");
DEFINE_bool(
    non_streaming_mode,
    false,
    "Forward non-streaming text mode to helper generation.");

DEFINE_int32(max_new_tokens, -1, "Optional max_new_tokens forwarded to helper.");
DEFINE_int32(top_k, -1, "Optional top_k forwarded to helper.");
DEFINE_double(top_p, -1.0, "Optional top_p forwarded to helper.");
DEFINE_double(temperature, -1.0, "Optional temperature forwarded to helper.");
DEFINE_double(
    repetition_penalty,
    -1.0,
    "Optional repetition_penalty forwarded to helper.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  qwen3_tts::Qwen3TTSRunner runner(FLAGS_model_path, FLAGS_data_path);

  std::string codes_path = FLAGS_codes_path;
  std::filesystem::path tmp_codes;
  if (codes_path.empty()) {
    if (FLAGS_text.empty()) {
      ET_LOG(Error, "Either --codes_path or --text must be provided.");
      return 1;
    }
    if (FLAGS_model_id_or_path.empty()) {
      ET_LOG(
          Error,
          "--model_id_or_path is required when --codes_path is not provided.");
      return 1;
    }

    tmp_codes = std::filesystem::temp_directory_path() /
        "qwen3_tts_codegen_codes.bin";
    codes_path = tmp_codes.string();

    qwen3_tts::CodeGenerationArgs helper_args;
    helper_args.python_executable = FLAGS_python_executable;
    helper_args.helper_script = FLAGS_helper_script;
    helper_args.model_id_or_path = FLAGS_model_id_or_path;
    helper_args.text = FLAGS_text;
    helper_args.language = FLAGS_language;
    helper_args.output_codes_path = codes_path;
    helper_args.ref_audio_path = FLAGS_ref_audio;
    helper_args.ref_text = FLAGS_ref_text;
    helper_args.x_vector_only_mode = FLAGS_x_vector_only_mode;
    helper_args.non_streaming_mode = FLAGS_non_streaming_mode;
    helper_args.max_new_tokens = FLAGS_max_new_tokens;
    helper_args.top_k = FLAGS_top_k;
    helper_args.top_p = static_cast<float>(FLAGS_top_p);
    helper_args.temperature = static_cast<float>(FLAGS_temperature);
    helper_args.repetition_penalty = static_cast<float>(FLAGS_repetition_penalty);

    if (!runner.run_code_generation(helper_args)) {
      return 1;
    }
  }

  std::vector<float> waveform;
  if (!runner.decode_codes_file(codes_path, &waveform)) {
    return 1;
  }

  if (!runner.write_wav_file(FLAGS_output_wav, waveform)) {
    ET_LOG(Error, "Failed to write wav output: %s", FLAGS_output_wav.c_str());
    return 1;
  }

  ET_LOG(
      Info,
      "Wrote %zu samples at %d Hz to %s",
      waveform.size(),
      runner.output_sample_rate(),
      FLAGS_output_wav.c_str());
  return 0;
}
