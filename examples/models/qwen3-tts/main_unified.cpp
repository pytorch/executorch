/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generated with assistance from Claude.

#include <chrono>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/runtime/platform/log.h>

#include "qwen3_tts_unified_runner.h"

DEFINE_string(
    model_path,
    "model.pte",
    "Path to unified qwen3-tts model (.pte).");
DEFINE_string(
    tokenizer_path,
    "",
    "Path to tokenizer.json (for text-to-audio mode).");
DEFINE_string(
    codes_path,
    "",
    "Path to pre-generated codec ids (.bin). "
    "If provided, runs decode-only mode.");
DEFINE_string(output_wav, "output.wav", "Path to output wav file.");
DEFINE_string(
    text,
    "",
    "Text for synthesis (requires --tokenizer_path).");
DEFINE_string(language, "English", "Language for synthesis.");

DEFINE_int32(max_new_tokens, 200, "Max codec tokens to generate.");
DEFINE_double(temperature, 1.0, "Sampling temperature.");
DEFINE_int32(top_k, -1, "Top-k sampling.");
DEFINE_double(top_p, -1.0, "Top-p sampling. Values <= 0 disable nucleus filtering.");
DEFINE_double(repetition_penalty, 1.05, "Repetition penalty for talker code_0 sampling.");
DEFINE_bool(
    trim_silence,
    true,
    "Trim leading silence from output audio.");
DEFINE_double(
    trim_threshold,
    0.005,
    "RMS threshold for silence trimming.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!FLAGS_codes_path.empty() && !FLAGS_text.empty()) {
    ET_LOG(Error, "Provide either --codes_path or --text, not both.");
    return 1;
  }
  if (FLAGS_codes_path.empty() && FLAGS_text.empty()) {
    ET_LOG(Error, "Either --codes_path or --text must be provided.");
    return 1;
  }
  if (!FLAGS_text.empty() && FLAGS_tokenizer_path.empty()) {
    ET_LOG(Error, "--text requires --tokenizer_path.");
    return 1;
  }

  auto t_start = std::chrono::steady_clock::now();

  qwen3_tts::Qwen3TTSUnifiedRunner runner(
      FLAGS_model_path, FLAGS_tokenizer_path);

  // Pre-load and warm up methods that will be used.
  if (!FLAGS_codes_path.empty()) {
    runner.warmup_decode();
  } else if (!FLAGS_text.empty()) {
    runner.warmup_all();
  }

  auto t_loaded = std::chrono::steady_clock::now();
  double load_ms = std::chrono::duration<double, std::milli>(
                       t_loaded - t_start)
                       .count();
  ET_LOG(Info, "Model loaded in %.1f ms", load_ms);

  std::vector<float> waveform;

  if (!FLAGS_codes_path.empty()) {
    // Decode-only mode: use precomputed codes.
    ET_LOG(Info, "Decode-only mode: %s", FLAGS_codes_path.c_str());
    auto t0 = std::chrono::steady_clock::now();
    if (!runner.decode_codes_file(FLAGS_codes_path, &waveform)) {
      ET_LOG(Error, "decode_codes_file failed.");
      return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double decode_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double audio_sec =
        static_cast<double>(waveform.size()) / runner.output_sample_rate();
    ET_LOG(
        Info,
        "Decoded %zu samples (%.2fs audio) in %.1f ms (%.2fx realtime)",
        waveform.size(),
        audio_sec,
        decode_ms,
        audio_sec / (decode_ms / 1000.0));
  } else if (!FLAGS_text.empty()) {
    // Full text-to-audio mode.
    qwen3_tts::SynthesizeConfig config;
    config.max_new_tokens = FLAGS_max_new_tokens;
    config.temperature = static_cast<float>(FLAGS_temperature);
    config.top_k = FLAGS_top_k;
    config.top_p = static_cast<float>(FLAGS_top_p);
    config.repetition_penalty = static_cast<float>(FLAGS_repetition_penalty);

    if (!runner.synthesize(FLAGS_text, FLAGS_language, config, &waveform)) {
      ET_LOG(Error, "Synthesis failed.");
      return 1;
    }
  }

  // Trim leading silence.
  if (FLAGS_trim_silence && !waveform.empty()) {
    float threshold = static_cast<float>(FLAGS_trim_threshold);
    size_t speech_start = 0;
    for (size_t i = 0; i < waveform.size(); ++i) {
      if (std::abs(waveform[i]) > threshold) {
        // Back up ~50ms for natural attack.
        size_t margin =
            static_cast<size_t>(0.05 * runner.output_sample_rate());
        speech_start = (i > margin) ? i - margin : 0;
        break;
      }
    }
    if (speech_start > 0) {
      double trimmed_sec =
          static_cast<double>(speech_start) / runner.output_sample_rate();
      ET_LOG(
          Info,
          "Trimmed %.2fs leading silence (%zu samples)",
          trimmed_sec,
          speech_start);
      waveform.erase(waveform.begin(), waveform.begin() + speech_start);
    }
  }

  if (!runner.write_wav_file(FLAGS_output_wav, waveform)) {
    ET_LOG(Error, "Failed to write wav: %s", FLAGS_output_wav.c_str());
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
