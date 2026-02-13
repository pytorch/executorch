/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CLI entry point for the Voxtral Realtime transcriber.
//
// Loads a .pte model, a preprocessor .pte, and a Tekken tokenizer.
// Processes a WAV file and prints transcribed text.
//
// Modes:
//   Default:     Offline transcription (full encoder, then decode)
//   --streaming: Streaming transcription (incremental mel + encoder + decode)

#include <cstdio>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/runtime/platform/log.h>

#include "voxtral_realtime_runner.h"

DEFINE_string(
    model_path,
    "model.pte",
    "Path to Voxtral Realtime model (.pte).");
DEFINE_string(tokenizer_path, "tekken.json", "Path to Tekken tokenizer file.");
DEFINE_string(preprocessor_path, "", "Path to mel preprocessor (.pte).");
DEFINE_string(audio_path, "", "Path to input audio file (.wav).");
DEFINE_double(temperature, 0.0, "Sampling temperature (0 = greedy).");
DEFINE_int32(max_new_tokens, 500, "Maximum number of tokens to generate.");
DEFINE_bool(streaming, false, "Use streaming transcription mode.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "audio_path flag must be provided.");
    return 1;
  }

  if (FLAGS_preprocessor_path.empty()) {
    ET_LOG(Error, "preprocessor_path flag must be provided.");
    return 1;
  }

  ::executorch::extension::llm::Stats stats;
  stats.model_load_start_ms = ::executorch::extension::llm::time_in_ms();

  voxtral_realtime::VoxtralRealtimeRunner runner(
      FLAGS_model_path, FLAGS_tokenizer_path, FLAGS_preprocessor_path);

  stats.model_load_end_ms = ::executorch::extension::llm::time_in_ms();
  stats.inference_start_ms = ::executorch::extension::llm::time_in_ms();

  ET_LOG(Info, "Loading audio from: %s", FLAGS_audio_path.c_str());
  auto audio_data =
      ::executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);

  voxtral_realtime::TranscribeConfig config;
  config.temperature = static_cast<float>(FLAGS_temperature);
  config.max_new_tokens = FLAGS_max_new_tokens;

  stats.num_prompt_tokens = 0;
  bool first_token = true;

  auto token_cb = [&](const std::string& piece) {
    if (first_token) {
      stats.first_token_ms = ::executorch::extension::llm::time_in_ms();
      stats.prompt_eval_end_ms = stats.first_token_ms;
      first_token = false;
    }
    ::executorch::extension::llm::safe_printf(piece.c_str());
    fflush(stdout);
  };

  int num_generated;
  if (FLAGS_streaming) {
    ET_CHECK_MSG(
        runner.is_streaming(),
        "Model was not exported with --streaming. Re-export with --streaming flag.");
    auto session = runner.create_streaming_session(config, token_cb);

    // Feed audio in 200ms chunks (simulates live microphone input).
    const int64_t chunk_size = 3200;
    for (int64_t offset = 0; offset < static_cast<int64_t>(audio_data.size());
         offset += chunk_size) {
      int64_t n = std::min(
          chunk_size, static_cast<int64_t>(audio_data.size()) - offset);
      session->feed_audio(audio_data.data() + offset, n);
    }
    num_generated = session->flush();
  } else {
    num_generated = runner.transcribe(
        audio_data.data(),
        static_cast<int64_t>(audio_data.size()),
        config,
        token_cb);
  }

  printf("\n");

  stats.num_generated_tokens = num_generated;
  stats.inference_end_ms = ::executorch::extension::llm::time_in_ms();

  ::executorch::extension::llm::print_report(stats);

  return 0;
}
